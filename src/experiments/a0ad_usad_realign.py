# -*- coding: utf-8 -*-
"""R15 修复 2: USAD fresh 全链重做 — (a) gate 标签 t>=8h; (b) 新分数尺度重整定
(层+optuna 3 seeds, 协议 M 分段); (c) 冻结新参数评 34 fresh run.
输出: a0z_usad_gate.json(新), a0z_usad_realign_params.csv, a0z_usad_newruns.csv(重写)
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
import experiments.e_r48_usad as u  # noqa
from a0_lib import (load_te_segments, interleave_mask, eval_params_M,  # noqa
                    train_layer_M, search_unified_M, protocol_M_metrics)
from kpi.panel import KPIConfig, kpi_panel  # noqa
from tuning.hard_ops import hard_pipeline  # noqa
from datasets.tep import (TE_DIR, FAULT_FILES, FAULT_ONSET_H, N_XMEAS)  # noqa
from common import CACHE_DIR  # noqa

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))
EXT = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "TE_ext"))
OLD = r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/datasets/TE"
W = 16
FRESH = [(i, r) for i in (2, 5, 7, 10, 13) for r in (1, 2)] + \
        [(i, r) for i in (3, 6, 8, 9, 12, 14, 15, 16, 17, 18, 19, 20)
         for r in (1, 2)]


def raw_stats():
    """原始量纲 mu/sd (a0w 口径) + 原始 X 与逐点故障标签(t>=8h)."""
    Xs, Ys = [], []
    df0 = pd.read_excel(os.path.join(TE_DIR, "mode1_normal_500.xlsx"))
    feat = [c for c in df0.columns if c != "Time"]
    Xs.append(df0[feat].values.astype(np.float32))
    Ys.append(np.zeros(len(df0), dtype=np.int8))
    for f in FAULT_FILES:
        df = pd.read_excel(os.path.join(TE_DIR, f"{f}.xlsx"))
        X = df[[c for c in df.columns
                if c.startswith("XMEAS")]].values.astype(np.float32)
        t = df["Time (h)"].values
        Xs.append(X)
        Ys.append((t >= FAULT_ONSET_H).astype(np.int8))   # R15: 前 480 点正常
    Xraw = np.vstack(Xs)
    Yraw = np.concatenate(Ys)
    nfit = int(len(Xraw) * 0.35)
    mu = Xraw[:nfit].mean(0, keepdims=True)
    sd = Xraw[:nfit].std(0, keepdims=True) + 1e-8
    return Xraw, Yraw, mu, sd


def main():
    import torch as _t
    _prev = _t.get_default_dtype()
    _t.set_default_dtype(_t.float32)
    Xraw, Yraw, mu, sd = raw_stats()
    Z = (Xraw - mu) / sd
    a, b = int(len(Z) * 0.35), int(len(Z) * 0.50)
    ae1, ae2 = u.train_usad(u.windows(Z[a:b])[:100000])
    s_main = u.score_stream(ae1, ae2, u.windows(Z[b:]))
    y_main = Yraw[b:][u.W - 1:]
    _t.set_default_dtype(_prev)

    # ── G1'' 闸门: 正确标签下的确定性 + AUROC ──
    from sklearn.metrics import roc_auc_score
    s_chk = u.score_stream(ae1, ae2, u.windows(Z[b:]))
    det_ok = bool(np.array_equal(s_main, s_chk))
    auc = float(roc_auc_score(y_main, s_main))
    gate = {"deterministic_rerun": det_ok,
            "main_auroc_correct_labels": round(auc, 4),
            "note": "labels t>=8h per fault file (R15 fix; earlier gate "
                    "flattened the 480-sample healthy prefix of each fault "
                    "run, mislabeling 2880 normal points)",
            "pass": det_ok and auc > 0.5}
    import json
    json.dump(gate, open(os.path.join(OUT, "a0z_usad_gate.json"), "w"),
              indent=1)
    print("G1'':", gate)
    if not gate["pass"]:
        return

    # ── 重整定: 协议 M 分段流 (与 iforest 坐标一致) ──
    s = np.asarray(s_main, np.float64)
    y = np.asarray(y_main)
    _, _, _, bounds, _ = load_te_segments()
    assert bounds[-1][1] == len(s), (bounds[-1], len(s))
    tune_m, test_m = interleave_mask(len(s), 60.0, bounds)
    prm, _, _ = train_layer_M(s, y, tune_m, bounds, 60.0, epochs=300)
    me = eval_params_M(prm, s, y, test_m, 60.0, bounds)
    rows_prm = [{"method": "layer", **prm,
                 "test_EF1": round(me["EF1_ov"], 4),
                 "test_det": round(me["det_ov"], 4)}]
    print(f"[realign layer] EF1={me['EF1_ov']:.3f} det={me['det_ov']:.2f} "
          f"theta_hi={prm['theta_hi']:.1f}")
    for seed in (1, 2, 3):
        picks, _ = search_unified_M(s, y, tune_m, bounds, 60.0, seed,
                                    budget=240, use_tpe=True)
        p_ = picks[240]["prm"]
        m_ = eval_params_M(p_, s, y, test_m, 60.0, bounds)
        rows_prm.append({"method": f"optuna_s{seed}", **p_,
                         "test_EF1": round(m_["EF1_ov"], 4),
                         "test_det": round(m_["det_ov"], 4)})
        print(f"[realign optuna s{seed}] EF1={m_['EF1_ov']:.3f} "
              f"det={m_['det_ov']:.2f}")
    pd.DataFrame(rows_prm).to_csv(
        os.path.join(OUT, "a0z_usad_realign_params.csv"), index=False,
        encoding="utf-8-sig")

    # ── 冻结新参数评 34 fresh run ──
    rows = []
    for idv, run in FRESH:
        path = (os.path.join(OLD, f"mode1_{idv}_{run}.xlsx")
                if idv in (2, 5, 7, 10, 13)
                else os.path.join(EXT, f"mode1_{idv}_{run}.xlsx"))
        df = pd.read_excel(path)
        xcols = [c for c in df.columns if c != "Time (h)"][:41]
        assert len(xcols) == N_XMEAS
        Xr = df[xcols].values.astype(np.float32)
        t = df["Time (h)"].values
        yf = (t >= FAULT_ONSET_H).astype(np.int8)
        Zf = (Xr - mu) / sd
        sc = u.score_stream(ae1, ae2, u.windows(Zf))
        yy = yf[u.W - 1:]
        for _, r in pd.DataFrame(rows_prm).iterrows():
            alarm = hard_pipeline(sc, r["theta_hi"], r["theta_lo"],
                                  int(r["T_on"]), int(r["T_off"]))
            m = kpi_panel(alarm, yy, cfg=KPIConfig(dt=60.0))
            rows.append({"idv": idv, "run": run, "method": r["method"],
                         "EventF1": m["EventF1"],
                         "det": m["event_detection_rate"],
                         "AAR": m["AAR_per_hour"]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "a0z_usad_newruns.csv"),
                              index=False, encoding="utf-8-sig")
    d = pd.DataFrame(rows)
    lay = d[d.method == "layer"]
    opt = d[d.method.str.startswith("optuna")]
    best = opt.groupby(["idv", "run"]).det.max().mean()
    print("fresh: layer %.3f/%.3f/%.4f | opt-mean %.3f/%.3f/%.4f | "
          "best det %.3f" % (
              lay.EventF1.mean(), lay.det.mean(), lay.AAR.mean(),
              opt.EventF1.mean(), opt.det.mean(), opt.AAR.mean(), best))


if __name__ == "__main__":
    main()
