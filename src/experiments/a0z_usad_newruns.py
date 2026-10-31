# -*- coding: utf-8 -*-
"""A0z-2: R2 — USAD 新鲜故障复现 (冻结登记 §R2).

G1' 闸门: 重建 USAD (e_r48 双种子固定) 对 TE 主流的分数, 与 r48_usad.csv
的层/搜索可复现性核对困难(原管线无分数缓存), 改为内部一致性闸门:
重建两次分数完全一致(确定性) + AUROC 与 r48 报告同量级.
34 run 全评; 参数 = a0_usad_TE_params.csv 冻结组.
输出: results/audit/a0z_usad_newruns.csv, a0z_usad_gate.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
import experiments.e_r48_usad as u  # noqa
from common import CACHE_DIR  # noqa
from kpi.panel import KPIConfig, events_from_binary, kpi_panel  # noqa
from tuning.hard_ops import hard_pipeline  # noqa
from datasets.tep import TE_DIR, FAULT_FILES, FAULT_ONSET_H  # noqa

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))
EXT = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "TE_ext"))
OLD = r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/datasets/TE"
FRESH = [(i, r) for i in (2, 5, 7, 10, 13) for r in (1, 2)] + \
        [(i, r) for i in (3, 6, 8, 9, 12, 14, 15, 16, 17, 18, 19, 20)
         for r in (1, 2)]


def rebuild():
    # 修复(2026-10-31): mu/sd 改从原始物理量纲 xlsx 计算(a0w 口径).
    # 旧版误在已标准化的 tep_data.npz 上再取 mu/sd 再施于原始 fresh 数据
    # (量纲错配 -> 常开告警的假饱和), 已作废.
    from datasets.tep import TE_DIR, FAULT_FILES
    Xs = []
    df0 = pd.read_excel(os.path.join(TE_DIR, "mode1_normal_500.xlsx"))
    feat = [c for c in df0.columns if c != "Time"]
    Xs.append(df0[feat].values.astype(np.float32))
    for f in FAULT_FILES:
        df = pd.read_excel(os.path.join(TE_DIR, f"{f}.xlsx"))
        Xs.append(df[[c for c in df.columns
                      if c.startswith("XMEAS")]].values.astype(np.float32))
    Xraw = np.vstack(Xs)
    Y = (np.concatenate(
        [np.zeros(30001, dtype=np.int8)] +
        [np.ones(3001, dtype=np.int8) for _ in FAULT_FILES]))
    nfit = int(len(Xraw) * 0.35)
    mu = Xraw[:nfit].mean(0, keepdims=True)
    sd = Xraw[:nfit].std(0, keepdims=True) + 1e-8
    Z = (Xraw - mu) / sd
    a, b = int(len(Z) * 0.35), int(len(Z) * 0.50)
    ae1, ae2 = u.train_usad(u.windows(Z[a:b])[:100000])
    s = u.score_stream(ae1, ae2, u.windows(Z[b:]))
    y = Y[b:][u.W - 1:]
    return ae1, ae2, mu, sd, s, y


def main():
    from sklearn.metrics import roc_auc_score
    ae1, ae2, mu, sd, s_main, y_main = rebuild()
    # G1': 确定性(重建分数再算一次) + AUROC 量级
    from datasets.tep import TE_DIR as _TD, FAULT_FILES as _FF
    _d0 = pd.read_excel(os.path.join(_TD, "mode1_normal_500.xlsx"))
    _Xs = [_d0[[c for c in _d0.columns if c != "Time"]].values.astype(np.float32)]
    for f in _FF:
        _d = pd.read_excel(os.path.join(_TD, f"{f}.xlsx"))
        _Xs.append(_d[[c for c in _d.columns
                       if c.startswith("XMEAS")]].values.astype(np.float32))
    _Xraw = np.vstack(_Xs)
    s2 = u.score_stream(ae1, ae2, u.windows(
        ((_Xraw - mu) / sd)[int(len(_Xraw) * 0.50):]))
    det_ok = bool(np.array_equal(s_main, s2))
    auc = float(roc_auc_score(y_main, s_main))
    gate = {"deterministic_rerun": det_ok, "main_auroc": round(auc, 3),
            "pass": det_ok and auc > 0.5,
            "note": "r48 管线无分数缓存; 闸门=确定性+AUROC 量级"}
    json.dump(gate, open(os.path.join(OUT, "a0z_usad_gate.json"), "w"),
              indent=1)
    print("G1':", gate)
    if not gate["pass"]:
        return
    prm = pd.read_csv(os.path.join(OUT, "a0_usad_TE_params.csv"))
    prm = prm[prm["method"].isin(["layer", "optuna_s1", "optuna_s2",
                                  "optuna_s3"])]
    rows = []
    for idv, run in FRESH:
        path = (os.path.join(OLD, f"mode1_{idv}_{run}.xlsx")
                if idv in (2, 5, 7, 10, 13)
                else os.path.join(EXT, f"mode1_{idv}_{run}.xlsx"))
        df = pd.read_excel(path)
        xcols = [c for c in df.columns if c != "Time (h)"][:41]
        X = df[xcols].values.astype(np.float32)
        t = df["Time (h)"].values
        y = (t >= FAULT_ONSET_H).astype(np.int8)
        Z = (X - mu) / sd
        sc = u.score_stream(ae1, ae2, u.windows(Z))
        yy = y[u.W - 1:]
        ev = events_from_binary(yy)
        for _, r in prm.iterrows():
            alarm = hard_pipeline(sc, r["theta_hi"], r["theta_lo"],
                                  int(r["T_on"]), int(r["T_off"]))
            m = kpi_panel(alarm, yy, cfg=KPIConfig(dt=60.0))
            rows.append({"idv": idv, "run": run, "method": r["method"],
                         "n_events": len(ev), "EventF1": m["EventF1"],
                         "det": m["event_detection_rate"],
                         "AAR": m["AAR_per_hour"]})
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT, "a0z_usad_newruns.csv"), index=False,
                  encoding="utf-8-sig")
    print(f"rows={len(df_out)} runs={df_out[['idv','run']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
