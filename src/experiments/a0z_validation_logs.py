# -*- coding: utf-8 -*-
"""A0z-1: R1 step1 — 4 条验证流的试验日志生成 + 测试侧重评估.

对 SWaT(cmhmil)/MetroPT3(iforest)/TE(cmhmil)/TE(usad) 各跑 optuna s1-s3 B240
(协议 M, 同 a0_single 口径), 保存全部 trial 的整定侧指标, 并对每个 trial
做测试侧 eval_params_M 重评估 (a0j 同口径). 层参照点一并记录.
输出: results/audit/a0z_trials_{tag}.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from a0_lib import (interleave_mask, eval_params_M, train_layer_M,  # noqa
                    search_unified_M)

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))

STREAMS = [("cmhmil", "SWaT", "SWaT_cmh"),
           ("iforest", "MetroPT3", "Metro_if"),
           ("cmhmil", "TE", "TE_cmh"),
           ("usad", "TE", "TE_usad")]
import sys as _sys
if len(_sys.argv) > 1:
    STREAMS = [x for x in STREAMS if x[2] in _sys.argv[1:]]


def load_stream(scorer, ds):
    # 修复(2026-10-31): TE 的 cmh/usad 验证流必须用与主比较相同的分段边界
    # (独立仿真 run 复位), 旧版 bounds=[(0,n)] 违背统一协议, 已作废重跑.
    if ds == "TE" and scorer in ("cmhmil", "usad"):
        from a0_lib import load_te_segments as _lts
        _, _, _, bounds, _meta = _lts()          # iforest 坐标 = 同一时间轴
        if scorer == "usad":
            import torch as _t
            _prev = _t.get_default_dtype()
            _t.set_default_dtype(_t.float32)
            from common import CACHE_DIR as _CC
            import experiments.e_r48_usad as _u
            d = np.load(_CC + "/tep_data.npz")
            X = d["X"].astype(np.float32)
            a_, b_ = int(len(X) * 0.35), int(len(X) * 0.50)
            ae1, ae2 = _u.train_usad(_u.windows(X[a_:b_])[:100000])
            sc = _u.score_stream(ae1, ae2, _u.windows(X[b_:]))
            _t.set_default_dtype(_prev)
            return (np.asarray(sc, np.float64),
                    np.asarray(d["Y"][b_:][_u.W - 1:]), 60.0, bounds)
        from scorers.adapter import get_scores
        s0, y0, dt0 = get_scores(scorer, ds)
        return np.asarray(s0, np.float64), np.asarray(y0), float(dt0), bounds
    if scorer == "usad":
        # e_r48 管线: npz 缓存 X,Y -> z-score(前35%) -> 窗16 -> 双AE 分数.
        # 种子固定 (e_r48 模块级 torch/np manual_seed), 确定性重建.
        import torch as _t
        _prev = _t.get_default_dtype()
        _t.set_default_dtype(_t.float32)   # a0_lib 全局 float64 会污染 USAD 权重
        from common import CACHE_DIR
        import experiments.e_r48_usad as u
        d = np.load(CACHE_DIR + "/tep_data.npz")
        X, Y = d["X"].astype(np.float32), d["Y"].astype(np.int8)
        nfit = int(len(X) * 0.35)
        mu = X[:nfit].mean(0, keepdims=True)
        sdv = X[:nfit].std(0, keepdims=True) + 1e-8
        Z = (X - mu) / sdv
        a, b = int(len(Z) * 0.35), int(len(Z) * 0.50)
        ae1, ae2 = u.train_usad(u.windows(Z[a:b])[:100000])
        s = u.score_stream(ae1, ae2, u.windows(Z[b:]))
        _t.set_default_dtype(_prev)
        return (np.asarray(s, np.float64),
                np.asarray(Y[b:][u.W - 1:]), 60.0, None)
    from scorers.adapter import get_scores
    s, y, dt = get_scores(scorer, ds)
    return np.asarray(s, np.float64), np.asarray(y), float(dt), None


def main():
    for scorer, ds, tag in STREAMS:
        t0 = time.time()
        s, y, dt, bounds = load_stream(scorer, ds)
        if bounds is None:
            bounds = [(0, len(s))]
        n = len(s)
        assert bounds[0][0] == 0 and bounds[-1][1] == n, (tag, bounds)
        for k in range(1, len(bounds)):
            assert bounds[k][0] == bounds[k - 1][1], (tag, 'gap', k)
        assert len(s) == len(y), (tag, len(s), len(y))
        tune_m, test_m = interleave_mask(n, dt, bounds)
        rows = []
        # 层参照
        prm, _, _ = train_layer_M(s, y, tune_m, bounds, dt, epochs=300)
        for side, m_ in [("tune", tune_m), ("test", test_m)]:
            me = eval_params_M(prm, s, y, m_, dt, bounds)
            rows.append({"seed": 0, "trial": -1, "layer": True, "side": side,
                         **{k: me[k] for k in
                            ("EF1_ov", "det_ov", "AAR_side", "FAR_side",
                             "feasible")}})
        # 搜索
        for seed in (1, 2, 3):
            picks, trials = search_unified_M(s, y, tune_m, bounds, dt, seed,
                                             budget=240, use_tpe=True)
            td = trials.to_dict("records")
            for r in td:
                p = {"theta_hi": float(np.quantile(s[tune_m], r["q"])),
                     "theta_lo": float(np.quantile(s[tune_m], r["q"]))
                     - r["mf"] * float(s[tune_m].std()),
                     "T_on": int(r["t_on"]), "T_off": int(r["t_off"])}
                mt = {"q": r["q"], "mf": r["mf"], "t_on": r["t_on"],
                      "t_off": r["t_off"], "tune_EF1": r["EF1_ov"],
                      "tune_AAR": r["AAR"], "tune_FAR": r["FAR"],
                      "tune_feas": r["feas"]}
                me_t = eval_params_M(p, s, y, tune_m, dt, bounds)
                me_e = eval_params_M(p, s, y, test_m, dt, bounds)
                rows.append({"seed": seed, "trial": r["trial"], "layer": False,
                             "side": "tune",
                             **mt,
                             "EF1_ov": me_t["EF1_ov"], "det_ov": me_t["det_ov"],
                             "AAR_side": me_t["AAR_side"],
                             "FAR_side": me_t["FAR_side"],
                             "feasible": me_t["feasible"]})
                rows.append({"seed": seed, "trial": r["trial"], "layer": False,
                             "side": "test",
                             "test_det": me_e["det_ov"],
                             "test_EF1": me_e["EF1_ov"],
                             "test_AAR": me_e["AAR_side"],
                             "test_TIA": me_e.get("TIA_side"),
                             "test_feas": me_e["feasible"]})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT, f"a0z_trials_{tag}.csv"), index=False,
                  encoding="utf-8-sig")
        print(f"[{tag}] n_trials_rows={len(df)} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
