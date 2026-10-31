# -*- coding: utf-8 -*-
"""A0.4: TE-iForest 端到端新协议核验 (设计文档 v2 全规则).

运行: 层×3重复 (运行方差) | Optuna/Random×5seeds×嵌套预算60/120/240 |
grid-delay | gyasi. 评估: 协议M 测试侧 + 整定侧可行性 + 软硬差(边沿).
输出: results/audit/a0_te_iforest.csv / a0_params.csv / a0_softgap.csv
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from a0_lib import (load_te_segments, interleave_mask, hard_chain_segmented,  # noqa
                    protocol_M_metrics, eval_params_M, train_layer_M,
                    search_unified_M, R_MAX, F_MAX)
from kpi.panel import events_from_binary  # noqa: E402

OUT_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    s, y, dt, bounds, meta = load_te_segments()
    tune_m, test_m = interleave_mask(len(s), dt, bounds)
    print(f"流长 {len(s)}, 段 {len(bounds)}, 整定样本 {tune_m.sum()}, "
          f"测试样本 {test_m.sum()}")
    ev_tune = [(a, b) for a, b in events_from_binary(y) if tune_m[a]]
    ev_test = [(a, b) for a, b in events_from_binary(y) if test_m[a]]
    print(f"整定侧事件 {len(ev_tune)}: {[(a, b) for a, b in ev_tune]}")
    print(f"测试侧事件 {len(ev_test)}: {[(a, b) for a, b in ev_test]}")

    rows, prm_rows, softgap_rows = [], [], []

    # ---- 层: 3 次独立重复 (运行方差, A0.3 核验4 结论) ----
    for rep in range(3):
        t0 = time.time()
        prm, hist, m_max = train_layer_M(s, y, tune_m, bounds, dt, epochs=300,
                                         verbose=(rep == 0))
        wall = time.time() - t0
        mt = eval_params_M(prm, s, y, tune_m, dt, bounds)
        me = eval_params_M(prm, s, y, test_m, dt, bounds)
        me.update({"method": f"layer_r{rep}", "wall_s": wall,
                   "train_EF1": mt["EF1_ov"], "train_feas": mt["feasible"]})
        rows.append(me)
        prm_rows.append({"method": f"layer_r{rep}", **prm, "m_max": m_max})
        print(f"[layer r{rep}] {wall:.0f}s tuneEF1={mt['EF1_ov']:.3f} "
              f"feas={mt['feasible']} | test EF1_ov={me['EF1_ov']:.3f} "
              f"EF1_iou={me['EF1_iou01']:.3f} det={me['det_ov']:.2f} "
              f"AAR={me['AAR_side']:.1f} FAR={me['FAR_side']:.4f}")
        # 软硬差 (A0.3-2): 终态参数下, 全流 + 报警边沿
        import torch
        from tuning.soft_ops import soft_pipeline
        from tuning.constrained import TuningParams
        a_hard = hard_chain_segmented(s, prm, bounds)
        gaps = []
        for (sg0, sg1) in bounds:
            tt = torch.as_tensor(s[sg0:sg1])
            model_gap = TuningParams(torch.as_tensor(s[tune_m]),
                                      m_max=m_max)  # 只为复用参数化
            # 直接以终态参数构造软链 (温度取退火地板)
            temp_s = 0.05 * float(np.std(s[tune_m]))
            a_soft = soft_pipeline(tt, prm["theta_hi"], prm["theta_lo"],
                                   float(prm["T_on"]), float(prm["T_off"]),
                                   temp_s, 0.02,
                                   K_on=24, K_off=48).numpy()
            gaps.append(np.abs(a_soft - a_hard[sg0:sg1]))
        g = np.concatenate(gaps)
        edges = rising = np.flatnonzero((a_hard[1:] == 1) & (a_hard[:-1] == 0)) + 1
        edge_gap = float(np.mean([g[i - 3:i + 4].max() for i in edges])) \
            if len(edges) else float("nan")
        softgap_rows.append({"method": f"layer_r{rep}",
                             "mean_gap": float(g.mean()),
                             "max_gap": float(g.max()),
                             "edge_gap_mean7": edge_gap,
                             "n_edges": len(edges)})

    # ---- 统一搜索: Optuna(TPE 约束感知) / Random, seeds 1-5 ----
    for method, use_tpe in [("optuna", True), ("random", False)]:
        for seed in range(1, 6):
            t0 = time.time()
            picks, trials = search_unified_M(s, y, tune_m, bounds, dt, seed,
                                             budget=240, use_tpe=use_tpe)
            wall = time.time() - t0
            trials.to_csv(os.path.join(
                OUT_DIR, f"a0_trials_{method}_s{seed}.csv"), index=False)
            for B, p in picks.items():
                me = eval_params_M(p["prm"], s, y, test_m, dt, bounds)
                me.update({"method": f"{method}_s{seed}_B{B}", "wall_s": wall,
                           "budget": B, "tune_feasible_pick": p["feasible"]})
                rows.append(me)
                prm_rows.append({"method": f"{method}_s{seed}_B{B}", **p["prm"]})
            print(f"[{method} s{seed}] {wall:.0f}s "
                  f"B60 EF1={picks[60]['prm'] and eval_params_M(picks[60]['prm'], s, y, test_m, dt, bounds)['EF1_ov']:.3f} "
                  f"B240 EF1={eval_params_M(picks[240]['prm'], s, y, test_m, dt, bounds)['EF1_ov']:.3f}")

    # ---- 经典基线 (域内): grid-delay; gyasi (解析, 各自域说明) ----
    sd = float(s[tune_m].std())
    best = None
    for q in [0.95, 0.99]:
        for t_on in [1, 5, 10]:
            for t_off in [1, 10, 30]:
                prm = {"theta_hi": float(np.quantile(s[tune_m], q)),
                       "theta_lo": float(np.quantile(s[tune_m], q)),
                       "T_on": t_on, "T_off": t_off}
                mt = eval_params_M(prm, s, y, tune_m, dt, bounds)
                key = (0 if mt["feasible"] else 1, -mt["EF1_ov"])
                if best is None or key < best[0]:
                    best = (key, prm, mt)
    me = eval_params_M(best[1], s, y, test_m, dt, bounds)
    me.update({"method": "grid-delay", "train_EF1": best[2]["EF1_ov"],
               "train_feas": best[2]["feasible"]})
    rows.append(me)
    prm_rows.append({"method": "grid-delay", **best[1]})
    print(f"[grid-delay] test EF1_ov={me['EF1_ov']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "a0_te_iforest.csv"),
              index=False, encoding="utf-8-sig")
    pd.DataFrame(prm_rows).to_csv(os.path.join(OUT_DIR, "a0_params.csv"),
                                  index=False, encoding="utf-8-sig")
    pd.DataFrame(softgap_rows).to_csv(os.path.join(OUT_DIR, "a0_softgap.csv"),
                                      index=False, encoding="utf-8-sig")
    cols = ["method", "EF1_ov", "EF1_iou01", "det_ov", "AAR_side", "FAR_side",
            "feasible", "MAD_s", "n_missed", "TIA_side", "n_stale_1h"]
    print("\n==== A0.4 主表 (测试侧, 协议 M) ====")
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
