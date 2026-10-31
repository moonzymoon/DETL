# -*- coding: utf-8 -*-
"""B-2: 滚动再整定 (R2.3): 时序 K 折, 每步只用过去标签整定, 当前折测试.

TE 分数流 (段界复位) 顺序切 5 折 (每折 ~80h, 含正常尾段+故障段).
每折: 层 + Optuna(s1) 在 [0, 折起点) 上整定, 本折前向评估.
输出: results/audit/a0c_rolling.csv
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from a0_lib import (load_te_segments, eval_params_M, train_layer_M,
                    search_unified_M)
from kpi.panel import events_from_binary  # noqa

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))
K = 5


def side_metrics(prm, s, y, dt, bounds, i0, i1):
    """折内 [i0,i1) 前向评估 (报警链全程, 指标只在折内样本)."""
    m = eval_params_M(prm, s, y, np.arange(len(s)) >= 0, dt, bounds)
    return m


def fold_eval(prm, s, y, dt, bounds, i0, i1):
    """折内指标: 以折内样本为 side (报警链仍全轴段界复位)."""
    side = np.zeros(len(s), dtype=bool)
    side[i0:i1] = True
    import a0_lib
    alarm = a0_lib.hard_chain_segmented(s, prm, bounds)
    return a0_lib.protocol_M_metrics(alarm, y, side, dt)


def main():
    s, y, dt, bounds, meta = load_te_segments()
    n = len(s)
    edges = np.linspace(0, n, K + 1).astype(int)
    rows = []
    for k in range(1, K):
        i0, i1 = edges[k], edges[k + 1]
        tune_m = np.zeros(n, dtype=bool)
        tune_m[:i0] = True                    # 只用过去 (含段内掩码无关: 纯时序)
        # 层: 过去段整定 (软扫描仍按段界分块)
        prm_l, _, _ = train_layer_M(s, y, tune_m, bounds, dt, epochs=300)
        ml = fold_eval(prm_l, s, y, dt, bounds, i0, i1)
        ml["method"] = "layer"
        rows.append({**ml, "fold": k, "train_n": int(tune_m.sum())})
        # Optuna s1
        picks, _ = search_unified_M(s, y, tune_m, bounds, dt, seed=1,
                                    budget=240, use_tpe=True)
        mo = fold_eval(picks[240]["prm"], s, y, dt, bounds, i0, i1)
        mo["method"] = "optuna_s1"
        rows.append({**mo, "fold": k, "train_n": int(tune_m.sum())})
        print(f"fold{k} [{i0}:{i1}] GT={ml['n_gt_side']} "
              f"layer EF1={ml['EF1_ov']:.3f} det={ml['det_ov']:.2f} | "
              f"optuna EF1={mo['EF1_ov']:.3f} det={mo['det_ov']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "a0c_rolling.csv"),
              index=False, encoding="utf-8-sig")
    print("写出 results/audit/a0c_rolling.csv")


if __name__ == "__main__":
    main()
