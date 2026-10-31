"""稳健性实验 (surrogate v2): 多种子方差 + 约束水平敏感性.

Part A 多种子: diff-tune x seeds {7,8,9} x 3 组合 -> EventF1/检出率 mean±std
  (回应"单次结果无方差"的审稿质疑)
Part B 约束敏感性: R_max ∈ {1,5,10,20} (条/h) x 3 组合 -> 负载-检出权衡曲线
  (方案 §5 E4 的乘子/约束水平敏感性)

输出: results/robustness/robustness_seeds.csv, robustness_rmax.csv
运行: cd src && python experiments/e_robustness.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e_pilot_tuning import interleave_split
from experiments.e3_baselines import _kmax

COMBOS = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]
F_MAX = 0.02


def run_one(scorer, ds, seed, r_max):
    s, y, dt = get_scores(scorer, ds, seed=SEED)
    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    K_on, K_off = _kmax(dt)
    params, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=r_max, F_max=F_MAX,
                             epochs=300, seed=seed, K_on=K_on, K_off=K_off)
    alarm = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                          params["T_on"], params["T_off"])
    p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
    p.update({"dataset": ds, "scorer": scorer, "seed": seed, "R_max": r_max})
    return p


def main():
    out_dir = os.path.join(RESULTS_DIR, "robustness")
    os.makedirs(out_dir, exist_ok=True)

    # Part A: 多种子 (R_max=10)
    rows = []
    for scorer, ds in COMBOS:
        for seed in (7, 8, 9):
            p = run_one(scorer, ds, seed, r_max=10.0)
            rows.append(p)
            print(f"[seeds] {scorer} x {ds} seed={seed}: EventF1={p['EventF1']:.3f} "
                  f"det={p['event_detection_rate']:.2f} AAR={p['AAR_per_hour']:.2f}",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "robustness_seeds.csv"), index=False,
              encoding="utf-8-sig")
    print("\n[seeds] 汇总 (mean±std):")
    for (ds,), g in df.groupby(["dataset"]):
        print(f"  {ds:9s} EventF1={g.EventF1.mean():.3f}±{g.EventF1.std():.3f} "
              f"det={g.event_detection_rate.mean():.2f}±{g.event_detection_rate.std():.2f}")

    # Part B: 约束水平敏感性 (seed 7). 含两个紧预算档 (0.02/0.1) 呈现 binding.
    rows = []
    for scorer, ds in COMBOS:
        for r_max in (0.02, 0.1, 1.0, 5.0, 10.0, 20.0):
            p = run_one(scorer, ds, SEED, r_max)
            rows.append(p)
            print(f"[rmax] {scorer} x {ds} R_max={r_max}: EventF1={p['EventF1']:.3f} "
                  f"AAR={p['AAR_per_hour']:.2f} det={p['event_detection_rate']:.2f} "
                  f"MAD={p['mean_alarm_delay_s']:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "robustness_rmax.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"\n写出 {out_dir}/ 下 2 个 csv")


if __name__ == "__main__":
    main()
