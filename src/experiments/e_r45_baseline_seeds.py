"""R45 (意见3): 黑盒基线在三条主流上的五种子跨度 (补齐主表统计).

Optuna(TPE) 与 Random 各以种子 {7,1,2,3,4} 重跑选参, 测试段硬评估.
与我方五种子逐位稳定 (r39) 和 BATADAL 跨度 (r35) 合并成完整披露.
输出: results/audit/r45_baseline_seeds.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from scorers.adapter import get_scores
from tuning.hard_ops import hard_pipeline
from experiments.e3_baselines import run_optuna
from experiments.e_pilot_tuning import interleave_split

SEEDS = (7, 1, 2, 3, 4)


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r45_baseline_seeds.csv")
    rows = []
    for ds, sc in [("SWaT", "cmhmil"), ("TE", "iforest"),
                   ("MetroPT3", "iforest")]:
        s, y, dt = get_scores(sc, ds, seed=7)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)
        cfg = KPIConfig(dt=dt)
        for name, use_tpe in [("optuna", True), ("random", False)]:
            f1s = []
            for sd in SEEDS:
                prm = run_optuna(s_tr, y_tr, ev_tr, dt, sd, use_tpe)
                th = np.quantile(s_tr, prm["q"])
                al = hard_pipeline(s_te, th, th - prm["margin"],
                                   int(prm["t_on"]), int(prm["t_off"]))
                p = kpi_panel(al, y_te, cfg=cfg)
                rows.append({"dataset": ds, "method": name, "seed": sd,
                             "F1": p["EventF1"],
                             "det": p["event_detection_rate"],
                             "AAR": p["AAR_per_hour"],
                             "n_ev": p["n_alarm_events"]})
                f1s.append(p["EventF1"])
            print(f"[r45] {ds:8s} {name:7s} 五种子F1: "
                  + " ".join(f"{v:.3f}" for v in f1s)
                  + f"  跨度 [{min(f1s):.3f},{max(f1s):.3f}]", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
