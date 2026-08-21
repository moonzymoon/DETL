"""R26 变量控制对照: 点级 relaxed-F1 + 同一约束组 (AAR/FAR 乘子) vs 本文目标.

回应外部审稿: "点级松弛 F1 未加约束, 只能证明加约束比不加好, 不能证明
事件级目标本身更优". 本脚本给 relaxed-F1 挂上与主方法完全相同的拉格朗日
约束项 (loss_mode='relaxed_f1_constrained'), 三数据集对照:
  - constrained (本文): miss + λ·lag + 乘子约束
  - relaxed_f1 (e6 已有): 点级松弛 F1, 无约束
  - relaxed_f1_constrained (新): 点级松弛 F1 + 同一乘子约束
若受约束的点级目标仍膨胀告警数 → 事件级目标的作用不止于约束;
若它与本文相当 → 约束做了大部分工作 (如实报).

输出: results/audit/r23_constrained_f1.csv
运行: cd src && python -u experiments/e_r23_constrained_f1.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e3_baselines import _kmax
from experiments.e_pilot_tuning import interleave_split

COMBOS = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r23_constrained_f1.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for scorer, ds in COMBOS:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)
        K_on, K_off = _kmax(dt)
        params, _ = train_tuning(
            s_tr, y_tr, ev_tr, dt, R_max=10.0, F_max=0.02, epochs=300,
            seed=SEED, K_on=K_on, K_off=K_off,
            loss_mode="relaxed_f1_constrained")
        alarm = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                              params["T_on"], params["T_off"])
        p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
        p.update({"dataset": ds, "scorer": scorer,
                  "loss_mode": "relaxed_f1_constrained"})
        rows.append(p)
        print(f"[r23] {ds:9s} relaxed-F1+constraints: F1={p['EventF1']:.3f} "
              f"AAR={p['AAR_per_hour']:.4f} n_ev={p['n_alarm_events']} "
              f"det={p['event_detection_rate']:.2f} "
              f"T={params['T_on']}/{params['T_off']}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
