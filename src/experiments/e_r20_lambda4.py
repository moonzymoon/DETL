"""R20 (R11-块3[3]): lambda=4 扩展扫描 —— 回应"未测试更大 lambda".

已有 lambda in {0.1,0.5,1,2}; 本脚本补 lambda=4 三数据集 (miss 项单位权重,
lambda=4 时 lag 项已 8 倍于初选值, 检验非单调项在极端权重下的稳定性).

输出: results/audit/r20_lambda4.csv
运行: cd src && python experiments/e_r20_lambda4.py
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
    out = os.path.join(RESULTS_DIR, "audit", "r20_lambda4.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for scorer, ds in COMBOS:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)
        K_on, K_off = _kmax(dt)
        params, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0,
                                 F_max=0.02, epochs=300, seed=SEED,
                                 K_on=K_on, K_off=K_off, lambda_lag=4.0)
        alarm = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                              params["T_on"], params["T_off"])
        p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
        rows.append({"dataset": ds, "lambda": 4.0, "F1": p["EventF1"],
                     "AAR": p["AAR_per_hour"], "det": p["event_detection_rate"],
                     "T_on": params["T_on"], "T_off": params["T_off"]})
        print(f"[r20] {ds:9s} lambda=4: F1={p['EventF1']:.3f} "
              f"det={p['event_detection_rate']:.2f} AAR={p['AAR_per_hour']:.3f}",
              flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
