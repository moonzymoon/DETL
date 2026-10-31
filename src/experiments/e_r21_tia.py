"""R22: 可微 time-in-alarm (TIA) 约束 —— 封堵常设告警角点.

动机: 论文自认 "time-in-alarm 无可微代理" (Limitations). 实际上
mean(ã_t) 天然可微, 作为第四拉格朗日约束 (μ_tia) 可直接封堵紧预算下
的常设告警解: MetroPT3 R_max=0.02 收敛到单个 1050 h 常设告警
(TIA≈100%, 见 metro_tight_audit.json).

设计:
  A. 默认路径回归: SWaT R_max=10, 无 tia —— 须与 robustness 主行一致
     (F1=0.333, AAR=0.052, MAD=65s, n=5), 验证代码扩展零副作用.
  B. 角点封堵: MetroPT3 R_max=0.02 + tia_max ∈ {0.2, 0.3, 0.5}
     —— 看 TIA / n_events / F1 / det 如何变化 (诚实报告).
  C. 不伤合法点: SWaT R_max=0.02 + tia_max=0.3 (SWaT-tight 硬 TIA=0.079
     < 0.3, 约束应不绑定 → 参数不变); TE R_max=0.02 + tia_max=0.3
     (TE-tight TIA=0.401 > 0.3, 约束绑定 —— 报告 F1 代价).

输出: results/audit/r21_tia.csv
运行: cd src && python -u experiments/e_r21_tia.py
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
from experiments.e3_baselines import _kmax
from experiments.e_pilot_tuning import interleave_split

COMBOS = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]
F_MAX = 0.02


def run_one(scorer, ds, r_max, tia_max):
    s, y, dt = get_scores(scorer, ds, seed=SEED)
    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    K_on, K_off = _kmax(dt)
    params, hist = train_tuning(
        s_tr, y_tr, ev_tr, dt, R_max=r_max, F_max=F_MAX, epochs=300,
        seed=SEED, K_on=K_on, K_off=K_off,
        tia_max=tia_max if tia_max is not None else None)
    alarm = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                          params["T_on"], params["T_off"])
    p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
    p.update({
        "dataset": ds, "R_max": r_max,
        "tia_max": tia_max if tia_max is not None else np.nan,
        "TIA_hard_test": float(alarm.mean()),
        "TIA_soft_last": hist[-1]["tia_soft"],
        "theta_hi": params["theta_hi"], "theta_lo": params["theta_lo"],
        "T_on": params["T_on"], "T_off": params["T_off"],
    })
    return p


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r21_tia.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    runs = ([("cmhmil", "SWaT", 10.0, None),            # A 回归
             ("iforest", "MetroPT3", 0.02, 0.2),        # B 角点
             ("iforest", "MetroPT3", 0.02, 0.3),
             ("iforest", "MetroPT3", 0.02, 0.5),
             ("cmhmil", "SWaT", 0.02, 0.3),             # C 不伤
             ("iforest", "TE", 0.02, 0.3)])
    rows = []
    for scorer, ds, r_max, tia in runs:
        p = run_one(scorer, ds, r_max, tia)
        rows.append(p)
        tia_str = "none" if tia is None else f"{tia:.1f}"
        print(f"[r21] {ds:9s} R_max={r_max:<5} tia_max={tia_str:>4}: "
              f"F1={p['EventF1']:.3f} det={p['event_detection_rate']:.2f} "
              f"AAR={p['AAR_per_hour']:.4f} n_ev={p['n_alarm_events']} "
              f"TIA_hard={p['TIA_hard_test']:.3f} "
              f"T_on={p['T_on']} T_off={p['T_off']} "
              f"theta={p['theta_hi']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
