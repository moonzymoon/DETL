"""R19 (最严审稿人): 副打分器检验 —— TE 的优势是否依赖打分器选择?

审稿质疑: 主表每数据集只用一个打分器 (SWaT-CMH-MIL / TE-iForest / Metro-iForest),
TE 上的 0.769 可能是 iForest 分数流的特性. 本实验在两个"次优配对"上跑
ours + fixed-window + optuna: (cmhmil, TE) [AUROC 0.817] 与 (iforest, SWaT)
[AUROC 0.665], 同交错协议同预算.

输出: results/audit/r19_second_scorer.csv
运行: cd src && python experiments/e_r19_second_scorer.py
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
from tuning.hard_ops import hard_pipeline, merge_gaps
from experiments.e3_baselines import run_optuna, apply_and_eval, _kmax
from experiments.e_pilot_tuning import interleave_split

COMBOS = [("cmhmil", "TE"), ("iforest", "SWaT")]
R_MAX, F_MAX = 10.0, 0.02


def fixedwin(s_tr, y_tr, s_te, y_te, dt):
    from kpi.panel import event_f1_one2one, rising_edges
    ev_tr = events_from_binary(y_tr)
    hours = len(s_tr) * dt / 3600.0
    best = None
    for q in [0.95, 0.99]:
        b = (s_tr > np.quantile(s_tr, q)).astype(int)
        if len(rising_edges(b)) / max(hours, 1e-6) > R_MAX:
            continue
        for w_s in [0, 60, 300, 600, 900]:
            m = merge_gaps(b, int(w_s / dt))
            f1 = (event_f1_one2one(ev_tr, events_from_binary(m))
                  if ev_tr else float("nan"))
            if best is None or (not np.isnan(f1) and f1 > best[0]):
                best = (f1 if not np.isnan(f1) else -1, q, int(w_s / dt))
    if best is None:
        best = (-1, 0.99, int(600 / dt))
    _, q, w = best
    alarm = merge_gaps((s_te > np.quantile(s_tr, q)).astype(int), w)
    p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
    return p


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r19_second_scorer.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for scorer, ds in COMBOS:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)
        K_on, K_off = _kmax(dt)
        params, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=R_MAX,
                                 F_max=F_MAX, epochs=300, seed=SEED,
                                 K_on=K_on, K_off=K_off)
        alarm = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                              params["T_on"], params["T_off"])
        p_ours = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
        prm = run_optuna(s_tr, y_tr, ev_tr, dt, SEED, True)
        p_opt, _ = apply_and_eval("optuna", prm, s_te, y_te, dt, s_tr)
        p_fix = fixedwin(s_tr, y_tr, s_te, y_te, dt)
        for name, p in [("ours", p_ours), ("optuna", p_opt), ("fixed-win", p_fix)]:
            rows.append({"dataset": ds, "scorer": scorer, "method": name,
                         "F1": p["EventF1"], "AAR": p["AAR_per_hour"],
                         "det": p["event_detection_rate"],
                         "MAD": p["mean_alarm_delay_s"],
                         "n_events": p["n_alarm_events"]})
        print(f"[r19] {scorer} x {ds}: ours F1={p_ours['EventF1']:.3f} "
              f"det={p_ours['event_detection_rate']:.2f} | "
              f"optuna {p_opt['EventF1']:.3f} | "
              f"fixed {p_fix['EventF1']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
