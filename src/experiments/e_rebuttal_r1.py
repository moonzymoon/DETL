"""轮次1修改补充实验: M1 纯时序划分 + M4 chatter 约束.

M1 (chronological split): 对每组合把分数流按时间对半切, 前半 tuning-train /
后半 tuning-test (无交错, 无跨块泄漏), 与交错划分结果对照. SWaT/MetroPT3 前半
无事件的情形如实报告 (即"仅用正常段整定"的真实工业场景).

M4 (chatter constraint): C_max=0.1 (ISA 常用管理线) 作为第三个拉格朗日约束
训练, 与无 chatter 约束版对照硬指标.

输出: results/rebuttal/{r1_chronological.csv, r1_chatter.csv}
运行: cd src && python experiments/e_rebuttal_r1.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import (KPIConfig, event_f1_one2one, events_from_binary,
                       kpi_panel, rising_edges)
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline, merge_gaps
from experiments.e3_baselines import _kmax

COMBOS = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]
R_MAX, F_MAX = 10.0, 0.02


def tune_eval(scorer, ds, s_tr, y_tr, s_te, y_te, dt, c_max=None):
    ev_tr = events_from_binary(y_tr)
    K_on, K_off = _kmax(dt)
    params, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=R_MAX, F_max=F_MAX,
                             epochs=300, seed=SEED, K_on=K_on, K_off=K_off,
                             C_max=c_max)
    alarm = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                          params["T_on"], params["T_off"])
    p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
    p.update({"dataset": ds, "scorer": scorer, "n_train_events": len(ev_tr),
              "params": str(params)})
    return p


def fixedwin_eval(s_tr, y_tr, s_te, y_te, dt, ds, scorer):
    """固定窗口基线 (同一时序划分下; 训练段无事件时按 AAR 最保守点回退)."""
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
    p.update({"dataset": ds, "scorer": scorer, "n_train_events": len(ev_tr)})
    return p


def main():
    out_dir = os.path.join(RESULTS_DIR, "rebuttal")
    os.makedirs(out_dir, exist_ok=True)
    rows_chrono, rows_chat = [], []

    for scorer, ds in COMBOS:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        half = len(s) // 2
        s_tr, y_tr, s_te, y_te = s[:half], y[:half], s[half:], y[half:]

        # --- M1: chronological ---
        p_ours = tune_eval(scorer, ds, s_tr, y_tr, s_te, y_te, dt)
        p_fix = fixedwin_eval(s_tr, y_tr, s_te, y_te, dt, ds, scorer)
        p_ours["method"], p_fix["method"] = "diff-tune(chrono)", "fixed-win(chrono)"
        rows_chrono += [p_ours, p_fix]
        print(f"[chrono] {scorer} x {ds}: train_events={p_ours['n_train_events']} | "
              f"ours F1={p_ours['EventF1']:.3f} det={p_ours['event_detection_rate']:.2f} "
              f"AAR={p_ours['AAR_per_hour']:.2f} | fixed F1={p_fix['EventF1']:.3f}",
              flush=True)

        # --- M4: chatter 约束 (交错划分, 与主表同协议) ---
        from experiments.e_pilot_tuning import interleave_split
        tr_m, te_m = interleave_split(len(s), dt)
        p_wo = tune_eval(scorer, ds, s[tr_m], y[tr_m], s[te_m], y[te_m], dt, c_max=None)
        p_wi = tune_eval(scorer, ds, s[tr_m], y[tr_m], s[te_m], y[te_m], dt, c_max=0.1)
        p_wo["method"], p_wi["method"] = "diff-tune", "diff-tune+chat-c"
        rows_chat += [p_wo, p_wi]
        print(f"[chat] {scorer} x {ds}: 无约束 chatter={p_wo['chatter_rate']:.3f} "
              f"F1={p_wo['EventF1']:.3f} | +chat-c chatter={p_wi['chatter_rate']:.3f} "
              f"F1={p_wi['EventF1']:.3f} AAR={p_wi['AAR_per_hour']:.2f}", flush=True)

    pd.DataFrame(rows_chrono).to_csv(
        os.path.join(out_dir, "r1_chronological.csv"), index=False,
        encoding="utf-8-sig")
    pd.DataFrame(rows_chat).to_csv(
        os.path.join(out_dir, "r1_chatter.csv"), index=False, encoding="utf-8-sig")
    print(f"写出 {out_dir}/ 下 2 个 csv")


if __name__ == "__main__":
    main()
