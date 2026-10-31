"""R37: 匹配阈值鲁棒性 —— event-F1 在 IoU>=tau (0/0.1/0.25/0.5) 下的排序稳定性.

攻击面: "any-overlap (IoU>0) 匹配使长 blanket 告警虚高 F1". 若主表的
方法排序在严格阈值下保持, 攻击失效; 若翻转, 如实报告 (宁严勿宽).
流重建: SWaT/TE/Metro 用 e3_selected_params.csv 缓存参数;
BATADAL 全部现跑 (ours/ours-mcap/pw+constr/optuna/random/grid/gyasi/fixed).
tau=0 行必须复现 tab:main (验证门).
输出: results/audit/r37_iou_robust.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from scorers.adapter import get_scores
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline, merge_gaps
from experiments.e3_baselines import (_kmax, run_grid_delay, run_optuna,
                                      run_gyasi_wang)
from experiments.e_pilot_tuning import interleave_split
from experiments.e_r36_mcap_sweep import batadal_stream

TAUS = (0.0, 0.10, 0.25, 0.50)


def f1_at_tau(gt, pred, tau):
    from scipy.optimize import linear_sum_assignment
    if not gt:
        return float("nan")
    if not pred:
        return 0.0
    IoU = np.zeros((len(gt), len(pred)))
    for i, (gs, ge) in enumerate(gt):
        for j, (ps, pe) in enumerate(pred):
            inter = max(0, min(ge, pe) - max(gs, ps))
            union = max(ge, pe) - min(gs, ps)
            IoU[i, j] = inter / union if union else 0.0
    rows, cols = linear_sum_assignment(-IoU)
    # τ=0 必须用严格 >0 (与 kpi.panel/match_breakdown 同规则); τ>0 用 >=τ
    tp = sum(1 for r, c in zip(rows, cols)
             if IoU[r, c] > 0 and IoU[r, c] >= tau)
    fp = len(pred) - tp
    fn = len(gt) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return float(2 * p * r / (p + r) if p + r else 0.0)


def fixed_window_pick(s_tr, y_tr, dt, R_max=10.0):
    """与 e_pilot_tuning 完全同规: AAR 在未合并流上判合规, F1 在合并流上选."""
    from kpi.panel import rising_edges, event_f1_one2one
    ev_tr = events_from_binary(y_tr)
    hours_tr = len(s_tr) * dt / 3600.0
    best = None
    for q in (0.90, 0.95, 0.99):
        thr = np.quantile(s_tr, q)
        b_tr = (s_tr > thr).astype(int)
        aar_tr = len(rising_edges(b_tr)) / max(hours_tr, 1e-6)
        if aar_tr > R_max:
            continue
        for w_s in (0, 60, 180, 300, 600, 900):
            w_pt = max(1, int(w_s / dt))
            m_tr = merge_gaps(b_tr, w_pt)
            f1 = event_f1_one2one(ev_tr, events_from_binary(m_tr))
            if best is None or f1 > best[0]:
                best = (f1, q, w_pt)
    if best is None:
        return 0.99, max(1, int(600 / dt))
    return best[1], best[2]


def eval_streams(ds, streams, gt, rows):
    for name, al in streams.items():
        pred = events_from_binary(al)
        for tau in TAUS:
            rows.append({"dataset": ds, "method": name, "tau": tau,
                         "F1": f1_at_tau(gt, pred, tau)})
        line = "  ".join(f"τ={t:g}:{f1_at_tau(gt, pred, t):.3f}"
                         for t in TAUS)
        print(f"[r37] {ds:8s} {name:12s} {line}", flush=True)


def primary(rows):
    prm = pd.read_csv(os.path.join(RESULTS_DIR, "e3", "e3_selected_params.csv"))
    for ds, sc in [("SWaT", "cmhmil"), ("TE", "iforest"),
                   ("MetroPT3", "iforest")]:
        s, y, dt = get_scores(sc, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
        gt = events_from_binary(y_te)
        streams = {}
        for m in ("diff-tune", "optuna", "random", "grid-delay", "gyasi"):
            pp = prm[(prm.dataset == ds) & (prm.method == m)].iloc[0]
            streams[m] = hard_pipeline(s_te, pp.theta_hi, pp.theta_lo,
                                       int(pp.T_on), int(pp.T_off))
        q, Wm = fixed_window_pick(s_tr, y_tr, dt)
        streams["fixed-window"] = merge_gaps(
            (s_te > np.quantile(s_tr, q)).astype(int), Wm)
        eval_streams(ds, streams, gt, rows)


def batadal(rows):
    s, y, dt = batadal_stream()
    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    gt = events_from_binary(y_te)
    K_on, K_off = _kmax(dt)
    std = float(np.std(s_tr))
    streams = {}
    for name, mode, kw in [
            ("ours", "constrained", {}),
            ("ours-mcap", "constrained", {"m_max": 0.05 * std}),
            ("pw+constr", "relaxed_f1_constrained", {})]:
        p, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0, F_max=0.02,
                            epochs=300, seed=SEED, K_on=K_on, K_off=K_off,
                            loss_mode=mode, **kw)
        streams[name] = hard_pipeline(s_te, p["theta_hi"], p["theta_lo"],
                                      p["T_on"], p["T_off"])
    for name, fn in [("optuna", lambda: run_optuna(s_tr, y_tr, ev_tr, dt,
                                                   SEED, True)),
                     ("random", lambda: run_optuna(s_tr, y_tr, ev_tr, dt,
                                                   SEED + 1, False)),
                     ("grid-delay", lambda: run_grid_delay(s_tr, y_tr,
                                                           ev_tr, dt)),
                     ("gyasi", lambda: run_gyasi_wang(s_tr, y_tr, ev_tr,
                                                      dt))]:
        prm = fn()
        th = np.quantile(s_tr, prm["q"])
        streams[name] = hard_pipeline(s_te, th, th - prm["margin"],
                                      int(prm["t_on"]), int(prm["t_off"]))
    q, Wm = fixed_window_pick(s_tr, y_tr, dt, R_max=10.0)
    streams["fixed-window"] = merge_gaps(
        (s_te > np.quantile(s_tr, q)).astype(int), Wm)
    eval_streams("BATADAL", streams, gt, rows)


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r37_iou_robust.csv")
    rows = []
    primary(rows)
    batadal(rows)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
