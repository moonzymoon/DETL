"""方案4 (R42): GT 事件边界扰动鲁棒性 —— 方法排序对标注边界主观性的稳定性.

对每个 GT 事件 (s,e), 长度 L=e-s:
  extend(α): (s-αL/2, e+αL/2)  边界外扩
  shrink(α): (s+αL/2, e-αL/2)  边界内缩 (保底 1 点)
α ∈ {0.1, 0.2}. 流重建与 r37 相同 (缓存参数; BATADAL 现跑).
报告: 各方法在 α=0/±10%/±20% 下的事件-F1, 检验排序翻转.
输出: results/audit/r42_boundary.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from scorers.adapter import get_scores
from tuning.hard_ops import hard_pipeline, merge_gaps
from experiments.e_pilot_tuning import interleave_split
from experiments.e_r37_iou_robust import fixed_window_pick, f1_at_tau
from experiments.e_r36_mcap_sweep import batadal_stream
from tuning.constrained import train_tuning
from experiments.e3_baselines import _kmax


def perturb(gt, alpha, mode):
    out = []
    for s, e in gt:
        L = e - s
        d = int(alpha * L / 2)
        if mode == "extend":
            s2, e2 = s - d, e + d
        else:
            s2, e2 = s + d, max(s + d + 1, e - d)
        out.append((max(0, s2), e2))
    return out


def eval_ds(ds, streams, gt, rows):
    for name, al in streams.items():
        pred = events_from_binary(al)
        vals = {}
        for mode, alpha in [("none", 0.0), ("ext10", 0.1), ("shr10", 0.1),
                            ("ext20", 0.2), ("shr20", 0.2)]:
            g = gt if mode == "none" else perturb(
                gt, alpha, "extend" if mode.startswith("ext") else "shrink")
            vals[mode] = f1_at_tau(g, pred, 0.0)
        rows.append({"dataset": ds, "method": name, **vals})
        print(f"[r42] {ds:8s} {name:12s} " + "  ".join(
            f"{k}={v:.3f}" for k, v in vals.items()), flush=True)


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r42_boundary.csv")
    rows = []
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
        eval_ds(ds, streams, gt, rows)
    # BATADAL: ours/ours-mcap/pw/optuna/grid 现跑 (复用 r37 配方)
    sb, yb, dtb = batadal_stream()
    tr_b, te_b = interleave_split(len(sb), dtb)
    s_tr, y_tr, s_te, y_te = sb[tr_b], yb[tr_b], sb[te_b], yb[te_b]
    ev_tr = events_from_binary(y_tr)
    gt = events_from_binary(y_te)
    K_on, K_off = _kmax(dtb)
    std = float(np.std(s_tr))
    streams = {}
    for name, mode, kw in [("ours", "constrained", {}),
                           ("ours-mcap", "constrained",
                            {"m_max": 0.05 * std}),
                           ("pw+constr", "relaxed_f1_constrained", {})]:
        p, _ = train_tuning(s_tr, y_tr, ev_tr, dtb, R_max=10.0, F_max=0.02,
                            epochs=300, seed=SEED, K_on=K_on, K_off=K_off,
                            loss_mode=mode, **kw)
        streams[name] = hard_pipeline(s_te, p["theta_hi"], p["theta_lo"],
                                      p["T_on"], p["T_off"])
    from experiments.e3_baselines import run_grid_delay
    pr = run_grid_delay(s_tr, y_tr, ev_tr, dtb)
    th = np.quantile(s_tr, pr["q"])
    streams["grid-delay"] = hard_pipeline(s_te, th, th - pr["margin"],
                                          int(pr["t_on"]), int(pr["t_off"]))
    q, Wm = fixed_window_pick(s_tr, y_tr, dtb)
    streams["fixed-window"] = merge_gaps(
        (s_te > np.quantile(s_tr, q)).astype(int), Wm)
    eval_ds("BATADAL", streams, gt, rows)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
