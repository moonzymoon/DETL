"""R36-A: m_max (死区上限) 敏感性 —— 卫兵自身不是新调参.

攻击面: "m_max 是新引入的超参, 你用卫兵整定替代了告警整定, 循环."
反驳需要两组证据:
  (a) BATADAL (卫兵起作用处): m_max 跨一个数量级 (0.01-0.20·std) 扫描,
      F1 / 占用 / 最长事件稳定 → 卫兵不挑值;
  (b) TE (锁存不发生处): 加同样上限后主结果逐位不动 → 上限只在
      优化器 otherwise 会利用常设告警时才收紧.
输出: results/audit/r36_mcap_sweep.csv
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
from tuning.hard_ops import hard_pipeline
from experiments.e3_baselines import _kmax
from experiments.e_pilot_tuning import interleave_split
from experiments.e_r33_batadal import W, load_batadal
from sklearn.ensemble import IsolationForest


def batadal_stream():
    X, Y, dt = load_batadal()
    n = len(X); a, b = int(n * 0.35), int(n * 0.50)
    Xt, Yt = X[b:], Y[b:]
    idx_w = np.arange(W)[None, :] + np.arange(b - a - W + 1)[:, None]
    featW = X[a:b][idx_w].reshape(len(idx_w), -1)
    ifo = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=0, n_jobs=-1).fit(featW)
    N = len(Xt) - W + 1
    idx = np.arange(W)[None, :] + np.arange(N)[:, None]
    s = -ifo.decision_function(Xt[idx].reshape(N, -1)).astype(np.float64)
    return s, Yt[W - 1:], dt


def occupancy_diag(alarm, dt):
    ev = events_from_binary(alarm)
    longest = max((e - b0) for b0, e in ev) * dt / 3600.0 if ev else 0.0
    return float(alarm.mean()), longest


def run(ds_name, s, y, dt, R_max, F_max, grid, out_rows):
    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    gt = events_from_binary(y_te)
    K_on, K_off = _kmax(dt)
    cfg = KPIConfig(dt=dt)
    std = float(np.std(s_tr))
    print(f"[r36] {ds_name}: 流长 {len(s)}, 整定段事件 {len(ev_tr)}, "
          f"测试事件 {len(gt)}, std={std:.4f}", flush=True)
    for label, mcap in grid:
        kw = dict(m_max=mcap) if mcap is not None else {}
        prm, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=R_max,
                              F_max=F_max, epochs=300, seed=SEED,
                              K_on=K_on, K_off=K_off,
                              loss_mode="constrained", **kw)
        al = hard_pipeline(s_te, prm["theta_hi"], prm["theta_lo"],
                           prm["T_on"], prm["T_off"])
        p = kpi_panel(al, y_te, cfg=cfg)
        occ, longest = occupancy_diag(al, dt)
        m_learned = float(prm["theta_hi"] - prm["theta_lo"])
        out_rows.append({"dataset": ds_name, "m_max_frac": label,
                         "m_max_abs": (mcap if mcap is not None else np.inf),
                         "m_learned": m_learned, "F1": p["EventF1"],
                         "det": p["event_detection_rate"],
                         "AAR": p["AAR_per_hour"],
                         "n_ev": p["n_alarm_events"], "occupancy": occ,
                         "longest_ev_h": longest})
        print(f"[r36] {ds_name:8s} m_max={label:>6s}: F1={p['EventF1']:.3f} "
              f"det={p['event_detection_rate']:.2f} "
              f"AAR={p['AAR_per_hour']:.4f} n_ev={p['n_alarm_events']} "
              f"m*={m_learned:.4f} occ={occ:.4f} 最长={longest:.1f}h",
              flush=True)


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r36_mcap_sweep.csv")
    rows = []
    # BATADAL: 卫兵起作用处, m_max = frac·std(整定段分数)
    s, y, dt = batadal_stream()
    tr_m, te_m = interleave_split(len(s), dt)
    std_b = float(np.std(s[tr_m]))
    grid_b = [("none", None)] + [(g, float(g) * std_b) for g in
                                 ("0.01", "0.02", "0.05", "0.10", "0.20")]
    run("BATADAL", s, y, dt, 10.0, 0.02, grid_b, rows)
    # TE 控制组: 锁存不发生处, 同一 0.05·std 上限应不改变结果
    s2, y2, dt2 = get_scores("iforest", "TE", seed=SEED)
    tr2, te2 = interleave_split(len(s2), dt2)
    std_t = float(np.std(s2[tr2]))
    run("TE", s2, y2, dt2, 10.0, 0.02,
        [("none", None), ("0.05", 0.05 * std_t)], rows)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
