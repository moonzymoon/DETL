"""R35 实验批: (a) BATADAL 时序划分 = 第二个零事件 commissioning 案例;
(b) TE-ext warm-polish 预算曲线 {5,10,20,40}; (c) κ 补点 {15,35} (ext+TE);
(d) BATADAL Optuna/Random 五种子跨度.

输出: results/audit/r35_batch_*.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SEED
from scorers.adapter import get_scores
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline, merge_gaps
from experiments.e3_baselines import _kmax, run_optuna
from experiments.e_pilot_tuning import interleave_split
from experiments.e_r28_polish import tpe_search
from experiments.e_r25b_diag import build_stream as build_te_ext
from experiments.e_r33_batadal import load_batadal
import tuning.losses as L
from sklearn.ensemble import IsolationForest

W = 16
OUT = os.path.join("..", "results", "audit")


def batadal_stream():
    X, Y, dt = load_batadal()
    n = len(X); a, b = int(n * 0.35), int(n * 0.50)
    Xt, Yt = X[b:], Y[b:]
    idx_w = np.arange(W)[None, :] + np.arange(b - a - W + 1)[:, None]
    ifo = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=0, n_jobs=-1).fit(
        X[a:b][idx_w].reshape(len(idx_w), -1))
    N = len(Xt) - W + 1
    idx = np.arange(W)[None, :] + np.arange(N)[:, None]
    s = -ifo.decision_function(Xt[idx].reshape(N, -1)).astype(np.float64)
    return s, Yt[W - 1:], dt


def main():
    cfgb = None
    # ---------- (a) BATADAL 时序划分 ----------
    s, y, dt = batadal_stream()
    n = len(s); half = n // 2
    s_tr, y_tr, s_te, y_te = s[:half], y[:half], s[half:], y[half:]
    ev_tr = events_from_binary(y_tr)
    cfg = KPIConfig(dt=dt)
    std = float(np.std(s_tr))
    rows = []
    for tag, kw in [("ours", {}),
                    ("ours-mcap", dict(m_max=0.05 * std)),
                    ("ours-mcap-floor", dict(m_max=0.05 * std, tia_box=0.3))]:
        prm, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0, F_max=0.02,
                              epochs=300, seed=SEED, K_on=4, K_off=8, **kw)
        al = hard_pipeline(s_te, prm["theta_hi"], prm["theta_lo"],
                           prm["T_on"], prm["T_off"])
        p = kpi_panel(al, y_te, cfg=cfg)
        p.update({"method": f"chrono-{tag}", "n_train_events": len(ev_tr),
                  "TIA_test": float(al.mean())})
        rows.append(p)
        print(f"[r35a] BATADAL chrono {tag:16s} F1={p['EventF1']:.3f} "
              f"det={p['event_detection_rate']:.2f} "
              f"AAR={p['AAR_per_hour']:.4f} n_ev={p['n_alarm_events']} "
              f"TIA={p['TIA_test']:.3f} train_ev={len(ev_tr)}", flush=True)
    # fixed-window chrono
    best = (None, -1)
    for q in (0.90, 0.95, 0.99):
        for wt in (300, 600, 1800):
            Wm = max(1, int(wt / dt))
            al = merge_gaps((s_tr > np.quantile(s_tr, q)).astype(int), Wm)
            p = kpi_panel(al, y_tr, cfg=cfg)
            if p["AAR_per_hour"] <= 10.0 and p["EventF1"] > best[1]:
                best = ((q, Wm), p["EventF1"])
    if best[0] is None:
        best = ((0.99, 1), -1)
    q, Wm = best[0]
    al = merge_gaps((s_te > np.quantile(s_tr, q)).astype(int), Wm)
    p = kpi_panel(al, y_te, cfg=cfg)
    p.update({"method": "chrono-fixed", "n_train_events": len(ev_tr),
              "TIA_test": float(al.mean())})
    rows.append(p)
    print(f"[r35a] BATADAL chrono fixed-window   F1={p['EventF1']:.3f} "
          f"det={p['event_detection_rate']:.2f} AAR={p['AAR_per_hour']:.4f} "
          f"n_ev={p['n_alarm_events']}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "r35_batadal_chrono.csv"),
                              index=False, encoding="utf-8-sig")

    # ---------- (b) TE-ext polish 预算曲线 ----------
    s2, y2, dt2 = build_te_ext()
    tr_m, te_m = interleave_split(len(s2), dt2)
    s_tr, y_tr, s_te, y_te = s2[tr_m], y2[tr_m], s2[te_m], y2[te_m]
    ev_tr = events_from_binary(y_tr)
    K_on, K_off = _kmax(dt2)
    cfg2 = KPIConfig(dt=dt2)
    prm, _ = train_tuning(s_tr, y_tr, ev_tr, dt2, R_max=10.0, F_max=0.02,
                          epochs=300, seed=SEED, K_on=K_on, K_off=K_off)
    std2 = float(np.std(s_tr))
    q_ours = float((s_tr < prm["theta_hi"]).mean())
    warm = {"q": min(max(q_ours, 0.50), 0.999),
            "margin": (prm["theta_hi"] - prm["theta_lo"]) / std2,
            "t_on": prm["T_on"], "t_off": prm["T_off"]}
    rows = []
    for bud in (5, 10, 20, 40):
        pr2 = tpe_search(s_tr, y_tr, ev_tr, dt2, SEED, n_trials=bud,
                         warm=warm)
        al = hard_pipeline(s_te, np.quantile(s_tr, pr2["q"]),
                           np.quantile(s_tr, pr2["q"]) - pr2["margin"],
                           int(pr2["t_on"]), int(pr2["t_off"]))
        p = kpi_panel(al, y_te, cfg=cfg2)
        rows.append({"budget": bud, "EventF1": p["EventF1"],
                     "det": p["event_detection_rate"],
                     "AAR": p["AAR_per_hour"], "n_ev": p["n_alarm_events"]})
        print(f"[r35b] TE-ext polish budget={bud:2d}: F1={p['EventF1']:.3f} "
              f"det={p['event_detection_rate']:.2f} "
              f"AAR={p['AAR_per_hour']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "r35_polish_budget.csv"),
                              index=False, encoding="utf-8-sig")

    # ---------- (c) κ 补点 ----------
    rows = []
    for kappa in (15.0, 35.0):
        L.BETA = kappa
        te_s, te_y, te_dt = get_scores("iforest", "TE", seed=SEED)
        for name, ss, yy, dd in [("TE-ext", s2, y2, dt2),
                                 ("TE", te_s, te_y, te_dt)]:
            trm, tem = interleave_split(len(ss), dd)
            str_, ytr, ste, yte = ss[trm], yy[trm], ss[tem], yy[tem]
            evtr = events_from_binary(ytr)
            prm, _ = train_tuning(str_, ytr, evtr, dd, R_max=10.0,
                                  F_max=0.02, epochs=300, seed=SEED,
                                  K_on=_kmax(dd)[0], K_off=_kmax(dd)[1])
            al = hard_pipeline(ste, prm["theta_hi"], prm["theta_lo"],
                               prm["T_on"], prm["T_off"])
            p = kpi_panel(al, yte, cfg=KPIConfig(dt=dd))
            rows.append({"kappa": kappa, "stream": name,
                         "EventF1": p["EventF1"],
                         "det": p["event_detection_rate"]})
            print(f"[r35c] κ={kappa:.0f} {name:7s} F1={p['EventF1']:.3f}",
                  flush=True)
    L.BETA = 10.0
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "r35_kappa.csv"),
                              index=False, encoding="utf-8-sig")

    # ---------- (d) BATADAL 五种子 ----------
    s3, y3, dt3 = batadal_stream()
    tr_m, te_m = interleave_split(len(s3), dt3)
    s_tr, y_tr, s_te, y_te = s3[tr_m], y3[tr_m], s3[te_m], y3[te_m]
    ev_tr = events_from_binary(y_tr)
    cfg3 = KPIConfig(dt=dt3)
    rows = []
    for method, use_tpe in [("optuna", True), ("random", False)]:
        for seed in (7, 8, 9, 10, 11):
            prm = run_optuna(s_tr, y_tr, ev_tr, dt3, seed, use_tpe)
            th = np.quantile(s_tr, prm["q"])
            al = hard_pipeline(s_te, th, th - prm["margin"],
                               int(prm["t_on"]), int(prm["t_off"]))
            p = kpi_panel(al, y_te, cfg=cfg3)
            rows.append({"method": method, "seed": seed,
                         "EventF1": p["EventF1"]})
        fs = [r["EventF1"] for r in rows if r["method"] == method]
        print(f"[r35d] BATADAL {method}: 五种子 F1 span "
              f"[{min(fs):.3f}, {max(fs):.3f}]", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "r35_batadal_seeds.csv"),
                              index=False, encoding="utf-8-sig")
    print("[r35] batch done")


if __name__ == "__main__":
    main()
