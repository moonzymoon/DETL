"""R60: NoBOOM 第五基准 (真实化工厂批式精馏, NeurIPS'25 D&B).

两条流 (acetone/butanol 三元精馏), 各含多 operating_point 的
train_normal + test_anormal 实验 (1s 采样, 标签列 Label (common/all)).
管线与主表一致: 滑窗 iForest (拟合只用 train_normal 实验, 打分拼接的
test 流) -> 1h 块交错划分 (块大小按流的实验尺度 1-8h 缩放; 日尺度工业流
用 12h) -> raw/fixed-window/grid-delay/random/Optuna/ours 六方法对比 +
IoU>=0.1 共主指标 + 点态 AUROC.
输出: results/audit/r60_noboom.csv
运行: cd src && python experiments/e_r60_noboom.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED, CACHE_DIR
from datasets.noboom import noboom_stream
from kpi.panel import (KPIConfig, event_f1_one2one, events_from_binary,
                       kpi_panel)
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline, merge_gaps
from experiments.e3_baselines import (_kmax, run_grid_delay, run_gyasi_wang)
from experiments.e_r37_iou_robust import f1_at_tau
from sklearn.ensemble import IsolationForest

R_MAX, F_MAX = 10.0, 0.02
WINDOW = 16
BLOCK_H_LOCAL = 1.0     # NoBOOM 实验尺度 1-8h -> 1h 块 (12h 用于日尺度流)
SEEDS = {"optuna": 7, "random": 8}


def score_stream(which, seed=0):
    key = f"noboom_{which}"
    npz = os.path.join(CACHE_DIR, f"{key}.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return d["scores"], d["labels"], float(d["dt"])
    d = noboom_stream(which)
    X, Y, Xtr = d["X"], d["Y"], d["X_train"]

    def win_feats(A):
        N = len(A) - WINDOW + 1
        idx = np.arange(WINDOW)[None, :] + np.arange(N)[:, None]
        return A[idx].reshape(N, -1)

    fitF = win_feats(Xtr)
    if len(fitF) > 50000:                      # 固定确定性子采样
        rng = np.random.default_rng(0)
        fitF = fitF[np.sort(rng.choice(len(fitF), 50000, replace=False))]
    ifo = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=seed, n_jobs=-1).fit(fitF)
    s = -ifo.decision_function(win_feats(X))
    y = Y[WINDOW - 1:]
    np.savez(npz, scores=s, labels=y, dt=1.0)
    return s, y, 1.0


def interleave_local(n, dt):
    blk = max(1, int(BLOCK_H_LOCAL * 3600 / dt))
    bid = np.arange(n) // blk
    return (bid % 2 == 0), (bid % 2 == 1)


def run_search(s_tr, y_tr, ev_tr, dt, seed, use_tpe):
    import optuna
    from experiments.e3_baselines import eval_params
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    K_on, K_off = _kmax(dt)
    sd = float(np.std(s_tr))
    samp = (optuna.samplers.TPESampler(seed=seed) if use_tpe
            else optuna.samplers.RandomSampler(seed=seed))

    def objective(trial):
        q = trial.suggest_float("q", 0.50, 0.999)
        margin = trial.suggest_float("margin", 0.0, 0.25) * sd
        t_on = trial.suggest_int("t_on", 1, K_on)
        t_off = trial.suggest_int("t_off", 1, K_off)
        obj, p = eval_params(s_tr, y_tr, ev_tr, dt, q, margin, t_on, t_off)
        if not np.isfinite(obj):
            obj = -10.0 * max(0.0, p["AAR_per_hour"] - R_MAX) / R_MAX
        return float(obj)

    st = optuna.create_study(direction="maximize", sampler=samp)
    st.optimize(objective, n_trials=60, show_progress_bar=False)
    b = st.best_params
    return {"q": b["q"], "margin": b["margin"] * sd,
            "t_on": b["t_on"], "t_off": b["t_off"]}


def eval_params_on(s_tr, s_te, y_te, dt, prm, tag, which):
    theta_hi = np.quantile(s_tr, prm["q"])
    alarm = hard_pipeline(s_te, theta_hi, theta_hi - prm["margin"],
                          int(prm["t_on"]), int(prm["t_off"]))
    p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
    p["method"] = tag
    p["stream"] = which
    p["F1_iou01"] = f1_at_tau(events_from_binary(y_te),
                              events_from_binary(alarm), 0.10)
    return p


def fixedwin_params(s_tr, y_tr, ev_tr, dt):
    hours = len(s_tr) * dt / 3600.0
    best = None
    for q in [0.95, 0.99]:
        b = (s_tr > np.quantile(s_tr, q)).astype(int)
        from kpi.panel import rising_edges
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
    return {"q": best[1], "margin": 0.0, "t_on": 1, "t_off": best[2] + 1}


def main():
    rows = []
    for which in ["acetone", "butanol"]:
        s, y, dt = score_stream(which)
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(y, s)
        tr_m, te_m = interleave_local(len(s), dt)
        s_tr, y_tr = s[tr_m], y[tr_m]
        s_te, y_te = s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)
        print(f"[{which}] n={len(s)} AUROC={auroc:.3f} "
              f"train_ev={len(ev_tr)} test_ev="
              f"{len(events_from_binary(y_te))}", flush=True)

        # raw q99
        b = (s_te > np.quantile(s_tr, 0.99)).astype(int)
        p = kpi_panel(b, y_te, cfg=KPIConfig(dt=dt))
        p.update({"method": "raw", "stream": which,
                  "F1_iou01": f1_at_tau(events_from_binary(y_te),
                                        events_from_binary(b), 0.10)})
        rows.append(p)

        for tag, prm in [
                ("fixed-window", fixedwin_params(s_tr, y_tr, ev_tr, dt)),
                ("grid-delay", run_grid_delay(s_tr, y_tr, ev_tr, dt)),
                ("random", run_search(s_tr, y_tr, ev_tr, dt,
                                      SEEDS["random"], use_tpe=False)),
                ("optuna", run_search(s_tr, y_tr, ev_tr, dt,
                                      SEEDS["optuna"], use_tpe=True))]:
            p = eval_params_on(s_tr, s_te, y_te, dt, prm, tag, which)
            rows.append(p)
            print(f"[{which}] {tag:12s} F1={p['EventF1']:.3f} "
                  f"IoU01={p['F1_iou01']:.3f} AAR={p['AAR_per_hour']:.2f}",
                  flush=True)

        # ours
        K_on, K_off = _kmax(dt)
        params, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=R_MAX,
                                 F_max=F_MAX, epochs=300, seed=SEED,
                                 K_on=K_on, K_off=K_off)
        alarm = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                              params["T_on"], params["T_off"])
        p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
        p.update({"method": "ours", "stream": which,
                  "F1_iou01": f1_at_tau(events_from_binary(y_te),
                                        events_from_binary(alarm), 0.10)})
        rows.append(p)
        print(f"[{which}] ours          F1={p['EventF1']:.3f} "
              f"IoU01={p['F1_iou01']:.3f} AAR={p['AAR_per_hour']:.2f}",
              flush=True)
        print(f"[{which}] AUROC={auroc:.3f}", flush=True)

    out = os.path.join(RESULTS_DIR, "audit", "r60_noboom.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
