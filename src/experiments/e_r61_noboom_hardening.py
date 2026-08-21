"""R61: NoBOOM 加固 (大修->小修冲刺).

(a) 搜索基线五种子跨度 (seeds 7/1/2/3/4, 交错划分, 两流);
(b) 第二 scorer: window-PCA95 重构误差 (r31 配方), 两流六方法;
(c) 时序划分复现 (前半整定后半测试, iForest scorer);
(d) 块尺寸敏感性: 层在 1/2/4h 块 (iForest scorer).
输出: results/audit/r61_noboom_hardening.csv
运行: cd src && python experiments/e_r61_noboom_hardening.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED, CACHE_DIR
from datasets.noboom import noboom_stream
from kpi.panel import (KPIConfig, event_f1_one2one, events_from_binary,
                       kpi_panel, rising_edges)
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline, merge_gaps
from experiments.e3_baselines import _kmax, eval_params
from experiments.e_r37_iou_robust import f1_at_tau
from experiments.e_r60_noboom import (run_search, fixedwin_params,
                                      eval_params_on, score_stream, WINDOW)
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

R_MAX = 10.0
SEEDS = (7, 1, 2, 3, 4)


def pca_scores(which):
    key = f"noboom_{which}_pca"
    npz = os.path.join(CACHE_DIR, f"{key}.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return d["scores"], d["labels"], float(d["dt"])
    d = noboom_stream(which)
    X, Y, Xtr = d["X"], d["Y"], d["X_train"]

    def win(A):
        N = len(A) - WINDOW + 1
        idx = np.arange(WINDOW)[None, :] + np.arange(N)[:, None]
        return A[idx].reshape(N, -1)

    fitF = win(Xtr)
    if len(fitF) > 50000:
        rng = np.random.default_rng(0)
        fitF = fitF[np.sort(rng.choice(len(fitF), 50000, replace=False))]
    p = PCA(n_components=0.95, random_state=0).fit(fitF)
    F = win(X)
    r = p.inverse_transform(p.transform(F)) - F
    s = np.linalg.norm(r, axis=1)
    y = Y[WINDOW - 1:]
    np.savez(npz, scores=s, labels=y, dt=1.0)
    return s, y, 1.0


def interleave_blk(n, dt, blk_h):
    blk = max(1, int(blk_h * 3600 / dt))
    bid = np.arange(n) // blk
    return (bid % 2 == 0), (bid % 2 == 1)


def layer_run(s_tr, y_tr, ev_tr, dt, s_te, y_te, which, tag, rows):
    K_on, K_off = _kmax(dt)
    params, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=R_MAX,
                             F_max=0.02, epochs=300, seed=SEED,
                             K_on=K_on, K_off=K_off)
    alarm = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                          params["T_on"], params["T_off"])
    p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
    p.update({"method": tag, "stream": which,
              "F1_iou01": f1_at_tau(events_from_binary(y_te),
                                    events_from_binary(alarm), 0.10)})
    rows.append(p)
    print(f"[{which}] {tag:22s} F1={p['EventF1']:.3f} "
          f"IoU01={p['F1_iou01']:.3f} AAR={p['AAR_per_hour']:.2f}",
          flush=True)


def main():
    rows = []
    for which in ["acetone", "butanol"]:
        s, y, dt = score_stream(which)
        tr_m, te_m = interleave_blk(len(s), dt, 1.0)
        s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)

        # (a) 五种子跨度
        for name, use_tpe in [("optuna", True), ("random", False)]:
            f1s = []
            for sd in SEEDS:
                prm = run_search(s_tr, y_tr, ev_tr, dt, sd, use_tpe)
                p = eval_params_on(s_tr, s_te, y_te, dt, prm,
                                   f"{name}-seed{sd}", which)
                rows.append(p)
                f1s.append(p["EventF1"])
            print(f"[{which}] {name} 5-seed F1: "
                  + " ".join(f"{v:.3f}" for v in f1s)
                  + f" span [{min(f1s):.3f},{max(f1s):.3f}]", flush=True)

        # (c) 时序划分复现 (iForest)
        half = len(s) // 2
        sT, yT, sE, yE = s[:half], y[:half], s[half:], y[half:]
        evT = events_from_binary(yT)
        layer_run(sT, yT, evT, dt, sE, yE, which, "ours(chrono)", rows)
        for name, use_tpe in [("optuna", True), ("random", False)]:
            prm = run_search(sT, yT, evT, dt, 7 if use_tpe else 8, use_tpe)
            p = eval_params_on(sT, sE, yE, dt, prm, f"{name}(chrono)", which)
            rows.append(p)
            print(f"[{which}] {name}(chrono)        F1={p['EventF1']:.3f} "
                  f"AAR={p['AAR_per_hour']:.2f}", flush=True)
        fw = fixedwin_params(sT, yT, evT, dt)
        p = eval_params_on(sT, sE, yE, dt, fw, "fixed(chrono)", which)
        rows.append(p)
        print(f"[{which}] fixed(chrono)        F1={p['EventF1']:.3f}",
              flush=True)

        # (d) 块尺寸敏感性 (仅层)
        for blk in [2.0, 4.0]:
            tm, em = interleave_blk(len(s), dt, blk)
            layer_run(s[tm], y[tm], events_from_binary(y[tm]), dt,
                      s[em], y[em], which, f"ours(blk{blk:.0f}h)", rows)

        # (b) PCA scorer 六方法
        sp, yp, _ = pca_scores(which)
        auroc = roc_auc_score(yp, sp)
        print(f"[{which}] PCA AUROC={auroc:.3f}", flush=True)
        tm, em = interleave_blk(len(sp), dt, 1.0)
        s_tr2, y_tr2 = sp[tm], yp[tm]
        s_te2, y_te2 = sp[em], yp[em]
        ev2 = events_from_binary(y_tr2)
        layer_run(s_tr2, y_tr2, ev2, dt, s_te2, y_te2, which,
                  "ours(pca)", rows)
        for name, use_tpe in [("optuna", True), ("random", False)]:
            prm = run_search(s_tr2, y_tr2, ev2, dt, 7 if use_tpe else 8,
                             use_tpe)
            p = eval_params_on(s_tr2, s_te2, y_te2, dt, prm,
                               f"{name}(pca)", which)
            rows.append(p)
            print(f"[{which}] {name}(pca) F1={p['EventF1']:.3f} "
                  f"AAR={p['AAR_per_hour']:.2f}", flush=True)
        fw = fixedwin_params(s_tr2, y_tr2, ev2, dt)
        p = eval_params_on(s_tr2, s_te2, y_te2, dt, fw, "fixed(pca)", which)
        rows.append(p)
        print(f"[{which}] fixed(pca) F1={p['EventF1']:.3f}", flush=True)
        gd = __import__("experiments.e3_baselines", fromlist=["run_grid_delay"]) \
            .run_grid_delay(s_tr2, y_tr2, ev2, dt)
        p = eval_params_on(s_tr2, s_te2, y_te2, dt, gd, "grid(pca)", which)
        rows.append(p)
        print(f"[{which}] grid(pca) F1={p['EventF1']:.3f}", flush=True)

    out = os.path.join(RESULTS_DIR, "audit", "r61_noboom_hardening.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
