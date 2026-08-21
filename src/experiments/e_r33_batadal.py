"""R33-W3: BATADAL 集成 —— 真实管网 (C-Town EPANET) SCADA 第四基准.

W3 突破口: 决定性胜利此前只在 TE 仿真上; BATADAL 是真实配水管网的水力
模型遥测 (小时级, 43 传感器), 训练集2 含 5 个标注攻击 (7-73h). 主管线
同构: training1(1年正常)+training2(6月含攻) 拼接, z-score 前35%, iForest
滑窗16 拟合 [35,50)% (纯正常段), 打分后50% (正常尾+全部攻击), 交错协议.

方法: ours / pw+constr / optuna / grid-delay / fixed-window + 留一.
输出: results/audit/r33_batadal.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_ROOT, RESULTS_DIR, SEED
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline, merge_gaps
from experiments.e3_baselines import _kmax, run_grid_delay, run_optuna
from experiments.e_pilot_tuning import interleave_split
from experiments.e_r22_jackknife import match_breakdown
from sklearn.ensemble import IsolationForest

W = 16
DATA = os.path.join(DATA_ROOT, "BATADAL")


def load_batadal():
    npz = os.path.join(RESULTS_DIR, "audit", "r33_batadal_data.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return d["X"], d["Y"], float(d["dt"])
    d1 = pd.read_csv(os.path.join(DATA, "BATADAL_training1.csv"))
    d2 = pd.read_csv(os.path.join(DATA, "BATADAL_training2.csv"))
    d1.columns = [c.strip() for c in d1.columns]
    d2.columns = [c.strip() for c in d2.columns]
    cols = [c for c in d1.columns if c not in ("DATETIME", "ATT_FLAG")
            and c in d2.columns]
    X1 = d1[cols].values.astype(np.float32)
    X2 = d2[cols].values.astype(np.float32)
    y2 = (d2["ATT_FLAG"].values == 1).astype(np.int8)
    X = np.vstack([X1, X2])
    Y = np.concatenate([np.zeros(len(X1), dtype=np.int8), y2])
    n_tr = int(len(X) * 0.35)
    mu = X[:n_tr].mean(0, keepdims=True)
    sd = X[:n_tr].std(0, keepdims=True) + 1e-8
    X = ((X - mu) / sd).astype(np.float32)
    np.savez_compressed(npz, X=X, Y=Y, dt=3600.0)
    return X, Y, 3600.0


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r33_batadal.csv")
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
    y = Yt[W - 1:]
    from sklearn.metrics import roc_auc_score
    print(f"[r33] BATADAL 分数流 {s.shape}, AUROC={roc_auc_score(y, s):.3f}",
          flush=True)

    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    gt = events_from_binary(y_te)
    K_on, K_off = _kmax(dt)
    cfg = KPIConfig(dt=dt)
    print(f"[r33] 整定训练事件 {len(ev_tr)}, 整定测试事件 {len(gt)}",
          flush=True)
    rows, streams = [], {}

    def panel(alarm, method):
        p = kpi_panel(alarm, y_te, cfg=cfg)
        p["method"] = method
        rows.append(p)
        streams[method] = events_from_binary(alarm)
        print(f"[r33] {method:16s} F1={p['EventF1']:.3f} "
              f"det={p['event_detection_rate']:.2f} "
              f"AAR={p['AAR_per_hour']:.3f} n_ev={p['n_alarm_events']} "
              f"MAD={p['mean_alarm_delay_s']:.0f}s", flush=True)

    for mode, tag in [("constrained", "ours"),
                      ("relaxed_f1_constrained", "pw+constr")]:
        prm, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0, F_max=0.02,
                              epochs=300, seed=SEED, K_on=K_on, K_off=K_off,
                              loss_mode=mode)
        panel(hard_pipeline(s_te, prm["theta_hi"], prm["theta_lo"],
                            prm["T_on"], prm["T_off"]), tag)
    for tag, fn in [("optuna", lambda: run_optuna(s_tr, y_tr, ev_tr, dt,
                                                  SEED, True)),
                    ("grid-delay", lambda: run_grid_delay(s_tr, y_tr,
                                                          ev_tr, dt))]:
        prm = fn()
        th = np.quantile(s_tr, prm["q"])
        panel(hard_pipeline(s_te, th, th - prm["margin"],
                            int(prm["t_on"]), int(prm["t_off"])), tag)
    best = (None, -1)
    for q in (0.90, 0.95, 0.99):
        for wt in (300, 600, 1800):
            Wm = max(1, int(wt / dt))
            al = merge_gaps((s_tr > np.quantile(s_tr, q)).astype(int), Wm)
            p = kpi_panel(al, y_tr, cfg=cfg)
            if p["AAR_per_hour"] <= 10.0 and p["EventF1"] > best[1]:
                best = ((q, Wm), p["EventF1"])
    q, Wm = best[0]
    panel(merge_gaps((s_te > np.quantile(s_tr, q)).astype(int), Wm),
          "fixed-window")

    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}", flush=True)
    for rival in [r for r in streams if r != "ours"]:
        f_o, f_r = [], []
        for i in range(len(gt)):
            gt_i = gt[:i] + gt[i + 1:]
            f_o.append(match_breakdown(gt_i, streams["ours"])[3])
            f_r.append(match_breakdown(gt_i, streams[rival])[3])
        gaps = [x - z for x, z in zip(f_o, f_r)]
        wins = sum(1 for g in gaps if g > 1e-9)
        print(f"[r33-LOO] ours-{rival:16s} n={len(gt)} 折: "
              f"[{min(gaps):+.3f}, {max(gaps):+.3f}]  领先 {wins}/{len(gaps)}",
              flush=True)


if __name__ == "__main__":
    main()
