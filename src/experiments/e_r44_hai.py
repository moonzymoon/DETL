"""方案3 (R44): HAI 22.04 第五工业基准 —— HIL 增强工控安全数据集.

数据: train1 26h 纯正常 (93601x88, 1s), test1 24h 含 7 攻击 (1.02%).
引用: Shin et al., "HAI 1.0: HIL-based Augmented ICS Security Dataset",
USENIX CSET 2020. 管线与 BATADAL 同构: iForest(W=16 展平, 拟合于训练集
20000 点子样本, 预注册) 打分测试流, 交错切分用 2h 块 (12 块交替,
预注册 —— 24h 流上 12h 块会退化成单块).
输出: results/audit/r44_hai.csv
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
from experiments.e3_baselines import (_kmax, run_grid_delay, run_gyasi_wang,
                                      run_optuna)
from experiments.e_r22_jackknife import match_breakdown
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

W = 16
DT = 1.0
DATA = os.path.join(DATA_ROOT, "HAI", "hai-22.04")
FIT_SUB = 20000


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r44_hai.csv")
    d_tr = pd.read_csv(os.path.join(DATA, "train1.csv"))
    d_te = pd.read_csv(os.path.join(DATA, "test1.csv"))
    cols = [c for c in d_tr.columns if c not in ("timestamp", "Attack")]
    Xtr = d_tr[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    Xte = d_te[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y_te = (d_te["Attack"] == 1).astype(np.int8).values
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    rng = np.random.RandomState(0)
    fit_idx = rng.choice(len(Xtr) - W + 1, FIT_SUB, replace=False)
    iw = np.arange(W)[None, :] + fit_idx[:, None]
    ifo = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=0, n_jobs=-1).fit(
        Xtr[iw].reshape(FIT_SUB, -1))
    N = len(Xte) - W + 1
    idx = np.arange(W)[None, :] + np.arange(N)[:, None]
    s = -ifo.decision_function(Xte[idx].reshape(N, -1)).astype(np.float64)
    y = y_te[W - 1:]
    print(f"[r44] HAI 分数流 {s.shape}, AUROC={roc_auc_score(y, s):.3f}",
          flush=True)

    # 交错切分: 2h 块 (7200 点) 交替, 奇数块训练
    blk = 7200
    nblk = len(s) // blk
    tr_m = np.zeros(len(s), dtype=bool)
    for i in range(nblk):
        if i % 2 == 0:
            tr_m[i * blk:(i + 1) * blk] = True
    s_tr, y_tr, s_te, y_te2 = s[tr_m], y[tr_m], s[~tr_m], y[~tr_m]
    ev_tr = events_from_binary(y_tr)
    gt = events_from_binary(y_te2)
    K_on, K_off = _kmax(DT)
    cfg = KPIConfig(dt=DT)
    std = float(np.std(s_tr))
    print(f"[r44] 整定段事件 {len(ev_tr)}, 整定测试事件 {len(gt)}, "
          f"std={std:.4f}", flush=True)
    rows, streams = [], {}

    def panel(alarm, method):
        p = kpi_panel(alarm, y_te2, cfg=cfg)
        p["method"] = method
        rows.append(p)
        streams[method] = events_from_binary(alarm)
        print(f"[r44] {method:14s} F1={p['EventF1']:.3f} "
              f"det={p['event_detection_rate']:.2f} "
              f"AAR={p['AAR_per_hour']:.3f} n_ev={p['n_alarm_events']} "
              f"MAD={p['mean_alarm_delay_s']:.0f}s", flush=True)

    for mode, tag, kw in [("constrained", "ours", {}),
                          ("constrained", "ours-mcap",
                           {"m_max": 0.05 * std}),
                          ("relaxed_f1_constrained", "pw+constr", {})]:
        prm, _ = train_tuning(s_tr, y_tr, ev_tr, DT, R_max=10.0, F_max=0.02,
                              epochs=300, seed=SEED, K_on=K_on, K_off=K_off,
                              loss_mode=mode, **kw)
        panel(hard_pipeline(s_te, prm["theta_hi"], prm["theta_lo"],
                            prm["T_on"], prm["T_off"]), tag)
    for tag, fn in [("optuna", lambda: run_optuna(s_tr, y_tr, ev_tr, DT,
                                                  SEED, True)),
                    ("random", lambda: run_optuna(s_tr, y_tr, ev_tr, DT,
                                                  SEED + 1, False)),
                    ("grid-delay", lambda: run_grid_delay(s_tr, y_tr,
                                                          ev_tr, DT)),
                    ("gyasi", lambda: run_gyasi_wang(s_tr, y_tr, ev_tr,
                                                     DT))]:
        prm = fn()
        th = np.quantile(s_tr, prm["q"])
        panel(hard_pipeline(s_te, th, th - prm["margin"], int(prm["t_on"]),
                            int(prm["t_off"])), tag)
    panel((s_te > np.quantile(s_tr, 0.99)).astype(int), "raw")
    best = (None, -1)
    from kpi.panel import rising_edges, event_f1_one2one
    for q in (0.90, 0.95, 0.99):
        b = (s_tr > np.quantile(s_tr, q)).astype(int)
        if len(rising_edges(b)) / (len(s_tr) * DT / 3600) > 10.0:
            continue
        for w_pt in (1, 5, 10, 30):
            f1 = event_f1_one2one(ev_tr,
                                  events_from_binary(merge_gaps(b, w_pt)))
            if f1 > best[1]:
                best = ((q, w_pt), f1)
    q, w_pt = best[0] if best[0] else (0.99, 10)
    panel(merge_gaps((s_te > np.quantile(s_tr, q)).astype(int), w_pt),
          "fixed-window")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}", flush=True)
    for rival in [r for r in streams if r not in ("ours", "ours-mcap")]:
        f_o, f_r = [], []
        for i in range(len(gt)):
            gt_i = gt[:i] + gt[i + 1:]
            base = streams.get("ours-mcap", streams["ours"])
            f_o.append(match_breakdown(gt_i, base)[3])
            f_r.append(match_breakdown(gt_i, streams[rival])[3])
        gaps = [a - b for a, b in zip(f_o, f_r)]
        wins = sum(1 for g in gaps if g > 1e-9)
        print(f"[r44-LOO] ours(mcap)-{rival:14s} n={len(gt)} 折: "
              f"[{min(gaps):+.3f}, {max(gaps):+.3f}]  领先 {wins}/{len(gaps)}",
              flush=True)


if __name__ == "__main__":
    main()
