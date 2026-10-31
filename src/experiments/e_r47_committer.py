"""R47 (意见4): Sudarshan 式 committer 概率基线适配到分数流.

思想 (Sudarshan et al. 2024 的合理移植): 用整定段标签学习
P(事件 | 分数流特征) —— "committer 概率", 把分数流升级为标签信息
驱动的概率流, 再在其上做同协议整定. 特征 (因果无泄漏): s_t 与
{5,15,60} 点滚动 mean/std/max, 一阶差分. 模型: LogisticRegression
(确定性, 无额外随机源); 类不平衡用 class_weight=balanced.
对比: committer×optuna / committer×ours vs 原分数上的 incumbents.
四数据集: SWaT/TE/Metro (缓存分数) + BATADAL (现跑).
输出: results/audit/r47_committer.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from scorers.adapter import get_scores
from sklearn.linear_model import LogisticRegression
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e3_baselines import _kmax, run_optuna
from experiments.e_pilot_tuning import interleave_split
from experiments.e_r36_mcap_sweep import batadal_stream


def committer_stream(s):
    """分数流 -> 因果特征矩阵 (n, 10)."""
    feats = [s]
    for w in (5, 15, 60):
        c = np.convolve(s, np.ones(w) / w, mode="full")[:len(s)]
        m = np.zeros_like(s)
        for i in range(w, len(s)):
            m[i] = s[i - w:i].mean()
        c = m
        v = np.zeros_like(s)
        for i in range(w, len(s)):
            v[i] = s[i - w:i].std()
        mx = np.zeros_like(s)
        for i in range(w, len(s)):
            mx[i] = s[i - w:i].max()
        feats += [c, v, mx]
    feats.append(np.diff(s, prepend=s[0]))
    return np.column_stack(feats)


def fast_committer(s):
    """向量化版本: 滚动统计 via cumsum / sliding_window_view."""
    from numpy.lib.stride_tricks import sliding_window_view
    feats = [s]
    for w in (5, 15, 60):
        pad = np.pad(s, (w - 1, 0), mode="edge")
        win = sliding_window_view(pad, w)
        feats += [win.mean(1), win.std(1), win.max(1)]
    feats.append(np.diff(s, prepend=s[0]))
    X = np.column_stack(feats)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-9
    return (X - mu) / sd


def run_ds(ds, s, y, dt, rows):
    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    gt = events_from_binary(y_te)
    K_on, K_off = _kmax(dt)
    cfg = KPIConfig(dt=dt)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(fast_committer(s_tr), y_tr)
    p_tr = clf.predict_proba(fast_committer(s_tr))[:, 1]
    p_te = clf.predict_proba(fast_committer(s_te))[:, 1]
    from sklearn.metrics import roc_auc_score
    au = roc_auc_score(y_te, p_te) if 0 < y_te.sum() < len(y_te) else float("nan")
    print(f"[r47] {ds:8s} committer 概率流 AUROC={au:.3f}", flush=True)

    def panel(alarm, method, stream):
        p = kpi_panel(alarm, y_te, cfg=cfg)
        rows.append({"dataset": ds, "method": method, "stream": stream,
                     "AUROC": au, "F1": p["EventF1"],
                     "det": p["event_detection_rate"],
                     "AAR": p["AAR_per_hour"],
                     "n_ev": p["n_alarm_events"]})
        print(f"[r47] {ds:8s} {method:16s} ({stream}): F1={p['EventF1']:.3f} "
              f"det={p['event_detection_rate']:.2f} "
              f"AAR={p['AAR_per_hour']:.4f} n_ev={p['n_alarm_events']}",
              flush=True)

    prm = run_optuna(p_tr, y_tr, ev_tr, dt, SEED, True)
    th = np.quantile(p_tr, prm["q"])
    panel(hard_pipeline(p_te, th, th - prm["margin"], int(prm["t_on"]),
                        int(prm["t_off"])), "committer+optuna", "p")
    params, _ = train_tuning(p_tr, y_tr, ev_tr, dt, R_max=10.0, F_max=0.02,
                             epochs=300, seed=SEED, K_on=K_on, K_off=K_off)
    panel(hard_pipeline(p_te, params["theta_hi"], params["theta_lo"],
                        params["T_on"], params["T_off"]), "committer+ours",
          "p")


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r47_committer.csv")
    rows = []
    for ds, sc in [("SWaT", "cmhmil"), ("TE", "iforest"),
                   ("MetroPT3", "iforest")]:
        s, y, dt = get_scores(sc, ds, seed=SEED)
        run_ds(ds, s, y, dt, rows)
    sb, yb, dtb = batadal_stream()
    run_ds("BATADAL", sb, yb, dtb, rows)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
