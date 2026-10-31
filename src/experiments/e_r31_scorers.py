"""R31: 检测器侧升级实验 —— 红线解除后, 用更强打分器救真实数据与扩展套件.

现状: MetroPT3 (唯一真实长期工业数据) iForest AUROC 0.684 → 所有人 F1<=0.138;
TE-ext 5 种新故障 iForest 10/16 运行低于 q90. 层的价值主张是打分器无关 ——
若换打分器能出好结果, 论文获得真实数据胜利; 若不能, "检测器质量是第一约束"
的论点更强. 两种候选 (与 local_iforest 完全同协议: 滑窗16, 拟合纯正常段):
  pca95 : 窗口 PCA(95%方差) 重构误差
  maxz  : 逐变量 |z| 的窗内最大 (平凡基线)

输出: results/audit/r31_scorers.csv (AUROC + (TE-ext) 分离度)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SEED
from sklearn.decomposition import PCA

W = 16


def win_feats(X, w=W):
    N = len(X) - w + 1
    idx = np.arange(w)[None, :] + np.arange(N)[:, None]
    return X[idx].reshape(N, -1)


def scorer_pca(fitF, F):
    p = PCA(n_components=0.95, random_state=0).fit(fitF)
    r = p.inverse_transform(p.transform(F)) - F
    return np.linalg.norm(r, axis=1)


def scorer_maxz(fitX, X):
    mu = fitX.mean(0); sd = fitX.std(0) + 1e-8
    z = np.abs((X - mu) / sd)
    # 滑窗内逐变量最大 → 窗口得分 = 全变量最大
    N = len(X) - W + 1
    from numpy.lib.stride_tricks import sliding_window_view
    zw = sliding_window_view(z, W, axis=0).max(axis=2)   # (N, D)
    return zw.max(axis=1)


def auroc(s, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


def main():
    out = os.path.join(RESULTS_DIR if (RESULTS_DIR := os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                     "results"))) else "", "audit", "r31_scorers.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []

    # --- MetroPT3 (协议同 local_iforest: 拟合 35-50% 段, 打分后 50%) ---
    from datasets.metropt3 import load_metropt3
    d = load_metropt3()
    X, Y, dt = d["X"], d["Y"], d["dt"]
    n = len(X); a, b = int(n * 0.35), int(n * 0.50)
    Xt, Yt = X[b:], Y[b:]
    F = win_feats(Xt); Ffit = win_feats(X[a:b])
    s_pca = scorer_pca(Ffit, F)
    s_mx = scorer_maxz(X[a:b], Xt)
    yw = Yt[W - 1:]
    for tag, s in [("iforest(ref)", None), ("pca95", s_pca), ("maxz", s_mx)]:
        if s is None:
            from scorers.adapter import get_scores
            s0, y0, _ = get_scores("iforest", "MetroPT3", seed=SEED)
            s, yw = s0, y0
        rows.append({"dataset": "MetroPT3", "scorer": tag,
                     "AUROC": auroc(s, yw)})
        print(f"[r31] MetroPT3 {tag:12s} AUROC={rows[-1]['AUROC']:.3f}",
              flush=True)

    # --- TE-ext (协议同 r25: 拟合正常块 35-50%, 打分 16 故障运行) ---
    from experiments.e_r25_te_ext import load_ext, FAULTS
    XE, YE, dtE = load_ext()
    n_norm = 30001
    a2, b2 = int(n_norm * 0.35), int(n_norm * 0.50)
    XtE = XE[n_norm:]
    FE = win_feats(XtE); FEfit = win_feats(XE[a2:b2])
    yE = YE[n_norm:][W - 1:]
    s_pcaE = scorer_pca(FEfit, FE)
    s_mxE = scorer_maxz(XE[a2:b2], XtE)
    for tag, s in [("pca95", s_pcaE), ("maxz", s_mxE)]:
        rows.append({"dataset": "TE-ext", "scorer": tag,
                     "AUROC": auroc(s, yE)})
        print(f"[r31] TE-ext   {tag:12s} AUROC={rows[-1]['AUROC']:.3f}",
              flush=True)
        # 分离度: 每故障运行事件内分数 max vs q90/q99 (整定段=交错半)
        from experiments.e_pilot_tuning import interleave_split
        tr_m, te_m = interleave_split(len(s), dtE)
        q90, q99 = np.quantile(s[tr_m], 0.90), np.quantile(s[tr_m], 0.99)
        above99 = above90 = 0
        for i in range(16):
            b0 = max(i * 3001 + 480 - (W - 1), 0)
            e0 = min((i + 1) * 3001 - (W - 1), len(yE))
            if e0 <= b0:
                continue
            m = yE[b0:e0] == 1
            if not m.any():
                continue
            mx = float(np.max(s[b0:e0][m]))
            above99 += mx > q99
            above90 += mx > q90
        rows[-1].update({"above_q99": above99, "above_q90": above90})
        print(f"       分离度: >q99 {above99}/16, >q90 {above90}/16 "
              f"(q90={q90:.3f} q99={q99:.3f})", flush=True)

    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
