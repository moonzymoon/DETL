# -*- coding: utf-8 -*-
"""A0z-4: P1 — 为何层泛化: 阈值裕度机理分析 (34 fresh run, 冻结参数/打分器).

裕度 = 每个故障 run 的故障段分数峰值 − 方法 theta_hi (正=故障分数越过阈值).
对 layer_r0 与 5 个 optuna B240 种子逐 run 计算; 分赢的故障族 / 平局族统计.
输出: results/audit/a0z_margin_analysis.{csv,json}
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from datasets.tep import TE_DIR, FAULT_FILES, FAULT_ONSET_H, N_XMEAS  # noqa
from sklearn.ensemble import IsolationForest  # noqa

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))
EXT = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "TE_ext"))
OLD = r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/datasets/TE"
W = 16
FRESH = [(i, r) for i in (2, 5, 7, 10, 13) for r in (1, 2)] + \
        [(i, r) for i in (3, 6, 8, 9, 12, 14, 15, 16, 17, 18, 19, 20)
         for r in (1, 2)]
# 赢的故障族 (iForest cohort, d_EF1>0): 7,12,14,17,18,20 (族均值); 2/6/13 平(双满分); 其余 0-0
WIN_IDV = {7, 12, 14, 17, 18, 20}
TIE_IDV = {2, 6, 13}


def rebuild_scorer():
    Xs = []
    df0 = pd.read_excel(os.path.join(TE_DIR, "mode1_normal_500.xlsx"))
    feat = [c for c in df0.columns if c != "Time"]
    Xs.append(df0[feat].values.astype(np.float32))
    for f in FAULT_FILES:
        df = pd.read_excel(os.path.join(TE_DIR, f"{f}.xlsx"))
        Xs.append(df[[c for c in df.columns if c.startswith("XMEAS")]]
                  .values.astype(np.float32))
    X = np.vstack(Xs)
    n_tr = int(len(X) * 0.35)
    mu = X[:n_tr].mean(0, keepdims=True)
    sd = X[:n_tr].std(0, keepdims=True) + 1e-8
    Z = (X - mu) / sd
    a, b = int(len(Z) * 0.35), int(len(Z) * 0.50)
    idx_w = (np.arange(W)[None, :] + np.arange(len(Z[a:b]) - W + 1)[:, None])
    featW = Z[a:b][idx_w].reshape(len(idx_w), -1)
    ifo = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=0, n_jobs=-1).fit(featW)
    return mu, sd, ifo


def main():
    mu, sd, ifo = rebuild_scorer()
    prm = pd.read_csv(os.path.join(OUT, "a0_params.csv"))
    methods = [("layer", prm[prm.method == "layer_r0"].iloc[0]["theta_hi"])]
    for s_ in range(1, 6):
        methods.append((f"opt_s{s_}",
                        prm[prm.method == f"optuna_s{s_}_B240"].iloc[0]["theta_hi"]))
    rows = []
    for idv, run in FRESH:
        path = (os.path.join(OLD, f"mode1_{idv}_{run}.xlsx")
                if idv in (2, 5, 7, 10, 13)
                else os.path.join(EXT, f"mode1_{idv}_{run}.xlsx"))
        df = pd.read_excel(path)
        xcols = [c for c in df.columns if c != "Time (h)"][:41]
        X = df[xcols].values.astype(np.float32)
        t = df["Time (h)"].values
        y = (t >= FAULT_ONSET_H).astype(np.int8)
        Z = (X - mu) / sd
        N = len(Z) - W + 1
        idx = np.arange(W)[None, :] + np.arange(N)[:, None]
        s = -ifo.decision_function(Z[idx].reshape(N, -1))
        yy = y[W - 1:]
        fault_peak = float(np.max(s[yy == 1])) if (yy == 1).any() else np.nan
        normal_q999 = float(np.quantile(s[yy == 0], 0.999))
        for m, th in methods:
            rows.append({"idv": idv, "run": run, "method": m,
                         "theta_hi": round(float(th), 4),
                         "fault_peak": round(fault_peak, 4),
                         "margin": round(fault_peak - float(th), 4),
                         "normal_q999": round(normal_q999, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "a0z_margin_analysis.csv"), index=False,
              encoding="utf-8-sig")
    lay = df[df.method == "layer"]
    opts = df[df.method.str.startswith("opt")]
    summary = {}
    for name, grp, th_col in [
            ("layer", lay, None),
            ("search_seed_mean", opts.groupby(["idv", "run"]).margin.mean()
             .reset_index(), None)]:
        g = grp
        win = g[g.idv.isin(WIN_IDV)]
        tie = g[g.idv.isin(TIE_IDV)]
        zero = g[~g.idv.isin(WIN_IDV | TIE_IDV)]
        summary[name] = {
            "win_faults_margin_median": round(float(win.margin.median()), 4),
            "frac_margin_pos_win": round(float((win.margin > 0).mean()), 3),
            "tie_faults_margin_median": round(float(tie.margin.median()), 4),
            "zero_faults_margin_median": round(float(zero.margin.median()), 4),
            "frac_margin_pos_zero": round(float((zero.margin > 0).mean()), 3),
        }
    import json
    json.dump(summary, open(os.path.join(OUT, "a0z_margin_analysis.json"),
                            "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))
    print("\ntheta_hi: layer=%.4f opt=[%s]" % (
        methods[0][1],
        ", ".join(f"{m}={th:.3f}" for m, th in methods[1:])))


if __name__ == "__main__":
    main()
