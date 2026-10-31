# -*- coding: utf-8 -*-
"""A0y: 边界图 (Fig.4) + 预算公平性曲线 — 全部数字来自冻结缓存.

(a) 各评测语境 layer−search 配对差 Δevent-F1 (点+CI 若有)
(b) 同 (a) 但 Δdetection
(c) 预算曲线: TE-iF 上 search EF1 vs budget {20,60,120,240} + layer 水平线
    (warm-20/cold-20/cold-60 来自 a0x, B60/B120/B240 来自 a0_te_iforest)
输出: paper/figures/fig_boundary.pdf
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))
R = os.path.join(ROOT, "results", "audit")
STYLE = r"D:/0科研/工作1/SCI论文方法论_可复用/TikZ统一风格/figurestyle.mplstyle"
plt.style.use(STYLE)

# ── (a)/(b) 配对差 ──────────────────────────────────────────────
# 5 主流 (协议 M): layer − optuna seed-mean
rows = []
spec = [("a0_te_iforest.csv", "TE iForest", "B240", 5),
        ("a0_cmhmil_TE.csv", "TE CMH-MIL", "", 3),
        ("a0_usad_TE.csv", "TE USAD", "", 3),
        ("a0_cmhmil_SWaT.csv", "SWaT CMH-MIL", "", 3),
        ("a0_iforest_MetroPT3.csv", "MetroPT3 iForest", "", 3)]
for f, tag, suf, n in spec:
    d = pd.read_csv(os.path.join(R, f))
    lay = d[d.method.str.match(r"layer(_r0)?$")].iloc[0]
    opt = d[d.method.str.startswith("optuna")]
    rows.append({"ctx": tag, "dEF1": lay.EF1_ov - opt.EF1_ov.mean(),
                 "dDet": lay.det_ov - opt.det_ov.mean(), "ci": None})
# BATADAL 9d
b = pd.read_csv(os.path.join(R, "a0k_regen_M_final.csv"))
b = b[b.stream == "BATADAL"]
lay = b[b.method.str.match(r"layer_r\d")].iloc[0]
opt = b[b.method.str.startswith("optuna") & b.method.str.contains("B240")]
rows.append({"ctx": "BATADAL", "dEF1": lay.EF1_ov - opt.EF1_ov.mean(),
             "dDet": lay.det_ov - opt.det_ov.mean(), "ci": None})
# TSB 120 (stream-level means; Wilcoxon 显著, 有域聚类 CI 的只有 det)
t = json.load(open(os.path.join(R, "a0o_tsb_domain.json"), encoding="utf-8"))
rows.append({"ctx": "TSB-AD 120", "dEF1": t["f1_diff"]["mean"],
             "dDet": t["det_diff"]["mean"], "ci": None})
# fresh-fault cohort (聚类 CI 在)
v = json.load(open(os.path.join(R, "a0v_cluster_bootstrap.json"),
                   encoding="utf-8"))
rows.append({"ctx": "Fresh faults (34)",
             "dEF1": v["idv_clustered"]["d_EF1"]["mean"],
             "dDet": v["idv_clustered"]["d_det"]["mean"],
             "ci": [v["idv_clustered"]["d_EF1"]["ci95"],
                    v["idv_clustered"]["d_det"]["ci95"]]})
df = pd.DataFrame(rows)
print(df.round(3).to_string(index=False))

# ── (c) 预算曲线 ────────────────────────────────────────────────
d = pd.read_csv(os.path.join(R, "a0_te_iforest.csv"))
budget_pts = {}
for B in (60, 120, 240):
    sub = d[d.method.str.contains(f"_B{B}$")]
    budget_pts[B] = sub.EF1_ov.mean()
x = pd.read_csv(os.path.join(R, "a0x_warmstart_M.csv"))
budget_pts[20] = x[x.config == "cold-20"].test_EF1.mean()
layer_ef1 = float(d[d.method == "layer_r0"].EF1_ov.iloc[0])
warm20 = x[x.config == "warm-20"].test_EF1.mean()
print("budget curve:", {k: round(v, 3) for k, v in budget_pts.items()},
      "layer", round(layer_ef1, 3), "warm20", round(warm20, 3))

# ── 画图 ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
labels = df.ctx.tolist()
ypos = np.arange(len(df))[::-1]
BLUE, RED, GREY = "#4878A8", "#C05B4D", "#8A8A8A"

for ax, col, ttl in [(axes[0], "dEF1", r"$\Delta$ event-$F_1$ (layer $-$ search)"),
                     (axes[1], "dDet", r"$\Delta$ detection (layer $-$ search)")]:
    for yp, (_, r) in zip(ypos, df.iterrows()):
        c = BLUE if r[col] > 0 else RED
        ax.scatter([r[col]], [yp], color=c, zorder=3, s=42)
        if r.ci is not None:
            lo, hi = r.ci[0 if col == "dEF1" else 1]
            ax.plot([lo, hi], [yp, yp], color=c, lw=1.4, zorder=2)
    ax.axvline(0, color="k", lw=0.8, ls="--", zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(ttl, fontsize=9)
    ax.set_xlabel("paired difference", fontsize=8)

ax = axes[2]
bs = sorted(budget_pts)
ax.plot(bs, [budget_pts[b_] for b_ in bs], "o-", color=GREY,
        label="cold search (TPE)", ms=4)
ax.scatter([20], [warm20], marker="s", color=BLUE, label="warm-20 (layer init)",
           zorder=3, s=36)
ax.axhline(layer_ef1, color=RED, ls=":", lw=1.4,
           label="layer (one training)")
ax.set_xscale("log")
ax.set_xticks(bs)
ax.set_xticklabels([str(b_) for b_ in bs], fontsize=8)
ax.minorticks_off()
ax.set_xlabel("hard-metric evaluations (log)", fontsize=8)
ax.set_ylabel("test event-$F_1$", fontsize=8)
ax.set_title("budget fairness, TE iForest", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="lower right")

for ax in axes:
    ax.tick_params(axis="x", labelsize=8)
fig.tight_layout()
out = os.path.join(ROOT, "paper", "figures", "fig_boundary.pdf")
fig.savefig(out, bbox_inches="tight")
print("saved", out)
