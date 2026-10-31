"""论文用图生成 (Nature 封面七色, 丰富形式) -> paper/figures/ (PDF 矢量).

配色 (用户指定 Nature 封面提取 7 色):
  pale mist #DDE6F3 / periwinkle #7FA4C8 / lilac #B59BC8 / orchid #D08AB6
  soft mint #8FB79B / muted violet #6A5A8F / indigo plum #3F345B

图:
  fig_architecture  方法架构示意 (训练软链路 + 约束 + 硬推理)
  fig_motivation    E2 放大箱线 + 负载爆炸条形 (双面板)
  fig_main_results  主结果水平条形 (3 面板, 方法排序, ours 高亮)
  fig_coupling      参数耦合轨迹 (阶段着色 + 温度副轴)
  fig_gap           软硬间隙 (偏差区着色)
  fig_rmax          预算敏感性 (slack/binding 区域 + EventF1 面板)

运行: cd src && python experiments/make_paper_figures.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR

FIG_DIR = os.path.normpath(os.path.join(RESULTS_DIR, "..", "paper", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

# Nature 封面 7 色
PALE, PERI, LILAC, ORCHID, MINT, VIOLET, PLUM = (
    "#DDE6F3", "#7FA4C8", "#B59BC8", "#D08AB6", "#8FB79B", "#6A5A8F", "#3F345B")
DSCOL = {"SWaT": PERI, "TE": ORCHID, "MetroPT3": MINT,
         "SMAP": LILAC, "WADI": VIOLET}
SCCOL = {"cmhmil": PERI, "AT": ORCHID, "iforest": MINT}
METHOD_COL = {"Raw threshold": PERI, "Fixed-window": LILAC, "Grid delay": "#9BB8D9",
              "Random (60)": ORCHID, "Optuna (60)": "#E0A8C6",
              "Univ.\ analytical": VIOLET,
              "Ours": PLUM, "Ours ($R_{\\max}{=}0.02$/h)": "#52406F"}

plt.rcParams.update({
    "font.size": 8.5, "axes.linewidth": 0.7, "figure.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": PLUM, "axes.labelcolor": PLUM,
    "xtick.color": PLUM, "ytick.color": PLUM, "text.color": "#2A2A3A",
    "axes.grid": True, "grid.color": PALE, "grid.linewidth": 0.5,
    "legend.frameon": False,
})


# ---------------------------------------------------------------- architecture
def fig_architecture():
    """方法架构示意 (网格化布局: 箭头只连框边锚点, 互不穿越)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 14.2); ax.set_ylim(0.2, 5.8); ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fc, fs=8.5, lw=1.0):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                           fc=fc, ec=PLUM, lw=lw, zorder=3)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=4, color="white" if fc == PLUM else PLUM,
                linespacing=1.25)
        return dict(l=x, r=x + w, b=y, t=y + h, cx=x + w / 2, cy=y + h / 2)

    def arrow(p1, p2, color=PLUM, lw=1.3, ls="-", head=True):
        ax.add_patch(FancyArrowPatch(p1, p2,
                                     arrowstyle="-|>" if head else "-",
                                     mutation_scale=11, color=color, lw=lw,
                                     linestyle=ls, zorder=2,
                                     joinstyle="miter", capstyle="butt"))

    def poly(pts, color=VIOLET, lw=1.1, head=True):
        for i in range(len(pts) - 1):
            arrow(pts[i], pts[i + 1], color=color, lw=lw,
                  head=head and i == len(pts) - 2)

    # 顶行: 训练软链路 (PALE->深 渐变)
    top_y, top_h = 4.35, 1.05
    b1 = box(0.55, top_y, 1.75, top_h, "score\nstream $s_t$", PALE)
    b2 = box(2.85, top_y, 1.95, top_h, "soft\nthreshold $\\theta$", "#D2DFEF")
    b3 = box(5.35, top_y, 2.15, top_h, "soft hysteresis\n$\\theta_{hi},\\theta_{lo}$",
             "#C8D6EA")
    b4 = box(8.05, top_y, 1.95, top_h, "soft on-delay\n$T_{on}$", "#BECDE4")
    b5 = box(10.55, top_y, 1.95, top_h, "soft off-delay\n$T_{off}$", "#B4C4DE")
    for a, b in [(b1, b2), (b2, b3), (b3, b4), (b4, b5)]:
        arrow((a["r"], a["cy"]), (b["l"], b["cy"]))
    ax.text(13.0, b5["cy"], "$\\tilde{a}_t$", ha="center", va="center",
            fontsize=10, color=PLUM)

    # 中行: 目标与约束
    ob = box(3.30, 2.45, 3.10, 0.95,
             "event objectives\n$\\mathcal{L}_{miss},\\;\\mathcal{L}_{lag}$", MINT)
    cb = box(7.10, 2.45, 3.30, 0.95,
             "ISA-KPI constraints\nAAR, FAR, chatter", ORCHID)
    arrow((b3["cx"] + 0.55, b3["b"]), (5.30, ob["t"]), ls="--", color=VIOLET)
    arrow((b4["cx"], b4["b"]), (8.75, cb["t"]), ls="--", color=VIOLET)

    # 底行: 拉格朗日 (与中行两框对齐)
    lb = box(3.30, 0.55, 7.10, 0.95,
             "Lagrangian + dual ascent ($\\mu_j\\!\\uparrow$)\n"
             "+ two-phase annealing", PLUM, fs=8.2)
    arrow((4.85, ob["b"]), (4.85, lb["t"]), color=VIOLET)
    arrow((8.75, cb["b"]), (8.75, lb["t"]), color=VIOLET)

    # 参数反馈: 沿左侧绕行, 从拉格朗日框回到软阈值框底边
    poly([(lb["l"], lb["cy"]), (1.05, lb["cy"]), (1.05, 3.90),
          (b2["cx"], 3.90), (b2["cx"], b2["b"])])
    ax.text(0.62, 2.35, "learned $\\theta, m, T_{on}, T_{off}$", fontsize=7,
            color=VIOLET, ha="center", va="center", rotation=90)

    # 右列: 硬推理 -> 告警
    hb = box(11.15, 2.45, 2.35, 0.95, "hard inference\n(ISA-18.2 exact)",
             "#EDF2FA", fs=8.2)
    ab = box(11.15, 0.55, 2.35, 0.95, "alarm events", MINT, fs=8.2)
    arrow((b5["cx"] + 0.55, b5["b"]), (hb["cx"], hb["t"]), ls="--")
    ax.text(11.05, 3.85, "rounded params", fontsize=6.8, color=PLUM,
            ha="right", va="center")
    arrow((hb["cx"], hb["b"]), (ab["cx"], ab["t"]))

    fig.savefig(os.path.join(FIG_DIR, "fig_architecture.pdf"),
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ---------------------------------------------------------------- motivation
def fig_motivation():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "e2", "e2_per_event_alarm_counts.csv"))
    kpi = pd.read_csv(os.path.join(RESULTS_DIR, "e2", "e2_kpi_panel.csv"))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7),
                                  gridspec_kw={"width_ratios": [1.6, 1]})
    datasets = ["SWaT", "TE", "MetroPT3", "SMAP", "WADI"]
    rng = np.random.default_rng(0)
    pos, tick_pos, tick_lab = 1.0, [], []
    for ds in datasets:
        first = True
        for sc in ["cmhmil", "AT", "iforest"]:
            sub = df[(df.dataset == ds) & (df.scorer == sc)]
            v = sub["alarm_points"].values + 1
            if not len(v):
                continue
            bp = ax.boxplot(v, positions=[pos], widths=0.72, patch_artist=True,
                            showfliers=False,
                            medianprops=dict(color=PLUM, lw=1.2),
                            whiskerprops=dict(color=VIOLET),
                            capprops=dict(color=VIOLET))
            for b in bp["boxes"]:
                b.set_facecolor(SCCOL[sc]); b.set_alpha(0.55)
                b.set_edgecolor(PLUM)
            ax.scatter(pos + rng.uniform(-0.22, 0.22, len(v)), v, s=6,
                       color=SCCOL[sc], edgecolors="white", linewidths=0.3,
                       zorder=5, alpha=0.9)
            if first:
                tick_pos.append(pos); tick_lab.append(ds); first = False
            pos += 1.05
        pos += 0.65
    ax.axhline(10, color=ORCHID, ls=":", lw=1.0)
    ax.text(0.6, 12, "10$\\times$ amplification", fontsize=7, color=ORCHID)
    ax.set_yscale("log")
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab)
    ax.set_ylabel("pointwise alarms per episode (+1)")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=SCCOL[s], alpha=0.65, ec=PLUM)
               for s in ["cmhmil", "AT", "iforest"]]
    ax.legend(handles, ["CMH-MIL", "Anom.\ Transf.", "iForest"], fontsize=6.6,
              ncol=3, loc="upper right")

    # 右面板: raw 阈值下的负载 (AAR + chatter)
    kpi = kpi[kpi.method.isna()] if "method" in kpi.columns else kpi
    labels, aars, chats, cols = [], [], [], []
    for _, r in kpi.iterrows():
        labels.append(f"{r.dataset}\n{r.scorer}")
        aars.append(r.AAR_per_hour); chats.append(r.chatter_rate)
        cols.append(DSCOL.get(r.dataset, PERI))
    y = np.arange(len(labels))
    ax2.barh(y, aars, color=cols, alpha=0.75, edgecolor=PLUM, lw=0.6,
             label="AAR (/h)")
    ax2.axvline(10, color=ORCHID, ls="--", lw=1.0)
    ax2.text(10.5, -0.45, "EEMUA\noverload", fontsize=6.2, color=ORCHID)
    for yi, ch in zip(y, chats):
        if ch > 0:
            ax2.text(0.3, yi, f"chat {ch:.2f}", fontsize=5.8, color="white",
                     va="center", zorder=5)
    ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=6.2)
    ax2.invert_yaxis()
    ax2.set_xlabel("average alarm rate (/h), raw threshold")
    ax2.set_xscale("symlog", linthresh=1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_motivation.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- main results
def fig_main_results():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "e3", "e3_baselines.csv"))
    rb = pd.read_csv(os.path.join(RESULTS_DIR, "robustness", "robustness_rmax.csv"))
    rb02 = rb[rb.R_max == 0.02]
    name_map = {"optuna": "Optuna (60)", "random": "Random (60)",
                "grid-delay": "Grid delay", "gyasi": "Univ.\\ analytical",
                "diff-tune": "Ours"}
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.9), sharey=False)
    for ax, ds in zip(axes, ["SWaT", "TE", "MetroPT3"]):
        sub = df[df.dataset == ds].copy()
        rows = []
        for _, r in sub.iterrows():
            rows.append((name_map.get(r.method, r.method), r.EventF1, False))
        if len(rb02[rb02.dataset == ds]):
            rows.append(("Ours ($R_{\\max}{=}0.02$/h)",
                         float(rb02[rb02.dataset == ds].EventF1.iloc[0]), True))
        raw = {"SWaT": 0.047, "TE": 0.118, "MetroPT3": 0.006}[ds]
        rows.insert(0, ("Raw threshold", raw, False))
        rows.sort(key=lambda t: t[1])
        labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
        cols = [METHOD_COL.get(l, PERI) for l in labels]
        y = np.arange(len(rows))
        ax.barh(y, vals, color=cols, edgecolor=PLUM, lw=0.6, alpha=0.9,
                height=0.62)
        for yi, v in zip(y, vals):
            ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=6.2,
                    color=PLUM)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.8)
        ax.set_xlim(0, 0.9)
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel("event $F_1$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_main_results.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- coupling
def fig_coupling():
    tdf = pd.read_csv(os.path.join(RESULTS_DIR, "e4", "e4_param_trajectory.csv"))
    datasets = ["SWaT", "TE", "MetroPT3"]
    keys = ["theta_hi", "T_on", "T_off", "margin"]
    labs = [r"$\theta_{\mathrm{hi}}$", r"$T_{\mathrm{on}}$",
            r"$T_{\mathrm{off}}$", r"deadband $m$"]
    fig, axes = plt.subplots(len(datasets), 4, figsize=(6.6, 5.0), sharex=True)
    for r, ds in enumerate(datasets):
        sub = tdf[tdf.dataset == ds]
        on_ep = float(sub.loc[sub.temp_frac <= 0.3, "epoch"].iloc[0])
        for c, (key, lab) in enumerate(zip(keys, labs)):
            ax = axes[r, c]
            ax.axvspan(0, on_ep, color=PALE, alpha=0.6, zorder=0)
            ax.plot(sub["epoch"], sub[key], lw=1.3, color=DSCOL[ds], zorder=3)
            ax.axvline(on_ep, color=VIOLET, ls="--", lw=0.8)
            if r == 0:
                ax.set_title(lab, fontsize=8.5)
            if c == 0:
                ax.set_ylabel(ds, fontsize=8)
            if r == len(datasets) - 1:
                ax.set_xlabel("epoch")
        # 温度副轴 (第一列): 只留 3 个刻度避免与主轴挤在一起
        ax2 = axes[r, 0].twinx()
        ax2.plot(sub["epoch"], sub["temp_frac"], lw=1.0, ls=":", color=VIOLET,
                 alpha=0.85)
        ax2.set_ylabel(r"$\tau_s/\sigma(s)$", fontsize=7, color=VIOLET)
        ax2.set_yticks([0.0, 0.5, 1.0])
        ax2.tick_params(labelsize=6, colors=VIOLET)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(VIOLET)
        ax2.spines["top"].set_visible(False)
        ax2.grid(False)
    axes[0, 2].plot([], [], ls=":", color=VIOLET, label=r"$\tau_s/\sigma(s)$")
    axes[0, 2].axvline(0, color=VIOLET, ls="--", label="constraints on")
    axes[0, 2].legend(fontsize=6.5, loc="upper right", markerscale=1.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_coupling.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- gap
def fig_gap():
    hdf = pd.read_csv(os.path.join(RESULTS_DIR, "e5", "e5_gap_history.csv"))
    hdf = hdf.dropna(subset=["hard_aar"])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    mk = {"SWaT": "o", "TE": "s", "MetroPT3": "^"}
    for panel, (ax, key, hkey, ttl) in enumerate(
            zip(axes, ["aar", "miss"], ["hard_aar", "hard_miss"],
                ["AAR surrogate gap (soft $-$ hard)",
                 "miss surrogate gap (soft $-$ hard)"])):
        ax.axvspan(0.3, 1.05, color=PALE, alpha=0.55, zorder=0)
        for ds in hdf.dataset.unique():
            sub = hdf[hdf.dataset == ds].sort_values("temp_frac")
            ax.plot(sub["temp_frac"], sub[key] - sub[hkey], marker=mk[ds],
                    ms=3, lw=1.0, color=DSCOL[ds], label=ds)
        ax.axhline(0, color=PLUM, lw=0.7)
        ax.set_title(ttl, fontsize=8.5)
        ax.set_xlabel(r"score-domain temperature / $\sigma$")
        ax.set_xlim(1.02, 0.04)
        ax.legend(fontsize=6.5, markerscale=1.3)
    axes[0].text(0.65, axes[0].get_ylim()[1] * 0.9, "soft-FAR biased zone",
                 fontsize=6.2, color=VIOLET, ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_gap.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- rmax
def fig_rmax():
    r = pd.read_csv(os.path.join(RESULTS_DIR, "robustness", "robustness_rmax.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5))
    mk = {"SWaT": "o", "TE": "s", "MetroPT3": "^"}
    for ds in r.dataset.unique():
        sub = r[r.dataset == ds].sort_values("R_max")
        axes[0].plot(sub["R_max"], sub["AAR_per_hour"], marker=mk[ds], ms=3.5,
                     lw=1.0, color=DSCOL[ds], label=ds)
        axes[1].plot(sub["R_max"], sub["event_detection_rate"], marker=mk[ds],
                     ms=3.5, lw=1.0, color=DSCOL[ds], label=ds)
        axes[2].plot(sub["R_max"], sub["EventF1"], marker=mk[ds], ms=3.5,
                     lw=1.0, color=DSCOL[ds], label=ds)
    axes[0].plot([0.02, 20], [0.02, 20], ":", color=ORCHID, lw=1.1,
                 label="binding line")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[1].set_xscale("log"); axes[2].set_xscale("log")
    axes[0].set_ylabel("achieved AAR (/h)")
    axes[1].set_ylabel("detection rate")
    axes[2].set_ylabel("event $F_1$")
    for ax in axes:
        ax.set_xlabel(r"alarm budget $R_{\max}$ (/h)")
        ax.legend(fontsize=6.2)
    deg = r[(r.dataset == "MetroPT3") & (r.R_max == 0.02)]   # 退化点: 三面板空心环
    if len(deg):
        for ax, col in zip(axes, ["AAR_per_hour", "event_detection_rate",
                                  "EventF1"]):
            ax.plot(deg["R_max"], deg[col], "o", ms=7, mfc="none", mec=MINT,
                    mew=1.2, zorder=5)
    axes[2].annotate("standing alarm\n(degenerate, see text)", xy=(0.02, 0.5),
                     xytext=(0.4, 0.42), fontsize=6.2, color=MINT,
                     arrowprops=dict(arrowstyle="->", color=MINT, lw=0.8))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_rmax.pdf"))
    plt.close(fig)


def fig_ton_sweep():
    """R12 新增: T_on 扫描一致性 (软目标 vs 硬 F1, 数据 results/audit/r7_ton_sweep.csv)."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "audit", "r7_ton_sweep.csv"))
    datasets = ["SWaT", "TE", "MetroPT3"]
    trained = {"SWaT": 7, "TE": 1, "MetroPT3": 13}
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.6))
    for j, ds in enumerate(datasets):
        d = df[df.dataset == ds].sort_values("T_on")
        axes[0][j].plot(d["T_on"], d["soft_obj"], "-o", ms=2.5, lw=1.0,
                        color=DSCOL[ds])
        axes[1][j].plot(d["T_on"], d["hard_F1_test"], "-s", ms=2.5, lw=1.0,
                        color=DSCOL[ds])
        for i in (0, 1):
            axes[i][j].axvline(trained[ds], ls=":", lw=0.9, color=ORCHID)
        axes[1][j].set_xlabel(r"$T_{\mathrm{on}}$ (samples)")
        axes[0][j].set_title(ds, fontsize=8)
        axes[0][j].tick_params(labelsize=7)
        axes[1][j].tick_params(labelsize=7)
    axes[0][0].set_ylabel("soft objective\n(final temp.)", fontsize=8)
    axes[1][0].set_ylabel("hard event $F_1$\n(tuning-test)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_ton_sweep.pdf"))
    plt.close(fig)


def fig_hometurf():
    """R12 新增: 解析基线 home-turf 预测 vs 实际 (r8/r9 CSV, log-log, y=x)."""
    r10 = pd.read_csv(os.path.join(RESULTS_DIR, "audit",
                                   "r8_gyasi_hometurf.csv"))
    r01 = pd.read_csv(os.path.join(RESULTS_DIR, "audit",
                                   "r9_gyasi_tight.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for ax, df, ttl, med in [(axes[0], r10, r"$R_{\max}=10$/h", 0.20),
                             (axes[1], r01, r"$R_{\max}=0.1$/h", 20.6)]:
        ax.scatter(df["r_pred_per_h"], df["far_achieved_per_h"], s=12,
                   color=PLUM, alpha=0.75, edgecolors="none")
        lo = min(df.r_pred_per_h.min(), df.far_achieved_per_h.min()) * 0.5
        hi = max(df.r_pred_per_h.max(), df.far_achieved_per_h.max()) * 2
        ax.plot([lo, hi], [lo, hi], ":", lw=1.0, color=ORCHID)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("i.i.d. design prediction (/h)", fontsize=8)
        ax.set_title(ttl + f"  (median {med:g}$\\times$)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("achieved rate, held-out\nnormal segment (/h)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_hometurf.pdf"))
    plt.close(fig)


def fig_alarm_timeline():
    """R22 新增: 告警时间线 —— 系统实际输出 (GT 事件 vs raw q99 告警 vs
    学习层告警事件). MetroPT3 面板加第 4 行: 紧预算常设告警; BATADAL 面板
    加第 4 行: 默认解的死区锁存 (全宽单条). 行标签内嵌事件计数."""
    import json

    from scorers.adapter import get_scores
    from tuning.hard_ops import hard_pipeline
    from kpi.panel import events_from_binary
    from experiments.e_pilot_tuning import interleave_split

    prm = pd.read_csv(os.path.join(RESULTS_DIR, "e3", "e3_selected_params.csv"))
    with open(os.path.join(RESULTS_DIR, "audit", "metro_tight_audit.json"),
              encoding="utf-8") as f:
        tight = json.load(f)
    combos = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3"),
              ("iforest", "BATADAL")]
    fig, axes = plt.subplots(
        4, 1, figsize=(7.0, 5.9),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.4, 1.4]})
    for ax, (scorer, ds) in zip(axes, combos):
        if ds == "BATADAL":
            from sklearn.ensemble import IsolationForest
            from experiments.e_r33_batadal import load_batadal
            X, Y, dt = load_batadal()
            n = len(X); a, b = int(n * 0.35), int(n * 0.50)
            Xt, Yt = X[b:], Y[b:]
            iw = np.arange(16)[None, :] + np.arange(b - a - 15)[:, None]
            ifo = IsolationForest(n_estimators=100, contamination="auto",
                                  random_state=0, n_jobs=-1).fit(
                X[a:b][iw].reshape(len(iw), -1))
            N = len(Xt) - 15
            idx = np.arange(16)[None, :] + np.arange(N)[:, None]
            s = -ifo.decision_function(Xt[idx].reshape(N, -1)).astype(np.float64)
            y = Yt[15:]
            tr_m, te_m = interleave_split(len(s), dt)
        else:
            s, y, dt = get_scores(scorer, ds, seed=7)
            tr_m, te_m = interleave_split(len(s), dt)
        s_tr, s_te, y_te = s[tr_m], s[te_m], y[te_m]
        ev_gt = events_from_binary(y_te)
        ev_raw = events_from_binary(
            (s_te > np.quantile(s_tr, 0.99)).astype(int))
        if ds == "BATADAL":
            from tuning.constrained import train_tuning
            from common import SEED as _SEED
            ev_tr = events_from_binary(y[tr_m])
            y_tr = y[tr_m]
            std = float(np.std(s_tr))
            p1, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0,
                                 F_max=0.02, epochs=300, seed=_SEED,
                                 K_on=4, K_off=8, m_max=0.05 * std)
            ev_mcap = events_from_binary(
                hard_pipeline(s_te, p1["theta_hi"], p1["theta_lo"],
                              p1["T_on"], p1["T_off"]))
            p2, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0,
                                 F_max=0.02, epochs=300, seed=_SEED,
                                 K_on=4, K_off=8)
            ev_latch = events_from_binary(
                hard_pipeline(s_te, p2["theta_hi"], p2["theta_lo"],
                              p2["T_on"], p2["T_off"]))
            tracks = [(ev_gt, "grey", 0.62, 0.55, "ground-truth episodes"),
                      (ev_raw, PERI, 1.0, 0.30, r"raw $q_{99}$ threshold"),
                      (ev_mcap, PLUM, 0.62, 0.80,
                       "learned (deadband ceiling)"),
                      (ev_latch, VIOLET, 0.62, 0.80,
                       "learned, default (latch)")]
        else:
            pp = prm[(prm.dataset == ds) & (prm.method == "diff-tune")].iloc[0]
            ev_our = events_from_binary(
                hard_pipeline(s_te, pp.theta_hi, pp.theta_lo,
                              int(pp.T_on), int(pp.T_off)))
            tracks = [(ev_gt, "grey", 0.62, 0.55, "ground-truth episodes"),
                      (ev_raw, PERI, 1.0, 0.30, r"raw $q_{99}$ threshold"),
                      (ev_our, PLUM, 0.62, 0.80, "learned layer (this work)")]
            if ds == "MetroPT3":
                tp = tight["params"]
                tracks.append((events_from_binary(
                    hard_pipeline(s_te, tp["theta_hi"], tp["theta_lo"],
                                  tp["T_on"], tp["T_off"])),
                    VIOLET, 0.62, 0.80,
                    r"learned, $R_{\max}{=}0.02$/h"))
        n = len(tracks)
        for row, (ev, col, hh, alpha, _) in enumerate(tracks):
            yc = n - 1 - row
            if ev:
                xs = np.array([e[0] for e in ev]) * dt / 3600.0
                ws = np.maximum(np.array([e[1] - e[0] for e in ev]) * dt / 3600.0,
                                0.35)        # 最小可视宽度 (h)
                ax.broken_barh([(x, w) for x, w in zip(xs, ws)], (yc - hh / 2, hh),
                               facecolors=col, alpha=alpha, linewidths=0)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"{lbl}  ({len(ev)})"
                            for ev, _, _, _, lbl in reversed(tracks)],
                           fontsize=7.5)
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_title(ds, fontsize=8.5, loc="left")
        ax.set_xlabel("tuning-test stream position (h)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_alarm_timeline.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_motivation()
    fig_main_results()
    fig_coupling()
    fig_gap()
    fig_rmax()
    fig_hometurf()
    fig_ton_sweep()
    fig_alarm_timeline()
    print(f"figures written to {FIG_DIR}")
