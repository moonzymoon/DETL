"""E2: 告警放大系数 — 固定阈值下每场 ground-truth 异常触发的点级告警条数.

阈值: 分数序列 99 分位 (固定规则, 与阈值校准无关; 记录在案便于复现).
产出:
  results/e2/e2_per_event_alarm_counts.csv  (每场事件的告警条数明细)
  results/e2/e2_amplification_summary.csv   (数据集 x 打分器汇总)
  results/e2/e2_kpi_panel.csv               (同阈值的 ISA-KPI 面板)
  results/e2/e2_boxplot.png                 (箱线图, Nature 配色)

运行: cd src && python experiments/e2_alarm_amplification.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import AVAILABILITY, get_scores
from kpi.panel import KPIConfig, events_from_binary, kpi_panel, rising_edges

DATASETS = ["SWaT", "TE", "MetroPT3", "SMAP", "WADI"]
SCORERS = ["cmhmil", "AT", "iforest"]
THRESH_Q = 0.99   # 固定阈值: 测试段分数 99 分位
NATURE_COLORS = {"cmhmil": "#0C5DA5", "AT": "#00B945", "iforest": "#FF9500"}


def per_event_counts(s, y, dt, thr_q=THRESH_Q):
    """每场 GT 异常事件内的点级告警条数 (告警点数与激活次数) + 事件时长."""
    thr = np.quantile(s, thr_q)
    alarm = (s > thr).astype(int)
    out = []
    for i, (s0, e0) in enumerate(events_from_binary(y)):
        seg = alarm[s0:e0]
        out.append({
            "event_id": i, "start": s0, "end": e0,
            "event_len_pts": e0 - s0,
            "event_duration_h": (e0 - s0) * dt / 3600.0,
            "alarm_points": int(seg.sum()),
            "alarm_activations": int(len(rising_edges(seg))
                                     + (1 if len(seg) and seg[0] == 1 else 0)),
        })
    return out, alarm, thr


def main():
    os.makedirs(os.path.join(RESULTS_DIR, "e2"), exist_ok=True)
    detail_rows, summary_rows, kpi_rows = [], [], []
    for ds in DATASETS:
        for sc in SCORERS:
            if not AVAILABILITY.get((sc, ds), False):
                continue
            try:
                s, y, dt = get_scores(sc, ds, seed=SEED)
            except Exception as e:
                print(f"[E2] FAIL {sc} x {ds}: {e}")
                continue
            counts, alarm, thr = per_event_counts(s, y, dt)
            for c in counts:
                c.update({"dataset": ds, "scorer": sc})
                detail_rows.append(c)
            # 阈值敏感性: q90/q95 下的每事件告警中位数 (q99 为主口径)
            sens = {}
            for q in (0.90, 0.95):
                thr_q = np.quantile(s, q)
                per_ev = [int(((s > thr_q)[s0:e0]).sum())
                          for s0, e0 in events_from_binary(y)]
                sens[q] = float(np.median(per_ev)) if per_ev else np.nan
            n_ev = len(counts)
            aps = [c["alarm_points"] for c in counts] or [np.nan]
            aas = [c["alarm_activations"] for c in counts] or [np.nan]
            summary_rows.append({
                "dataset": ds, "scorer": sc, "thr_q": THRESH_Q, "thr": thr,
                "n_gt_events": n_ev,
                "alarm_points_per_event_median": float(np.median(aps)),
                "alarm_points_per_event_mean": float(np.mean(aps)),
                "alarm_points_per_event_max": float(np.max(aps)) if n_ev else np.nan,
                "alarm_activations_per_event_median": float(np.median(aas)),
                "alarm_points_per_event_median_q90": sens[0.90],
                "alarm_points_per_event_median_q95": sens[0.95],
                "false_alarm_points_per_hour": float(
                    alarm[y == 0].sum() / (len(y) * dt / 3600.0)),
            })
            panel = kpi_panel(alarm, y, cfg=KPIConfig(dt=dt))
            panel.update({"dataset": ds, "scorer": sc})
            kpi_rows.append(panel)
            print(f"[E2] {sc} x {ds}: {n_ev} 场异常, 点级告警/事件 中位="
                  f"{np.median(aps):.0f}, AAR={panel['AAR_per_hour']:.1f}/h")

    ddf = pd.DataFrame(detail_rows)
    sdf = pd.DataFrame(summary_rows)
    kdf = pd.DataFrame(kpi_rows)
    out = os.path.join(RESULTS_DIR, "e2")
    ddf.to_csv(os.path.join(out, "e2_per_event_alarm_counts.csv"),
               index=False, encoding="utf-8-sig")
    sdf.to_csv(os.path.join(out, "e2_amplification_summary.csv"),
               index=False, encoding="utf-8-sig")
    kdf.to_csv(os.path.join(out, "e2_kpi_panel.csv"), index=False, encoding="utf-8-sig")

    # 箱线图: x=数据集, 颜色=打分器, y=每场事件的点级告警条数+1 (log 刻度, +1 处理零值)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    combos = [(ds, sc) for ds in DATASETS for sc in SCORERS
              if len(ddf[(ddf.dataset == ds) & (ddf.scorer == sc)])]
    positions, pos = [], 1.0
    tick_pos, tick_lab = [], []
    for ds in DATASETS:
        first = True
        for sc in SCORERS:
            v = ddf[(ddf.dataset == ds) & (ddf.scorer == sc)]["alarm_points"].values + 1
            if not len(v):
                continue
            bp = ax.boxplot(v, positions=[pos], widths=0.8, patch_artist=True,
                            showfliers=True, flierprops=dict(markersize=3, alpha=0.5))
            for b in bp["boxes"]:
                b.set_facecolor(NATURE_COLORS[sc]); b.set_alpha(0.85)
            for med in bp["medians"]:
                med.set_color("black")
            if first:
                tick_pos.append(pos); tick_lab.append(ds); first = False
            positions.append((pos, sc)); pos += 1.1
        pos += 0.8
    ax.set_yscale("log")
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab)
    ax.set_ylabel("Point-level alarms per GT event (+1, log)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=NATURE_COLORS[sc]) for sc in SCORERS]
    ax.legend(handles, SCORERS, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "e2_boxplot.png"), dpi=300)
    print(f"[E2] 写出 {out}/ 下 3 个 csv + e2_boxplot.png")
    if len(sdf):
        print(sdf[["dataset", "scorer", "n_gt_events",
                   "alarm_points_per_event_median", "false_alarm_points_per_hour"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
