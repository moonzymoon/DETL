"""E4: 消融 — 逐个拆整定算子 + 参数耦合轨迹分析 (方案 §5 E4).

变体 (同一训练协议, 只冻结对应参数为算子退化值):
  full        : threshold + hysteresis + on-delay + off-delay (完整)
  no-hyst     : margin -> 0 (无迟滞)
  no-on       : T_on -> 1 (无持续确认)
  no-off      : T_off -> 1 (无保持)
  thr-only    : 三者全禁 (退化为纯阈值, 与 raw 的可学习版对照)

参数耦合轨迹: full 变体的 (theta_hi, T_on, T_off, margin) 随 epoch 曲线
(方案 §4.2: on-delay 与 hysteresis 参数耦合退化需显式报告).

输出: results/e4/e4_ablation.csv, e4_param_trajectory.csv, e4_coupling.png
运行: cd src && python experiments/e4_ablation.py
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
from scorers.adapter import get_scores
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e_pilot_tuning import interleave_split
from experiments.e3_baselines import _kmax

COMBOS = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]
VARIANTS = [("full", ()), ("no-hyst", ("hysteresis",)),
            ("no-on", ("on_delay",)), ("no-off", ("off_delay",)),
            ("thr-only", ("hysteresis", "on_delay", "off_delay"))]
R_MAX, F_MAX = 10.0, 0.02


def main():
    out_dir = os.path.join(RESULTS_DIR, "e4")
    os.makedirs(out_dir, exist_ok=True)
    rows, traj_rows = [], []
    for scorer, ds in COMBOS:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)
        K_on, K_off = _kmax(dt)
        print(f"\n[E4] {scorer} x {ds}: train_events={len(ev_tr)}")
        for name, disable in VARIANTS:
            params, hist = train_tuning(
                s_tr, y_tr, ev_tr, dt, R_max=R_MAX, F_max=F_MAX,
                epochs=300, seed=SEED, K_on=K_on, K_off=K_off, disable=disable)
            alarm = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                                  params["T_on"], params["T_off"])
            p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
            p.update({"dataset": ds, "scorer": scorer, "variant": name})
            rows.append(p)
            print(f"  {name:9s} EventF1={p['EventF1']:.3f} AAR={p['AAR_per_hour']:.2f} "
                  f"MAD={p['mean_alarm_delay_s']:.0f}s det={p['event_detection_rate']:.2f} "
                  f"n_ev={p['n_alarm_events']} params={params}")
            if name == "full":                    # 参数轨迹 (full 变体)
                for h in hist:
                    traj_rows.append({"dataset": ds, "scorer": scorer, **h})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "e4_ablation.csv"), index=False,
              encoding="utf-8-sig")
    tdf = pd.DataFrame(traj_rows)
    tdf.to_csv(os.path.join(out_dir, "e4_param_trajectory.csv"), index=False,
               encoding="utf-8-sig")

    # 参数耦合轨迹图 (每个数据集一行: theta_hi / T_on / T_off / margin)
    datasets = [c[1] for c in COMBOS]
    fig, axes = plt.subplots(len(datasets), 4, figsize=(13, 3.0 * len(datasets)),
                             sharex=True)
    for r, ds in enumerate(datasets):
        sub = tdf[tdf.dataset == ds]
        for c, key in enumerate(["theta_hi", "T_on", "T_off", "margin"]):
            ax = axes[r, c]
            ax.plot(sub["epoch"], sub[key], lw=1.2, color="#0C5DA5")
            on_ep = sub.loc[sub.temp_frac <= 0.3, "epoch"]
            if len(on_ep):
                ax.axvline(float(on_ep.iloc[0]), color="gray", ls="--",
                           lw=0.8, label="约束启用")
            if r == 0:
                ax.set_title(key)
            if c == 0:
                ax.set_ylabel(ds)
            if r == len(datasets) - 1:
                ax.set_xlabel("epoch")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("参数耦合轨迹 (full 变体, 虚线=约束项启用)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "e4_coupling.png"), dpi=200)
    print(f"\n[E4] 写出 {out_dir}/ 下 3 个文件")
    cols = ["dataset", "scorer", "variant", "EventF1", "AAR_per_hour",
            "mean_alarm_delay_s", "event_detection_rate", "n_alarm_events"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
