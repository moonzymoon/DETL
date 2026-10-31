"""E5: 软硬间隙专节 — 软代理指标 vs 推理硬指标的偏差及其随退火温度的变化.

设计 (方案 §5 E5):
  - 训练中每 10 个 epoch 在同一训练流上做一次硬规则评估 (hard_pipeline + kpi_panel),
    得到同 epoch 的 (软, 硬) 指标对: AAR_soft vs AAR_hard, miss_soft vs (1-检出率),
    FAR_soft vs 硬正常点告警率;
  - 最终测试段上报告"结束软指标 vs 推理硬指标"的间隙表;
  - 温度选择依据: 间隙随 temp_frac 的变化曲线;
  - chatter 无软代理 (本文未建模), 按方案预案诚实报告为代理局限.

输出: results/e5/e5_gap_history.csv, e5_test_gap.csv, e5_gap_curves.png
运行: cd src && python experiments/e5_soft_hard_gap.py
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
from tuning.constrained import TuningParams, train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e_pilot_tuning import interleave_split
from experiments.e3_baselines import _kmax

COMBOS = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]
R_MAX, F_MAX = 10.0, 0.02


def make_hard_eval(s, y, events, dt):
    """返回闭包: 给定 model -> 训练流上的硬规则指标."""
    def fn(model):
        hp = model.hard_params()
        alarm = hard_pipeline(s, hp["theta_hi"], hp["theta_lo"],
                              hp["T_on"], hp["T_off"])
        p = kpi_panel(alarm, y, cfg=KPIConfig(dt=dt))
        return {"aar": p["AAR_per_hour"],
                "far": float((alarm[y == 0] > 0).mean()) if (y == 0).any() else 0.0,
                "miss": 1.0 - p["event_detection_rate"],
                "event_f1": p["EventF1"]}
    return fn


def main():
    out_dir = os.path.join(RESULTS_DIR, "e5")
    os.makedirs(out_dir, exist_ok=True)
    hist_rows, test_rows = [], []
    for scorer, ds in COMBOS:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)
        K_on, K_off = _kmax(dt)
        print(f"\n[E5] {scorer} x {ds}")
        params, hist = train_tuning(
            s_tr, y_tr, ev_tr, dt, R_max=R_MAX, F_max=F_MAX, epochs=300,
            seed=SEED, K_on=K_on, K_off=K_off,
            hard_eval_fn=make_hard_eval(s_tr, y_tr, ev_tr, dt),
            hard_eval_every=10)
        for h in hist:
            h.update({"dataset": ds, "scorer": scorer})
            hist_rows.append(h)

        # 测试段最终间隙: 结束软指标 (最后 epoch) vs 硬指标 (测试段)
        alarm_te = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                                 params["T_on"], params["T_off"])
        p = kpi_panel(alarm_te, y_te, cfg=KPIConfig(dt=dt))
        last = hist[-1]
        test_rows.append({
            "dataset": ds, "scorer": scorer,
            "soft_aar_train": last["aar"], "hard_aar_test": p["AAR_per_hour"],
            "soft_far_train": last["far"],
            "hard_far_test": float((alarm_te[y_te == 0] > 0).mean()),
            "soft_miss_train": last["miss"],
            "hard_miss_test": 1.0 - p["event_detection_rate"],
            "hard_event_f1_test": p["EventF1"],
            "final_temp_frac": last["temp_frac"],
        })
        print(f"  软AAR={last['aar']:.3f} vs 硬AAR={p['AAR_per_hour']:.3f} | "
              f"软miss={last['miss']:.3f} vs 硬miss={1-p['event_detection_rate']:.3f} | "
              f"硬EventF1={p['EventF1']:.3f}")

    hdf = pd.DataFrame(hist_rows)
    hdf.to_csv(os.path.join(out_dir, "e5_gap_history.csv"), index=False,
               encoding="utf-8-sig")
    tdf = pd.DataFrame(test_rows)
    tdf.to_csv(os.path.join(out_dir, "e5_test_gap.csv"), index=False,
               encoding="utf-8-sig")

    # 间隙-温度曲线: AAR 与 miss 的 (软-硬) 差 vs temp_frac
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    colors = {"SWaT": "#0C5DA5", "TE": "#00B945", "MetroPT3": "#FF9500"}
    for ds in hdf.dataset.unique():
        sub = hdf.dropna(subset=["hard_aar"])
        sub = sub[sub.dataset == ds]
        if not len(sub):
            continue
        axes[0].plot(sub["temp_frac"], sub["aar"] - sub["hard_aar"],
                     "o-", ms=3, lw=1, color=colors.get(ds), label=ds)
        axes[1].plot(sub["temp_frac"], sub["miss"] - sub["hard_miss"],
                     "o-", ms=3, lw=1, color=colors.get(ds), label=ds)
    for ax, ttl in zip(axes, ["AAR 间隙 (软-硬)", "miss 间隙 (软-硬)"]):
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("temp_frac (分数域退火温度/std)")
        ax.set_title(ttl)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "e5_gap_curves.png"), dpi=200)
    print(f"\n[E5] 写出 {out_dir}/ 下 3 个文件")
    print(tdf.to_string(index=False))


if __name__ == "__main__":
    main()
