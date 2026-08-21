"""E8: 跨数据集参数迁移 — SWaT 上学到的整定参数直接迁移到 WADI / MetroPT3 (方案 §5 E8).

迁移协议 (跨数据集/打分器的尺度归一, 均不接触目标段标签):
  - 阈值: 源 θ_hi 的训练分位数语义 q_src -> 目标 tuning-train 段分数的同分位数;
  - deadband margin: 按分数 std 比例缩放 (margin_tgt = margin_src·std_tgt/std_src);
  - T_on/T_off: 按秒换算 (T_tgt = T_src·dt_src/dt_tgt).
参照: 在目标数据集上直接整定 (同协议) 的结果. 报告性能保持率
  retention = EventF1_transfer / EventF1_direct-tuned.

输出: results/e8/e8_transfer.csv
运行: cd src && python experiments/e8_transfer.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e_pilot_tuning import interleave_split
from experiments.e3_baselines import _kmax

SOURCE = ("cmhmil", "SWaT")
TARGETS = [("cmhmil", "WADI"), ("iforest", "MetroPT3")]
R_MAX, F_MAX = 10.0, 0.02


def tune_and_params(scorer, ds):
    """在该组合上按标准协议整定, 返回 (hard_params, train_stream_stats, 分割)."""
    s, y, dt = get_scores(scorer, ds, seed=SEED)
    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr = s[tr_m], y[tr_m]
    ev_tr = events_from_binary(y_tr)
    K_on, K_off = _kmax(dt)
    params, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=R_MAX, F_max=F_MAX,
                             epochs=300, seed=SEED, K_on=K_on, K_off=K_off)
    stats = {"q": float((s_tr <= params["theta_hi"]).mean()),
             "margin": params["theta_hi"] - params["theta_lo"],
             "std": float(np.std(s_tr)), "dt": dt}
    return params, stats, (s, y, dt, tr_m, te_m)


def transfer_params(src_stats, src_params, tgt_tr_scores, tgt_dt):
    """按分位数/std/秒 语义把源参数映射到目标域."""
    theta_hi = np.quantile(tgt_tr_scores, src_stats["q"])
    margin = src_stats["margin"] * (float(np.std(tgt_tr_scores)) / src_stats["std"])
    t_on = max(1, int(round(src_params["T_on"] * src_stats["dt"] / tgt_dt)))
    t_off = max(1, int(round(src_params["T_off"] * src_stats["dt"] / tgt_dt)))
    return {"theta_hi": theta_hi, "theta_lo": theta_hi - margin,
            "T_on": t_on, "T_off": t_off}


def main():
    out_dir = os.path.join(RESULTS_DIR, "e8")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[E8] 源: {SOURCE[0]} x {SOURCE[1]}")
    src_params, src_stats, _ = tune_and_params(*SOURCE)
    print(f"  源参数 {src_params} | q={src_stats['q']:.4f} margin={src_stats['margin']:.4f}")

    rows = []
    for scorer, ds in TARGETS:
        tgt_params, tgt_stats, (s, y, dt, tr_m, te_m) = tune_and_params(scorer, ds)
        s_te, y_te = s[te_m], y[te_m]
        # 直接整定 (参照)
        alarm_dir = hard_pipeline(s_te, tgt_params["theta_hi"], tgt_params["theta_lo"],
                                  tgt_params["T_on"], tgt_params["T_off"])
        p_dir = kpi_panel(alarm_dir, y_te, cfg=KPIConfig(dt=dt))
        # 迁移参数 (只用目标训练段分数做尺度映射, 不用标签)
        t_params = transfer_params(src_stats, src_params, s[tr_m], dt)
        alarm_tr = hard_pipeline(s_te, t_params["theta_hi"], t_params["theta_lo"],
                                 t_params["T_on"], t_params["T_off"])
        p_tr = kpi_panel(alarm_tr, y_te, cfg=KPIConfig(dt=dt))
        retention = (p_tr["EventF1"] / p_dir["EventF1"]
                     if p_dir["EventF1"] > 0 else np.nan)
        rows.append({
            "dataset": ds, "scorer": scorer,
            "direct_EventF1": p_dir["EventF1"], "transfer_EventF1": p_tr["EventF1"],
            "retention": retention,
            "direct_AAR": p_dir["AAR_per_hour"], "transfer_AAR": p_tr["AAR_per_hour"],
            "direct_det": p_dir["event_detection_rate"],
            "transfer_det": p_tr["event_detection_rate"],
            "direct_MAD_s": p_dir["mean_alarm_delay_s"],
            "transfer_MAD_s": p_tr["mean_alarm_delay_s"],
            "transfer_params": str(t_params),
        })
        print(f"  -> {ds}: direct EventF1={p_dir['EventF1']:.3f} "
              f"transfer={p_tr['EventF1']:.3f} retention={retention:.2f} | "
              f"transfer AAR={p_tr['AAR_per_hour']:.2f} det={p_tr['event_detection_rate']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "e8_transfer.csv"), index=False,
              encoding="utf-8-sig")
    print(f"\n[E8] 写出 {out_dir}/e8_transfer.csv")


if __name__ == "__main__":
    main()
