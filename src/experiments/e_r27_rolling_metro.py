"""R29-A4: MetroPT3 滚动重整定 —— 量化 "periodic re-tuning" 建议.

时序协议下 MetroPT3 双方法崩溃至 ~0.001 (非平稳漂移击败静态阈值).
论文已建议 "periodic re-tuning ... operationally cheap" 但无实验支撑.
本脚本做滚动窗口重整定: 每步用过去 W 小时训练, 预测未来 Δ 小时,
拼接全程告警后评估 —— 与静态时序 (0.001) 对照. 两方法同协议:
  ours-rolling   : train_tuning (constrained) 于尾窗
  fixed-rolling  : q×合并窗 网格于尾窗 (预算内)
参数: W=700h, Δ=175h (流总长 2106h → 测试覆盖 700h 以后全程).

输出: results/audit/r27_rolling_metro.csv
运行: cd src && python -u experiments/e_r27_rolling_metro.py
"""
import os
import sys
import time as _time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline, merge_gaps
from experiments.e3_baselines import _kmax

W_H, D_H = 700.0, 175.0


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r27_rolling_metro.csv")
    s, y, dt = get_scores("iforest", "MetroPT3", seed=SEED)
    T = len(s)
    hpph = 3600.0 / dt                      # 点/小时
    W, D = int(W_H * hpph), int(D_H * hpph)
    K_on, K_off = _kmax(dt)
    cfg = KPIConfig(dt=dt)
    rows = []
    for method in ("ours", "fixed"):
        alarm = np.zeros(T, dtype=np.int8)
        t0 = _time.time()
        folds = 0
        for start in range(W, T, D):
            s_w, y_w = s[start - W:start], y[start - W:start]
            end = min(start + D, T)
            if method == "ours":
                prm, _ = train_tuning(s_w, y_w, events_from_binary(y_w),
                                      dt, R_max=10.0, F_max=0.02, epochs=300,
                                      seed=SEED, K_on=K_on, K_off=K_off,
                                      init_t_on=13.0)
                alarm[start:end] = hard_pipeline(
                    s[start:end], prm["theta_hi"], prm["theta_lo"],
                    prm["T_on"], prm["T_off"])
            else:
                best = (None, -1)
                for q in (0.90, 0.95, 0.99):
                    for wt in (300, 600, 1800):
                        Wm = max(1, int(wt / dt))
                        al = merge_gaps((s_w > np.quantile(s_w, q)).astype(int),
                                        Wm)
                        p = kpi_panel(al, y_w, cfg=cfg)
                        if p["AAR_per_hour"] <= 10.0 and p["EventF1"] > best[1]:
                            best = ((q, Wm), p["EventF1"])
                q, Wm = best[0]
                alarm[start:end] = merge_gaps(
                    (s[start:end] > np.quantile(s_w, q)).astype(int), Wm)
            folds += 1
        wall = _time.time() - t0
        p = kpi_panel(alarm[W:], y[W:], cfg=cfg)
        p.update({"method": f"{method}-rolling", "folds": folds,
                  "wall_s_per_fold": wall / folds,
                  "window_h": W_H, "step_h": D_H})
        rows.append(p)
        print(f"[r27] {method}-rolling ({folds} 折, {wall/folds:.0f}s/折): "
              f"F1={p['EventF1']:.3f} det={p['event_detection_rate']:.2f} "
              f"AAR={p['AAR_per_hour']:.3f} n_ev={p['n_alarm_events']} "
              f"MAD={p['mean_alarm_delay_s']:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}  (静态时序参照: 两方法 F1≈0.001)")


if __name__ == "__main__":
    main()
