"""E-pilot: 可微整定层 vs 原始阈值 vs 固定窗口合并 (第5周 Go/No-Go 检查).

三种方法在同一分数流上对比 (均为推理侧 100% 硬规则, 统一 KPI 面板评估):
  raw        : 阈值 = 整定训练段分数 99 分位 (E2 固定规则), 无整定算子.
  fixed-win  : 经典固定窗口合并 — 网格 (阈值分位数 x 合并窗 W) 在整定训练段上按
               Event-F1 (s.t. AAR<=10/h) 选最优, 测试段应用.
  diff-tune  : 本文可微整定层 (soft 迟滞 + on/off-delay, 拉格朗日约束训练),
               学到的硬参数在测试段应用.

数据划分 (诚实性): 交错块划分 — 12h 块交替分给 tuning-train / tuning-test,
避免纯时序对半切把事件全分到一侧 (SWaT/MetroPT3 的事件集中在流后段).
固定窗口基线的网格搜索同样只用 tuning-train 段, 与可微层信息量对齐.

输出: results/e_pilot/e_pilot_tuning.csv + 训练曲线数据
运行: cd src && python experiments/e_pilot_tuning.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_delay_timer, hard_pipeline, merge_gaps

COMBOS = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]
BLOCK_H = 12.0        # 交错块大小
R_MAX = 10.0          # AAR 约束 (条/h, EEMUA 设计目标)
F_MAX = 0.02          # 正常点软告警率约束
# 固定窗口基线网格
Q_GRID = [0.90, 0.95, 0.99]
W_GRID_S = [0, 60, 180, 300, 600, 900]     # 合并窗 (秒)


def interleave_split(n, dt):
    """12h 块交替 -> (train_idx, test_idx) bool 掩码."""
    blk = max(1, int(BLOCK_H * 3600 / dt))
    idx = np.arange(n)
    block_id = idx // blk
    return (block_id % 2 == 0), (block_id % 2 == 1)


def eval_alarm(alarm, y, dt, tag):
    p = kpi_panel(alarm, y, cfg=KPIConfig(dt=dt))
    p["method"] = tag
    return p


def main():
    out_dir = os.path.join(RESULTS_DIR, "e_pilot")
    os.makedirs(out_dir, exist_ok=True)
    rows, train_logs = [], []
    for scorer, ds in COMBOS:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr = s[tr_m], y[tr_m]
        s_te, y_te = s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)
        print(f"\n[pilot] {scorer} x {ds}: train={tr_m.sum()} test={te_m.sum()} "
              f"train_events={len(ev_tr)} test_events={len(events_from_binary(y_te))}")

        # --- 方法1: raw (E2 固定规则: 训练段 q99, 无整定) ---
        thr_raw = np.quantile(s_tr, 0.99)
        alarm_raw = (s_te > thr_raw).astype(int)

        # --- 方法2: fixed-window merge (网格在训练段选) ---
        best = None
        hours_tr = tr_m.sum() * dt / 3600.0
        for q in Q_GRID:
            thr = np.quantile(s_tr, q)
            b_tr = (s_tr > thr).astype(int)
            from kpi.panel import rising_edges
            aar_tr = len(rising_edges(b_tr)) / max(hours_tr, 1e-6)
            if aar_tr > R_MAX:
                continue                       # 违反 AAR 约束的网格点弃用
            from kpi.panel import event_f1_one2one
            for w_s in W_GRID_S:
                w_pt = int(w_s / dt)
                m_tr = merge_gaps(b_tr, w_pt)
                f1 = event_f1_one2one(ev_tr, events_from_binary(m_tr))
                if best is None or f1 > best[0]:
                    best = (f1, q, w_pt)
        if best is None:                       # 全网格违反 AAR -> 用最保守点
            best = (float("nan"), 0.99, int(600 / dt))
        _, q_best, w_best = best
        alarm_fix = merge_gaps((s_te > np.quantile(s_tr, q_best)).astype(int), w_best)

        # --- 方法3: 可微整定层 ---
        t0 = time.time()
        params, hist = train_tuning(
            s_tr, y_tr, ev_tr, dt,
            R_max=R_MAX, F_max=F_MAX, epochs=300, seed=SEED,
            K_on=min(24, max(4, int(120 / dt))),
            K_off=min(48, max(8, int(300 / dt))))
        train_logs.append({"dataset": ds, "scorer": scorer, "history": hist})
        alarm_diff = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                                   params["T_on"], params["T_off"])
        print(f"  diff-tune 参数: θ={params['theta_hi']:.3f} margin={params['theta_hi']-params['theta_lo']:.3f} "
              f"T_on={params['T_on']} T_off={params['T_off']} ({time.time()-t0:.0f}s)")
        print(f"  fixed-win 选点: q={q_best} W={w_best * dt:.0f}s | raw thr={thr_raw:.3f}")

        for tag, alarm in [("raw", alarm_raw), ("fixed-win", alarm_fix),
                           ("diff-tune", alarm_diff)]:
            p = eval_alarm(alarm, y_te, dt, tag)
            p.update({"dataset": ds, "scorer": scorer})
            rows.append(p)
            print(f"  {tag:9s} EventF1={p['EventF1']:.3f} AAR={p['AAR_per_hour']:.1f} "
                  f"chatter={p['chatter_rate']:.2f} MAD={p['mean_alarm_delay_s']:.0f}s "
                  f"det={p['event_detection_rate']:.2f} n_ev={p['n_alarm_events']}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "e_pilot_tuning.csv"), index=False,
              encoding="utf-8-sig")
    # 训练历史落盘 (E5 软硬间隙专节素材: 软指标随温度变化轨迹)
    hrows = []
    for log in train_logs:
        for h in log["history"]:
            hrows.append({"dataset": log["dataset"], "scorer": log["scorer"], **h})
    pd.DataFrame(hrows).to_csv(os.path.join(out_dir, "e_pilot_train_history.csv"),
                               index=False, encoding="utf-8-sig")
    print(f"\n[pilot] 写出 {out_dir}/e_pilot_tuning.csv + e_pilot_train_history.csv")
    cols = ["dataset", "scorer", "method", "EventF1", "AAR_per_hour",
            "chatter_rate", "mean_alarm_delay_s", "event_detection_rate",
            "n_alarm_events"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
