"""第 4 轮 (外部 AI 审稿) 审计实验: 对 DeepSeek 三轮意见中的数值/数学疑点逐条取证.

Part 1 (快, 无需训练):
  A. theta 边界: e4 轨迹终值 vs 调优段分数 [q50, q99.9] —— 是否全部严格内部
  B. FAR 代理间隙: e5_gap_history 软/硬 FAR 随温度 —— 两阶段退火机理量化
  C. 软硬算子一致性: 终值参数 + 终值温度, soft_pipeline vs hard_pipeline
     + on-delay 窗口语义: tau_v->0 极限是 cap-K 最小值 (DeepSeek 指出的极限语义),
       工作温度下近似 T 窗最小值 —— 两组数值都给出
  D. AT 1s vs 5s 时间归一化: 每分钟告警点数 (E2 动机数字口径)
Part 2 (慢, ~5min): MetroPT3 x iforest x R_max=0.02 复现 + 事件级审计
  (n_alarm=1/n_gt=3/det=1.0/F1=0.5 的机制: 常驻告警事件?)

输出: results/audit/*.csv|.json|.txt
运行: cd src && python experiments/e_audit_r4.py [--metro-only|--quick-only]
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.soft_ops import (soft_pipeline, soft_on_delay, window_weights)
from tuning.hard_ops import hard_pipeline
from experiments.e_pilot_tuning import interleave_split
from experiments.e3_baselines import _kmax

OUT = os.path.join(RESULTS_DIR, "audit")
os.makedirs(OUT, exist_ok=True)


def part_a_theta_bounds():
    traj = pd.read_csv(os.path.join(RESULTS_DIR, "e4", "e4_param_trajectory.csv"))
    rows = []
    for (ds, sc), g in traj.groupby(["dataset", "scorer"]):
        fin = g.iloc[-1]
        s, y, dt = get_scores(sc, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr = s[tr_m]
        q50, q999 = np.quantile(s_tr, 0.50), np.quantile(s_tr, 0.999)
        th = float(fin["theta_hi"])
        rows.append(dict(dataset=ds, scorer=sc, theta_hi=th, q50=q50, q999=q999,
                         lo_gap=th - q50, hi_gap=q999 - th,
                         interior=(th > q50) and (th < q999),
                         rel_pos=(th - q50) / max(q999 - q50, 1e-12)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "theta_bounds.csv"), index=False,
              encoding="utf-8-sig")
    print("[A] theta 边界核查 (rel_pos = (θ-q50)/(q99.9-q50), 内部应 ∈ (0,1)):")
    print(df.to_string(index=False))
    return df


def part_b_far_gap():
    hdf = pd.read_csv(os.path.join(RESULTS_DIR, "e5", "e5_gap_history.csv"))
    hdf = hdf.dropna(subset=["hard_far"])
    rows = []
    for ds, g in hdf.groupby("dataset"):
        for tf in (1.0, 0.5, 0.3, 0.1, 0.05):
            sub = g.iloc[(g["temp_frac"] - tf).abs().argsort()[:1]]
            if not len(sub):
                continue
            r = sub.iloc[0]
            rows.append(dict(dataset=ds, temp_frac=tf,
                             soft_far=float(r["far"]), hard_far=float(r["hard_far"]),
                             ratio=float(r["far"]) / max(float(r["hard_far"]), 1e-9)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "far_gap.csv"), index=False,
              encoding="utf-8-sig")
    print("\n[B] 软/硬 FAR 随温度 (两阶段退火机理量化):")
    print(df.to_string(index=False))
    return df


def part_c_soft_hard():
    traj = pd.read_csv(os.path.join(RESULTS_DIR, "e4", "e4_param_trajectory.csv"))
    rows = []
    for (ds, sc), g in traj.groupby(["dataset", "scorer"]):
        fin = g.iloc[-1]
        s, y, dt = get_scores(sc, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_te = torch.tensor(s[te_m], dtype=torch.float32)
        K_on, K_off = _kmax(dt)
        t_hi = float(fin["theta_hi"]); m = float(fin["margin"])
        T_on = float(fin["T_on"]); T_off = float(fin["T_off"])
        temp_s = float(fin["temp_frac"]) * float(np.std(s[te_m]))
        temp_v = float(fin["temp_v"])
        a_soft = soft_pipeline(s_te, t_hi, t_hi - m, T_on, T_off,
                               temp_s, temp_v, K_on, K_off).numpy()
        a_hard = hard_pipeline(s[te_m], t_hi, t_hi - m,
                               max(1, round(T_on)), max(1, round(T_off)))
        rows.append(dict(dataset=ds, scorer=sc, temp_s=temp_s, temp_v=temp_v,
                         T_on=round(T_on, 2), K_on=K_on,
                         mean_abs_diff=float(np.abs(a_soft - a_hard).mean()),
                         max_abs_diff=float(np.abs(a_soft - a_hard).max()),
                         corr=float(np.corrcoef(a_soft, a_hard)[0, 1])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "soft_hard_agreement.csv"), index=False,
              encoding="utf-8-sig")
    print("\n[C] 终值温度下软链 vs 硬链一致性 (tuning-test 段):")
    print(df.to_string(index=False))

    # on-delay 窗口语义: h = 前 3 点高, 之后 0; T_on=3, K_on=8
    h = torch.tensor([1.0, 1.0, 1.0] + [0.0] * 5)
    for tv in (1e-3, 0.05, 0.15):
        u = float(soft_on_delay(h, torch.tensor(3.0), tv, 8)[-1])
        print(f"    on-delay 语义: h=[1,1,1,0,0,0,0,0], T_on=3, K_on=8, "
              f"tau_v={tv}: u_t={u:.4f}  (硬 T 窗最小值=0, 硬 cap 最小值=0; "
              f"参考: 窗内保持值=1)")
    # 窗内全高时输出应接近 1 (窗口语义在工作温度下成立)
    h2 = torch.ones(8)
    for tv in (1e-3, 0.05):
        u2 = float(soft_on_delay(h2, torch.tensor(3.0), tv, 8)[-1])
        print(f"    on-delay 语义: h=全1, T_on=3, K_on=8, tau_v={tv}: "
              f"u_t={u2:.4f} (硬输出=1)")
    w = window_weights(torch.tensor(3.0), 8).numpy()
    print(f"    w_k(T=3,K=8) = {np.round(w, 4)}  (tau_w=0.5 固定)")
    return df


def part_d_at_normalization():
    e2 = pd.read_csv(os.path.join(RESULTS_DIR, "e2",
                                  "e2_per_event_alarm_counts.csv"))
    s, y, dt = get_scores("cmhmil", "SWaT", seed=SEED)
    tr_m, te_m = interleave_split(len(s), dt)
    dur_h = y[te_m].sum() * dt / 3600.0
    rows = []
    for sc, native_dt in (("cmhmil", 5.0), ("iforest", 5.0), ("AT", 1.0)):
        sub = e2[(e2.dataset == "SWaT") & (e2.scorer == sc)]
        if not len(sub):
            continue
        cnt = float(sub["alarm_points"].max())
        rows.append(dict(dataset="SWaT", scorer=sc, native_dt_s=native_dt,
                         alarm_points=cnt, episode_hours=dur_h,
                         per_minute=cnt / (dur_h * 60)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "at_normalization.csv"), index=False,
              encoding="utf-8-sig")
    print(f"\n[D] SWaT 攻击段时长 {dur_h:.1f} h; 各打分器原生采样口径下的归一化:")
    print(df.to_string(index=False))
    return df


def part_e_metro_tight():
    """MetroPT3 x iforest x R_max=0.02 复现 + 事件级审计 (慢)."""
    from scipy.optimize import linear_sum_assignment
    s, y, dt = get_scores("iforest", "MetroPT3", seed=SEED)
    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    K_on, K_off = _kmax(dt)
    params, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=0.02, F_max=0.02,
                             epochs=300, seed=SEED, K_on=K_on, K_off=K_off)
    alarm = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                          params["T_on"], params["T_off"])
    gt_ev = events_from_binary(y_te)
    al_ev = events_from_binary(alarm)
    hours = len(s_te) * dt / 3600.0
    # 事件区间重合矩阵 (交并比)
    iou = np.zeros((len(gt_ev), len(al_ev)))
    ov = np.zeros_like(iou)
    for i, (a0, a1) in enumerate(gt_ev):
        for j, (b0, b1) in enumerate(al_ev):
            inter = max(0, min(a1, b1) - max(a0, b0))
            union = max(a1, b1) - min(a0, b0)
            ov[i, j] = inter
            iou[i, j] = inter / union if union else 0.0
    r, c = linear_sum_assignment(-iou)
    matched = [(int(i), int(j), float(iou[i, j]))
               for i, j in zip(r, c) if iou[i, j] > 0]
    report = dict(
        params={k: (float(v) if not isinstance(v, (int, np.integer)) else int(v))
                for k, v in params.items()},
        tuning_test_hours=hours, n_gt_events=len(gt_ev),
        n_alarm_events=len(al_ev),
        gt_events_h=[[a0 * dt / 3600, a1 * dt / 3600,
                      (a1 - a0) * dt / 3600] for a0, a1 in gt_ev],
        alarm_events_h=[[b0 * dt / 3600, b1 * dt / 3600,
                         (b1 - b0) * dt / 3600] for b0, b1 in al_ev],
        time_in_alarm_frac=float(alarm.mean()),
        one2one_matched=matched,
        tp=len(matched), fp=len(al_ev) - len(matched),
        fn=len(gt_ev) - len(matched),
        EventF1=(2 * len(matched) /
                 (2 * len(matched) +
                  (len(al_ev) - len(matched)) + (len(gt_ev) - len(matched)))),
        AAR=len(al_ev) / hours,
        kpi=kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt)))
    import json
    with open(os.path.join(OUT, "metro_tight_audit.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print("[E] MetroPT3 紧预算审计:")
    print(f"    params={report['params']}")
    print(f"    n_gt={report['n_gt_events']} n_alarm={report['n_alarm_events']} "
          f"time_in_alarm={report['time_in_alarm_frac']:.4f}")
    print(f"    matched(1-1)={report['one2one_matched']}")
    print(f"    F1={report['EventF1']:.3f} AAR={report['AAR']:.5f}/h "
          f"MAD={report['kpi']['mean_alarm_delay_s']:.0f}s "
          f"det={report['kpi']['event_detection_rate']:.2f}")
    for k, (a, b, d) in enumerate(report["alarm_events_h"]):
        print(f"    alarm event {k}: [{a:.1f}h, {b:.1f}h) dur={d:.1f}h")
    for k, (a, b, d) in enumerate(report["gt_events_h"]):
        print(f"    gt event    {k}: [{a:.1f}h, {b:.1f}h) dur={d:.1f}h")


if __name__ == "__main__":
    args = set(sys.argv[1:])
    if not args or "--quick-only" in args:
        part_a_theta_bounds()
        part_b_far_gap()
        part_c_soft_hard()
        part_d_at_normalization()
    if not args or "--metro-only" in args:
        part_e_metro_tight()
    print(f"\n写出 {OUT}/")
