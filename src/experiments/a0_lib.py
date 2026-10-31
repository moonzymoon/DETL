# -*- coding: utf-8 -*-
"""A0 新协议核心库 (设计文档 v2 冻结版实现).

组件:
  1. TE 流分段加载 (7 段: normal_500 + 6 故障批, 段界状态复位)
  2. 协议 M: 全时间轴逐段硬链 + 12h 掩码 + 跨块事件唯一归属规则
     (GT/报警均按起始样本归属侧; 匹配/延迟/FP 在该侧掩码样本上计算)
  3. 统一参数域: margin ∈ [0, 2σ]; 搜索=约束感知 TPE + 可行优先两级选解;
     层=sigmoid 箱 (m_max=2σ), float64, 分段软扫描, 掩码损失
  4. 指标: any-overlap F1 (主) + IoU>=0.1 F1 (并报) + 检出/延迟(连漏报)/AAR/FAR/
     time-in-alarm/报警时长/stale(>=1h) 四件套
预注册: 00返修/A0_固定设计文档.md v2. 本库任何行为变更须登记修正册.
"""
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from kpi.panel import (KPIConfig, events_from_binary, rising_edges)  # noqa: E402
from tuning.constrained import TuningParams  # noqa: E402
from tuning.losses import soft_chatter  # noqa: E402
from tuning.soft_ops import soft_pipeline  # noqa: E402
from tuning.hard_ops import hard_pipeline  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

torch.set_default_dtype(torch.float64)      # A0.3 决策: 全 float64

R_MAX, F_MAX = 10.0, 0.02                   # 硬约束 (整定段硬值判定)
BLOCK_H = 12.0                              # 交错块
M_SIGMA = 2.0                               # margin 上限 = 2·std(s_tune)
STALE_MIN = 60.0                            # stale 报警阈值: 持续 >= 1h


# ---------------------------------------------------------------- 数据 ----
def load_te_segments():
    """返回 (scores, labels, dt, seg_bounds, seg_meta).

    分数流 = 全时间轴后 50% (35/15/50 协议: 打分器拟合用前 35%, 分数从 b=int(0.5n)
    起按窗尾对齐). 段边界映射到分数流坐标: 正常尾段 + 6 个故障批 (段界=独立仿真).
    """
    from datasets.tep import load_tep, FAULT_FILES
    d = load_tep()
    X, Y, dt = d["X"], d["Y"], d["dt"]
    from scorers.adapter import get_scores, local_iforest
    s, y, dt = get_scores("iforest", "TE")     # 分数缓存 (投稿版同源)
    # 复算坐标映射 (与 local_iforest 相同的 35/15/50 + window=16 窗尾对齐)
    window = 16
    n_full = len(X)
    b = int(n_full * 0.50)
    off = b + window - 1                        # 分数 j <-> 全轴 j+off
    seg_bounds, seg_meta = [], []
    # 段 0: normal 尾段 (全轴 [off, 30001))
    e0 = 30001 - off
    if e0 > 0:
        seg_bounds.append((0, e0)); meta = [("normal_tail", 0)]
    p = 30001
    for f in FAULT_FILES:
        idv = int(f.split("_")[1])
        s0, e1 = max(0, p - off), p + 3001 - off
        if e1 > s0:
            seg_bounds.append((s0, e1)); meta.append((f, idv))
        p += 3001
    assert seg_bounds[-1][1] == len(s), \
        f"段边界与分数流长不符: {seg_bounds[-1][1]} vs {len(s)}"
    return np.asarray(s, dtype=np.float64), np.asarray(y), float(dt), \
        seg_bounds, meta


def interleave_mask(n, dt, bounds):
    """12h 块交替掩码 (块编号按完整时间轴索引), 返回 (tune_m, test_m)."""
    blk = max(1, int(BLOCK_H * 3600 / dt))
    bid = np.arange(n) // blk
    return bid % 2 == 0, bid % 2 == 1


# ------------------------------------------------------------ 硬链分段 ----
def hard_chain_segmented(s, prm, bounds):
    """段界复位的硬报警链. prm: dict(theta_hi, theta_lo, T_on, T_off) 整数窗."""
    out = np.zeros(len(s), dtype=np.int8)
    for s0, e0 in bounds:
        out[s0:e0] = hard_pipeline(s[s0:e0], prm["theta_hi"], prm["theta_lo"],
                                   int(prm["T_on"]), int(prm["T_off"]))
    return out


# ------------------------------------------------------- 协议 M 指标 ------
def _iou_on_mask(gs, ge, ps, pe, side):
    """事件 [gs,ge) 与报警 [ps,pe) 在 side 掩码样本上的 IoU."""
    inter_idx = np.arange(max(gs, ps), min(ge, pe))
    if len(inter_idx) == 0:
        return 0.0, 0
    inter = int(side[inter_idx].sum())
    union = (int(side[np.arange(gs, ge)].sum()) + int(side[np.arange(ps, pe)].sum())
             - inter)
    return (inter / union if union else 0.0), inter


def protocol_M_metrics(alarm, y, side, dt, iou_thr=0.1):
    """协议 M 指标面板. side: 本侧掩码 (bool). 归属=起始样本在本侧."""
    a = np.asarray(alarm)
    gt = [(s, e) for s, e in events_from_binary(y) if side[s]]
    pr = [(s, e) for s, e in events_from_binary(a) if side[s]]
    n = len(a)
    res = {"n_gt_side": len(gt), "n_alarm_side": len(pr)}

    # 一对一指派 (any-overlap 于 side 样本)
    def match_matrix(thr):
        if not gt or not pr:
            return 0
        M = np.zeros((len(gt), len(pr)))
        for i, (gs, ge) in enumerate(gt):
            for j, (ps, pe) in enumerate(pr):
                iou, inter = _iou_on_mask(gs, ge, ps, pe, side)
                M[i, j] = 1.0 if (inter > 0 and (thr is None or iou >= thr)) else 0.0
        rows, cols = linear_sum_assignment(-M)
        return int(sum(1 for r, c in zip(rows, cols) if M[r, c] > 0))

    for tag, thr in [("ov", None), ("iou01", iou_thr)]:
        tp = match_matrix(thr)
        fp, fn = len(pr) - tp, len(gt) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        res[f"EF1_{tag}"] = float(2 * p * r / (p + r) if p + r else 0.0)
        res[f"det_{tag}"] = tp / len(gt) if gt else float("nan")

    # 延迟: 事件进入本侧首样本 -> 本侧首个交集报警首样本 (与漏报率同报)
    lags = []
    for gs, ge in gt:
        idx = np.arange(gs, ge)[side[gs:ge]]
        hit = -1
        for t in idx:
            if a[t]:
                hit = t
                break
        if hit >= 0:
            lags.append((hit - gs) * dt)
    res["MAD_s"] = float(np.mean(lags)) if lags else float("nan")
    res["n_missed"] = len(gt) - len(lags)
    res["miss_rate"] = res["n_missed"] / len(gt) if gt else float("nan")

    # 负荷类 (本侧样本)
    side_h = side.sum() * dt / 3600.0
    rises = [t for t in rising_edges(a) if side[t]]
    res["AAR_side"] = len(rises) / side_h if side_h else float("nan")
    normal_side = side & (y == 0)
    res["FAR_side"] = float(a[normal_side].mean()) if normal_side.any() else 0.0
    res["feasible"] = bool(res["AAR_side"] <= R_MAX and res["FAR_side"] <= F_MAX)
    # R2.4 四件套
    res["TIA_side"] = float(a[side].mean())
    durs_h = [(e - s) * dt / 3600.0 for s, e in pr]
    res["alarm_dur_med_h"] = float(np.median(durs_h)) if durs_h else float("nan")
    res["n_stale_1h"] = int(sum(1 for d_ in durs_h if d_ >= 1.0))
    return res


def eval_params_M(prm, s, y, side, dt, bounds):
    alarm = hard_chain_segmented(s, prm, bounds)
    m = protocol_M_metrics(alarm, y, side, dt)
    m["theta_hi"] = prm["theta_hi"]
    m["margin"] = prm["theta_hi"] - prm["theta_lo"]
    m["T_on"], m["T_off"] = int(prm["T_on"]), int(prm["T_off"])
    return m


# ------------------------------------------------ 层: 分段软扫描 + 掩码损失 -
def train_layer_M(s, y, tune_m, bounds, dt, epochs=300, lr=0.05,
                  lambda_lag=0.5, eta_c=0.05, seed=7, verbose=False,
                  disable=(), loss_mode="constrained"):
    """协议 M 下的层训练: 段界复位软扫描, 损失只计整定侧样本与其起始事件.

    disable/loss_mode 为 B 核心 (协议 M 重跑消融/损失对比) 移植,
    语义与旧 train_tuning 一致 (冻结登记_补实验_20261025.md).
    """
    torch.manual_seed(seed)
    s_t = torch.as_tensor(s)
    y_np = np.asarray(y)
    sd = float(s[tune_m].std())
    m_max = M_SIGMA * sd
    K_on, K_off = 24, 48
    # 初始化分位数用整定侧样本 (与投稿版语义一致: 仅整定数据)
    s_tune = s_t[torch.as_tensor(tune_m)]
    model = TuningParams(s_tune, K_on=K_on, K_off=K_off, m_max=m_max)
    with torch.no_grad():
        if "hysteresis" in disable:                  # margin -> ~0
            model._margin_raw.fill_(-20.0)
            model._margin_raw.requires_grad_(False)
        if "on_delay" in disable:                    # T_on -> 1
            model._t_on.fill_(-20.0)
            model._t_on.requires_grad_(False)
        if "off_delay" in disable:                   # T_off -> 1
            model._t_off.fill_(-20.0)
            model._t_off.requires_grad_(False)
    # 事件: 起始样本在整定侧 (完整区间参与, 信用取整定侧片段)
    gt_all = events_from_binary(y_np)
    ev_tune = [(a, b) for a, b in gt_all if tune_m[a]]
    tm = torch.as_tensor(tune_m)
    nm = tune_m & (y_np == 0)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    mu = np.array([1.0, 1.0])
    hist = []
    BETA = 10.0
    for ep in range(epochs):
        temp_frac = max(0.05, 1.0 * (0.985 ** ep))
        temp_v = max(0.02, 0.15 * (0.985 ** ep))
        opt.zero_grad()
        # 段界复位软扫描 -> 拼接全流 ã
        a_full = torch.cat([
            model(s_t[s0:e0], temp_frac, temp_v) for s0, e0 in bounds])
        temp_s = max(temp_frac, 0.01) * model.std
        # --- 目标: 整定侧事件 (miss + lag, 信用取整定侧片段) ---
        miss_terms, lag_terms = [], []
        for g0, g1 in ev_tune:
            idx = np.arange(g0, g1)[tune_m[g0:g1]]
            L = max(len(idx), 1)
            seg = a_full[torch.as_tensor(idx)]
            miss_terms.append(1.0 - torch.logsumexp(BETA * seg, 0) / BETA)
            pm = torch.logcumsumexp(BETA * seg, 0) / BETA
            pm_prev = torch.cat([torch.zeros(1), pm[:-1]])
            p = seg * (1 - pm_prev).clamp(0, 1)
            frac = torch.arange(L, dtype=torch.float64) / L
            lag_terms.append((p * frac).sum() + (1 - p.sum().clamp(0, 1)))
        miss = torch.stack(miss_terms).mean() if miss_terms else torch.zeros(())
        lag = torch.stack(lag_terms).mean() if lag_terms else torch.zeros(())
        # --- 约束: 整定侧样本上的软 AAR / FAR ---
        # AAR 边沿只在真实相邻样本对上计 (两侧都在整定侧), 不跨掩码拼接缝
        rises_full = torch.relu(a_full[1:] - a_full[:-1])
        pair = tm[1:] & tm[:-1]
        rises = (rises_full * pair.double()).sum()
        aar = rises / max(tm.sum().item() * dt / 3600.0, 1e-6)
        a_nm = a_full[torch.as_tensor(nm)]
        far = a_nm.mean() if nm.any() else torch.zeros(())
        viol = [max(0.0, float(aar) - R_MAX) / R_MAX,
                max(0.0, float(far) - F_MAX) / F_MAX]
        constraints_on = (temp_frac <= 0.3) or (not ev_tune)
        pen = (mu[0] * torch.relu(aar - R_MAX) / R_MAX
               + mu[1] * torch.relu(far - F_MAX) / F_MAX)
        if loss_mode == "relaxed_f1":
            # B2 对照: 点级 relaxed F1 (软 TP/FP/FN, 整定侧样本), 无 ISA 约束项
            a_tu = a_full[tm]
            y_tu = torch.as_tensor(y_np[tune_m]).double()
            tp = (a_tu * y_tu).sum()
            fp = (a_tu * (1 - y_tu)).sum()
            fn = ((1 - a_tu) * y_tu).sum()
            prec = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            f1s = 2 * prec * rec / (prec + rec + 1e-6)
            loss = 1.0 - f1s
        else:
            loss = miss + lambda_lag * lag + (pen if constraints_on else 0.0)
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        else:
            opt.zero_grad()
        if constraints_on:
            mu = np.minimum(mu * np.exp(eta_c * np.array(viol)), 100.0)
        hist.append({"ep": ep, "loss": float(loss.detach()),
                     "miss": float(miss.detach()), "lag": float(lag.detach()),
                     "aar": float(aar), "far": float(far),
                     "theta_hi": float(model.theta_hi),
                     "margin": float(model.margin()),
                     "T_on": float(model.T_on()), "T_off": float(model.T_off())})
        if verbose and ep % 50 == 0:
            print(f"  ep{ep} miss={float(miss):.3f} aar={float(aar):.2f} "
                  f"far={float(far):.4f} m={float(model.margin()):.4f}")
    prm = {"theta_hi": float(model.theta_hi),
           "theta_lo": float(model.theta_hi - model.margin()),
           "T_on": max(1, int(round(float(model.T_on())))),
           "T_off": max(1, int(round(float(model.T_off()))))}
    return prm, hist, float(m_max)


# --------------------------------------------- 统一搜索: 约束感知 + 可行优先 -
def search_unified_M(s, y, tune_m, bounds, dt, seed, budget=240, use_tpe=True,
                     m_sigma=M_SIGMA):
    """统一参数域搜索. 返回 (选解@嵌套前缀60/120/240, 试验表)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sd = float(s[tune_m].std())
    m_max = m_sigma * sd
    K_on, K_off = 24, 48

    def hard_kpi_tune(q, mf, t_on, t_off):
        prm = {"theta_hi": float(np.quantile(s[tune_m], q)),
               "theta_lo": float(np.quantile(s[tune_m], q)) - mf * sd,
               "T_on": int(t_on), "T_off": int(t_off)}
        m = eval_params_M(prm, s, y, tune_m, dt, bounds)
        return m

    trials_log = []

    def objective(trial):
        q = trial.suggest_float("q", 0.50, 0.999)
        mf = trial.suggest_float("mf", 0.0, m_sigma)     # margin/σ 连续
        t_on = trial.suggest_int("t_on", 1, K_on)
        t_off = trial.suggest_int("t_off", 1, K_off)
        t0 = time.time()
        m = hard_kpi_tune(q, mf, t_on, t_off)
        viol_aar = max(0.0, m["AAR_side"] - R_MAX) / R_MAX
        viol_far = max(0.0, m["FAR_side"] - F_MAX) / F_MAX
        trials_log.append({"trial": trial.number, "q": q, "mf": mf,
                           "t_on": t_on, "t_off": t_off,
                           "EF1_ov": m["EF1_ov"], "AAR": m["AAR_side"],
                           "FAR": m["FAR_side"], "feas": m["feasible"],
                           "wall": time.time() - t0})
        trial.set_user_attr("viol", (viol_aar, viol_far))
        trial.set_user_attr("ef1", m["EF1_ov"])
        return m["EF1_ov"]

    def constraints_fun(trial):
        v = trial.user_attrs["viol"]
        return np.array(v)

    sampler = (optuna.samplers.TPESampler(seed=seed, constraints_func=constraints_fun)
               if use_tpe else
               optuna.samplers.RandomSampler(seed=seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    with __import__("optuna").progress.disable_progress_bar() if False else _noop():
        study.optimize(objective, n_trials=budget, show_progress_bar=False)

    def pick(feasible, ef1, aar, far, n_pref):
        """可行优先两级: 可行中 EF1 最大(平局 AAR 低, 再 FAR 低); 否则违约字典序."""
        best = None
        for r in [t for t in trials_log if t["trial"] < n_pref]:
            key = (0 if r["feas"] else 1, -r["EF1_ov"] if r["feas"] else 0,
                   r["AAR"] if r["feas"] else 0, r["FAR"] if r["feas"] else 0,
                   0 if r["feas"] else r["AAR"], 0 if r["feas"] else r["FAR"],
                   r["trial"])
            if best is None or key < best[0]:
                best = (key, r)
        return best[1]

    out = {}
    for B in (60, 120, 240):
        r = pick(None, None, None, None, B)
        out[B] = {"q": r["q"], "mf": r["mf"], "t_on": r["t_on"],
                  "t_off": r["t_off"], "feasible": r["feas"],
                  "prm": {"theta_hi": float(np.quantile(s[tune_m], r["q"])),
                          "theta_lo": float(np.quantile(s[tune_m], r["q"]))
                          - r["mf"] * sd,
                          "T_on": int(r["t_on"]), "T_off": int(r["t_off"])}}
    return out, pd.DataFrame(trials_log)


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ------------------------------------------------------------ 玩具验证 ----
def toy_protocol_test():
    """设计文档 §4 玩具验证: 2 段 x 1 跨块事件 x 3 报警位置, 断言唯一规则输出."""
    n = 48
    s = np.full(n, 0.0)
    y = np.zeros(n, dtype=np.int8)
    bounds = [(0, 24), (24, 48)]
    # 掩码: 12 样本块交替 (dt=60s -> 12h 块缩为 12 样本)
    blk = 12
    bid = np.arange(n) // blk
    tune_m, test_m = bid % 2 == 0, bid % 2 == 1
    # GT 事件: [8, 30) 起始于块0(整定侧)但跨到块1
    y[8:30] = 1
    checks = []
    # 情形 A: 报警 [10,14) 起始于整定侧块0, 延伸进入测试块1
    alarmA = np.zeros(n, dtype=np.int8); alarmA[10:14] = 1
    mA = protocol_M_metrics(alarmA, y, tune_m, 60.0)
    # 事件起始 8 在整定侧 -> 归整定侧; 报警起始 10 在整定侧 -> 整定侧;
    # 整定侧交集样本 10,11 -> any-overlap 命中; 延迟 = (10-8)*60
    checks.append(("A: 事件归整定侧, 检出", mA["n_gt_side"] == 1 and mA["det_ov"] == 1))
    checks.append(("A: 延迟=(10-8)*60", abs(mA["MAD_s"] - 2 * 60) < 1e-9))
    mAt = protocol_M_metrics(alarmA, y, test_m, 60.0)
    checks.append(("A: 测试侧无 GT 且报警起始不在 -> 全空",
                   mAt["n_gt_side"] == 0 and mAt["n_alarm_side"] == 0))
    # 情形 B: 报警 [26,28) 完全在块2 (整定侧, 事件整定侧片段内)
    alarmB = np.zeros(n, dtype=np.int8); alarmB[26:28] = 1
    mB = protocol_M_metrics(alarmB, y, tune_m, 60.0)
    # 事件整定侧样本 = [8,12)∪[24,30); 报警 [26,28) 交集命中; 延迟从事件整定侧
    # 首样本(8)计 = (26-8)*60
    checks.append(("B: 整定侧后段报警命中, 延迟=(26-8)*60",
                   mB["det_ov"] == 1 and abs(mB["MAD_s"] - 18 * 60) < 1e-9))
    # 情形 B2: 报警 [14,16) 起始在测试块1, 事件起始在整定侧
    alarmB2 = np.zeros(n, dtype=np.int8); alarmB2[14:16] = 1
    mB2 = protocol_M_metrics(alarmB2, y, tune_m, 60.0)
    mB2t = protocol_M_metrics(alarmB2, y, test_m, 60.0)
    checks.append(("B2: 整定侧看: 无报警 -> 未检出", mB2["det_ov"] == 0))
    checks.append(("B2: 测试侧看: 报警起始在测试侧 -> 计为测试侧 FP",
                   mB2t["n_gt_side"] == 0 and mB2t["n_alarm_side"] == 1
                   and mB2t["EF1_ov"] == 0.0))
    # 情形 C: 报警 [5,7) 在事件前 (整定侧)
    alarmC = np.zeros(n, dtype=np.int8); alarmC[5:7] = 1
    mC = protocol_M_metrics(alarmC, y, tune_m, 60.0)
    checks.append(("C: 事件前报警=整定侧 FP, 事件未检出",
                   mC["det_ov"] == 0 and mC["n_alarm_side"] == 1))
    ok = all(c[1] for c in checks)
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return ok


if __name__ == "__main__":
    print("== 协议 M 玩具验证 ==")
    assert toy_protocol_test(), "玩具验证失败 — 规则实现与设计文档不符!"
    print("全部通过")
