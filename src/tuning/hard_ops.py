"""硬规则告警整定算子 (numpy 状态机) — 推理侧 100% 硬逻辑 (ISA-18.2 合规声明),
也是经典基线 (固定窗口合并 / 网格 on-off-delay) 的实现底座.

  hard_hysteresis(s, θ_hi, θ_lo): s>=θ_hi 置位, s<=θ_lo 复位, 之间保持 (deadband).
  hard_delay_timer(b, T_on, T_off): 连续 T_on 点激活才通报 (delay-on),
      通报后连续 T_off 点静默才撤销 (delay-off). T_on=1 且 T_off=1 退化为原始阈值.
  merge_gaps(b, W): 固定窗口合并 — 间隔 <=W 的告警段并成一段 (经典后处理基线).
"""
import numpy as np


def hard_hysteresis(s, theta_hi, theta_lo):
    s = np.asarray(s)
    out = np.zeros(len(s), dtype=np.int8)
    state = 0
    for t in range(len(s)):
        if s[t] >= theta_hi:
            state = 1
        elif s[t] <= theta_lo:
            state = 0
        out[t] = state
    return out


def hard_delay_timer(b, T_on, T_off):
    b = (np.asarray(b) > 0).astype(np.int8)
    T_on = max(1, int(round(T_on)))
    T_off = max(1, int(round(T_off)))
    out = np.zeros(len(b), dtype=np.int8)
    state, run1, run0 = 0, 0, 0
    for t in range(len(b)):
        if b[t]:
            run1 += 1
            run0 = 0
        else:
            run0 += 1
            run1 = 0
        if state == 0 and run1 >= T_on:
            state = 1
            run0 = 0
        elif state == 1 and run0 >= T_off:
            state = 0
            run1 = 0
        out[t] = state
    return out


def merge_gaps(b, W):
    """间隔(连续0段长度) <= W 的告警段合并. W 单位: 点."""
    b = (np.asarray(b) > 0).astype(np.int8)
    if b.sum() == 0:
        return b
    W = max(0, int(W))
    edges = np.flatnonzero(np.diff(np.concatenate([[0], b, [0]])))
    starts, ends = edges[::2], edges[1::2]
    out = np.zeros_like(b)
    cur_s = starts[0]
    for i in range(1, len(starts)):
        if starts[i] - ends[i - 1] <= W:
            continue                    # 并入当前段
        out[cur_s:ends[i - 1]] = 1
        cur_s = starts[i]
    out[cur_s:ends[-1]] = 1
    return out


def hard_pipeline(s, theta_hi, theta_lo, T_on, T_off):
    """推理侧硬链路 (与 soft_pipeline 对应): 迟滞 -> on/off 延迟定时器."""
    h = hard_hysteresis(s, theta_hi, theta_lo)
    return hard_delay_timer(h, T_on, T_off)
