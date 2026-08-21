"""可微代理损失: 事件漏报 + 检测滞后 (目标项) 与 AAR / 正常点误报 (约束项).

设计 (方案 §4.2: 承认代理局限, 训练后一律用 kpi.panel 硬指标验证):
  soft_missed_e   = 1 - LME_{t∈e}(ã_t)   (事件漏报, 漏检零信用)
      LME = (1/β)·log mean(e^{β·ã}) 为长度校正的 soft max:
      持续高覆盖 -> 接近 max; 单点尖峰 -> 按 ln(覆盖率) 折减.
      (v1 用 LSE 近似 max, 长事件上过冲 ln(L)/β 且部分覆盖被记全检出,
       E5 定位为 MetroPT3 检出率反降根因, v2 改用 LME.)
  soft_lag_e      = Σ_t p_t·(t-s_e)/L_e + (1-Σp_t)      (归一化检测滞后, 未检出=1)
      p_t = ã_t·(1 - 前缀 LME_{<t} ã)  —— 软"首次命中"分布 (前缀亦长度校正)
  soft_AAR        = Σ_t relu(ã_t - ã_{t-1}) / 小时数    (激活次数代理)
  soft_FAR        = mean(ã | y=0)                       (正常点误报代理)
  soft_chatter    = Σ_t r_t·q_t / (Σ_t r_t + ε)         (震颤代理, 可选约束)
      r_t = relu(ã_t - ã_{t-1}) 软激活; q_t = 1-Π_{k=1..K_c}(1-r_{t-k})
      为过去 K_c 点内存在软激活的精确 soft-OR (log 域前缀和计算, 长窗稳定)
      —— r_t 落在过去 K_c 内即计为 chatter; 激活锐化时收敛到 ISA 二值指标.
      (v1 用 q=1-exp(-Σr) 指数近似, n=1 时读 0.63 偏低, R7 改为精确 OR.)

所有量对 ã 全程可微; smooth 聚合用 β-softmax (β=10) 近似.
"""
import math

import torch

BETA = 10.0   # smooth-max 锐度


def _lme(x):
    """log-mean-exp: 长度校正 soft max. 上界 max(x); 稀疏尖峰按覆盖率折减."""
    return torch.logsumexp(BETA * x, dim=0) / BETA - math.log(len(x)) / BETA


def soft_chatter(a_hat, K_c):
    """可微 chatter 代理: 软激活中落在过去 K_c 点内已有软激活处的占比.
    窗口严格取 (t-K_c, t) 之前段 (不含当前激活, 避免孤立激活自配对).
    q_t = 1 - Π_{k=1..K_c}(1-r_{t-k}) 精确 soft-OR, log 域前缀和实现:
    log(1-r) <= 0 累加只会更负, expm1 下溢到 -1 => q 到 1, 长窗无溢出."""
    r = torch.relu(a_hat[1:] - a_hat[:-1])                 # (n-1,)
    if K_c <= 1 or len(r) < 2:
        return torch.zeros((), dtype=a_hat.dtype)
    rr = r.clamp(max=1.0 - 1e-6)                           # log(1-r) 有限
    logr = torch.log1p(-rr)                                # (n-1,), <= 0
    P = torch.cat([torch.zeros(1, dtype=r.dtype), torch.cumsum(logr, 0)])
    n_r = len(r)
    past = torch.cat([torch.zeros(min(K_c, n_r), dtype=r.dtype),
                      P[:max(n_r - K_c, 0)]])              # P[t-K_c] (截 0)
    q = -torch.expm1(P[:n_r] - past)                       # 1 - Π(1-r)
    return (r * q).sum() / (r.sum() + 1e-6)


def soft_kpi(a_hat, y, events, dt, K_chatter=None, miss_mode="max"):
    """返回 dict of 0-dim tensors: miss, lag, aar, far[, chatter].
    K_chatter>0 时计算 chatter 代理 (窗口 K_chatter 点, 对应 ISA 常用 60s).
    miss_mode: 'max' — LSE 近似 max (默认; 与硬检出率"事件内任意告警"同构);
               'lme' — log-mean-exp 持续覆盖语义 (消融用: 语义更严但实测更差,
                       见实验记录 E5)."""
    n = len(a_hat)
    hours = n * dt / 3600.0

    # 约束: AAR (软激活次数/小时) 与 FAR (正常点软告警均值)
    rising = torch.relu(a_hat[1:] - a_hat[:-1])
    aar = rising.sum() / max(hours, 1e-6)
    normal = y == 0
    far = a_hat[normal].mean() if normal.any() else torch.zeros((), dtype=a_hat.dtype)

    # 目标: 每场 GT 事件的漏报与归一化检测滞后
    miss_terms, lag_terms = [], []
    for s0, e0 in events:
        seg = a_hat[s0:e0]
        L = max(e0 - s0, 1)
        if miss_mode == "lme":
            miss_terms.append(1.0 - _lme(seg).clamp(0.0, 1.0))   # 持续覆盖语义
            pm = torch.logcumsumexp(BETA * seg, dim=0) / BETA \
                - torch.log(torch.arange(1, L + 1, dtype=a_hat.dtype)) / BETA
        else:                                                      # max 语义 (默认)
            # 不 clamp: LSE 过冲产生负 miss 是记录在案的代理偏差 (E5)
            miss_terms.append(1.0 - torch.logsumexp(BETA * seg, dim=0) / BETA)
            pm = torch.logcumsumexp(BETA * seg, dim=0) / BETA
        pm_prev = torch.cat([torch.zeros(1, dtype=a_hat.dtype), pm[:-1]])
        p = seg * (1 - pm_prev).clamp(0.0, 1.0)                # 软首次命中分布
        frac = torch.arange(L, dtype=a_hat.dtype) / L
        lag_terms.append((p * frac).sum() + (1 - p.sum().clamp(0.0, 1.0)))

    miss = torch.stack(miss_terms).mean() if miss_terms else torch.zeros((), dtype=a_hat.dtype)
    lag = torch.stack(lag_terms).mean() if lag_terms else torch.zeros((), dtype=a_hat.dtype)
    out = {"miss": miss, "lag": lag, "aar": aar, "far": far}
    if K_chatter:
        out["chatter"] = soft_chatter(a_hat, K_chatter)
    return out
