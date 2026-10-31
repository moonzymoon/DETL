"""可微告警整定算子 (torch, 第9篇核心组件).

实现方案 §4.1 的 ①②②③ 链路 (洪泛聚合 ④ 留待 E4 阶段):
  点级分数流 s --soft threshold--> 软告警 a
       --soft hysteresis (双阈值迟滞)--> h
       --soft on-delay (持续确认)--> u
       --soft off-delay (保持)--> 整定后软告警 ã

可微化设计 (每个算子都有硬规则对应物, 见 hard_ops.py; 训练用软, 推理用硬):
  - soft threshold: a = sigmoid((s - θ)/temp_s)  [temp_s 为分数域温度, 按分数 std 缩放]
  - soft hysteresis: h_t = max_{j<=t} [ hi_j * Π_{j<i<=t}(1 - lo_i) ]
      hi = sigmoid((s-θ_hi)/temp_s) 置位倾向, lo = sigmoid((θ_lo-s)/temp_s) 复位倾向;
      用 log 域前缀和 + cummax 精确并行计算, 与硬迟滞状态机同构.
  - soft on-delay: u_t = -temp_v·log( Σ_k ŵ_k(T_on)·exp(-h_{t-k}/temp_v) )
      即对过去窗内软告警做加权 softmin —— 持续 T_on 点高才输出高 (持续确认).
  - soft off-delay: ã_t = temp_v·log( Σ_k ṽ_k(T_off)·exp(u_{t-k}/temp_v) )
      即加权 softmax —— 一旦确认, 在 T_off 窗内保持 (告警保持/延迟撤销).
  - 窗长参数 T 连续化: 权 w_k = sigmoid((T-1-k)/τ_w) (滞后 k=0..K-1 的平滑盒窗),
      归一化后加权, 梯度可流到 T.

两个温度 (方案 §4.2 坑位预案):
  temp_s ∈ 分数域 (阈值/迟滞 sigmoid), 训练中按 std(s) 比例退火 —— 温度与分数
    尺度不匹配会导致 sigmoid 全饱和 (梯度消失) 或全平滑 (FAR 虚高把约束乘子推爆);
  temp_v ∈ 值域 [0,1] (softmin/softmax), 独立退火.
"""
import math

import torch
import torch.nn.functional as F

EPS = 1e-9


def soft_threshold(s, theta, temp_s):
    """a = sigmoid((s - theta)/temp_s), 越大越异常."""
    return torch.sigmoid((s - theta) / max(temp_s, 1e-6))


def window_weights(T, K, temp_w=0.5):
    """平滑盒窗: 滞后 k=0..K-1, 权重 sigmoid((T-1-k)/temp_w), 归一化. T 可微."""
    k = torch.arange(K, dtype=torch.float32, device=_dev(T))
    w = torch.sigmoid((T - 1.0 - k) / temp_w)
    return w / (w.sum() + EPS)


def _dev(*tensors):
    for t in tensors:
        if torch.is_tensor(t):
            return t.device
    return torch.device("cpu")


def _lag_matrix(a, K):
    """(N,) -> (N,K), 第 k 列 = a 右移 k (头部 replicate padding, 避免边界 0 假复位)."""
    a = a.view(1, 1, -1)
    pad = F.pad(a, (K - 1, 0), mode="replicate")
    M = pad.unfold(-1, K, 1)[0, 0]          # (N, K), 列 j = 滞后 K-1-j
    return M.flip(1)                        # 翻转为 列 k = 滞后 k


def soft_on_delay(h, T_on, temp_v, K_on):
    """持续确认 (delay-on): 对窗内 h 加权 softmin."""
    w = window_weights(T_on, K_on)
    x = -_lag_matrix(h, K_on) / max(temp_v, 1e-3)       # (N,K)
    out = -max(temp_v, 1e-3) * torch.logsumexp(
        x + torch.log(w + EPS), dim=1)
    return out.clamp(0.0, 1.0)


def soft_off_delay(u, T_off, temp_v, K_off):
    """告警保持 (delay-off): 对窗内 u 加权 softmax."""
    v = window_weights(T_off, K_off)
    x = _lag_matrix(u, K_off) / max(temp_v, 1e-3)
    out = max(temp_v, 1e-3) * torch.logsumexp(
        x + torch.log(v + EPS), dim=1)
    return out.clamp(0.0, 1.0)


def soft_hysteresis(s, theta_hi, theta_lo, temp_s):
    """软迟滞 (deadband): log 域闭式递推 h_t = max_{j<=t}[hi_j·Π_{i>j}(1-lo_i)].
    与硬迟滞状态机同构: 置位后只要未跌破 θ_lo 就保持.
    数值: float32 下 clamp 边界取 1e-6 (1-1e-9 会舍入回 1 导致 log(-0))."""
    hi = soft_threshold(s, theta_hi, temp_s).clamp(EPS, 1.0)
    lo = torch.sigmoid((theta_lo - s) / max(temp_s, 1e-6)).clamp(0.0, 1.0 - 1e-6)
    log_hi = torch.log(hi)
    log_keep = torch.log1p(-lo)                          # log(1-lo) >= log(1e-6)
    A = torch.cumsum(log_keep, dim=0)
    m = torch.cummax(log_hi - A, dim=0).values
    return torch.exp((A + m).clamp(-40.0, 0.0))


def soft_pipeline(s, theta_hi, theta_lo, T_on, T_off, temp_s, temp_v,
                  K_on=24, K_off=48):
    """完整可微链路: 阈值/迟滞 -> on-delay -> off-delay. 返回 ã ∈ (0,1)."""
    h = soft_hysteresis(s, theta_hi, theta_lo, temp_s)
    u = soft_on_delay(h, T_on, temp_v, K_on)
    return soft_off_delay(u, T_off, temp_v, K_off)
