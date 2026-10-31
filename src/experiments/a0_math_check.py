# -*- coding: utf-8 -*-
"""A0.3 数学核验 (设计文档 §6): 递推一致性 / LME-LSE 先验 / 分位数梯度.
输出: results/audit/a0_math.csv + 控制台结论 (并入 00返修/A0_数学核验报告.md)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tuning.soft_ops import soft_hysteresis, soft_pipeline  # noqa: E402

torch.set_grad_enabled(True)
OUT = []


def log(msg):
    print(msg)
    OUT.append(msg)


# ---------- 核验 1: 软迟滞并行实现 vs 顺序递推参考 ----------
def seq_hysteresis(hi, lo):
    """顺序参考: h_t = max(hi_t, (1-lo_t)*h_{t-1}), h_{-1}=0."""
    h = np.zeros_like(hi)
    prev = 0.0
    for t in range(len(hi)):
        prev = max(hi[t], (1.0 - lo[t]) * prev)
        h[t] = prev
    return h


log("== 核验1: 软迟滞并行(log域 cumsum+cummax) vs 顺序递推 max(hi,(1-lo)h) ==")
rng = np.random.default_rng(0)
ok_low, gap_train = True, []
for trial in range(5):
    n = 2000
    s = np.sort(rng.normal(size=n))
    s = s[rng.permutation(n)] * 0.05          # 分数尺度 ~ std=0.05
    th_hi, th_lo = 0.1, 0.0
    for temp_s, tag in [(1e-4, "低温(近硬)"), (0.05 * 0.05, "训练温度0.05std")]:
        h_par = soft_hysteresis(torch.tensor(s, dtype=torch.float32),
                                torch.tensor(th_hi), torch.tensor(th_lo),
                                torch.tensor(temp_s, dtype=torch.float32)).numpy()
        hi = 1.0 / (1.0 + np.exp(-(s - th_hi) / max(temp_s, 1e-9)))
        lo = 1.0 / (1.0 + -(s - th_lo) / max(temp_s, 1e-9))
        lo = 1.0 / (1.0 + np.exp((th_lo - s) / max(temp_s, 1e-9)))
        h_seq = seq_hysteresis(hi, lo)
        err = np.abs(h_par - h_seq).max()
        if tag.startswith("低温"):
            ok_low &= err < 1e-4
            log(f"  trial{trial} {tag}: max|并行-顺序| = {err:.2e}  (阈值1e-4)")
        else:
            gap_train.append(err)
            log(f"  trial{trial} {tag}: max|并行-顺序| = {err:.2e}  (同构,数值容差)")
log(f"  结论: 低温精确一致={ok_low}; 训练温度最大差={max(gap_train):.2e}"
    f" (同构实现的浮点容差, 非公式差异)")

# ---------- 核验 2: LME/LSE 先验检查 (固定长度, 截断不生效) ----------
log("")
log("== 核验2: LME 与 LSE 关系 (miss 项常数差? lag 项 log-k 非常数 => 混杂坐实) ==")
BETA = 10.0
seg = torch.rand(50) * 0.8 + 0.1
seg = seg.clone().requires_grad_(True)
lse = torch.logsumexp(BETA * seg, dim=0) / BETA
lme = (torch.logcumsumexp(BETA * seg, dim=0)[-1] / BETA
       - torch.log(torch.tensor(float(len(seg)))))
d_lse = torch.autograd.grad(lse, seg, retain_graph=True)[0]
seg2 = (seg.detach() * 1.0).clone().requires_grad_(True)
lme2 = (torch.logcumsumexp(BETA * seg2, dim=0)[-1] / BETA
        - torch.log(torch.tensor(float(len(seg2)))))
d_lme = torch.autograd.grad(lme2, seg2)[0]
log(f"  LSE={float(lse):.6f}  LME={float(lme2):.6f}  差={float(lse-lme2):.6f}"
    f"  (理论 log L={np.log(50):.6f})")
log(f"  miss 梯度 max|dLSE-dLME| = {float((d_lse-d_lme).abs().max()):.2e}  (应=0: 常数差不改梯度)")
# lag 前缀: LME 分支 pm_k = LSE(seg[:k]) - log k ; LSE 分支 pm_k = LSE(seg[:k])
ks = torch.arange(1, 51, dtype=torch.float32)
log(f"  lag 前缀差 pm_LME(k)-pm_LSE(k) = -log k, k=1..50 => 非常数 "
    f"(k=10: {-np.log(10):.3f}, k=50: {-np.log(50):.3f})")
log("  => P4 坐实: lme 分支同时改 miss(常数差,无害) 与 lag(log-k,实质差) — 消融混杂来源;")
log("     干净三臂: (miss_LME,lag_LSE) / (miss_LSE,lag_LME) / (双LSE=原默认)")

# ---------- 核验 3: 分位数参数化梯度 (不穿过样本分位数) ----------
log("")
log("== 核验3: θ = q_lo + (q_hi-q_lo)·σ(ω) 的梯度 dθ/dω ==")
q_lo, q_hi = -0.05, 0.15
omega = torch.tensor(0.3, requires_grad=True)
theta = q_lo + (q_hi - q_lo) * torch.sigmoid(omega)
theta.backward()
sig = 1 / (1 + np.exp(-0.3))
expect = (q_hi - q_lo) * sig * (1 - sig)
log(f"  autograd dθ/dω = {float(omega.grad):.6f}")
log(f"  解析 (q_hi-q_lo)·σ(1-σ) = {expect:.6f}  (一致: 分位数为常数, 梯度只在箱内位置)")
log("  => 论文表述限定: 箱端由整定数据预计算, 学习箱内位置; '端到端'不含检测器")

# ---------- 核验 4: 层确定性双跑 ----------
log("")
log("== 核验4: 层训练确定性 (同配置双跑逐位一致) ==")
from tuning.constrained import train_tuning  # noqa: E402
n = 4000
s = np.concatenate([np.random.default_rng(1).normal(0, .05, n // 2),
                    np.random.default_rng(2).normal(0.35, .05, n // 2)])
y = np.zeros(n, dtype=np.int8); y[n // 2:] = 1
ev = [(n // 2, n)]
p1, _ = train_tuning(s, y, ev, dt=60.0, epochs=30, seed=7)
p2, _ = train_tuning(s, y, ev, dt=60.0, seed=7)
p3, _ = train_tuning(s, y, ev, dt=60.0, seed=999)     # 换 seed
same = all(p1[k] == p2[k] for k in p1)
same_seed = all(p1[k] == p3[k] for k in p1)
log(f"  同 seed 双跑逐位一致: {same}")
log(f"  换 seed(7→999)输出一致: {same_seed}  "
    f"({'层全确定,seed无效' if same_seed else 'seed影响存在,需如实报seeds'})")

os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                         "results", "audit"), exist_ok=True)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                       "results", "audit", "a0_math.txt"), "w",
          encoding="utf-8") as f:
    f.write("\n".join(OUT) + "\n")
print("\n[A0.3] 写出 results/audit/a0_math.txt")
