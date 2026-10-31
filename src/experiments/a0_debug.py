# -*- coding: utf-8 -*-
"""调试: 核验1 迟滞差异定位 + 核验4 非确定性来源 (线程?)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tuning.soft_ops import soft_hysteresis  # noqa: E402

# ---- 核验1: 小样例逐步对照 ----
rng = np.random.default_rng(3)
n = 12
s = np.sort(rng.normal(size=n)) * 0.05
s = s[rng.permutation(n)]
th_hi, th_lo, temp = 0.08, 0.0, 1e-3
h_par = soft_hysteresis(torch.tensor(s, dtype=torch.float32),
                        torch.tensor(th_hi), torch.tensor(th_lo),
                        torch.tensor(temp, dtype=torch.float32)).numpy()
hi = 1 / (1 + np.exp(-(s - th_hi) / temp))
lo = 1 / (1 + np.exp((th_lo - s) / temp))
prev = 0.0
h_seq = []
for t in range(n):
    prev = max(hi[t], (1 - lo[t]) * prev)
    h_seq.append(prev)
h_seq = np.array(h_seq)
print("t |   s    hi    lo  | par   seq")
for t in range(n):
    flag = " <== " if abs(h_par[t] - h_seq[t]) > 1e-3 else ""
    print(f"{t:2d} | {s[t]:+.3f} {hi[t]:.2f} {lo[t]:.2f} | "
          f"{h_par[t]:.4f} {h_seq[t]:.4f}{flag}")

# ---- 核验4: 线程数影响 ----
from tuning.constrained import train_tuning  # noqa: E402
n2 = 4000
s2 = np.concatenate([np.random.default_rng(1).normal(0, .05, n2 // 2),
                     np.random.default_rng(2).normal(0.35, .05, n2 // 2)])
y2 = np.zeros(n2, dtype=np.int8); y2[n2 // 2:] = 1
ev2 = [(n2 // 2, n2)]
for nt in [torch.get_num_threads(), 1]:
    torch.set_num_threads(nt)
    pa, _ = train_tuning(s2, y2, ev2, dt=60.0, epochs=30, seed=7)
    pb, _ = train_tuning(s2, y2, ev2, dt=60.0, seed=7)
    pc, _ = train_tuning(s2, y2, ev2, dt=60.0, seed=999)
    same = all(pa[k] == pb[k] for k in pa)
    same_seed = all(pa[k] == pc[k] for k in pa)
    print(f"threads={nt}: 双跑一致={same} seed无关={same_seed} "
          f"theta_hi={pa['theta_hi']:.6f}/{pb['theta_hi']:.6f}/{pc['theta_hi']:.6f}")
