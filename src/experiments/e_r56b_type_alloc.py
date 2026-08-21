"""R56b: te-ext 故障类型分配核查 (审核报告 §3.3). 统计每个 episode
属于 tuning-train 还是 tuning-test 块, 以及各侧的故障类型集合."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.e_r25_te_ext import load_ext, FAULTS
from experiments.e_pilot_tuning import interleave_split
from kpi.panel import events_from_binary

X, Y, dt = load_ext()
n_norm = 30001
w = 16
y = Y[n_norm:][w - 1:]
tr_m, te_m = interleave_split(len(y), dt)

# 每个 run 的正段起点即事件; 直接从标签找正段并映射类型
evs = events_from_binary(y)
print(f"scored length {len(y)}, episodes {len(evs)}")

# 用 X 的行数按 run 累计 (load_ext 中 Xs 逐 run vstack)
# 重新从 npz 元数据推: 直接读原始 excel 行数太重; 用 Y 的 0/1 结构:
# normal 段 (前 30001-w) 全 0, 每 run 一个正段. 找正段并映射.
pos = np.where(y > 0)[0]
segs = []
st = pos[0]
for a, b in zip(pos[:-1], pos[1:]):
    if b - a > 1:
        segs.append((st, a + 1))
        st = b
segs.append((st, pos[-1] + 1))
print(f"positive segments: {len(segs)}")

for (b, f), (s, e) in zip(
        [(0, None)] + list(enumerate(FAULTS)), segs):
    typ = "normal" if f is None else f.split("_")[1]
    in_tr = tr_m[s:e].sum() / (e - s)
    in_te = te_m[s:e].sum() / (e - s)
    side = "TRAIN" if in_tr > 0.5 else ("TEST" if in_te > 0.5 else "SPLIT")
    print(f"IDV {typ:>3s} [{s:7d},{e:7d}) tr={in_tr:.2f} te={in_te:.2f} -> {side}")
