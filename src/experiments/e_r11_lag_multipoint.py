"""R11 大改实验 2: 检测滞后代理的多点协同单调性测试.

审稿质疑 (R8-任务二[2]/任务三[5]): 已有 2e5 单点增加测试 (96.8% 下降),
但多点协同增加未被排除. 本实验用 losses.soft_kpi 的同一实现,
对随机事件段做多点随机增加 (m in {2,5,20,100}), 统计 ℓ 下降占比.

输出: results/audit/r11_lag_multipoint.csv + 控制台汇总
运行: cd src && python experiments/e_r11_lag_multipoint.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR
from tuning.losses import soft_kpi

TRIALS = 20000
LENS = [50, 500, 5000]
MS = [2, 5, 20, 100]


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r11_lag_multipoint.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []
    for m in MS:
        fell = 0
        for _ in range(TRIALS):
            L = int(rng.choice(LENS))
            base = torch.tensor(rng.beta(1.5, 4.0, size=L) * rng.choice([0.2, 1.0]),
                                dtype=torch.float32)
            base = base.clamp(0.0, 1.0)
            y = torch.zeros(L, dtype=torch.int8)
            idx = rng.choice(L, size=min(m, L), replace=False)
            inc = torch.tensor(rng.uniform(0.05, 1.0, size=len(idx)),
                               dtype=torch.float32)
            pert = base.clone()
            pert[idx] = (pert[idx] + inc).clamp(0.0, 1.0)
            l0 = float(soft_kpi(base, y, [(0, L)], 60.0)["lag"])
            l1 = float(soft_kpi(pert, y, [(0, L)], 60.0)["lag"])
            fell += int(l1 < l0)
        rows.append({"m_points": m, "trials": TRIALS, "fell_frac": fell / TRIALS})
        print(f"m={m:3d}: 多点增加后 ℓ 下降占比 = {fell / TRIALS:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"总下降占比 = {df.fell_frac.mean():.3f}; 写出 {out}")


if __name__ == "__main__":
    main()
