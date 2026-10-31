"""R11 大改实验 3: 滞环 log 域 guard 的实际绑定占比与梯度量级.

审稿质疑 (R8-任务三[2]): guard 截断把微小梯度截为零, 未验证训练轨迹中
这些区域的实际影响. 本探针在三个数据集的收敛参数 + 末温下, 统计
a_t <= 1e-9 或 lo_t >= 1-1e-6 的样本占比, 及这些样本处的 logistic 梯度.

输出: results/audit/r11_guard_probe.csv + 控制台汇总
运行: cd src && python experiments/e_r11_guard_probe.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR
from scorers.adapter import get_scores
from experiments.e_pilot_tuning import interleave_split

PARAMS = {  # 收敛参数 (r1_chatter.csv 存档, 默认预算交错协议)
    "SWaT": ("cmhmil", 0.6899768710136414, 0.6820564270019531),
    "TE": ("iforest", 0.0030330829322338104, -0.07433068752288818),
    "MetroPT3": ("iforest", 0.18404000997543335, 0.1602478176355362),
}


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r11_guard_probe.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for ds, (scorer, th, tl) in PARAMS.items():
        s, y, dt = get_scores(scorer, ds, seed=7)
        tr_m, _ = interleave_split(len(s), dt)
        s_tr = torch.tensor(s[tr_m], dtype=torch.float32)
        temp_s = 0.05 * (float(s_tr.std()) + 1e-6)
        a = torch.sigmoid((s_tr - th) / temp_s)
        lo = torch.sigmoid((tl - s_tr) / temp_s)
        bind_a = a <= 1e-9
        bind_lo = lo >= 1.0 - 1e-6
        frac = float((bind_a | bind_lo).float().mean())
        grad_a = float(a[bind_a].mul(1 - a[bind_a]).max()) if bind_a.any() else 0.0
        grad_lo = float((lo * (1 - lo))[bind_lo].max()) if bind_lo.any() else 0.0
        rows.append({"dataset": ds, "n": int(len(s_tr)),
                     "bind_frac_a": float(bind_a.float().mean()),
                     "bind_frac_lo": float(bind_lo.float().mean()),
                     "bind_frac": frac,
                     "max_grad_where_bind_a": grad_a,
                     "max_grad_where_bind_lo": grad_lo})
        print(f"{ds:9s}: guard 绑定占比 = {frac:.4f} "
              f"(a 下界 {float(bind_a.float().mean()):.4f}, lo 上界 "
              f"{float(bind_lo.float().mean()):.4f}); 绑定处最大梯度 "
              f"a:{grad_a:.1e} lo:{grad_lo:.1e}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
