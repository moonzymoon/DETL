"""R36-C: BATADAL 锁存几何核查 —— 默认解的 θ_hi/θ_lo/m* 与分数范围实测.

论文 §4.5 现称 "deadband 停在 0.2·std 初始化"; r36 实测 m*=0.183=8.2·std.
本脚本打印默认收敛解的完整几何: θ_hi 相对箱顶/分位, θ_lo 相对分数最小值,
以确定正确表述.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SEED
from kpi.panel import events_from_binary
from tuning.constrained import train_tuning
from experiments.e3_baselines import _kmax
from experiments.e_pilot_tuning import interleave_split
from experiments.e_r36_mcap_sweep import batadal_stream


def main():
    s, y, dt = batadal_stream()
    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    K_on, K_off = _kmax(dt)
    std = float(np.std(s_tr))
    prm, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0, F_max=0.02,
                          epochs=300, seed=SEED, K_on=K_on, K_off=K_off,
                          loss_mode="constrained")
    th_hi, th_lo = float(prm["theta_hi"]), float(prm["theta_lo"])
    m = th_hi - th_lo
    qs = {q: float(np.quantile(s_tr, q)) for q in (0.50, 0.90, 0.99, 0.999)}
    print(f"std(s_tr)={std:.4f}  0.2·std={0.2*std:.4f}")
    print(f"θ_hi={th_hi:.4f}  θ_lo={th_lo:.4f}  m*={m:.4f} = {m/std:.1f}·std")
    print("分位: " + "  ".join(f"q{int(q*1000)/10:g}={v:.4f}"
                              for q, v in qs.items()))
    print(f"分数范围: 整定段 [{s_tr.min():.4f},{s_tr.max():.4f}]  "
          f"测试段 [{s_te.min():.4f},{s_te.max():.4f}]")
    print(f"θ_lo 低于整定段最小值? {th_lo < s_tr.min()}  "
          f"(θ_lo - min = {th_lo - s_tr.min():+.4f})")
    print(f"θ_lo 低于测试段最小值? {th_lo < s_te.min()}  "
          f"(θ_lo - min = {th_lo - s_te.min():+.4f})")
    above = (s_te > th_hi).mean()
    print(f"测试段超 θ_hi 占比={above:.4f}")
    print(f"T_on={prm['T_on']:.0f} T_off={prm['T_off']:.0f}")


if __name__ == "__main__":
    main()
