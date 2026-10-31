# -*- coding: utf-8 -*-
"""B-1: 标签鲁棒性实验 (R2.1): 整定侧事件标签扰动后重整定, 干净测试侧评分.

扰动类型 (按事件时长比例, 跨数据集统一规则):
  shift ±5% / ±10% (整体平移) | dilate +10%/+20% | erode 10%/20%
漏记事件: 随机丢 25% / 50% 整定侧事件 (其样本按正常计), 重整定.
每配置: 层重整定(float64 确定性) + 测试侧干净评估. 输出 results/audit/.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from a0_lib import (load_te_segments, interleave_mask, eval_params_M,
                    train_layer_M)
from kpi.panel import events_from_binary  # noqa

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))
RNG = np.random.default_rng(11)


def perturbed_labels(y, events, kind, frac):
    y2 = y.copy()
    for s0, e0 in events:
        L = e0 - s0
        d = int(round(L * frac))
        y2[s0:e0] = 0
        if kind == "shift":
            a = int(np.clip(s0 + d, 0, len(y) - L))
            y2[a:a + L] = 1
        elif kind == "dilate":
            a, b = max(0, s0 - d), min(len(y), e0 + d)
            y2[a:b] = 1
        elif kind == "erode":
            a, b = s0 + d, e0 - d
            if b > a:
                y2[a:b] = 1
    return y2


def main():
    s, y, dt, bounds, meta = load_te_segments()
    tune_m, test_m = interleave_mask(len(s), dt, bounds)
    ev_all = events_from_binary(y)
    ev_tune = [(a, b) for a, b in ev_all if tune_m[a]]

    rows = []
    configs = ([("none", 0.0)] + [("shift", f) for f in (0.05, 0.10, -0.05, -0.10)]
               + [("dilate", f) for f in (0.10, 0.20)]
               + [("erode", f) for f in (0.10, 0.20)])
    for kind, frac in configs:
        if kind == "none":
            y_t = y
        elif kind in ("shift", "dilate", "erode"):
            y_t = y.copy()
            # 只扰动整定侧事件 (测试侧标签保持干净)
            y_p = perturbed_labels(y, ev_tune, kind, abs(frac) if kind != "shift" else frac)
            y_t = y_p
        prm, _, _ = train_layer_M(s, y_t, tune_m, bounds, dt, epochs=300)
        me = eval_params_M(prm, s, y, test_m, dt, bounds)   # 干净测试标签
        me.update({"exp": "boundary", "kind": kind, "frac": frac})
        rows.append(me)
        print(f"[{kind} {frac:+.2f}] EF1={me['EF1_ov']:.3f} det={me['det_ov']:.2f} "
              f"AAR={me['AAR_side']:.2f} margin={prm['theta_hi']-prm['theta_lo']:.4f}")

    # 漏记事件
    for drop in (0.25, 0.50):
        y2 = y.copy()
        k = max(1, int(round(len(ev_tune) * drop)))
        drop_idx = RNG.choice(len(ev_tune), size=k, replace=False)
        for i in drop_idx:
            a, b = ev_tune[i]
            y2[a:b] = 0                      # 漏记 -> 按正常
        prm, _, _ = train_layer_M(s, y2, tune_m, bounds, dt, epochs=300)
        me = eval_params_M(prm, s, y, test_m, dt, bounds)
        me.update({"exp": "missing", "kind": "drop_events", "frac": -drop})
        rows.append(me)
        print(f"[missing {drop:.0%}] EF1={me['EF1_ov']:.3f} det={me['det_ov']:.2f} "
              f"AAR={me['AAR_side']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "a0b_robustness.csv"),
              index=False, encoding="utf-8-sig")
    print("写出 results/audit/a0b_robustness.csv")


if __name__ == "__main__":
    main()
