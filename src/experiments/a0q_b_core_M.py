# -*- coding: utf-8 -*-
"""A0q: B 核心 — 消融/损失对比/软硬差距 按协议 M 重跑 (冻结登记 §B).

- B1 消融: full/no-hyst/no-on/no-off/thr-only, train_layer_M(disable=...)
- B2 损失对比: constrained vs relaxed_f1 (点级软 F1, 无 ISA 约束)
- B3 软硬差距: full 参数下 soft(退火地板温度) vs hard 逐样本差 + 边沿差
流: TE-iForest (分段) + SWaT-cmh + MetroPT3-if (单段) — 与旧 e4/e6 同三流.
输出: results/audit/a0q_b_core_M.csv, a0q_b_gap.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from a0_lib import (load_te_segments, interleave_mask, eval_params_M,  # noqa
                    train_layer_M, hard_chain_segmented)

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))
VARIANTS = [("full", ()), ("no-hyst", ("hysteresis",)),
            ("no-on", ("on_delay",)), ("no-off", ("off_delay",)),
            ("thr-only", ("hysteresis", "on_delay", "off_delay"))]
LOSS_MODES = ["constrained", "relaxed_f1"]


def streams():
    # (标签, (s,y,dt,bounds))
    s, y, dt, bounds, meta = load_te_segments()
    yield "TE", (s, y, dt, bounds)
    for scorer, ds in [("cmhmil", "SWaT"), ("iforest", "MetroPT3")]:
        from scorers.adapter import get_scores
        ss, yy, dtt = get_scores(scorer, ds)
        ss = np.asarray(ss, np.float64)
        n = len(ss)
        yield ds, (ss, np.asarray(yy), float(dtt), [(0, n)])


def soft_hard_gap(s, prm, bounds, tune_m):
    import torch
    from tuning.soft_ops import soft_pipeline
    a_hard = hard_chain_segmented(s, prm, bounds)
    gaps = []
    for (g0, g1) in bounds:
        tt = torch.as_tensor(s[g0:g1])
        temp_s = 0.05 * float(np.std(s[tune_m]))
        a_soft = soft_pipeline(tt, prm["theta_hi"], prm["theta_lo"],
                               float(prm["T_on"]), float(prm["T_off"]),
                               temp_s, 0.02, K_on=24, K_off=48).numpy()
        gaps.append(np.abs(a_soft - a_hard[g0:g1]))
    g = np.concatenate(gaps)
    edges = np.flatnonzero((a_hard[1:] == 1) & (a_hard[:-1] == 0)) + 1
    edge_gap = float(np.mean([g[i - 3:i + 4].max() for i in edges])) \
        if len(edges) else float("nan")
    return {"mean_gap": float(g.mean()), "max_gap": float(g.max()),
            "n_edges": int(len(edges)), "edge_gap_max7": edge_gap,
            "frac_gap_gt01": float((g > 0.1).mean())}


def main():
    rows, gap_rows = [], []
    for tag, (s, y, dt, bounds) in streams():
        tune_m, test_m = interleave_mask(len(s), dt, bounds)
        print(f"\n== {tag}: n={len(s)} tune={tune_m.sum()} ==")
        # B1 消融
        for name, dis in VARIANTS:
            t0 = time.time()
            prm, _, _ = train_layer_M(s, y, tune_m, bounds, dt, epochs=300,
                                      disable=dis)
            me = eval_params_M(prm, s, y, test_m, dt, bounds)
            me.update({"stream": tag, "block": "B1_ablation", "variant": name,
                       "wall_s": round(time.time() - t0, 1), **{
                       k: prm[k] for k in
                       ("theta_hi", "theta_lo", "T_on", "T_off")}})
            rows.append(me)
            print(f"  [B1 {name:8s}] EF1={me['EF1_ov']:.3f} "
                  f"det={me['det_ov']:.2f} AAR={me['AAR_side']:.2f} "
                  f"feas={me['feasible']} {me['wall_s']}s")
            if name == "full":
                gap_rows.append({"stream": tag,
                                 **soft_hard_gap(s, prm, bounds, tune_m)})
        # B2 损失对比 (full 变体, 两种损失)
        for mode in LOSS_MODES:
            t0 = time.time()
            prm, _, _ = train_layer_M(s, y, tune_m, bounds, dt, epochs=300,
                                      loss_mode=mode)
            me = eval_params_M(prm, s, y, test_m, dt, bounds)
            me.update({"stream": tag, "block": "B2_loss", "variant": mode,
                       "wall_s": round(time.time() - t0, 1), **{
                       k: prm[k] for k in
                       ("theta_hi", "theta_lo", "T_on", "T_off")}})
            rows.append(me)
            print(f"  [B2 {mode:11s}] EF1={me['EF1_ov']:.3f} "
                  f"det={me['det_ov']:.2f} AAR={me['AAR_side']:.2f} "
                  f"feas={me['feasible']}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "a0q_b_core_M.csv"),
                              index=False, encoding="utf-8-sig")
    pd.DataFrame(gap_rows).to_csv(os.path.join(OUT, "a0q_b_gap.csv"),
                                  index=False, encoding="utf-8-sig")
    print("\nDONE -> a0q_b_core_M.csv / a0q_b_gap.csv")


if __name__ == "__main__":
    main()
