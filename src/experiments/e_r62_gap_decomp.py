"""R62a: 软硬差距分解 (审核条目7/9/11).

三个序列: ã (软链), a* (cap 语义硬链: u*=滑窗min(h,K_on) 后
hard_delay_timer(u*, 1, K_off)), â (部署硬链, 学习窗口 W).
输出 mean|ã-a*|, mean|a*-â|, mean|ã-â| —— 把 composition/window
语义差从总差距中分离.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from tuning.constrained import train_tuning
from tuning.soft_ops import soft_pipeline
from tuning.hard_ops import hard_pipeline, hard_hysteresis
from experiments.e3_baselines import _kmax
from experiments.e_pilot_tuning import interleave_split


def sliding_min(b, K):
    out = b.copy()
    for k in range(1, K):
        out = np.minimum(out, np.roll(b, k))
        out[:k] = 1  # 头部窗口不足按 1 (不构成清零证据), 与 replicate 精神一致
    return out


def cap_chain(s, th, tl, K_on, K_off):
    h = hard_hysteresis(s, th, tl)
    u = sliding_min(h, K_on)
    from tuning.hard_ops import hard_delay_timer
    return hard_delay_timer(u, 1, K_off)


def main():
    rows = []
    for scorer, ds in [("cmhmil", "SWaT"), ("iforest", "TE"),
                       ("iforest", "MetroPT3")]:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr = s[tr_m], y[tr_m]
        ev = __import__("kpi.panel", fromlist=["events_from_binary"]) \
            .events_from_binary(y_tr)
        K_on, K_off = _kmax(dt)
        params, _ = train_tuning(s_tr, y_tr, ev, dt, R_max=10.0,
                                 F_max=0.02, epochs=300, seed=SEED,
                                 K_on=K_on, K_off=K_off)
        th, tl = params["theta_hi"], params["theta_lo"]
        Ton, Toff = params["T_on"], params["T_off"]
        s_te = s[te_m]
        import torch
        temp_s = 0.05 * float(np.std(s_tr))
        with torch.no_grad():
            soft = soft_pipeline(
                torch.tensor(s_te, dtype=torch.float32), th, tl, Ton, Toff,
                temp_s, 0.02, K_on=K_on, K_off=K_off).numpy()
        a_cap = cap_chain(s_te, th, tl, K_on, K_off)
        a_dep = hard_pipeline(s_te, th, tl, Ton, Toff)
        d_prop = np.mean(np.abs(soft - a_cap))
        d_sem = np.mean(np.abs(a_cap - a_dep))
        d_tot = np.mean(np.abs(soft - a_dep))
        rows.append({"dataset": ds, "mean|soft-cap|": d_prop,
                     "mean|cap-deployed|": d_sem,
                     "mean|soft-deployed|": d_tot,
                     "T_on": Ton, "T_off": Toff, "K_on": K_on, "K_off": K_off})
        print(f"[{ds}] |soft-cap|={d_prop:.4f} |cap-dep|={d_sem:.4f} "
              f"|soft-dep|={d_tot:.4f} (W={Ton}/{Toff}, K={K_on}/{K_off})",
              flush=True)
    out = os.path.join(RESULTS_DIR, "audit", "r62_gap_decomp.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
