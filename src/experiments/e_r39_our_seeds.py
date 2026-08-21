"""R39: 我们自己方法的训练种子稳定性 (对称公平性).

攻击面: "你们报告了黑盒基线的五种子跨度, 但可微整定本身也是单种子 ——
梯度训练跨种子稳定吗?" 对四条主流各以 5 个训练种子重训, 报告
event-F1 跨度. BATADAL 另报 m-cap 配置.
输出: results/audit/r39_our_seeds.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from scorers.adapter import get_scores
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e3_baselines import _kmax
from experiments.e_pilot_tuning import interleave_split
from experiments.e_r36_mcap_sweep import batadal_stream

SEEDS = (7, 1, 2, 3, 4)


def run(ds, s, y, dt, variants, rows):
    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    K_on, K_off = _kmax(dt)
    cfg = KPIConfig(dt=dt)
    std = float(np.std(s_tr))
    for vname, mode, kw in variants:
        f1s = []
        for sd in SEEDS:
            prm, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0,
                                  F_max=0.02, epochs=300, seed=sd,
                                  K_on=K_on, K_off=K_off,
                                  loss_mode=mode, **kw)
            al = hard_pipeline(s_te, prm["theta_hi"], prm["theta_lo"],
                               prm["T_on"], prm["T_off"])
            p = kpi_panel(al, y_te, cfg=cfg)
            rows.append({"dataset": ds, "variant": vname, "seed": sd,
                         "F1": p["EventF1"],
                         "det": p["event_detection_rate"],
                         "AAR": p["AAR_per_hour"],
                         "n_ev": p["n_alarm_events"]})
            f1s.append(p["EventF1"])
            print(f"[r39] {ds:8s} {vname:10s} seed={sd}: F1={p['EventF1']:.3f} "
                  f"det={p['event_detection_rate']:.2f} "
                  f"AAR={p['AAR_per_hour']:.4f} n_ev={p['n_alarm_events']}",
                  flush=True)
        print(f"[r39] {ds:8s} {vname:10s} 跨度 [{min(f1s):.3f}, {max(f1s):.3f}]",
              flush=True)


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r39_our_seeds.csv")
    rows = []
    base = [("ours", "constrained", {})]
    for ds, sc in [("SWaT", "cmhmil"), ("TE", "iforest"),
                   ("MetroPT3", "iforest")]:
        s, y, dt = get_scores(sc, ds, seed=SEED)
        run(ds, s, y, dt, base, rows)
    sb, yb, dtb = batadal_stream()
    run("BATADAL", sb, yb, dtb,
        [("ours", "constrained", {}),
         ("ours-mcap", "constrained",
          {"m_max": 0.05 * float(np.std(sb[interleave_split(len(sb), dtb)[0]]))})],
        rows)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
