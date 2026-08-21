"""R36-B: m_max 通用常数搜索 —— 上限能否一个值同时护住四条流?

r36 发现: BATADAL 锁存裕度 ≈ 8.2·std, TE 健康收敛裕度 ≈ 1.6·std,
0.05·std 上限护 BATADAL (F1 不变) 但毁 TE (0.769→0.072).
假设: m_max ≈ 2-5·std 落在两者之间 → BATADAL 去锁存 且 TE/SWaT/Metro 不动.
输出: results/audit/r36b_mcap_universal.csv
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
from experiments.e_r36_mcap_sweep import batadal_stream, occupancy_diag


def run(ds_name, s, y, dt, fracs, out_rows):
    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    K_on, K_off = _kmax(dt)
    cfg = KPIConfig(dt=dt)
    std = float(np.std(s_tr))
    print(f"[r36b] {ds_name}: std={std:.4f}", flush=True)
    for label, mcap in [("none", None)] + [(f"{g:g}", g * std) for g in fracs]:
        kw = dict(m_max=mcap) if mcap is not None else {}
        prm, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0,
                              F_max=0.02, epochs=300, seed=SEED,
                              K_on=K_on, K_off=K_off,
                              loss_mode="constrained", **kw)
        al = hard_pipeline(s_te, prm["theta_hi"], prm["theta_lo"],
                           prm["T_on"], prm["T_off"])
        p = kpi_panel(al, y_te, cfg=cfg)
        occ, longest = occupancy_diag(al, dt)
        out_rows.append({"dataset": ds_name, "m_max_frac": label,
                         "m_learned": float(prm["theta_hi"] - prm["theta_lo"]),
                         "F1": p["EventF1"],
                         "det": p["event_detection_rate"],
                         "AAR": p["AAR_per_hour"],
                         "n_ev": p["n_alarm_events"],
                         "occupancy": occ, "longest_ev_h": longest})
        print(f"[r36b] {ds_name:8s} m_max={label:>5s}·std: "
              f"F1={p['EventF1']:.3f} det={p['event_detection_rate']:.2f} "
              f"AAR={p['AAR_per_hour']:.4f} n_ev={p['n_alarm_events']} "
              f"occ={occ:.4f} 最长={longest:.1f}h", flush=True)


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r36b_mcap_universal.csv")
    rows = []
    fracs = [1.0, 2.0, 3.0, 5.0]
    s, y, dt = batadal_stream()
    run("BATADAL", s, y, dt, fracs, rows)
    for ds in ("TE", "MetroPT3"):
        s2, y2, dt2 = get_scores("iforest", ds, seed=SEED)
        run(ds, s2, y2, dt2, [3.0], rows)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
