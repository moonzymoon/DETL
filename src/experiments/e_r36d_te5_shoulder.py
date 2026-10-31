"""R36-D: 收尾两组 —— (a) TE @ m_max=5·std (非绑定箱是否零扰动),
(b) BATADAL FAR 肩部核查 (Limitations(3) 的 flat-shoulder 判据第四流验证:
正常段超限率-分位曲线在运行区是否平坦).
输出: results/audit/r36d_te5_shoulder.csv
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


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r36d_te5_shoulder.csv")
    rows = []
    # (a) TE @ 5·std
    s, y, dt = get_scores("iforest", "TE", seed=SEED)
    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    K_on, K_off = _kmax(dt)
    std = float(np.std(s_tr))
    cfg = KPIConfig(dt=dt)
    for label, mcap in [("none", None), ("5std", 5.0 * std)]:
        kw = dict(m_max=mcap) if mcap is not None else {}
        prm, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0,
                              F_max=0.02, epochs=300, seed=SEED,
                              K_on=K_on, K_off=K_off,
                              loss_mode="constrained", **kw)
        al = hard_pipeline(s_te, prm["theta_hi"], prm["theta_lo"],
                           prm["T_on"], prm["T_off"])
        p = kpi_panel(al, y_te, cfg=cfg)
        occ, longest = occupancy_diag(al, dt)
        rows.append({"dataset": "TE", "variant": label,
                     "m_learned": float(prm["theta_hi"] - prm["theta_lo"]),
                     "F1": p["EventF1"], "det": p["event_detection_rate"],
                     "AAR": p["AAR_per_hour"], "n_ev": p["n_alarm_events"],
                     "occupancy": occ, "longest_ev_h": longest})
        print(f"[r36d] TE {label:5s}: F1={p['EventF1']:.3f} "
              f"det={p['event_detection_rate']:.2f} "
              f"n_ev={p['n_alarm_events']} "
              f"m*={prm['theta_hi'] - prm['theta_lo']:.4f}", flush=True)
    # (b) BATADAL FAR 肩部: 正常整定段在各分位阈值的点级超限率
    sb, yb, dtb = batadal_stream()
    trb, teb = interleave_split(len(sb), dtb)
    s_trb, y_trb = sb[trb], yb[trb]
    normal = s_trb[y_trb == 0]
    for q in (0.90, 0.95, 0.99, 0.999):
        th = np.quantile(s_trb, q)
        rate = float((normal > th).mean())
        rows.append({"dataset": "BATADAL", "variant": f"shoulder_q{q}",
                     "F1": np.nan, "det": np.nan, "AAR": np.nan,
                     "n_ev": np.nan, "occupancy": rate,
                     "longest_ev_h": np.nan, "m_learned": float(th)})
        print(f"[r36d] BATADAL 肩部 q={q:g}: 正常段超限率={rate:.5f}",
              flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
