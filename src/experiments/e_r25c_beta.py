"""R29-A3c: 扩展 TE 每故障分解 + β(smax 锐度)敏感性.

诊断结论(机制): 边缘事件上 smax 地板 (log L)/β 掩盖漏报 → 训练不降阈值.
β 敏感性: β∈{10,25,50} 于 ext; β=50 于三个主管线流验证不破坏
(若破坏, 如实记录; 不改论文默认, 仅报告敏感性 —— 同 λ 扫描性质).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
import tuning.losses as L
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from scorers.adapter import get_scores
from experiments.e3_baselines import _kmax
from experiments.e_pilot_tuning import interleave_split
from experiments.e_r25b_diag import build_stream

FAULTS = ["1_1", "1_2", "4_1", "4_2", "11_1", "11_2",
          "2_1", "2_2", "5_1", "5_2", "7_1", "7_2",
          "10_1", "10_2", "13_1", "13_2"]
RUN_LEN = 3001
ONSET = 480          # 8h × 60min


def ext_eval(prm, s_te, y_te, dt):
    al = hard_pipeline(s_te, prm["theta_hi"], prm["theta_lo"],
                       prm["T_on"], prm["T_off"])
    return al, kpi_panel(al, y_te, cfg=KPIConfig(dt=dt))


def main():
    s, y, dt = build_stream()
    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    K_on, K_off = _kmax(dt)

    # β 扫描 (ext)
    for beta in (10.0, 25.0, 50.0):
        L.BETA = beta
        prm, hist = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0,
                                 F_max=0.02, epochs=300, seed=SEED,
                                 K_on=K_on, K_off=K_off)
        al, p = ext_eval(prm, s_te, y_te, dt)
        print(f"[beta] ext β={beta:4.0f}: θ={prm['theta_hi']:.4f} "
              f"F1={p['EventF1']:.3f} det={p['event_detection_rate']:.2f} "
              f"AAR={p['AAR_per_hour']:.3f} n_ev={p['n_alarm_events']} "
              f"miss_final={hist[-1]['miss']:.3f}", flush=True)
        if beta == 50.0:
            print("[beta] ext β=50 每故障检出:", flush=True)
            for i, f in enumerate(FAULTS):
                b0 = i * RUN_LEN + ONSET - 15
                e0 = (i + 1) * RUN_LEN - 15
                if b0 >= len(y_te):
                    break
                seg = slice(max(b0, 0), min(e0, len(y_te)))
                if y_te[seg].max():
                    hit = "Y" if al[seg].max() else "-"
                    smax_in = float(np.max(s_te[seg][y_te[seg] == 1]))
                    print(f"   IDV({f:5s}): {hit}  事件内分数max={smax_in:.3f}",
                          flush=True)

    # β=50 不破坏主管线验证 (TE 原始 / SWaT / MetroPT3)
    for scorer, ds in [("iforest", "TE"), ("cmhmil", "SWaT"),
                       ("iforest", "MetroPT3")]:
        s2, y2, dt2 = get_scores(scorer, ds, seed=SEED)
        tr2, te2 = interleave_split(len(s2), dt2)
        prm, _ = train_tuning(s2[tr2], y2[tr2], events_from_binary(y2[tr2]),
                              dt2, R_max=10.0, F_max=0.02, epochs=300,
                              seed=SEED, K_on=_kmax(dt2)[0],
                              K_off=_kmax(dt2)[1])
        al = hard_pipeline(s2[te2], prm["theta_hi"], prm["theta_lo"],
                           prm["T_on"], prm["T_off"])
        p = kpi_panel(al, y2[te2], cfg=KPIConfig(dt=dt2))
        print(f"[beta] β=50 主管线 {ds:9s}: F1={p['EventF1']:.3f} "
              f"det={p['event_detection_rate']:.2f} "
              f"AAR={p['AAR_per_hour']:.3f}", flush=True)
    L.BETA = 10.0


if __name__ == "__main__":
    main()
