"""E6: 损失函数对比 — 工业约束 KPI 损失 vs 点级 relaxed F1 损失 (方案 §5 E6).

两种损失各训一版同一可微整定层 (同架构/同退火/同种子), 统一硬规则测试:
  constrained : miss + lag + 拉格朗日约束 (AAR<=10/h, FAR<=0.02)  <- 本文
  relaxed_f1  : 点级 relaxed F1 (软 TP/FP/FN), 无 ISA 约束项        <- VAD/TSAD 常用

报告五项: 告警率 / chatter / 负载(误报点/h) / 检测滞后 / Event-F1.
卖点检验: F1 损失版是否违反 ISA 约束而本文守住.

输出: results/e6/e6_loss_comparison.csv
运行: cd src && python experiments/e6_loss_comparison.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import KPIConfig, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e_pilot_tuning import interleave_split
from experiments.e3_baselines import _kmax

COMBOS = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]
R_MAX = 10.0
F_MAX = 0.02


def main():
    out_dir = os.path.join(RESULTS_DIR, "e6")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for scorer, ds in COMBOS:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
        from kpi.panel import events_from_binary
        ev_tr = events_from_binary(y_tr)
        K_on, K_off = _kmax(dt)
        for mode in ["constrained", "relaxed_f1"]:
            params, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=R_MAX,
                                     F_max=F_MAX, epochs=300, seed=SEED,
                                     K_on=K_on, K_off=K_off, loss_mode=mode)
            alarm = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                                  params["T_on"], params["T_off"])
            p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
            p.update({"dataset": ds, "scorer": scorer, "loss_mode": mode,
                      **params})
            rows.append(p)
            print(f"[E6] {scorer} x {ds} [{mode:11s}] EventF1={p['EventF1']:.3f} "
                  f"AAR={p['AAR_per_hour']:.2f} chatter={p['chatter_rate']:.2f} "
                  f"MAD={p['mean_alarm_delay_s']:.0f}s det={p['event_detection_rate']:.2f} "
                  f"n_ev={p['n_alarm_events']}")
    df = pd.DataFrame(rows)
    # ISA 约束判定 (硬规则): AAR<=10/h 且 chatter<=0.1 (ISA 常用管理线)
    df["AAR_ok"] = df["AAR_per_hour"] <= R_MAX
    df["chatter_ok"] = df["chatter_rate"] <= 0.1
    df.to_csv(os.path.join(out_dir, "e6_loss_comparison.csv"), index=False,
              encoding="utf-8-sig")
    print(f"\n[E6] 写出 {out_dir}/e6_loss_comparison.csv")
    cols = ["dataset", "scorer", "loss_mode", "EventF1", "AAR_per_hour",
            "chatter_rate", "mean_alarm_delay_s", "event_detection_rate",
            "n_alarm_events", "AAR_ok", "chatter_ok"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
