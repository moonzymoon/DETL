"""R7 反馈实验: chatter 代理换成精确 soft-OR 后重跑 M4 (chatter 约束对比).

背景: v1 代理 q=1-exp(-sum r) 在 n=1 时读 0.63, 低估重触发比例 (R7-任务三[4]).
losses.soft_chatter 已改为精确 OR q=1-prod(1-r) (log 域前缀和, 数值稳定).
本脚本与 e_rebuttal_r1.py 的 M4 块同协议重跑, 输出硬指标对照.

输出: results/audit/r7_chatter_or.csv
运行: cd src && python experiments/e_r7_chatter_or.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import KPIConfig, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e3_baselines import _kmax
from experiments.e_pilot_tuning import interleave_split

COMBOS = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]
R_MAX, F_MAX = 10.0, 0.02


def tune_eval(scorer, ds, s_tr, y_tr, s_te, y_te, dt, c_max=None):
    from kpi.panel import events_from_binary
    ev_tr = events_from_binary(y_tr)
    K_on, K_off = _kmax(dt)
    params, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=R_MAX, F_max=F_MAX,
                             epochs=300, seed=SEED, K_on=K_on, K_off=K_off,
                             C_max=c_max)
    alarm = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                          params["T_on"], params["T_off"])
    p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
    p.update({"dataset": ds, "scorer": scorer, "n_train_events": len(ev_tr),
              "params": str(params)})
    return p


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r7_chatter_or.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for scorer, ds in COMBOS:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        p_wo = tune_eval(scorer, ds, s[tr_m], y[tr_m], s[te_m], y[te_m], dt,
                         c_max=None)
        p_wi = tune_eval(scorer, ds, s[tr_m], y[tr_m], s[te_m], y[te_m], dt,
                         c_max=0.1)
        p_wo["method"], p_wi["method"] = "diff-tune", "diff-tune+chat-c(OR)"
        rows += [p_wo, p_wi]
        print(f"[r7-or] {scorer} x {ds}: "
              f"chat={p_wi['chatter_rate']:.3f} F1={p_wi['EventF1']:.3f} "
              f"AAR={p_wi['AAR_per_hour']:.2f} MAD={p_wi['mean_alarm_delay_s']:.0f}",
              flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
