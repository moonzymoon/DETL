"""R11 大改实验 1: T_on 初始化敏感性 —— 窗口长度是"学出来的"还是"初值给的"?

审稿核心质疑 (R8-任务二[1]/任务四[1]): TE 上 T_on 梯度 7e-6 且自失效为 1,
窗口长度可能实际未被有效学习. 本实验把 init_T_on 拉到 {2, 12, 20} (默认 4)
重训全部三个数据集, 看收敛 T_on 是否稳定 (学出) 还是跟随初值 (没学).
附带验证: 显式 init=4 的运行与存档参数逐位一致 (代码改动无副作用).

输出: results/audit/r11_ton_init.csv + 控制台汇总
运行: cd src && python experiments/e_r11_ton_init.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e3_baselines import _kmax
from experiments.e_pilot_tuning import interleave_split

COMBOS = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]
INITS = [4, 2, 12, 20]
ARCHIVED = {"SWaT": (0.6899768710136414, 0.6820564270019531, 7, 1),
            "TE": (0.0030330829322338104, -0.07433068752288818, 1, 1),
            "MetroPT3": (0.18404000997543335, 0.1602478176355362, 13, 1)}


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r11_ton_init.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for scorer, ds in COMBOS:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)
        K_on, K_off = _kmax(dt)
        for it in INITS:
            params, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0,
                                     F_max=0.02, epochs=300, seed=SEED,
                                     K_on=K_on, K_off=K_off, init_t_on=float(it))
            alarm = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                                  params["T_on"], params["T_off"])
            p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
            rows.append({"dataset": ds, "init_T_on": it,
                         "conv_T_on": params["T_on"], "conv_T_off": params["T_off"],
                         "theta_hi": params["theta_hi"], "F1": p["EventF1"],
                         "AAR": p["AAR_per_hour"], "MAD": p["mean_alarm_delay_s"]})
            print(f"[r11-init] {ds:9s} init={it:2d} -> conv T_on={params['T_on']:2d} "
                  f"T_off={params['T_off']:2d} F1={p['EventF1']:.3f}", flush=True)
        if ds in ARCHIVED and any(r["dataset"] == ds and r["init_T_on"] == 4 for r in rows):
            r4 = [r for r in rows if r["dataset"] == ds and r["init_T_on"] == 4][0]
            a = ARCHIVED[ds]
            ok = (abs(r4["theta_hi"] - a[0]) < 1e-12 and r4["conv_T_on"] == a[2]
                  and r4["conv_T_off"] == a[3])
            print(f"[r11-init] {ds}: init=4 与存档一致 = {ok}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
