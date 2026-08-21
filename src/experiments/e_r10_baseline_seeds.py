"""R10 审稿补充: 随机搜索 / TPE 黑盒基线的多种子散布.

审稿意见: Random(60) 与 Optuna(60) 是随机采样器, 单种子结果无法排除运气
成分. 本脚本对 3 个主数据集 x {TPE, Random} x 种子{7..11} 各跑一次
(与主表同协议: 交错划分, 60 次硬评估, 训练段选参/测试段评估),
报告 event-F1 的种子间散布, 检验主表结论对采样器运气的敏感性.

输出: results/audit/r10_baseline_seeds.csv + 控制台汇总
运行: cd src && python experiments/e_r10_baseline_seeds.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR
from scorers.adapter import get_scores
from kpi.panel import events_from_binary
from experiments.e3_baselines import run_optuna, apply_and_eval
from experiments.e_pilot_tuning import interleave_split

COMBOS = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]
SEEDS = [7, 8, 9, 10, 11]


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r10_baseline_seeds.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for scorer, ds in COMBOS:
        s, y, dt = get_scores(scorer, ds, seed=7)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr = s[tr_m], y[tr_m]
        s_te, y_te = s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)
        for use_tpe, name in [(True, "optuna"), (False, "random")]:
            for sd_ in SEEDS:
                prm = run_optuna(s_tr, y_tr, ev_tr, dt, sd_, use_tpe)
                p, _ = apply_and_eval(name, prm, s_te, y_te, dt, s_tr)
                rows.append({"dataset": ds, "method": name, "seed": sd_,
                             "EventF1": p["EventF1"],
                             "AAR": p["AAR_per_hour"],
                             "MAD": p["mean_alarm_delay_s"],
                             "det": p["event_detection_rate"]})
            sub = [r for r in rows if r["dataset"] == ds and r["method"] == name]
            f1 = [r["EventF1"] for r in sub]
            print(f"{ds:9s} {name:7s}: F1 min={min(f1):.3f} "
                  f"med={np.median(f1):.3f} max={max(f1):.3f}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
