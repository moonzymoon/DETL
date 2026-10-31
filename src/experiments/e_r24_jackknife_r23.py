"""R28-2: 留一事件外扩展 —— 覆盖点级+约束变体 (r23 rival).

§4.3 现有断言 "the layer stays ahead of every baseline in all six folds"
只覆盖 tab:main 四基线; §4.5 引入的 pointwise+constraints 变体是
TE 全集上第二强的学习配置 (0.500), 最坏审稿人会要求同一检验覆盖它.
本脚本在 TE (6 折) 与 MetroPT3 (3 折) 上重训该变体 (确定性) 并做
评测侧留一, 与主方法逐折对比, 结果无论胜负均如实写入.

输出: results/audit/r24_jackknife_r23.csv
运行: cd src && python -u experiments/e_r24_jackknife_r23.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import events_from_binary
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e3_baselines import _kmax
from experiments.e_pilot_tuning import interleave_split
from experiments.e_r22_jackknife import match_breakdown

COMBOS = [("iforest", "TE"), ("iforest", "MetroPT3")]


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r24_jackknife_r23.csv")
    prm = pd.read_csv(os.path.join(RESULTS_DIR, "e3", "e3_selected_params.csv"))
    rows = []
    for scorer, ds in COMBOS:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)
        gt = events_from_binary(y_te)
        K_on, K_off = _kmax(dt)
        pp = prm[(prm.dataset == ds) & (prm.method == "diff-tune")].iloc[0]
        ev_ours = events_from_binary(
            hard_pipeline(s_te, pp.theta_hi, pp.theta_lo,
                          int(pp.T_on), int(pp.T_off)))
        params, _ = train_tuning(
            s_tr, y_tr, ev_tr, dt, R_max=10.0, F_max=0.02, epochs=300,
            seed=SEED, K_on=K_on, K_off=K_off,
            loss_mode="relaxed_f1_constrained")
        ev_alt = events_from_binary(
            hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                          params["T_on"], params["T_off"]))
        for tag, ev in [("ours", ev_ours), ("pw+constr", ev_alt)]:
            tp, fp, fn, f1 = match_breakdown(gt, ev)
            rows.append({"dataset": ds, "method": tag, "fold": "full",
                         "n_gt": len(gt), "tp": tp, "fp": fp, "fn": fn,
                         "F1": f1})
        f_o, f_a = [], []
        for i in range(len(gt)):
            gt_i = gt[:i] + gt[i + 1:]
            for tag, ev in [("ours", ev_ours), ("pw+constr", ev_alt)]:
                _, _, _, f1 = match_breakdown(gt_i, ev)
                rows.append({"dataset": ds, "method": tag,
                             "fold": f"drop-{i}", "n_gt": len(gt_i),
                             "tp": None, "fp": None, "fn": None, "F1": f1})
            f_o.append(match_breakdown(gt_i, ev_ours)[3])
            f_a.append(match_breakdown(gt_i, ev_alt)[3])
        gaps = [a - b for a, b in zip(f_o, f_a)]
        wins = sum(1 for g in gaps if g > 1e-9)
        print(f"[r24] {ds:8s} ours vs pw+constr 逐折差: "
              f"[{min(gaps):+.3f}, {max(gaps):+.3f}]  ours 领先 {wins}/{len(gaps)} 折",
              flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
