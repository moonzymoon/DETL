"""R23-M1 应对: 事件级留一敏感性 (leave-one-event-out jackknife).

最坏审稿人攻击: 事件级 F1 结论建立在 1/6/3 个 GT 事件上, 单个事件即可
翻转差距. 本脚本在评测侧做留一事件外: 告警流固定 (整定参数来自缓存,
不重训), 每次从 GT 集合中去掉一个事件重算 F1 —— 量化 "0.769 vs 0.500"
这类差距对单个事件的依赖程度.

范围: TE (6 事件) 与 MetroPT3 (3 事件); SWaT 仅 1 个事件, 留一后 GT
为空 (F1 无定义), 属退化情形不入表. 方法: 学习层 (diff-tune) 与全部
四个整定基线 (optuna/random/grid-delay/gyasi, 参数取自
results/e3/e3_selected_params.csv), 关键对照为各数据集最强基线
(TE: grid-delay 0.500; MetroPT3: grid-delay 0.098).

输出: results/audit/r22_jackknife.csv + 控制台
运行: cd src && python -u experiments/e_r22_jackknife.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import events_from_binary
from tuning.hard_ops import hard_pipeline
from experiments.e_pilot_tuning import interleave_split

COMBOS = [("iforest", "TE"), ("iforest", "MetroPT3")]


def match_breakdown(gt_events, pred_events):
    """一对一 IoU>0 指派, 返回 (tp, fp, fn, F1). 与 kpi.panel 同规则."""
    if not gt_events:
        return (0, len(pred_events), 0, float("nan"))
    if not pred_events:
        return (0, 0, len(gt_events), 0.0)
    IoU = np.zeros((len(gt_events), len(pred_events)))
    for i, (gs, ge) in enumerate(gt_events):
        for j, (ps, pe) in enumerate(pred_events):
            inter = max(0, min(ge, pe) - max(gs, ps))
            union = max(ge, pe) - min(gs, ps)
            IoU[i, j] = inter / union if union else 0.0
    rows, cols = linear_sum_assignment(-IoU)
    tp = sum(1 for r, c in zip(rows, cols) if IoU[r, c] > 0)
    fp = len(pred_events) - tp
    fn = len(gt_events) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (tp, fp, fn, float(2 * p * r / (p + r) if p + r else 0.0))


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r22_jackknife.csv")
    prm = pd.read_csv(os.path.join(RESULTS_DIR, "e3", "e3_selected_params.csv"))
    rows = []
    for scorer, ds in COMBOS:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_te, y_te = s[te_m], y[te_m]
        gt = events_from_binary(y_te)
        streams = {}
        for method in ("diff-tune", "optuna", "random", "grid-delay", "gyasi"):
            pp = prm[(prm.dataset == ds) & (prm.method == method)].iloc[0]
            streams[method] = events_from_binary(
                hard_pipeline(s_te, pp.theta_hi, pp.theta_lo,
                              int(pp.T_on), int(pp.T_off)))
        for method, ev in streams.items():
            tp, fp, fn, f1 = match_breakdown(gt, ev)
            rows.append({"dataset": ds, "method": method, "fold": "full",
                         "n_gt": len(gt), "tp": tp, "fp": fp, "fn": fn,
                         "F1": f1})
            print(f"[r22] {ds:8s} {method:9s} full: TP={tp} FP={fp} FN={fn} "
                  f"F1={f1:.3f} (n_gt={len(gt)})", flush=True)
        # 留一事件外: 去掉第 i 个 GT 事件
        for i in range(len(gt)):
            gt_i = gt[:i] + gt[i + 1:]
            for method, ev in streams.items():
                _, _, _, f1 = match_breakdown(gt_i, ev)
                rows.append({"dataset": ds, "method": method,
                             "fold": f"drop-{i}", "n_gt": len(gt_i),
                             "tp": None, "fp": None, "fn": None, "F1": f1})
        for method in streams:
            fs = [r["F1"] for r in rows
                  if r["dataset"] == ds and r["method"] == method
                  and r["fold"] != "full"]
            print(f"[r22] {ds:8s} {method:9s} LOO F1 span: "
                  f"[{min(fs):.3f}, {max(fs):.3f}]", flush=True)
        f_ours = [r["F1"] for r in rows if r["dataset"] == ds
                  and r["method"] == "diff-tune" and r["fold"] != "full"]
        for rival in ("grid-delay", "optuna", "random", "gyasi"):
            f_alt = [r["F1"] for r in rows if r["dataset"] == ds
                     and r["method"] == rival and r["fold"] != "full"]
            gaps = [a - b for a, b in zip(f_ours, f_alt)]
            wins = sum(1 for g in gaps if g > 1e-9)
            print(f"[r22] {ds:8s} 学习层-{rival:10s} 逐折差: "
                  f"[{min(gaps):+.3f}, {max(gaps):+.3f}]  "
                  f"学习层领先 {wins}/{len(gaps)} 折", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
