"""方案2 (R41): best-F1 阈值 oracle —— 测试段上带标签的最优裸阈值上界.

定义 (Lipton et al. 2014 的 best-F1 线, 事件化版本): 在测试段分数上
扫阈值 θ (分位网格), 告警 = s_te > θ (裸阈值, 无死区/计时器),
事件级 F1 (匈牙利匹配 IoU>0), 取最大. 这是"阈值参数单独可达上界"
的 oracle —— 使用测试标签, 不是可部署方法, 仅作参照行.
四数据集: SWaT(cmhmil)/TE(iforest)/MetroPT3(iforest)/BATADAL(现跑).
输出: results/audit/r41_oracle.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from kpi.panel import events_from_binary
from scorers.adapter import get_scores
from experiments.e_pilot_tuning import interleave_split
from experiments.e_r22_jackknife import match_breakdown
from experiments.e_r36_mcap_sweep import batadal_stream

QGRID = np.linspace(0.30, 0.9995, 400)


def oracle_f1(s_te, y_te):
    gt = events_from_binary(y_te)
    best = (None, -1.0, None, None)
    for q in QGRID:
        th = np.quantile(s_te, q)
        al = (s_te > th).astype(int)
        tp, fp, fn, f1 = match_breakdown(gt, events_from_binary(al))
        det = tp / len(gt) if gt else 0.0
        if f1 > best[1]:
            best = (q, f1, tp, fp, fn, det)
    return best  # (q*, F1, tp, fp, fn, det)


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r41_oracle.csv")
    rows = []
    for ds, sc in [("SWaT", "cmhmil"), ("TE", "iforest"),
                   ("MetroPT3", "iforest")]:
        s, y, dt = get_scores(sc, ds, seed=SEED)
        _, te_m = interleave_split(len(s), dt)
        q, f1, tp, fp, fn, det = oracle_f1(s[te_m], y[te_m])
        rows.append({"dataset": ds, "q_star": q, "F1_oracle": f1,
                     "tp": tp, "fp": fp, "fn": fn, "det": det})
        print(f"[r41] {ds:8s} oracle: q*={q:.4f} F1={f1:.3f} "
              f"tp={tp} fp={fp} fn={fn} det={det:.2f}", flush=True)
    sb, yb, dtb = batadal_stream()
    _, te_b = interleave_split(len(sb), dtb)
    q, f1, tp, fp, fn, det = oracle_f1(sb[te_b], yb[te_b])
    rows.append({"dataset": "BATADAL", "q_star": q, "F1_oracle": f1,
                 "tp": tp, "fp": fp, "fn": fn, "det": det})
    print(f"[r41] BATADAL  oracle: q*={q:.4f} F1={f1:.3f} "
          f"tp={tp} fp={fp} fn={fn} det={det:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
