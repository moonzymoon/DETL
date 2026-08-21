"""E1: 点级分数质量基准 — 3 打分器 x 5 数据集, AUROC + 点级 F1 (分位数扫描阈值).

输出: results/e1/e1_point_quality.csv
Go/No-Go: SWaT + TE + MetroPT3 中至少两个数据集有打分器 AUROC > 0.7.

运行: cd src && python experiments/e1_point_quality.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import AVAILABILITY, get_scores

DATASETS = ["SWaT", "TE", "MetroPT3", "SMAP", "WADI"]
SCORERS = ["cmhmil", "AT", "iforest"]
# 阈值 = 分数分布分位数 (扫描), 与任何校准机制无关, 只作打分器质量基准
QUANTILES = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]


def point_f1(y, alarm):
    tp = int(((alarm == 1) & (y == 1)).sum())
    fp = int(((alarm == 1) & (y == 0)).sum())
    fn = int(((alarm == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def main():
    rows = []
    for ds in DATASETS:
        for sc in SCORERS:
            if not AVAILABILITY.get((sc, ds), False):
                print(f"[E1] SKIP {sc} x {ds}: 无 checkpoint/落盘分数 (记录为不可用)")
                rows.append({"dataset": ds, "scorer": sc, "available": False})
                continue
            t0 = time.time()
            try:
                s, y, dt = get_scores(sc, ds, seed=SEED)
            except Exception as e:
                print(f"[E1] FAIL {sc} x {ds}: {e}")
                rows.append({"dataset": ds, "scorer": sc, "available": False,
                             "note": f"error: {type(e).__name__}: {e}"})
                continue
            row = {"dataset": ds, "scorer": sc, "available": True,
                   "AUROC": auroc(y, s), "n_test": len(y),
                   "anomaly_rate": float(y.mean()), "dt_s": dt}
            f1s = {}
            for q in QUANTILES:
                thr = np.quantile(s, q)
                f1, _, _ = point_f1(y, (s > thr).astype(int))
                f1s[q] = f1
                row[f"F1_q{q}"] = f1
            row["F1_max"] = max(f1s.values())
            row["best_q"] = max(f1s, key=f1s.get)
            rows.append(row)
            print(f"[E1] {sc} x {ds}: AUROC={row['AUROC']:.4f} F1_max={row['F1_max']:.4f} "
                  f"({time.time()-t0:.0f}s)")
    out = os.path.join(RESULTS_DIR, "e1")
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "e1_point_quality.csv"), index=False, encoding="utf-8-sig")
    print(f"[E1] 写出 {out}/e1_point_quality.csv")
    print(df[["dataset", "scorer", "available", "AUROC", "F1_max"]].to_string(index=False))


if __name__ == "__main__":
    main()
