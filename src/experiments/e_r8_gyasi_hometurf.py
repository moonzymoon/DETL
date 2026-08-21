"""R8 审稿要求 M2: 解析基线 (Gyasi& Wang 简化复现) 的 home-turf 验证.

问题: 主表中解析基线在分数流上全面失败 (碎裂 37-6213 段). 该失败是
"移植越界"还是"实现错误"? 本实验把同一个 run_gyasi_wang 实现放回其
设计场景 —— 原始单变量过程信号 (TE 41 个 XMEAS):

  设计段 = 250h 纯正常 + 每故障run前半故障段
  测试段 = 后 250h 纯正常 + 每故障run后半故障段
  指标   = 设计预测误报率 r_false (iid 公式) vs 持出正常段实际误报率;
           各故障段检出 (任意重叠) 与故障点漏报比例; 变量 lag-1 自相关.
  对照   = iid 洗牌控制 (独立打乱调优/测试正常样本, 保持边缘分布,
           破坏自相关) —— 区分"公式/代码自洽"与"自相关越界".

输出: results/audit/r8_gyasi_hometurf.csv (逐变量) + 控制台汇总
运行: cd src && python experiments/e_r8_gyasi_hometurf.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR
from datasets.tep import load_tep
from experiments.e3_baselines import run_gyasi_wang
from kpi.panel import events_from_binary, rising_edges
from tuning.hard_ops import hard_pipeline

R_MAX = 10.0


def lag1(x):
    x = x - x.mean()
    return float((x[:-1] * x[1:]).sum() / max((x * x).sum(), 1e-12))


def auroc(x_neg, x_pos):
    n = len(x_neg) + len(x_pos)
    ranks = pd.Series(np.concatenate([x_neg, x_pos])).rank().values
    return float((ranks[len(x_neg):].sum()
                  - len(x_pos) * (len(x_pos) + 1) / 2.0) / (len(x_neg) * len(x_pos)))


def main():
    d = load_tep()
    X, dt = d["X"], d["dt"]
    n_norm = 30001
    runs = [n_norm + i * 3001 for i in range(6)]        # 各故障 run 起点
    on = int(8 * 60)                                     # 故障注入 (第8h, 点)

    tune_norm = np.arange(0, 15000)
    test_norm = np.arange(15000, 30000)
    tune_abn = np.concatenate([np.arange(r + on + 200, r + on + 1200) for r in runs])
    test_abn = [(r + on + 1200, r + 3001) for r in runs]

    rows = []
    for v in range(X.shape[1]):
        x = X[:, v].astype(np.float64)
        # --- 调优流: 正常段 + 故障段拼接 ---
        s_tr = np.concatenate([x[tune_norm], x[tune_abn]])
        y_tr = np.concatenate([np.zeros(len(tune_norm), dtype=np.int8),
                               np.ones(len(tune_abn), dtype=np.int8)])
        ev_tr = events_from_binary(y_tr)
        prm = run_gyasi_wang(s_tr, y_tr, ev_tr, dt)
        theta_hi = np.quantile(s_tr, prm["q"])
        theta_lo = theta_hi - prm["margin"]
        # 设计预测 (iid 公式, 与 e3 实现一致)
        Pn = float((s_tr[y_tr == 0] > theta_hi).mean())
        r_pred = (Pn ** prm["t_on"]) * (1 - Pn) * (3600.0 / dt)
        # --- 持出评估: 正常段误报率 ---
        al_n = hard_pipeline(x[test_norm], theta_hi, theta_lo,
                             int(prm["t_on"]), int(prm["t_off"]))
        h_norm = len(test_norm) * dt / 3600.0
        far = len(rising_edges(al_n)) / max(h_norm, 1e-9)
        # --- 故障段检出与点级漏报 ---
        dets, miss_pts = [], []
        for a, b in test_abn:
            al_f = hard_pipeline(x[a:b], theta_hi, theta_lo,
                                 int(prm["t_on"]), int(prm["t_off"]))
            dets.append(float(al_f.max() > 0))
            miss_pts.append(float(1.0 - al_f.mean()))
        rows.append({"variable": v + 1,
                     "auroc": auroc(x[tune_norm], x[tune_abn]),
                     "lag1": lag1(x[test_norm]),
                     "q": prm["q"], "t_on": prm["t_on"],
                     "margin_std": prm["margin"] / (np.std(s_tr) + 1e-9),
                     "r_pred_per_h": r_pred, "far_achieved_per_h": far,
                     "ratio": far / max(r_pred, 1e-9),
                     "det_rate": float(np.mean(dets)),
                     "point_miss": float(np.mean(miss_pts))})

    df = pd.DataFrame(rows)
    # --- iid 控制: 洗牌正常样本 (同一设计代码) ---
    rng = np.random.default_rng(7)
    ctrl = []
    for v in range(X.shape[1]):
        x = X[:, v].astype(np.float64)
        tn = rng.permutation(x[tune_norm])
        te = rng.permutation(x[test_norm])
        s_tr = np.concatenate([tn, x[tune_abn]])
        y_tr = np.concatenate([np.zeros(len(tn), dtype=np.int8),
                               np.ones(len(tune_abn), dtype=np.int8)])
        prm = run_gyasi_wang(s_tr, y_tr, events_from_binary(y_tr), dt)
        theta_hi = np.quantile(s_tr, prm["q"])
        Pn = float((tn > theta_hi).mean())
        r_pred = (Pn ** prm["t_on"]) * (1 - Pn) * (3600.0 / dt)
        al_n = hard_pipeline(te, theta_hi, theta_hi - prm["margin"],
                             int(prm["t_on"]), int(prm["t_off"]))
        far = len(rising_edges(al_n)) / (len(te) * dt / 3600.0)
        ctrl.append(far / max(r_pred, 1e-9))
    df["iid_ctrl_ratio"] = ctrl

    out = os.path.join(RESULTS_DIR, "audit", "r8_gyasi_hometurf.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    top = df.sort_values("auroc", ascending=False).head(6)
    print("=== 全 41 变量汇总 ===")
    print(f"achieved/predicted FAR 比: 中位数 {df.ratio.median():.2f}x  "
          f"IQR [{df.ratio.quantile(.25):.2f}, {df.ratio.quantile(.75):.2f}]")
    print(f"iid 洗牌控制 比: 中位数 {np.median(ctrl):.2f}x  "
          f"IQR [{np.quantile(ctrl, .25):.2f}, {np.quantile(ctrl, .75):.2f}]")
    print(f"故障段检出率: 中位数 {df.det_rate.median():.2f}  "
          f"lag-1 自相关: 中位数 {df.lag1.median():.3f}")
    print("=== 分离度前 6 变量 ===")
    print(top[["variable", "auroc", "lag1", "t_on", "r_pred_per_h",
               "far_achieved_per_h", "ratio", "det_rate", "point_miss"]]
          .to_string(index=False))
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
