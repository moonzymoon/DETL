"""R9 审稿补充: 紧预算 (R_MAX=0.1/h) 下的 home-turf 复验 —— 强制 T_on>1.

R8 验证中 41 个变量全部选 T_on=1 (宽松预算下单阈值即达标, 设计取最小延迟).
审稿人指出延迟计时器路径未被行使. 本脚本把 R_MAX 压到 0.1/h 重跑:
设计将被迫选更大 T_on / 更高死区, 再验 iid 对照是否仍命中预测.

输出: results/audit/r9_gyasi_tight.csv + 控制台汇总
运行: cd src && python experiments/e_r9_gyasi_tight.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR
from datasets.tep import load_tep
import experiments.e3_baselines as e3
from kpi.panel import events_from_binary, rising_edges
from tuning.hard_ops import hard_pipeline

TUNE_NORM = 15000


def run(r_max):
    e3.R_MAX = r_max                       # run_gyasi_wang 引用模块级 R_MAX
    d = load_tep()
    X, dt = d["X"], d["dt"]
    n_norm = 30001
    runs = [n_norm + i * 3001 for i in range(6)]
    on = int(8 * 60)
    tune_norm = np.arange(0, TUNE_NORM)
    test_norm = np.arange(TUNE_NORM, 30000)
    tune_abn = np.concatenate([np.arange(r + on + 200, r + on + 1200) for r in runs])
    test_abn = [(r + on + 1200, r + 3001) for r in runs]
    rows = []
    for v in range(X.shape[1]):
        x = X[:, v].astype(np.float64)
        s_tr = np.concatenate([x[tune_norm], x[tune_abn]])
        y_tr = np.concatenate([np.zeros(TUNE_NORM, dtype=np.int8),
                               np.ones(len(tune_abn), dtype=np.int8)])
        prm = e3.run_gyasi_wang(s_tr, y_tr, events_from_binary(y_tr), dt)
        theta_hi = np.quantile(s_tr, prm["q"])
        Pn = float((s_tr[y_tr == 0] > theta_hi).mean())
        r_pred = (Pn ** prm["t_on"]) * (1 - Pn) * (3600.0 / dt)
        al_n = hard_pipeline(x[test_norm], theta_hi, theta_hi - prm["margin"],
                             int(prm["t_on"]), int(prm["t_off"]))
        far = len(rising_edges(al_n)) / (len(test_norm) * dt / 3600.0)
        dets = []
        for a, b in test_abn:
            al_f = hard_pipeline(x[a:b], theta_hi, theta_hi - prm["margin"],
                                 int(prm["t_on"]), int(prm["t_off"]))
            dets.append(float(al_f.max() > 0))
        rows.append({"variable": v + 1, "t_on": prm["t_on"],
                     "margin_std": prm["margin"] / (np.std(s_tr) + 1e-9),
                     "r_pred_per_h": r_pred, "far_achieved_per_h": far,
                     "ratio": far / max(r_pred, 1e-9),
                     "det_rate": float(np.mean(dets))})
    return pd.DataFrame(rows)


def iid_ctrl(r_max, df_raw):
    """同一紧预算下的 iid 洗牌对照 (仅误报率)."""
    e3.R_MAX = r_max
    d = load_tep()
    X, dt = d["X"], d["dt"]
    n_norm = 30001
    runs = [n_norm + i * 3001 for i in range(6)]
    on = int(8 * 60)
    tune_abn = np.concatenate([np.arange(r + on + 200, r + on + 1200) for r in runs])
    rng = np.random.default_rng(7)
    out = []
    for v in range(X.shape[1]):
        x = X[:, v].astype(np.float64)
        tn = rng.permutation(x[:TUNE_NORM])
        te = rng.permutation(x[TUNE_NORM:30000])
        s_tr = np.concatenate([tn, x[tune_abn]])
        y_tr = np.concatenate([np.zeros(TUNE_NORM, dtype=np.int8),
                               np.ones(len(tune_abn), dtype=np.int8)])
        prm = e3.run_gyasi_wang(s_tr, y_tr, events_from_binary(y_tr), dt)
        theta_hi = np.quantile(s_tr, prm["q"])
        Pn = float((tn > theta_hi).mean())
        r_pred = (Pn ** prm["t_on"]) * (1 - Pn) * (3600.0 / dt)
        al_n = hard_pipeline(te, theta_hi, theta_hi - prm["margin"],
                             int(prm["t_on"]), int(prm["t_off"]))
        far = len(rising_edges(al_n)) / (len(te) * dt / 3600.0)
        out.append(far / max(r_pred, 1e-9))
    return np.array(out)


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r9_gyasi_tight.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df = run(0.1)
    ctrl = iid_ctrl(0.1, df)
    df["iid_ctrl_ratio"] = ctrl
    df.to_csv(out, index=False, encoding="utf-8-sig")
    t_dist = df.t_on.value_counts().sort_index()
    print(f"R_MAX=0.1/h | T_on 分布: {dict(t_dist)}")
    print(f"achieved/predicted: 中位 {df.ratio.median():.2f}x  "
          f"IQR [{df.ratio.quantile(.25):.2f}, {df.ratio.quantile(.75):.2f}]")
    print(f"iid 对照: 中位 {np.median(ctrl):.2f}x  "
          f"IQR [{np.quantile(ctrl, .25):.2f}, {np.quantile(ctrl, .75):.2f}]")
    print(f"det_rate: 中位 {df.det_rate.median():.2f}  margin_std 中位 "
          f"{df.margin_std.median():.2f}")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
