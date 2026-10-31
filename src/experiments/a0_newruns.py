# -*- coding: utf-8 -*-
"""A0.5: 未见故障类型独立 run 验证 (本地 10 个 run: IDV 2/5/7/10/13 x 2).

协议 (设计文档 §5): z-score 常数/检测器/分数变换全部沿用整定阶段设置;
新 run 全段前向; 逐 run 配对 (层 vs 各搜索), 不造置信区间.
前置: a0_te_iforest.py 产出 results/audit/a0_params.csv
输出: results/audit/a0_newruns.csv
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from a0_lib import hard_chain_segmented, load_te_segments  # noqa: E402
from kpi.panel import KPIConfig, events_from_binary, kpi_panel  # noqa: E402
from tuning.hard_ops import hard_pipeline  # noqa: E402
from datasets.tep import TE_DIR, FAULT_ONSET_H, N_XMEAS, XMEAS_COLS  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))
UNSEEN = [(idv, run) for idv in (2, 5, 7, 10, 13) for run in (1, 2)]
WINDOW = 16


def rebuild_scorer():
    """精确复现整定阶段: 全流 z-score(前35%) + iForest(拟合 35-50% 段, seed=0)."""
    from datasets.tep import FAULT_FILES
    Xs = []
    df0 = pd.read_excel(os.path.join(TE_DIR, "mode1_normal_500.xlsx"))
    feat = [c for c in df0.columns if c != "Time"]
    Xs.append(df0[feat].values.astype(np.float32))
    for f in FAULT_FILES:
        df = pd.read_excel(os.path.join(TE_DIR, f"{f}.xlsx"))
        Xs.append(df[[c for c in df.columns if c.startswith("XMEAS")]]
                  .values.astype(np.float32))
    X = np.vstack(Xs)
    n_tr = int(len(X) * 0.35)
    mu = X[:n_tr].mean(0, keepdims=True)
    sd = X[:n_tr].std(0, keepdims=True) + 1e-8
    Z = (X - mu) / sd
    a, b = int(len(Z) * 0.35), int(len(Z) * 0.50)
    idx_w = (np.arange(WINDOW)[None, :]
             + np.arange(len(Z[a:b]) - WINDOW + 1)[:, None])
    featW = Z[a:b][idx_w].reshape(len(idx_w), -1)
    ifo = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=0, n_jobs=-1)
    ifo.fit(featW)
    return mu, sd, ifo


def score_run(path, mu, sd, ifo):
    """新 run 前向打分 (z-score 用整定常数, iForest 不重拟合)."""
    df = pd.read_excel(path)
    X = df[[c for c in df.columns if c.startswith("XMEAS")]].values
    t = df["Time (h)"].values
    y = (t >= FAULT_ONSET_H).astype(np.int8)
    Z = (X - mu) / sd
    N = len(Z) - WINDOW + 1
    idx = np.arange(WINDOW)[None, :] + np.arange(N)[:, None]
    feats = Z[idx].reshape(N, -1)
    s = -ifo.decision_function(feats)
    return s.astype(np.float64), y[WINDOW - 1:], 60.0


def eval_run(prm, s, y, dt):
    alarm = hard_pipeline(s, prm["theta_hi"], prm["theta_lo"],
                          int(prm["T_on"]), int(prm["T_off"]))
    return kpi_panel(alarm, y, cfg=KPIConfig(dt=dt))


def main():
    mu, sd, ifo = rebuild_scorer()
    prm_df = pd.read_csv(os.path.join(OUT, "a0_params.csv"))
    prm_df = prm_df[prm_df["method"].str.match(
        r"layer_r\d|optuna_s\d_B240|random_s\d_B240|grid-delay")]
    rows = []
    for idv, run in UNSEEN:
        path = os.path.join(TE_DIR, f"mode1_{idv}_{run}.xlsx")
        s, y, dt = score_run(path, mu, sd, ifo)
        ev = events_from_binary(y)
        for _, r in prm_df.iterrows():
            prm = {"theta_hi": r["theta_hi"], "theta_lo": r["theta_lo"],
                   "T_on": int(r["T_on"]), "T_off": int(r["T_off"])}
            m = eval_run(prm, s, y, dt)
            rows.append({"idv": idv, "run": run, "method": r["method"],
                         "n_events": len(ev), "EventF1": m["EventF1"],
                         "det": m["event_detection_rate"],
                         "AAR": m["AAR_per_hour"],
                         "MAD_s": m["mean_alarm_delay_s"],
                         "n_alarm": m["n_alarm_events"]})
        print(f"IDV({idv}) run{run}: 完成 {len(prm_df)} 配置")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "a0_newruns.csv"),
              index=False, encoding="utf-8-sig")
    piv = df.pivot_table(index="method", values=["EventF1", "det", "AAR"],
                         aggfunc="mean")
    print("\n==== A0.5 逐 run 均值概览 (10 未见 run) ====")
    print(piv.round(3).to_string())
    print("\n逐 run 明细已写 results/audit/a0_newruns.csv")


if __name__ == "__main__":
    main()
