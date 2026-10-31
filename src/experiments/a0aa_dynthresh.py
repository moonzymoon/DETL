# -*- coding: utf-8 -*-
"""A0aa: S2 — 动态阈值基线 (滚动分位, 仅历史窗口) 协议 M 评估 (冻结 §S2).

变体 A: q99 / B: q99.9, w=7天(原生采样), T_on=T_off=1, 无死区.
评估: 5 主流测试侧 + 34 新鲜 run (iForest 冻结打分器).
输出: results/audit/a0aa_dynthresh.{csv,json}
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from a0_lib import interleave_mask, eval_params_M, hard_chain_segmented  # noqa
from kpi.panel import KPIConfig, events_from_binary, kpi_panel  # noqa
from tuning.hard_ops import hard_pipeline  # noqa
from datasets.tep import TE_DIR, FAULT_FILES, FAULT_ONSET_H  # noqa
from sklearn.ensemble import IsolationForest  # noqa

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))
EXT = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "TE_ext"))
OLD = r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/datasets/TE"
W = 16
QS = [("roll-q99", 0.99), ("roll-q999", 0.999)]


def rolling_threshold(s, y, dt, q, days=7.0):
    """仅历史滚动窗口分位阈值; 窗口不足 1 天用全部已有历史."""
    win = max(int(days * 86400.0 / dt), 1)
    minw = max(int(1.0 * 86400.0 / dt), 1)
    n = len(s)
    thr = np.empty(n)
    lo = thr[0]  # placeholder
    # 用滑动分位近似: 累积前缀 + 分块近似(每 minw 步重算一次, 块内沿用)
    block = max(minw, 1)
    th = np.quantile(s[:max(minw, 2)], q)
    for i in range(n):
        if i % block == 0:
            j0 = max(0, i - win)
            seg = s[j0:i]
            th = np.quantile(seg, q) if len(seg) >= min(minw, 2) else th
        thr[i] = th
    return thr


def eval_dyn(s, y, dt, q, bounds=None):
    thr = rolling_threshold(s, y, dt, q)
    alarm = (s > thr).astype(np.int8)
    # T_on=T_off=1 => 直接用 binary
    return kpi_panel(alarm, y, cfg=KPIConfig(dt=dt))


def main():
    rows = []
    # ── 5 主流 ──
    from scorers.adapter import get_scores
    streams = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]
    # TE 若为分段加载 (load_te_segments), 单独处理
    for scorer, ds in streams:
        s, y, dt = get_scores(scorer, ds)
        s = np.asarray(s, np.float64)
        y = np.asarray(y)
        n = len(s)
        bounds = [(0, n)]
        _, test_m = interleave_mask(n, dt, bounds)
        st, en = np.flatnonzero(test_m)[0], np.flatnonzero(test_m)[-1] + 1
        for tag, q in QS:
            m = eval_dyn(s[st:en], y[st:en], dt, q)
            rows.append({"suite": "main", "stream": f"{ds}-{scorer}",
                         "variant": tag,
                         "EF1": round(m["EventF1"], 4),
                         "det": round(m["event_detection_rate"], 4),
                         "AAR": round(m["AAR_per_hour"], 4)})
        print(rows[-2], rows[-1])
    # TE iForest 分段流
    from a0_lib import load_te_segments
    s, y, dt, bounds, meta = load_te_segments()
    _, test_m = interleave_mask(len(s), dt, bounds)
    idx = np.flatnonzero(test_m)
    # 分段处理(段界复位): 段内滚动
    for tag, q in QS:
        alarms = np.zeros(len(s), np.int8)
        for a, b in bounds:
            thr = rolling_threshold(s[a:b], None, dt, q)
            alarms[a:b] = (s[a:b] > thr)
        yy = y[idx]
        m = kpi_panel(alarms[idx], yy, cfg=KPIConfig(dt=dt))
        rows.append({"suite": "main", "stream": "TE-iforest", "variant": tag,
                     "EF1": round(m["EventF1"], 4),
                     "det": round(m["event_detection_rate"], 4),
                     "AAR": round(m["AAR_per_hour"], 4)})
    print(rows[-2], rows[-1])
    # cmhmil TE / usad TE (adapter 分数)
    for scorer, ds in [("cmhmil", "TE"), ("usad", "TE")]:
        try:
            s, y, dt = get_scores(scorer, ds)
        except Exception as e:
            print("skip", scorer, ds, e)
            continue
        s = np.asarray(s, np.float64)
        y = np.asarray(y)
        n = len(s)
        _, test_m = interleave_mask(n, dt, [(0, n)])
        st, en = np.flatnonzero(test_m)[0], np.flatnonzero(test_m)[-1] + 1
        for tag, q in QS:
            m = eval_dyn(s[st:en], y[st:en], dt, q)
            rows.append({"suite": "main", "stream": f"{ds}-{scorer}",
                         "variant": tag,
                         "EF1": round(m["EventF1"], 4),
                         "det": round(m["event_detection_rate"], 4),
                         "AAR": round(m["AAR_per_hour"], 4)})

    # ── 34 新鲜 run (iForest 冻结) ──
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
    Z0 = (X - mu) / sd
    a, b = int(len(Z0) * 0.35), int(len(Z0) * 0.50)
    idx_w = (np.arange(W)[None, :] + np.arange(b - a - W + 1)[:, None])
    ifo = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=0, n_jobs=-1).fit(
        Z0[a:b][idx_w].reshape(len(idx_w), -1))
    FRESH = [(i, r) for i in (2, 5, 7, 10, 13) for r in (1, 2)] + \
            [(i, r) for i in (3, 6, 8, 9, 12, 14, 15, 16, 17, 18, 19, 20)
             for r in (1, 2)]
    for idv, run in FRESH:
        path = (os.path.join(OLD, f"mode1_{idv}_{run}.xlsx")
                if idv in (2, 5, 7, 10, 13)
                else os.path.join(EXT, f"mode1_{idv}_{run}.xlsx"))
        df = pd.read_excel(path)
        xcols = [c for c in df.columns if c != "Time (h)"][:41]
        Xr = df[xcols].values.astype(np.float32)
        t = df["Time (h)"].values
        y = (t >= FAULT_ONSET_H).astype(np.int8)
        Zz = (Xr - mu) / sd
        N = len(Zz) - W + 1
        idx = np.arange(W)[None, :] + np.arange(N)[:, None]
        s = -ifo.decision_function(Zz[idx].reshape(N, -1))
        yy = y[W - 1:]
        for tag, q in QS:
            m = eval_dyn(np.asarray(s, np.float64), yy, 60.0, q)
            rows.append({"suite": "fresh", "stream": f"IDV{idv}", "variant": tag,
                         "EF1": round(m["EventF1"], 4),
                         "det": round(m["event_detection_rate"], 4),
                         "AAR": round(m["AAR_per_hour"], 4)})
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT, "a0aa_dynthresh.csv"), index=False,
                  encoding="utf-8-sig")
    print(df_out.groupby(["suite", "variant"])[["EF1", "det", "AAR"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
