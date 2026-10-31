# -*- coding: utf-8 -*-
"""A0ab: 动态阈值基线 v2 — 修复 GPT 终审两项 P0 (因果性 + 协议 M 评估).

修复协议 (2026-10-26 冻结, 见登记):
1. 因果阈值: 逐 1 天块重算, th = quantile(s[max(0,i-win):i], q) 仅用严格过去样本;
   首小时(min_hist=1h)为校准期不出告警; 不足 2 个历史样本时阈值=+inf.
   (v1 缺陷: i=0 用首个 24h 初始化含未来/故障样本 — 已废弃为 legacy)
2. 协议 M 评估: 主流全时间轴生成告警后调 protocol_M_metrics(alarm,y,test_m,dt),
   与层/搜索同一口径 (v1 缺陷: 首末测试样本之间的连续区间含整定块 — 已废弃).
3. TE 分段流逐段因果; 34 fresh run 全 run 因果 (kpi_panel, 同 a0u 口径).
输出: results/audit/a0ab_dynthresh_v2.csv (含 stream_source 字段, 无静默回退)
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from a0_lib import (load_te_segments, interleave_mask, protocol_M_metrics)  # noqa
from kpi.panel import KPIConfig, kpi_panel  # noqa
from datasets.tep import TE_DIR, FAULT_FILES, FAULT_ONSET_H  # noqa
from sklearn.ensemble import IsolationForest  # noqa

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))
EXT = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "TE_ext"))
OLD = r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/datasets/TE"
W = 16
QS = [("roll-q99", 0.99), ("roll-q999", 0.999)]
DAYS, MIN_H = 7.0, 1.0


def causal_alarms(s, dt, q):
    """逐 1 天块, 仅严格过去样本的分位阈值; 首 1 小时校准期无告警."""
    n = len(s)
    win = max(int(DAYS * 86400.0 / dt), 2)
    step = max(int(MIN_H * 3600.0 / dt), 2)   # 每小时重算(校准期=1h)
    alarm = np.zeros(n, np.int8)
    th = np.inf
    for i in range(n):
        if i >= step and (i - step) % step == 0:
            j0 = max(0, i - win)
            seg = s[j0:i]                       # 严格过去, >=1h 历史
            th = float(np.quantile(seg, q))
        alarm[i] = 1 if (i >= step and s[i] > th) else 0
    return alarm


def main():
    rows = []
    from scorers.adapter import get_scores
    # ── 单段主流: SWaT / Metro ──
    for scorer, ds in [("cmhmil", "SWaT"), ("iforest", "MetroPT3")]:
        s, y, dt = get_scores(scorer, ds)
        s = np.asarray(s, np.float64)
        y = np.asarray(y)
        n = len(s)
        _, test_m = interleave_mask(n, dt, [(0, n)])
        for tag, q in QS:
            a = causal_alarms(s, dt, q)
            m = protocol_M_metrics(a, y, test_m, dt)
            rows.append({"suite": "main", "stream": f"{ds}-{scorer}",
                         "source": "adapter", "variant": tag,
                         "EF1": round(m["EF1_ov"], 4),
                         "det": round(m["det_ov"], 4),
                         "AAR": round(m["AAR_side"], 4)})
        print(rows[-2]["stream"], "v2 done")
    # ── TE 三打分器: 分段流 (iForest) + adapter 单段 (cmh/usad 走分段重放不可行,
    #    cmh/usad 为旧协议整定参数对照, 其流构造同 v1 采 adapter 单段) ──
    s, y, dt, bounds, meta = load_te_segments()
    _, test_m = interleave_mask(len(s), dt, bounds)
    for tag, q in QS:
        a = np.zeros(len(s), np.int8)
        for g0, g1 in bounds:
            a[g0:g1] = causal_alarms(s[g0:g1], dt, q)
        m = protocol_M_metrics(a, y, test_m, dt)
        rows.append({"suite": "main", "stream": "TE-iforest",
                     "source": "segments", "variant": tag,
                     "EF1": round(m["EF1_ov"], 4),
                     "det": round(m["det_ov"], 4),
                     "AAR": round(m["AAR_side"], 4)})
    print("TE-iforest segments v2 done")
    def usad_te_scores():
        # 真 USAD 分数 (adapter 无 usad, 会静默回退 iForest — 禁用)
        import torch as _t
        _prev = _t.get_default_dtype()
        _t.set_default_dtype(_t.float32)
        from common import CACHE_DIR as _CC
        import experiments.e_r48_usad as _u
        d = np.load(_CC + "/tep_data.npz")
        X = d["X"].astype(np.float32)
        a_, b_ = int(len(X) * 0.35), int(len(X) * 0.50)
        ae1_, ae2_ = _u.train_usad(_u.windows(X[a_:b_])[:100000])
        sc = _u.score_stream(ae1_, ae2_, _u.windows(X[b_:]))
        _t.set_default_dtype(_prev)
        return (np.asarray(sc, np.float64),
                np.asarray(d["Y"][b_:][_u.W - 1:]), 60.0)

    # R15 修复: cmh/usad 同样按独立 run 分段复位(load_te_segments 同构边界,
    # 三流坐标已验证逐位一致), 不再连续滚动整条拼接流
    _, _, _, te_bounds, _ = load_te_segments()
    for scorer in ("cmhmil", "usad"):
        if scorer == "usad":
            s2, y2, dt2 = usad_te_scores()
        else:
            s2, y2, dt2 = get_scores(scorer, "TE")
        s2 = np.asarray(s2, np.float64)
        y2 = np.asarray(y2)
        n2 = len(s2)
        assert te_bounds[-1][1] == n2, (scorer, te_bounds[-1], n2)
        _, tm2 = interleave_mask(n2, dt2, te_bounds)
        for tag, q in QS:
            a = np.zeros(n2, np.int8)
            for g0, g1 in te_bounds:
                a[g0:g1] = causal_alarms(s2[g0:g1], dt2, q)
            m = protocol_M_metrics(a, y2, tm2, dt2)
            rows.append({"suite": "main", "stream": f"TE-{scorer}",
                         "source": "adapter", "variant": tag,
                         "EF1": round(m["EF1_ov"], 4),
                         "det": round(m["det_ov"], 4),
                         "AAR": round(m["AAR_side"], 4)})
        print(f"TE-{scorer} v2 done")

    # ── 34 fresh run (iForest 冻结) ──
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
        y3 = (t >= FAULT_ONSET_H).astype(np.int8)
        Zz = (Xr - mu) / sd
        N = len(Zz) - W + 1
        idx = np.arange(W)[None, :] + np.arange(N)[:, None]
        s3 = np.asarray(-ifo.decision_function(Zz[idx].reshape(N, -1)),
                        np.float64)
        yy = y3[W - 1:]
        for tag, q in QS:
            a3 = causal_alarms(s3, 60.0, q)
            m = kpi_panel(a3, yy, cfg=KPIConfig(dt=60.0))
            rows.append({"suite": "fresh", "stream": f"IDV{idv}",
                         "source": "fresh", "variant": tag,
                         "EF1": round(m["EventF1"], 4),
                         "det": round(m["event_detection_rate"], 4),
                         "AAR": round(m["AAR_per_hour"], 4)})
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(OUT, "a0ab_dynthresh_v2.csv"), index=False,
                  encoding="utf-8-sig")
    print(df_out.groupby(["suite", "variant"])[["EF1", "det", "AAR"]]
          .mean().round(3).to_string())
    print(df_out[df_out.suite == "main"].to_string(index=False))


if __name__ == "__main__":
    main()
