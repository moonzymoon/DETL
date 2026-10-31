# -*- coding: utf-8 -*-
"""A0r: C — BATADAL 机理一致性 (按冻结_C_BATADAL映射_20261025.md 执行).

层参数冻结 = a0k_bat_aco_params.csv layer_r0; 分数流按 r33 配方确定性重建;
判定 = 事件起始时刻 |z| top-3 列 vs 冻结直接/级联集合.
输出: results/audit/a0r_batadal_mechconsistency.{csv,json}
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from a0_lib import interleave_mask, hard_chain_segmented  # noqa
from kpi.panel import events_from_binary  # noqa
from common import DATA_ROOT  # noqa

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))
W = 16

# 冻结映射 (攻击编号由标注时长唯一匹配: 42/60/37/7/73)
MAP = {  # seg_len_hours -> (attack, direct, cascade)
    42: ("A1", {"L_T7"}, {"F_PU10", "S_PU10", "F_PU11", "S_PU11"}),
    60: ("A3", {"L_T1"}, {"F_PU1", "S_PU1", "F_PU2", "S_PU2"}),
    37: ("A4", {"L_T1"}, {"F_PU1", "S_PU1", "F_PU2", "S_PU2"}),
    7:  ("A5", {"S_PU7"}, {"F_PU7", "L_T4"}),
    73: ("A6", {"L_T4"}, {"F_PU7", "S_PU7"}),
}


def rebuild_stream():
    d = np.load(os.path.join(OUT, "r33_batadal_data.npz"))
    X, Y, dt = d["X"], d["Y"], float(d["dt"])
    n = len(X)
    a, b = int(n * 0.35), int(n * 0.50)
    idx_w = np.arange(W)[None, :] + np.arange(b - a - W + 1)[:, None]
    featW = X[a:b][idx_w].reshape(len(idx_w), -1)
    ifo = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=0, n_jobs=-1).fit(featW)
    N = n - b - W + 1
    idx = np.arange(W)[None, :] + np.arange(N)[:, None]
    s = -ifo.decision_function(X[b:][idx].reshape(N, -1)).astype(np.float64)
    y = Y[b + W - 1:]
    # 列名 (与 r33 相同的交集顺序)
    d1 = pd.read_csv(os.path.join(DATA_ROOT, "BATADAL",
                                  "BATADAL_training1.csv"), nrows=2)
    cols = [c.strip() for c in d1.columns
            if c.strip() not in ("DATETIME", "ATT_FLAG")]
    assert len(cols) == X.shape[1], (len(cols), X.shape)
    return s, y, dt, X, cols, b


def main():
    s, y, dt, X, cols, b = rebuild_stream()
    prm_df = pd.read_csv(os.path.join(OUT, "a0k_bat_aco_params.csv"))
    row = prm_df[(prm_df.stream == "BATADAL") & (prm_df.method == "layer_r0")].iloc[0]
    prm = {"theta_hi": float(row.theta_hi), "theta_lo": float(row.theta_lo),
           "T_on": int(row.T_on), "T_off": int(row.T_off)}
    bounds = [(0, len(s))]
    tune_m, test_m = interleave_mask(len(s), dt, bounds)
    alarm = hard_chain_segmented(s, prm, bounds)

    gt = events_from_binary(y)
    al = events_from_binary(alarm)
    rows = []
    for (g0, g1) in gt:
        in_test = bool(test_m[g0])
        ovl = [(a0, a1) for (a0, a1) in al if a0 < g1 and g0 < a1]
        detected = len(ovl) > 0
        if detected:
            t_hit = ovl[0][0]
            inst = t_hit
            mode = "annunciation"
        else:
            inst = g0 + int(np.argmax(s[g0:g1]))
            mode = "score-peak"
        z = np.abs(X[b + W - 1 + inst])
        top3 = [cols[i] for i in np.argsort(-z)[:3]]
        # 该 GT 段的攻击编号: 用段内 y 连续长度 (小时) 匹配
        seg_h = (g1 - g0) * dt / 3600.0
        key = min(MAP, key=lambda k: abs(k - seg_h))
        atk, direct, casc = MAP[key]
        acc = "direct" if top3[0] in direct else (
            "cascade" if top3[0] in casc else (
                "partial" if any(c in direct | casc for c in top3) else "none"))
        rows.append({"attack": atk, "gt_len_h": seg_h, "test_side": in_test,
                     "detected": detected, "mode": mode,
                     "top1": top3[0], "top2": top3[1], "top3": top3[2],
                     "top1_z": round(float(z[cols.index(top3[0])]), 2),
                     "verdict": acc,
                     "onset_score_idx": int(inst)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "a0r_batadal_mechconsistency.csv"),
              index=False, encoding="utf-8-sig")
    te = df[df.test_side]
    n_ok = int((te.verdict.isin(["direct", "cascade"])).sum())
    summary = {
        "n_gt_events_total": len(df),
        "n_test_side": len(te),
        "n_test_direct_or_cascade": n_ok,
        "frac": round(n_ok / len(te), 3) if len(te) else None,
        "per_event": df.to_dict("records"),
        "protocol": "frozen map 00返修/冻结_C_BATADAL映射_20261025.md",
    }
    import json
    with open(os.path.join(OUT, "a0r_batadal_mechconsistency.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(df.to_string(index=False))
    print(f"\ntest-side direct/cascade: {n_ok}/{len(te)}")


if __name__ == "__main__":
    main()
