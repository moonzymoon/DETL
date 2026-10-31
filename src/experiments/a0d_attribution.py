# -*- coding: utf-8 -*-
"""B-3 (R2.2): 报警触发时的操作员可读演示 — TE-iforest 层报警的变量级上下文.

预先说明的选择规则 (跑之前固定):
  案例 A = 层检出的第一个测试侧事件 (首发报警时刻);
  案例 B = 层漏检的第一个测试侧事件 (分数峰值 vs 阈值);
  案例 C = dilate 层在未见 run 上的一个检出 (IDV(2) run1 首发报警).
贡献 = 报警时刻窗口(16)内逐变量 |z| 排序 top-3 (检测器侧描述性统计,
控制室可即时计算; 不声称等于 iForest 内在归因或故障根因).
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from a0_lib import load_te_segments, interleave_mask, hard_chain_segmented  # noqa
from datasets.tep import load_tep, TE_DIR, FAULT_ONSET_H  # noqa

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))

# Downs & Vogel (1993) 标准 XMEAS 表 (变量号: 名称 单位)
XMEAS = {
    1: "A feed (str.1) km3/h", 2: "D feed (str.2) kg/h", 3: "E feed (str.3) kg/h",
    4: "A+C feed (str.4) km3/h", 5: "Recirc flow (str.8) km3/h",
    6: "Reactor feed (str.6) km3/h", 7: "Reactor pressure kPa",
    8: "Reactor level %", 9: "Reactor temperature C",
    10: "Purge rate (str.9) km3/h", 11: "Sep. temperature C",
    12: "Sep. level %", 13: "Sep. pressure kPa", 14: "Sep. underflow (str.10) m3/h",
    15: "Stripper level %", 16: "Stripper pressure kPa",
    17: "Stripper underflow (str.11) m3/h", 18: "Stripper temperature C",
    19: "Stripper steam kg/h", 20: "Compressor work kW",
    21: "Reactor CW outlet temp C", 22: "Condenser CW outlet temp C",
}
IDV_NAME = {1: "A/C feed ratio step", 4: "Reactor CW inlet temp step",
            11: "Reactor CW inlet temp random variation",
            2: "B composition step (str.4)", 5: "Condenser CW inlet temp step",
            7: "C header pressure loss", 10: "C feed temperature step",
            13: "Reaction kinetics drift"}


def top3_at(Xz, t, window=16):
    """报警时刻 t 的窗口内逐变量 |z| top-3 (只用当时可见数据)."""
    seg = Xz[t - window + 1: t + 1]
    mean_abs = np.abs(seg).mean(axis=0)
    order = np.argsort(-mean_abs)[:3]
    return [(i + 1, float(mean_abs[i])) for i in order]


def main():
    d = load_tep()
    X, Y, dt = d["X"], d["Y"], d["dt"]          # X 已 z-score (整定段常数)
    s, y, _, bounds, meta = load_te_segments()
    tune_m, test_m = interleave_mask(len(s), dt, bounds)
    prm = pd.read_csv(os.path.join(OUT, "a0_params.csv"))
    lay = prm[prm.method == "layer_r0"].iloc[0]
    theta_hi = float(lay["theta_hi"])
    pl = {"theta_hi": theta_hi, "theta_lo": float(lay["theta_lo"]),
          "T_on": int(lay["T_on"]), "T_off": int(lay["T_off"])}
    alarm = hard_chain_segmented(s, pl, bounds)
    off = 24018                                        # 分数流->全轴偏移 (a0_lib)
    rows = []

    # 测试侧事件 (按起始样本归属)
    from kpi.panel import events_from_binary
    ev_test = [(a, b) for a, b in events_from_binary(y) if test_m[a]]
    ev_hit, ev_miss = [], []
    for a, b in ev_test:
        idx = np.arange(a, b)[test_m[a:b]]
        hit_t = next((t for t in idx if alarm[t]), None)
        (ev_hit if hit_t is not None else ev_miss).append((a, b, hit_t))

    # 案例 A: 首个检出事件
    a0, b0, ht = ev_hit[0]
    seg_id = next(i for i, (x0, x1) in enumerate(bounds) if x0 <= a0 < x1)
    idv = meta[seg_id][1]
    delay_h = (ht - a0) * dt / 3600.0
    for var, z in top3_at(X, off + ht):
        rows.append({"case": "A hit", "idv": idv, "alarm_at_h": (off + ht) * dt / 3600,
                     "delay_h": round(delay_h, 1), "score": float(s[ht]),
                     "theta_hi": theta_hi, "var": var,
                     "name_unit": XMEAS.get(var, f"comp. analysis {var}"),
                     "mean_abs_z": round(z, 1)})

    # 案例 B: 首个漏检事件 (分数峰值)
    a0, b0, _ = ev_miss[0]
    seg_id = next(i for i, (x0, x1) in enumerate(bounds) if x0 <= a0 < x1)
    idv = meta[seg_id][1]
    peak = float(s[a0:b0].max())
    for var, z in top3_at(X, off + a0 + int(np.argmax(s[a0:b0]))):
        rows.append({"case": "B miss", "idv": idv, "alarm_at_h": np.nan,
                     "delay_h": np.nan, "score": peak,
                     "theta_hi": theta_hi, "var": var,
                     "name_unit": XMEAS.get(var, f"comp. analysis {var}"),
                     "mean_abs_z": round(z, 1)})
    print(f"案例B: IDV({idv}) 事件内分数峰值 {peak:.4f} vs θ_hi {theta_hi:.4f} "
          f"({'低于阈值' if peak < theta_hi else '高于但未持续'})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "a0d_attribution.csv"),
              index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print("\n案例A: IDV", IDV_NAME.get(idv), "| 延迟(h):", round(delay_h, 1))


if __name__ == "__main__":
    main()
