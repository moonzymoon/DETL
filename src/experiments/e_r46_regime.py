"""R46 (意见2): 先验工况判别规则 —— 整定段事件尺度预测 层 vs 搜索 的胜负.

特征 (先验可用, 不用测试标签): 整定段事件数 n_tr, 中位事件长度 (点),
中位事件长度 / K_off (计时器上限, 无量纲化).
标签: TSB-AD 120 流上 ours-optuna 的 F1 差.
验证: 四工业基准 + TE-ext (规则在它们上的预测 vs 实际).
输出: results/audit/r46_regime.csv
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
from experiments.e_r43_tsbad import TSB_DIR, interleave_blocks, load_stream

OUT = os.path.join(RESULTS_DIR, "audit", "r46_regime.csv")


def episode_stats(y):
    ev = events_from_binary(y)
    lens = [e - b for b, e in ev] or [0]
    return len(ev), float(np.median(lens))


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "audit", "r43_tsbad.csv"))
    if "n_tr_ev" not in df:
        from experiments.e_r43b_summary import main as _m  # 已含 n_tr_ev 列
    try:
        n_tr = df["n_tr_ev"]
    except KeyError:
        n_tr = None
    rows = []
    from experiments.e_r43_tsbad import interleave_blocks, load_stream
    for _, r in df.iterrows():
        try:
            s, y = load_stream(os.path.join(TSB_DIR, r["file"]))
            tr_m, _ = interleave_blocks(len(s))
            n_ev, med_len = episode_stats(y[tr_m])
        except Exception:
            n_ev, med_len = -1, 0.0
        rows.append({"file": r["file"], "n_tr_ev": n_ev,
                     "med_ep_tr": med_len, "ours": r["ours"],
                     "optuna": r["optuna"]})
    st = pd.DataFrame(rows)
    st["diff"] = st["ours"] - st["optuna"]
    st = st[st.n_tr_ev >= 0]
    # 扫阈值: 整定段中位事件长度 >= thr -> 预测层不输
    best = (None, -1)
    for thr in np.unique(st.med_ep_tr.values):
        pred_win = st.med_ep_tr >= thr          # 预测层>=搜索
        actual_win = st["diff"] > -1e-9
        acc = (pred_win == actual_win).mean()
        if acc > best[1]:
            best = (float(thr), acc)
    thr, acc = best
    print(f"[r46] 阈值=整定段中位事件长度 >= {thr:g} 点 -> 预测层不输; "
          f"120 流准确率 {acc:.3f} ({int(acc*len(st))}/{len(st)})", flush=True)
    for tag, cond in [(">=thr", st.med_ep_tr >= thr), ("<thr", ~(
            st.med_ep_tr >= thr))]:
        sub = st[cond]
        w = (sub["diff"] > 1e-9).sum(); t = (abs(sub["diff"]) <= 1e-9).sum()
        l = (sub["diff"] < -1e-9).sum()
        print(f"[r46] {tag:5s} n={len(sub)}: 层胜 {w}/平 {t}/负 {l}",
              flush=True)
    # 工业基准 + TE-ext 上的同一统计 (先验视角)
    print("[r46] 工业流整定段事件统计:", flush=True)
    for ds, sc in [("SWaT", "cmhmil"), ("TE", "iforest"),
                   ("MetroPT3", "iforest")]:
        s, y, dt = get_scores(sc, ds, seed=SEED)
        tr_m, _ = interleave_split(len(s), dt)
        n_ev, med_len = episode_stats(y[tr_m])
        med_h = med_len * dt / 3600
        print(f"[r46] {ds:8s}: n_tr_ev={n_ev} 中位事件长 {med_len} 点 "
              f"({med_h:.1f} h) -> 规则预测: "
              f"{'层' if med_len >= thr else '搜索'}", flush=True)
    st.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"写出 {OUT}", flush=True)


if __name__ == "__main__":
    main()
