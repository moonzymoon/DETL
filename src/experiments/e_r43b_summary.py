"""r43b: TSB-AD 批次汇总 —— 胜/平/负 + 二项符号检验 + 域分解."""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR

OUT = os.path.join(RESULTS_DIR, "audit", "r43_tsbad.csv")


def wtl(ours, rival, tol=1e-9):
    w = int((ours - rival > tol).sum())
    l = int((rival - ours > tol).sum())
    t = len(ours) - w - l
    # 双侧二项检验 (非平局折)
    n = w + l
    p = stats.binomtest(w, n, 0.5).pvalue if n else float("nan")
    return w, t, l, n, p


def main():
    df = pd.read_csv(OUT)
    df = df[~df.get("error", pd.Series([None] * len(df))).notna()
             ] if "error" in df else df
    df["domain"] = df["file"].str.split("_").str[1]
    print(f"[r43b] 纳入流数: {len(df)}  (排除见 r43_log)")
    print(f"[r43b] 域分布: {df['domain'].value_counts().to_dict()}")
    for r in ("optuna", "random", "grid", "fixed"):
        w, t, l, n, p = wtl(df["ours"].values, df[r].values)
        med = np.median(df["ours"] - df[r])
        print(f"[r43b] ours vs {r:8s}: 胜 {w} / 平 {t} / 负 {l} "
              f"(非平局 n={n}, 双侧二项 p={p:.4g}), 中位差 {med:+.3f}")
    # 按整定段事件数分层 (>=2 vs 1) —— n_tr_ev 事后重算 (同 loader/切分)
    from kpi.panel import events_from_binary
    from experiments.e_r43_tsbad import TSB_DIR, interleave_blocks, load_stream
    tr_evs = {}
    for f in df["file"]:
        try:
            s, y = load_stream(os.path.join(TSB_DIR, f))
            tr_m, _ = interleave_blocks(len(s))
            tr_evs[f] = len(events_from_binary(y[tr_m]))
        except Exception:
            tr_evs[f] = -1
    df["n_tr_ev"] = df["file"].map(tr_evs)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    for cond, tag in [(df.n_tr_ev >= 2, "train_ev>=2"),
                      (df.n_tr_ev == 1, "train_ev==1")]:
        sub = df[cond]
        if len(sub) < 5:
            continue
        w, t, l, n, p = wtl(sub["ours"].values, sub["optuna"].values)
        print(f"[r43b] {tag} (n={len(sub)}): vs optuna "
              f"胜 {w}/平 {t}/负 {l}, p={p:.4g}")
    # 域级 vs optuna
    for dom, sub in df.groupby("domain"):
        if len(sub) < 5:
            continue
        w, t, l, n, p = wtl(sub["ours"].values, sub["optuna"].values)
        print(f"[r43b] 域 {dom:12s} (n={len(sub)}): vs optuna "
              f"{w}/{t}/{l}, p={p:.3g}")
    df.to_csv(os.path.join(RESULTS_DIR, "audit", "r43_tsbad_final.csv"),
              index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
