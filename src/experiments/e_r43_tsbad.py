"""方案1 (R43): TSB-AD-MV 200 流大规模验证 —— 跨域统计陈述.

预注册协议 (在看到任何结果前固定):
  - 流: TSB-AD-MV 全部 200 条 (NeurIPS 2024 D&B 基准多元子集),
    末列 Label 为点级 GT; dt=1s 名义 (仅用于速率记账).
  - 打分: z-score 后 iForest(n=100, seed=0), 拟合于前 35% (无标签使用),
    打分剩余 65% —— 与主管线同构.
  - 切分: 24 个交替块 (块长 n//24), 奇数块整定训练 / 偶数块整定测试
    (主协议 12h 块精神的流长自适应版).
  - 纳入规则 (预注册): 流长>=1000 且 测试段 GT 事件>=2 且 训练段 GT 事件>=1.
    排除的流按原因计数报告.
  - 方法: ours(constrained) / optuna(60,TPE) / random(60) / grid-delay /
    fixed-window(q∈{0.90,0.95,0.99}×W 同主网格), 预算 R_max=10, F_max=0.02.
  - 统计: 逐流事件-F1 (匈牙利 IoU>0), 胜/平/负 (容差 1e-9) +
    二项符号检验 (双侧).
用法: python e_r43_tsbad.py [start] [end]   # 文件索引切片, 断点续跑
输出: results/audit/r43_tsbad.csv (追加), 全部完成后汇总
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_ROOT, RESULTS_DIR, SEED
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline, merge_gaps
from experiments.e3_baselines import (_kmax, run_grid_delay, run_optuna)
from sklearn.ensemble import IsolationForest

TSB_DIR = os.path.join(DATA_ROOT, "TSB-AD-MV", "TSB-AD-M")
OUT = os.path.join(RESULTS_DIR, "audit", "r43_tsbad.csv")
DT = 1.0


def interleave_blocks(n, nblk=24):
    idx = np.arange(n)
    blocks = np.array_split(idx, nblk)
    tr_m = np.zeros(n, dtype=bool)
    for i, b in enumerate(blocks):
        if i % 2 == 0:
            tr_m[b] = True
    return tr_m, ~tr_m


def load_stream(path):
    d = pd.read_csv(path)
    X = d.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X.values.astype(np.float32)
    y = pd.to_numeric(d.iloc[:, -1], errors="coerce").fillna(0).values
    y = (y == 1).astype(np.int8)
    # 前 35% 拟合打分器 (不使用标签), 其余打分 —— 同主管线
    n_fit = int(len(X) * 0.35)
    mu = X[:n_fit].mean(0, keepdims=True)
    sd = X[:n_fit].std(0, keepdims=True) + 1e-8
    Xz = (X - mu) / sd
    ifo = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=0, n_jobs=-1).fit(Xz[:n_fit])
    s = -ifo.decision_function(Xz[n_fit:]).astype(np.float64)
    return s, y[n_fit:]


def f1_hungarian(gt, pred):
    from scipy.optimize import linear_sum_assignment
    if not gt or not pred:
        return 0.0 if gt else float("nan")
    IoU = np.zeros((len(gt), len(pred)))
    for i, (gs, ge) in enumerate(gt):
        for j, (ps, pe) in enumerate(pred):
            inter = max(0, min(ge, pe) - max(gs, ps))
            union = max(ge, pe) - min(gs, ps)
            IoU[i, j] = inter / union if union else 0.0
    r, c = linear_sum_assignment(-IoU)
    tp = sum(1 for a, b in zip(r, c) if IoU[a, b] > 0)
    p = tp / len(pred)
    q = tp / len(gt)
    return float(2 * p * q / (p + q) if p + q else 0.0)


def main():
    args = [int(a) for a in sys.argv[1:]]
    start, end = (args if len(args) == 2 else (0, 10 ** 9))
    files = sorted(glob.glob(os.path.join(TSB_DIR, "*.csv")))
    files = files[start:end]
    rows_new = []
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["file"])
    else:
        done = set()
    n_excl = {"short": 0, "test_ev": 0, "train_ev": 0}
    for k, f in enumerate(files):
        name = os.path.basename(f)
        if name in done:
            continue
        try:
            s, y = load_stream(f)
            tr_m, te_m = interleave_blocks(len(s))
            s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
            ev_tr = events_from_binary(y_tr)
            gt = events_from_binary(y_te)
            if len(s) < 1000:
                n_excl["short"] += 1
                continue
            if len(gt) < 2:
                n_excl["test_ev"] += 1
                continue
            if len(ev_tr) < 1:
                n_excl["train_ev"] += 1
                continue
            K_on, K_off = _kmax(DT)
            t0 = time.time()
            prm, _ = train_tuning(s_tr, y_tr, ev_tr, DT, R_max=10.0,
                                  F_max=0.02, epochs=300, seed=SEED,
                                  K_on=K_on, K_off=K_off)
            al = hard_pipeline(s_te, prm["theta_hi"], prm["theta_lo"],
                               prm["T_on"], prm["T_off"])
            f_ours = f1_hungarian(gt, events_from_binary(al))
            res = {"file": name, "n": len(s), "n_te_ev": len(gt),
                   "ours": f_ours}
            for tag, use_tpe in [("optuna", True), ("random", False)]:
                p = run_optuna(s_tr, y_tr, ev_tr, DT,
                               SEED if use_tpe else SEED + 1, use_tpe)
                th = np.quantile(s_tr, p["q"])
                a2 = hard_pipeline(s_te, th, th - p["margin"],
                                   int(p["t_on"]), int(p["t_off"]))
                res[tag] = f1_hungarian(gt, events_from_binary(a2))
            p = run_grid_delay(s_tr, y_tr, ev_tr, DT)
            th = np.quantile(s_tr, p["q"])
            a2 = hard_pipeline(s_te, th, th - p["margin"],
                               int(p["t_on"]), int(p["t_off"]))
            res["grid"] = f1_hungarian(gt, events_from_binary(a2))
            best = (None, -1)
            for q in (0.90, 0.95, 0.99):
                from kpi.panel import rising_edges, event_f1_one2one
                b_tr = (s_tr > np.quantile(s_tr, q)).astype(int)
                hours = len(s_tr) * DT / 3600.0
                if len(rising_edges(b_tr)) / hours > 10.0:
                    continue
                for w_pt in (1, 5, 10, 30):
                    m_tr = merge_gaps(b_tr, w_pt)
                    f1 = event_f1_one2one(ev_tr, events_from_binary(m_tr))
                    if f1 > best[1]:
                        best = ((q, w_pt), f1)
            if best[0]:
                q, w_pt = best[0]
                a2 = merge_gaps((s_te > np.quantile(s_tr, q)).astype(int),
                                w_pt)
                res["fixed"] = f1_hungarian(gt, events_from_binary(a2))
            else:
                res["fixed"] = 0.0
            rows_new.append(res)
            pd.DataFrame(rows_new).to_csv(
                OUT + ".part", index=False, encoding="utf-8-sig")
            print(f"[r43 {start}] {name}: n={len(s)} ev_te={len(gt)} "
                  f"ours={f_ours:.3f} opt={res['optuna']:.3f} "
                  f"rnd={res['random']:.3f} grid={res['grid']:.3f} "
                  f"fix={res['fixed']:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
        except Exception as exc:  # 单流失败不拖垮批次, 照报
            print(f"[r43 {start}] {name}: EXC {type(exc).__name__}: {exc}",
                  flush=True)
            rows_new.append({"file": name, "error": str(exc)})
    # 合并 part 到主 CSV
    if os.path.exists(OUT + ".part"):
        part = pd.read_csv(OUT + ".part")
        if os.path.exists(OUT):
            main_df = pd.read_csv(OUT)
            part = pd.concat([main_df[~main_df["file"].isin(set(part["file"]))],
                              part], ignore_index=True)
        part.to_csv(OUT, index=False, encoding="utf-8-sig")
        os.remove(OUT + ".part")
    print(f"[r43 {start}] 批次完成, 排除: {n_excl}, 累计 "
          f"{len(pd.read_csv(OUT)) if os.path.exists(OUT) else 0} 流",
          flush=True)


if __name__ == "__main__":
    main()
