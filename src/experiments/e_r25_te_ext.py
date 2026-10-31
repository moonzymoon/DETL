"""R29-A3: 扩展 TE 套件 —— 8 种故障 × 2 批 = 16 个事件, 逃离小样本区.

动机: 三基准整定测试流仅 1/6/3 个事件, 事件级结论的统计粒度粗.
TE 仿真源 (mv-per/tennessee-eastman-dataset) 含全部 20 种故障多批次;
本脚本新增 5 种故障 (IDV 2/5/7/10/13, 按 Downs-Vogel 类型多样性预注册:
阶跃×2, 可用性损失, 随机变化, 缓慢漂移 —— 不依据结果挑选), 连同原有
IDV 1/4/11 构成 16 个故障运行.

与主管线的一致性与偏差 (如实记录):
  - z-score 锚定流前 35% (纯正常块内, 同原口径);
  - iForest (滑窗16, 拟合正常块 35%-50% 切片) —— 扩展后原比例的
    35-50% 将落入故障区, 故拟合段锚定正常块内部 (偏差已声明);
  - 打分范围: 正常块结束起的全部 16 个故障运行 (原口径 50% 测试段
    恰含全部故障; 扩展后按比例会丢前 3 个故障, 故锚定正常块尾部);
  - 交错协议/硬面板/预算同主实验.

方法: ours(diff-tune), pointwise+constraints, Optuna, grid-delay,
fixed-window + 留一事件外 (ours vs 全部对手).

输出: results/audit/r25_te_ext.csv (+ 控制台逐折)
运行: cd src && python -u experiments/e_r25_te_ext.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_ROOT, RESULTS_DIR, SEED
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline, merge_gaps
from tuning.losses import soft_kpi
from experiments.e3_baselines import _kmax, run_grid_delay, run_optuna
from experiments.e_pilot_tuning import interleave_split
from experiments.e_r22_jackknife import match_breakdown
from scorers.adapter import local_iforest

TE_DIR = os.path.join(DATA_ROOT, "TE")
FAULTS = ["mode1_1_1", "mode1_1_2", "mode1_4_1", "mode1_4_2",
          "mode1_11_1", "mode1_11_2",
          "mode1_2_1", "mode1_2_2", "mode1_5_1", "mode1_5_2",
          "mode1_7_1", "mode1_7_2", "mode1_10_1", "mode1_10_2",
          "mode1_13_1", "mode1_13_2"]
ONSET_H = 8.0
N_XMEAS = 41
XMEAS = [f"XMEAS_{i}" for i in range(1, N_XMEAS + 1)]


def load_ext():
    npz = os.path.join(RESULTS_DIR, "audit", "r25_te_ext_data.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return d["X"], d["Y"], float(d["dt"])
    df = pd.read_excel(os.path.join(TE_DIR, "mode1_normal_500.xlsx"))
    feat = [c for c in df.columns if c != "Time"]
    df = df.rename(columns={c: XMEAS[i] for i, c in enumerate(feat)})
    Xn = df[XMEAS].values.astype(np.float32)
    Xs, Ys = [Xn], [np.zeros(len(Xn), dtype=np.int8)]
    for f in FAULTS:
        d2 = pd.read_excel(os.path.join(TE_DIR, f"{f}.xlsx"))
        t = d2["Time (h)"].values
        Xs.append(d2[[c for c in d2.columns if c.startswith("XMEAS")]]
                  .values.astype(np.float32))
        Ys.append((t >= ONSET_H).astype(np.int8))
    X = np.vstack(Xs); Y = np.concatenate(Ys)
    n_tr = int(len(X) * 0.35)
    mu = X[:n_tr].mean(0, keepdims=True)
    sd = X[:n_tr].std(0, keepdims=True) + 1e-8
    X = ((X - mu) / sd).astype(np.float32)
    np.savez_compressed(npz, X=X, Y=Y, dt=60.0)
    return X, Y, 60.0


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r25_te_ext.csv")
    X, Y, dt = load_ext()
    print(f"[r25] 扩展 TE: X{X.shape} 事件数(全流)="
          f"{len(events_from_binary(Y))}", flush=True)
    # iForest: 拟合段锚定正常块内部 (原 35-50% 比例), 打分正常块尾部起
    n_norm = 30001
    a, b = int(n_norm * 0.35), int(n_norm * 0.50)
    from sklearn.ensemble import IsolationForest
    w = 16
    idx_w = (np.arange(w)[None, :]
             + np.arange(b - a - w + 1)[:, None])
    featW = X[a:b][idx_w].reshape(len(idx_w), -1)
    ifo = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=0, n_jobs=-1).fit(featW)
    Xt = X[n_norm:]
    N = len(Xt) - w + 1
    idx = np.arange(w)[None, :] + np.arange(N)[:, None]
    s = -ifo.decision_function(Xt[idx].reshape(N, -1)).astype(np.float64)
    y = Y[n_norm:][w - 1:]
    print(f"[r25] 分数流 {s.shape}, 整定测试事件数="
          f"{len(events_from_binary(y[interleave_split(len(s), dt)[1]]))}",
          flush=True)

    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    gt = events_from_binary(y_te)
    K_on, K_off = _kmax(dt)
    F_MAX, R_MAX = 0.02, 10.0
    cfg = KPIConfig(dt=dt)
    rows = []
    streams = {}

    def panel(alarm, method):
        p = kpi_panel(alarm, y_te, cfg=cfg)
        p["method"] = method
        rows.append(p)
        streams[method] = events_from_binary(alarm)
        print(f"[r25] {method:16s} F1={p['EventF1']:.3f} "
              f"det={p['event_detection_rate']:.2f} "
              f"AAR={p['AAR_per_hour']:.3f} "
              f"n_ev={p['n_alarm_events']} MAD={p['mean_alarm_delay_s']:.0f}s",
              flush=True)

    # ours / pointwise+constraints
    for mode, tag in [("constrained", "ours"),
                      ("relaxed_f1_constrained", "pw+constr")]:
        prm, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=R_MAX,
                              F_max=F_MAX, epochs=300, seed=SEED,
                              K_on=K_on, K_off=K_off, loss_mode=mode)
        panel(hard_pipeline(s_te, prm["theta_hi"], prm["theta_lo"],
                            prm["T_on"], prm["T_off"]), tag)
    # 黑盒/经典
    for tag, fn in [("optuna", lambda: run_optuna(s_tr, y_tr, ev_tr, dt,
                                                  SEED, True)),
                    ("grid-delay", lambda: run_grid_delay(s_tr, y_tr,
                                                          ev_tr, dt))]:
        prm = fn()
        th = np.quantile(s_tr, prm["q"])
        panel(hard_pipeline(s_te, th, th - prm["margin"],
                            int(prm["t_on"]), int(prm["t_off"])), tag)
    # fixed-window (q × 合并窗网格, 整定段选型, 预算内)
    best = (None, -1)
    for q in (0.90, 0.95, 0.99):
        for wt in (300, 600, 1800):
            W = max(1, int(wt / dt))
            al = merge_gaps((s_tr > np.quantile(s_tr, q)).astype(int), W)
            p = kpi_panel(al, y_tr, cfg=cfg)
            if p["AAR_per_hour"] <= R_MAX and p["EventF1"] > best[1]:
                best = ((q, W), p["EventF1"])
    q, W = best[0]
    panel(merge_gaps((s_te > np.quantile(s_tr, q)).astype(int), W),
          "fixed-window")

    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}", flush=True)

    # 留一事件外: ours vs 每个对手
    for rival in [r for r in streams if r != "ours"]:
        f_o, f_r = [], []
        for i in range(len(gt)):
            gt_i = gt[:i] + gt[i + 1:]
            f_o.append(match_breakdown(gt_i, streams["ours"])[3])
            f_r.append(match_breakdown(gt_i, streams[rival])[3])
        gaps = [a2 - b2 for a2, b2 in zip(f_o, f_r)]
        wins = sum(1 for g in gaps if g > 1e-9)
        print(f"[r25-LOO] ours-{rival:16s} n={len(gt)} 折: "
              f"[{min(gaps):+.3f}, {max(gaps):+.3f}]  领先 {wins}/{len(gaps)}",
              flush=True)


if __name__ == "__main__":
    main()
