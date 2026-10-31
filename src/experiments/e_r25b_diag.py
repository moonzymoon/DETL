"""R29-A3 诊断: 扩展 TE 上 ours 崩溃的机制 + 论文预注册三起点政策的效果.

r25 结果: ours 0.207 (det 0.25) vs optuna 0.560 —— 先诊断再照报.
诊断项: (1) 收敛参数与 θ 在箱内相对位置 (是否撞顶/贴底);
(2) 每故障类型的检出分解 (ours vs optuna);
(3) 论文自带三起点初始化策略 {2,4,20} (§4.4 预注册政策, 非调参) 的结果.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e3_baselines import _kmax, run_optuna
from experiments.e_pilot_tuning import interleave_split
from experiments.e_r25_te_ext import load_ext

from sklearn.ensemble import IsolationForest


def build_stream():
    X, Y, dt = load_ext()
    n_norm = 30001
    a, b = int(n_norm * 0.35), int(n_norm * 0.50)
    w = 16
    idx_w = np.arange(w)[None, :] + np.arange(b - a - w + 1)[:, None]
    ifo = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=0, n_jobs=-1).fit(
        X[a:b][idx_w].reshape(len(idx_w), -1))
    Xt = X[n_norm:]
    N = len(Xt) - w + 1
    idx = np.arange(w)[None, :] + np.arange(N)[:, None]
    s = -ifo.decision_function(Xt[idx].reshape(N, -1)).astype(np.float64)
    y = Y[n_norm:][w - 1:]
    return s, y, dt


def main():
    s, y, dt = build_stream()
    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    gt = events_from_binary(y_te)
    K_on, K_off = _kmax(dt)
    cfg = KPIConfig(dt=dt)
    q50, q999 = np.quantile(s_tr, 0.50), np.quantile(s_tr, 0.999)

    print(f"[diag] 分位: q50={q50:.4f} q90={np.quantile(s_tr,0.9):.4f} "
          f"q99={np.quantile(s_tr,0.99):.4f} q99.9={q999:.4f}", flush=True)
    print(f"[diag] 事件内分数中位 vs 事件外: "
          f"{np.median(s_te[y_te==1]):.4f} / {np.median(s_te[y_te==0]):.4f}",
          flush=True)

    # 三起点策略 (论文 §4.4 预注册: {2,4,20})
    for init in (4, 2, 20):
        prm, hist = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0,
                                 F_max=0.02, epochs=300, seed=SEED,
                                 K_on=K_on, K_off=K_off, init_t_on=float(init))
        alarm = hard_pipeline(s_te, prm["theta_hi"], prm["theta_lo"],
                              prm["T_on"], prm["T_off"])
        p = kpi_panel(alarm, y_te, cfg=cfg)
        rel = (prm["theta_hi"] - q50) / max(q999 - q50, 1e-9)
        print(f"[diag] init={init:2d}: θ={prm['theta_hi']:.4f} "
              f"(箱内相对位置 {rel:.2f}) T={prm['T_on']}/{prm['T_off']} "
              f"F1={p['EventF1']:.3f} det={p['event_detection_rate']:.2f} "
              f"AAR={p['AAR_per_hour']:.3f} n_ev={p['n_alarm_events']} "
              f"miss_final={hist[-1]['miss']:.3f}", flush=True)

    # 每故障检出分解 (ours 默认 init=4 vs optuna)
    prm, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0, F_max=0.02,
                          epochs=300, seed=SEED, K_on=K_on, K_off=K_off,
                          init_t_on=4.0)
    al_ours = hard_pipeline(s_te, prm["theta_hi"], prm["theta_lo"],
                            prm["T_on"], prm["T_off"])
    po = run_optuna(s_tr, y_tr, ev_tr, dt, SEED, True)
    th = np.quantile(s_tr, po["q"])
    al_opt = hard_pipeline(s_te, th, th - po["margin"],
                           int(po["t_on"]), int(po["t_off"]))
    # 故障运行边界: 每运行 3001 样本 (对齐 X), 分数流去掉滑窗头
    FAULTS = ["1_1", "1_2", "4_1", "4_2", "11_1", "11_2",
              "2_1", "2_2", "5_1", "5_2", "7_1", "7_2",
              "10_1", "10_2", "13_1", "13_2"]
    print("[diag] 每运行检出 (alarm 覆盖该运行标注段与否):")
    for i, f in enumerate(FAULTS):
        b0 = i * 3001 + 480 - (16 - 1)      # 标注起点在分数流中的近似位置
        e0 = (i + 1) * 3001 - (16 - 1)
        seg = slice(max(b0, 0), e0)
        lab = y_te[seg].max() if e0 > 0 else 0
        u = al_ours[seg].max() if e0 > 0 else 0
        o = al_opt[seg].max() if e0 > 0 else 0
        if lab:
            print(f"  IDV({f:5s}): ours={'Y' if u else '-'} "
                  f"optuna={'Y' if o else '-'}  "
                  f"(事件内分数max={s_te[seg][y_te[seg]==1].max() if (y_te[seg]==1).any() else float('nan'):.3f})",
                  flush=True)


if __name__ == "__main__":
    main()
