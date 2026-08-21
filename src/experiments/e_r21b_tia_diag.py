"""R22b: TIA 约束失败机制诊断 + 占空比阈值下界 (box) 变体.

e_r21 发现: MetroPT3 R_max=0.02 + tia_max=0.2 软罚未能拆掉常设告警
(硬 TIA=1.000, n_ev=1). 本脚本鉴别机制并测试结构修复:

  A. 轨迹诊断: Metro tia_max=0.2, 每 30 epoch 打印 theta/tia_soft/miss/
     mu_tia —— 判别 (i) 软占空比在整定流已<=界 (约束满足, 测试流泛化失败)
     还是 (ii) 软占空比>界但罚项与 miss 项平衡 (均衡未移动阈值).
  B. box 变体: tia_box ∈ {0.2,0.3} —— θ 箱下端 q50→q_{1-tia_box},
     整定段点级占空比按构造受限; 看常设告警是否消失及 F1 代价.
  C. box 不伤合法点: SWaT R_max=0.02 + tia_box=0.3 (SWaT-tight 占空比
     7.9% < 30%, 下界 q70 应不绑定, 参数不变).

输出: results/audit/r21b_tia_diag.csv + r21b_traj.csv
运行: cd src && python -u experiments/e_r21b_tia_diag.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e3_baselines import _kmax
from experiments.e_pilot_tuning import interleave_split

F_MAX = 0.02


def load(scorer, ds):
    s, y, dt = get_scores(scorer, ds, seed=SEED)
    tr_m, te_m = interleave_split(len(s), dt)
    return (s[tr_m], y[tr_m], s[te_m], y[te_m], dt)


def evaluate(s_tr, y_tr, s_te, y_te, dt, params):
    alarm_tr = hard_pipeline(s_tr, params["theta_hi"], params["theta_lo"],
                             params["T_on"], params["T_off"])
    alarm_te = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                             params["T_on"], params["T_off"])
    p = kpi_panel(alarm_te, y_te, cfg=KPIConfig(dt=dt))
    p["TIA_hard_tune"] = float(alarm_tr.mean())
    p["TIA_hard_test"] = float(alarm_te.mean())
    return p


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r21b_tia_diag.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []

    # --- A. 轨迹诊断 (Metro, 软罚 tia_max=0.2) ---
    s_tr, y_tr, s_te, y_te, dt = load("iforest", "MetroPT3")
    ev_tr = events_from_binary(y_tr)
    K_on, K_off = _kmax(dt)
    params, hist = train_tuning(
        s_tr, y_tr, ev_tr, dt, R_max=0.02, F_max=F_MAX, epochs=300,
        seed=SEED, K_on=K_on, K_off=K_off, tia_max=0.2)
    p = evaluate(s_tr, y_tr, s_te, y_te, dt, params)
    p.update({"variant": "penalty", "dataset": "MetroPT3",
              "tia_max": 0.2, "tia_box": np.nan, **params})
    rows.append(p)
    print(f"[r21b-A penalty] F1={p['EventF1']:.3f} det={p['event_detection_rate']:.2f} "
          f"n_ev={p['n_alarm_events']} TIA_tune={p['TIA_hard_tune']:.3f} "
          f"TIA_test={p['TIA_hard_test']:.3f} theta={params['theta_hi']:.4f}",
          flush=True)
    traj = pd.DataFrame([
        {"epoch": r["epoch"], "theta_hi": r["theta_hi"], "T_on": r["T_on"],
         "tia_soft": r["tia_soft"], "miss": r["miss"], "mu_tia": r["mu_tia"],
         "aar_soft": r["aar"]} for r in hist])
    traj.to_csv(os.path.join(RESULTS_DIR, "audit", "r21b_traj.csv"),
                index=False, encoding="utf-8-sig")
    for _, r in traj.iloc[::30].iterrows():
        print(f"  ep{int(r.epoch):3d} theta={r.theta_hi:.4f} "
              f"tia_soft={r.tia_soft:.3f} miss={r.miss:.3f} "
              f"mu_tia={r.mu_tia:.1f}", flush=True)

    # --- B. box 变体 (Metro) ---
    for box in (0.2, 0.3):
        params, _ = train_tuning(
            s_tr, y_tr, ev_tr, dt, R_max=0.02, F_max=F_MAX, epochs=300,
            seed=SEED, K_on=K_on, K_off=K_off, tia_box=box)
        p = evaluate(s_tr, y_tr, s_te, y_te, dt, params)
        p.update({"variant": "box", "dataset": "MetroPT3",
                  "tia_max": np.nan, "tia_box": box, **params})
        rows.append(p)
        print(f"[r21b-B box={box}] F1={p['EventF1']:.3f} "
              f"det={p['event_detection_rate']:.2f} n_ev={p['n_alarm_events']} "
              f"TIA_tune={p['TIA_hard_tune']:.3f} TIA_test={p['TIA_hard_test']:.3f} "
              f"theta={params['theta_hi']:.4f} T_on={params['T_on']} "
              f"T_off={params['T_off']}", flush=True)

    # --- C. box 不伤合法点 (SWaT tight) ---
    s_tr, y_tr, s_te, y_te, dt = load("cmhmil", "SWaT")
    ev_tr = events_from_binary(y_tr)
    K_on, K_off = _kmax(dt)
    params, _ = train_tuning(
        s_tr, y_tr, ev_tr, dt, R_max=0.02, F_max=F_MAX, epochs=300,
        seed=SEED, K_on=K_on, K_off=K_off, tia_box=0.3)
    p = evaluate(s_tr, y_tr, s_te, y_te, dt, params)
    p.update({"variant": "box", "dataset": "SWaT",
              "tia_max": np.nan, "tia_box": 0.3, **params})
    rows.append(p)
    print(f"[r21b-C SWaT box=0.3] F1={p['EventF1']:.3f} "
          f"n_ev={p['n_alarm_events']} TIA_test={p['TIA_hard_test']:.3f} "
          f"theta={params['theta_hi']:.4f} (无箱参照 F1=0.400, n=4, TIA=0.079)",
          flush=True)

    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
