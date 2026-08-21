"""E3: 基线对比 — 黑盒 (Optuna TPE / 随机搜索, 同评估预算) + 经典 (on/off-delay 网格)
+ Gyasi & Wang 2022 简化复现 (解析式串行 deadband+delay-timer 设计) vs 本文可微整定层.

方法与选参信息量 (全部只在 tuning-train 段选参, 12h 交错块划分同 e_pilot):
  optuna     : TPE 贝叶斯优化, BUDGET 次硬评估, 目标 = Event-F1 - 10·relu(AAR-R_max)/R_max
  random     : 均匀随机搜索, 同预算
  grid-delay : 经典固定 on/off-delay 网格 (q x T_on x T_off), 按训练段目标选优
  gyasi      : Gyasi & Wang (J. Process Control 2022) 简化复现 —— 在正常/异常分数的
               经验分布上按 iid 假设解析估计 (误报激活率, 检出概率), 选满足
               AAR<=R_max 且检出>=0.9 的最小延迟参数. 完整贝叶斯设计按方案预案
               降级, iid 假设对自相关分数流的偏差在实验记录中注明.
  diff-tune  : 本文可微整定层 (复跑 train_tuning 保证同表可比)

输出: results/e3/e3_baselines.csv
运行: cd src && python experiments/e3_baselines.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import KPIConfig, event_f1_one2one, events_from_binary, kpi_panel, rising_edges
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e_pilot_tuning import interleave_split, BLOCK_H

COMBOS = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]
R_MAX = 10.0
BUDGET = 60                     # 黑盒评估预算 (硬评估次数)


def _kmax(dt):
    return min(24, max(4, int(120 / dt))), min(48, max(8, int(300 / dt)))


def eval_params(s_tr, y_tr, ev_tr, dt, q, margin, t_on, t_off):
    """训练段上硬评估一组参数. 返回 (objective, kpi)."""
    theta_hi = np.quantile(s_tr, q)
    theta_lo = theta_hi - margin
    alarm = hard_pipeline(s_tr, theta_hi, theta_lo, int(t_on), int(t_off))
    p = kpi_panel(alarm, y_tr, cfg=KPIConfig(dt=dt))
    obj = p["EventF1"] - 10.0 * max(0.0, p["AAR_per_hour"] - R_MAX) / R_MAX
    return obj, p


def run_optuna(s_tr, y_tr, ev_tr, dt, seed, use_tpe):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    K_on, K_off = _kmax(dt)
    sd = float(np.std(s_tr))
    sampler = (optuna.samplers.TPESampler(seed=seed) if use_tpe
               else optuna.samplers.RandomSampler(seed=seed))

    def objective(trial):
        q = trial.suggest_float("q", 0.50, 0.999)
        margin = trial.suggest_float("margin", 0.0, 0.25) * sd
        t_on = trial.suggest_int("t_on", 1, K_on)
        t_off = trial.suggest_int("t_off", 1, K_off)
        obj, _ = eval_params(s_tr, y_tr, ev_tr, dt, q, margin, t_on, t_off)
        return obj

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=BUDGET, show_progress_bar=False)
    b = study.best_params
    return {"q": b["q"], "margin": b["margin"] * sd,
            "t_on": b["t_on"], "t_off": b["t_off"]}


def run_grid_delay(s_tr, y_tr, ev_tr, dt):
    """经典 on/off-delay 网格 (固定阈值分位)."""
    sd = float(np.std(s_tr))
    best = None
    for q in [0.95, 0.99]:
        for t_on in [1, 5, 10]:
            for t_off in [1, 10, 30]:
                obj, _ = eval_params(s_tr, y_tr, ev_tr, dt, q, 0.0, t_on, t_off)
                if best is None or obj > best[0]:
                    best = (obj, {"q": q, "margin": 0.0,
                                  "t_on": t_on, "t_off": t_off})
    return best[1]


def run_gyasi_wang(s_tr, y_tr, ev_tr, dt):
    """Gyasi & Wang 2022 简化复现: 解析式串行 deadband + delay-timer 设计.

    iid 假设下的解析估计:
      误报激活率/h  r_false ≈ P_n(θ)^{T_on}·(1-P_n(θ))·(3600/dt)
      事件检出概率  P_det ≈ 1-(1-P_a(θ)^{T_on})^{L̄/2}   (L̄=平均事件长)
    选参: 满足 r_false<=R_max 且 P_det>=0.9 中最小 T_on·dt (延迟最小),
          deadband margin 取能保持 r_false 达标的最小值; 无可行解则逐级放宽 P_det.
    """
    sn = s_tr[y_tr == 0]
    sa = s_tr[y_tr == 1]
    sd = float(np.std(s_tr)) + 1e-9
    L_bar = float(np.mean([e - s0 for s0, e in ev_tr])) if ev_tr else 10.0
    K_on, K_off = _kmax(dt)
    margins = [0.0, 0.05, 0.1, 0.25]            # deadband 宽度 (std 单位)
    for target_det in [0.9, 0.7, 0.5]:
        best = None
        for q in np.arange(0.90, 1.0, 0.005):
            theta = np.quantile(sn, q)
            Pn = float((sn > theta).mean())
            Pa = float((sa > theta).mean()) if len(sa) else 0.5
            for t_on in range(1, K_on + 1):
                r_false = (Pn ** t_on) * (1 - Pn) * (3600.0 / dt)
                p_det = 1 - (1 - Pa ** t_on) ** max(1, int(L_bar / 2))
                if r_false > R_MAX or p_det < target_det:
                    continue
                for m in margins:                # 最小可行 deadband
                    if best is None:
                        best = (t_on, m, q)
                    break
                break
            if best:
                break
        if best:
            t_on, m, q = best
            return {"q": float(q), "margin": m * sd, "t_on": t_on,
                    "t_off": min(K_off, max(1, t_on))}   # T_off 对称取小
    return {"q": 0.99, "margin": 0.0, "t_on": 1, "t_off": 1}


def apply_and_eval(method, prm, s_te, y_te, dt, s_tr):
    theta_hi = np.quantile(s_tr, prm["q"])
    theta_lo = theta_hi - prm["margin"]
    alarm = hard_pipeline(s_te, theta_hi, theta_lo, int(prm["t_on"]), int(prm["t_off"]))
    p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
    p["method"] = method
    return p, {"theta_hi": theta_hi, "theta_lo": theta_lo,
               "T_on": int(prm["t_on"]), "T_off": int(prm["t_off"])}


def main():
    out_dir = os.path.join(RESULTS_DIR, "e3")
    os.makedirs(out_dir, exist_ok=True)
    rows, prm_rows = [], []
    for scorer, ds in COMBOS:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)
        print(f"\n[E3] {scorer} x {ds}: train_events={len(ev_tr)}")

        methods = {}
        for name, fn in [("optuna", lambda: run_optuna(s_tr, y_tr, ev_tr, dt, SEED, True)),
                         ("random", lambda: run_optuna(s_tr, y_tr, ev_tr, dt, SEED + 1, False)),
                         ("grid-delay", lambda: run_grid_delay(s_tr, y_tr, ev_tr, dt)),
                         ("gyasi", lambda: run_gyasi_wang(s_tr, y_tr, ev_tr, dt))]:
            t0 = time.time()
            methods[name] = fn()
            print(f"  {name:10s} 选参 {methods[name]} ({time.time()-t0:.0f}s)")

        # diff-tune (复跑保证同表)
        t0 = time.time()
        K_on, K_off = _kmax(dt)
        params, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=R_MAX, F_max=0.02,
                                 epochs=300, seed=SEED, K_on=K_on, K_off=K_off)
        print(f"  diff-tune  参数 {params} ({time.time()-t0:.0f}s)")

        for name, prm in methods.items():
            p, hp = apply_and_eval(name, prm, s_te, y_te, dt, s_tr)
            p.update({"dataset": ds, "scorer": scorer})
            rows.append(p)
            prm_rows.append({"dataset": ds, "scorer": scorer, "method": name, **hp})
            print(f"  {name:10s} EventF1={p['EventF1']:.3f} AAR={p['AAR_per_hour']:.1f} "
                  f"MAD={p['mean_alarm_delay_s']:.0f}s det={p['event_detection_rate']:.2f}")

        theta_hi, theta_lo = params["theta_hi"], params["theta_lo"]
        alarm = hard_pipeline(s_te, theta_hi, theta_lo, params["T_on"], params["T_off"])
        p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
        p.update({"dataset": ds, "scorer": scorer, "method": "diff-tune"})
        rows.append(p)
        prm_rows.append({"dataset": ds, "scorer": scorer, "method": "diff-tune", **params})
        print(f"  diff-tune  EventF1={p['EventF1']:.3f} AAR={p['AAR_per_hour']:.1f} "
              f"MAD={p['mean_alarm_delay_s']:.0f}s det={p['event_detection_rate']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "e3_baselines.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(prm_rows).to_csv(os.path.join(out_dir, "e3_selected_params.csv"),
                                  index=False, encoding="utf-8-sig")
    print(f"\n[E3] 写出 {out_dir}/e3_baselines.csv + e3_selected_params.csv")
    cols = ["dataset", "scorer", "method", "EventF1", "AAR_per_hour",
            "chatter_rate", "mean_alarm_delay_s", "event_detection_rate", "n_alarm_events"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
