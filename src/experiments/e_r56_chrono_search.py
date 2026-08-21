"""R56 (外部审核报告 P0-泄漏质疑): 严格时序划分下的黑盒搜索基线.

审核报告质疑 TE 0.769 vs 0.500 可能受益于交错块内同事件跨块. 已有的
r1_chronological 只跑了 Ours+Fixed-window. 本实验在同一时序划分下补跑
Optuna(seed 7)/Random(seed 8)/Grid-delay, 检验比较结论在无交错泄漏协议下
是否保持. 划分逻辑与 e_rebuttal_r1 完全一致 (整流对半, 前半 tuning,
后半 test), BATADAL 用 e_r36 的 batadal_stream.

输出: results/audit/r56_chrono_search.csv
运行: cd src && python experiments/e_r56_chrono_search.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.hard_ops import hard_pipeline
from experiments.e3_baselines import run_optuna, run_grid_delay, _kmax
from experiments.e_r36_mcap_sweep import batadal_stream

R_MAX = 10.0


def eval_on_test(s_tr, y_tr, s_te, y_te, dt, params):
    theta_hi = np.quantile(s_tr, params["q"])
    theta_lo = theta_hi - params["margin"]
    alarm = hard_pipeline(s_te, theta_hi, theta_lo,
                          int(params["t_on"]), int(params["t_off"]))
    p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
    return p


def run_search_nansafe(s_tr, y_tr, ev_tr, dt, seed, use_tpe):
    """NaN 安全搜索: 训练段无事件时目标退化为约束合规 (负违规), 与
    fixed-window 的最保守回退同理."""
    import optuna
    from experiments.e3_baselines import eval_params
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
        obj, p = eval_params(s_tr, y_tr, ev_tr, dt, q, margin, t_on, t_off)
        if (not np.isfinite(obj)) or obj is None:
            obj = -10.0 * max(0.0, p["AAR_per_hour"] - R_MAX) / R_MAX
        return float(obj)

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=60, show_progress_bar=False)
    b = study.best_params
    return {"q": b["q"], "margin": b["margin"] * sd,
            "t_on": b["t_on"], "t_off": b["t_off"]}


def main():
    combos = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]
    rows = []
    for scorer, ds in combos:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        half = len(s) // 2
        s_tr, y_tr = s[:half], y[:half]
        s_te, y_te = s[half:], y[half:]
        ev_tr = events_from_binary(y_tr)
        for name, seed in [("optuna", 7), ("random", 8)]:
            prm = run_search_nansafe(s_tr, y_tr, ev_tr, dt, seed, use_tpe=(name == "optuna"))
            p = eval_on_test(s_tr, y_tr, s_te, y_te, dt, prm)
            p.update({"dataset": ds, "method": f"{name}(chrono)",
                      "n_train_events": len(ev_tr)})
            rows.append(p)
            print(f"[chrono-search] {ds:8s} {name:7s} F1={p['EventF1']:.3f} "
                  f"det={p['event_detection_rate']:.2f} AAR={p['AAR_per_hour']:.2f}",
                  flush=True)
        # grid-delay: 复用 e3 的网格选择 (在训练段上选, 返回最佳参数)
        gp = run_grid_delay(s_tr, y_tr, ev_tr, dt)
        # run_grid_delay 返回的是训练段最优参数 dict (q/margin/t_on/t_off 同构)
        p = eval_on_test(s_tr, y_tr, s_te, y_te, dt, gp)
        p.update({"dataset": ds, "method": "grid-delay(chrono)",
                  "n_train_events": len(ev_tr)})
        rows.append(p)
        print(f"[chrono-search] {ds:8s} grid    F1={p['EventF1']:.3f} "
              f"det={p['event_detection_rate']:.2f} AAR={p['AAR_per_hour']:.2f}",
              flush=True)

    # BATADAL
    s, y, dt = batadal_stream()
    half = len(s) // 2
    s_tr, y_tr, s_te, y_te = s[:half], y[:half], s[half:], y[half:]
    ev_tr = events_from_binary(y_tr)
    for name, seed in [("optuna", 7), ("random", 8)]:
        prm = run_search_nansafe(s_tr, y_tr, ev_tr, dt, seed, use_tpe=(name == "optuna"))
        p = eval_on_test(s_tr, y_tr, s_te, y_te, dt, prm)
        p.update({"dataset": "BATADAL", "method": f"{name}(chrono)",
                  "n_train_events": len(ev_tr)})
        rows.append(p)
        print(f"[chrono-search] BATADAL  {name:7s} F1={p['EventF1']:.3f} "
              f"det={p['event_detection_rate']:.2f} AAR={p['AAR_per_hour']:.2f}",
              flush=True)
    gp = run_grid_delay(s_tr, y_tr, ev_tr, dt)
    p = eval_on_test(s_tr, y_tr, s_te, y_te, dt, gp)
    p.update({"dataset": "BATADAL", "method": "grid-delay(chrono)",
              "n_train_events": len(ev_tr)})
    rows.append(p)
    print(f"[chrono-search] BATADAL  grid    F1={p['EventF1']:.3f} "
          f"det={p['event_detection_rate']:.2f} AAR={p['AAR_per_hour']:.2f}",
          flush=True)

    out = os.path.join(RESULTS_DIR, "audit", "r56_chrono_search.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
