"""R30: 代理训练 + 硬指标小预算抛光 (warm-polish) —— 框架内修复扩展套件失败.

R29 诊断: 代理对阈下事件失明 (smax 地板), 硬指标搜索不失明但从头搜在
主管线失败 (Optuna60: TE 0.286 vs ours 0.769). 混合方案: 以代理收敛解为
温启动, TPE 追加 19 次硬评估 (共 20, 预算为基线 1/3), 目标与基线完全
一致 (整定段硬 F1, AAR 超预算罚项). 对照: 冷启动 TPE-20.

流: SWaT/TE/MetroPT3 (ours 参数取 e3 缓存) + TE-ext (重训).
输出: results/audit/r28_polish.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
import experiments.e3_baselines as e3
from scorers.adapter import get_scores
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e3_baselines import _kmax, eval_params
from experiments.e_pilot_tuning import interleave_split

POLISH_BUDGET = 20


def tpe_search(s_tr, y_tr, ev_tr, dt, seed, n_trials, warm=None):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    K_on, K_off = _kmax(dt)
    sd = float(np.std(s_tr))

    def objective(trial):
        q = trial.suggest_float("q", 0.50, 0.999)
        margin = trial.suggest_float("margin", 0.0, 0.25) * sd
        t_on = trial.suggest_int("t_on", 1, K_on)
        t_off = trial.suggest_int("t_off", 1, K_off)
        obj, _ = eval_params(s_tr, y_tr, ev_tr, dt, q, margin, t_on, t_off)
        return obj

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    if warm is not None:
        study.enqueue_trial(warm)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    b = study.best_params
    return {"q": b["q"], "margin": b["margin"] * sd,
            "t_on": b["t_on"], "t_off": b["t_off"]}


def to_test(theta_hi, theta_lo, t_on, t_off, s_te, y_te, dt, method, rows):
    alarm = hard_pipeline(s_te, theta_hi, theta_lo, int(t_on), int(t_off))
    p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
    p["method"] = method
    rows.append(p)
    print(f"[r28] {method:22s} F1={p['EventF1']:.3f} "
          f"det={p['event_detection_rate']:.2f} AAR={p['AAR_per_hour']:.3f} "
          f"n_ev={p['n_alarm_events']}", flush=True)
    return p


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r28_polish.csv")
    prm_tab = pd.read_csv(os.path.join(RESULTS_DIR, "e3",
                                       "e3_selected_params.csv"))
    combos = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]
    rows = []
    global s_tr
    for scorer, ds in combos:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)
        sd = float(np.std(s_tr))
        pp = prm_tab[(prm_tab.dataset == ds)
                     & (prm_tab.method == "diff-tune")].iloc[0]
        # ours
        to_test(pp.theta_hi, pp.theta_lo, int(pp.T_on), int(pp.T_off),
                s_te, y_te, dt, f"{ds}:ours", rows)
        # warm 起点 (代理解映射入搜索空间)
        q_ours = float((s_tr < pp.theta_hi).mean())
        warm = {"q": min(max(q_ours, 0.50), 0.999),
                "margin": (pp.theta_hi - pp.theta_lo) / sd,
                "t_on": int(pp.T_on), "t_off": int(pp.T_off)}
        for tag, kw in [
                ("warm20", dict(n_trials=POLISH_BUDGET, warm=warm)),
                ("cold20", dict(n_trials=POLISH_BUDGET, warm=None))]:
            prm = tpe_search(s_tr, y_tr, ev_tr, dt, SEED, **kw)
            alarm = hard_pipeline(s_te, np.quantile(s_tr, prm["q"]),
                                  np.quantile(s_tr, prm["q"]) - prm["margin"],
                                  int(prm["t_on"]), int(prm["t_off"]))
            p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
            p["method"] = f"{ds}:{tag}"
            rows.append(p)
            print(f"[r28] {ds}:{tag:6s} F1={p['EventF1']:.3f} "
                  f"det={p['event_detection_rate']:.2f} "
                  f"AAR={p['AAR_per_hour']:.3f} "
                  f"n_ev={p['n_alarm_events']}", flush=True)

    # TE-ext: 重训 ours 后同流程
    from experiments.e_r25b_diag import build_stream
    s, y, dt = build_stream()
    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    K_on, K_off = _kmax(dt)
    prm, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0, F_max=0.02,
                          epochs=300, seed=SEED, K_on=K_on, K_off=K_off)
    to_test(prm["theta_hi"], prm["theta_lo"], prm["T_on"], prm["T_off"],
            s_te, y_te, dt, "TEext:ours", rows)
    sd = float(np.std(s_tr))
    q_ours = float((s_tr < prm["theta_hi"]).mean())
    warm = {"q": min(max(q_ours, 0.50), 0.999),
            "margin": (prm["theta_hi"] - prm["theta_lo"]) / sd,
            "t_on": prm["T_on"], "t_off": prm["T_off"]}
    for tag, kw in [("warm20", dict(n_trials=POLISH_BUDGET, warm=warm)),
                    ("cold20", dict(n_trials=POLISH_BUDGET, warm=None))]:
        pr2 = tpe_search(s_tr, y_tr, ev_tr, dt, SEED, **kw)
        alarm = hard_pipeline(s_te, np.quantile(s_tr, pr2["q"]),
                              np.quantile(s_tr, pr2["q"]) - pr2["margin"],
                              int(pr2["t_on"]), int(pr2["t_off"]))
        p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
        p["method"] = f"TEext:{tag}"
        rows.append(p)
        print(f"[r28] TEext:{tag:6s} F1={p['EventF1']:.3f} "
              f"det={p['event_detection_rate']:.2f} "
              f"AAR={p['AAR_per_hour']:.3f} n_ev={p['n_alarm_events']}",
              flush=True)

    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
