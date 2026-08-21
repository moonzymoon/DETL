"""R18 (DeepSeek R10 [4][6]): λ 敏感性 + 初始化×miss 轨迹.

[4] 检测滞后代理非单调 -> lambda_lag in {0.1, 0.5(默认), 1.0, 2.0} 三数据集,
    看 F1/AAR/MAD 与末段 lag 是否稳定 (lambda=0.5 已有主表结果).
[6] miss 负值是否只在已检出事件出现 -> init_T_on in {2,12,20} 重训,
    记录轨迹中 miss 最小值 / 首次转负 epoch / 末段检出状态.

输出: results/audit/r18_lambda_init.csv
运行: cd src && python experiments/e_r18_lambda_init.py
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

COMBOS = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]


def run(scorer, ds, dt_parts, lam=None, init=None):
    s, y, dt = get_scores(scorer, ds, seed=SEED)
    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    K_on, K_off = _kmax(dt)
    kw = {}
    if lam is not None:
        kw["lambda_lag"] = lam
    if init is not None:
        kw["init_t_on"] = float(init)
    params, hist = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0, F_max=0.02,
                                epochs=300, seed=SEED, K_on=K_on, K_off=K_off,
                                **kw)
    alarm = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                          params["T_on"], params["T_off"])
    p = kpi_panel(alarm, y_te, cfg=KPIConfig(dt=dt))
    miss_traj = [h["miss"] for h in hist]
    lag_traj = [h["lag"] for h in hist]
    first_neg = next((h["epoch"] for h in hist if h["miss"] < 0), None)
    return {"dataset": ds,
            "lambda" if lam is not None else "init_T_on":
                lam if lam is not None else init,
            "F1": p["EventF1"], "AAR": p["AAR_per_hour"],
            "MAD": p["mean_alarm_delay_s"], "det": p["event_detection_rate"],
            "n_events": p["n_alarm_events"],
            "min_miss": float(min(miss_traj)), "first_neg_epoch": first_neg,
            "final_lag": lag_traj[-1], "min_lag": float(min(lag_traj)),
            "final_miss": miss_traj[-1],
            "T_on": params["T_on"], "T_off": params["T_off"]}


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r18_lambda_init.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for scorer, ds in COMBOS:
        for lam in [0.1, 1.0, 2.0]:
            r = run(scorer, ds, None, lam=lam)
            rows.append(r)
            print(f"[lam] {ds:9s} λ={lam}: F1={r['F1']:.3f} det={r['det']:.2f} "
                  f"AAR={r['AAR']:.3f} min_miss={r['min_miss']:.3f} "
                  f"final_lag={r['final_lag']:.3f}", flush=True)
        for it in [2, 12, 20]:
            r = run(scorer, ds, None, init=it)
            rows.append(r)
            print(f"[init] {ds:9s} init={it}: F1={r['F1']:.3f} "
                  f"det={r['det']:.2f} min_miss={r['min_miss']:.3f} "
                  f"first_neg_ep={r['first_neg_epoch']}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
