"""两阶段退火阈值消融 (R5 审稿 B2-3): 约束启用温度阈值敏感性.

变体: always (全程启用约束) / 0.5 / 0.3 (默认) / 0.15 / never (纯目标)
配置: 3 组主配置 x seed 7 x R_max=10, F_max=0.02
输出: results/audit/phase_ablation.csv
运行: cd src && python experiments/e_phase_ablation.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e_pilot_tuning import interleave_split
from experiments.e3_baselines import _kmax

# 通过禁用两阶段逻辑模拟不同阈值: 复制 train_tuning 但参数化 constraints_on 阈值
import torch
import numpy as np
from tuning.losses import soft_kpi
from tuning.constrained import TuningParams, _soft_point_f1


def train_tune_phased(s, y, events, dt, R_max=10.0, F_max=0.02, epochs=300,
                      lr=0.05, eta_c=0.05, lambda_lag=0.5, seed=7,
                      K_on=24, K_off=48, phase_th=0.3):
    """phase_th: None=全程启用; float=温度阈值; 'never'=从不启用."""
    torch.manual_seed(seed)
    s_t = torch.as_tensor(s, dtype=torch.float32)
    y_t = torch.as_tensor(y)
    model = TuningParams(s_t, K_on=K_on, K_off=K_off)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    mu = np.array([1.0, 1.0])
    for ep in range(epochs):
        temp_frac = max(0.05, 1.0 * (0.985 ** ep))
        temp_v = max(0.02, 0.15 * (0.985 ** ep))
        opt.zero_grad()
        a = model(s_t, temp_frac, temp_v)
        k = soft_kpi(a, y_t, events, dt)
        if phase_th == "never":
            on = False
        elif phase_th is None:
            on = True
        else:
            on = (temp_frac <= phase_th) or (not events)
        with torch.no_grad():
            viol = [max(0.0, float(k["aar"]) - R_max) / R_max,
                    max(0.0, float(k["far"]) - F_max) / F_max]
        pen = (mu[0] * torch.relu(k["aar"] - R_max) / R_max
               + mu[1] * torch.relu(k["far"] - F_max) / F_max)
        loss = k["miss"] + lambda_lag * k["lag"] + (pen if on else 0.0)
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if on:
            mu = np.minimum(mu * np.exp(eta_c * np.array(viol)), 100.0)
    return model.hard_params()


def main():
    out = os.path.join(RESULTS_DIR, "audit", "phase_ablation.csv")
    rows = []
    variants = [("always", None), ("0.5", 0.5), ("0.3 (default)", 0.3),
                ("0.15", 0.15), ("never", "never")]
    for sc, ds in (("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")):
        s, y, dt = get_scores(sc, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        ev_tr = events_from_binary(y[tr_m])
        K_on, K_off = _kmax(dt)
        for name, th in variants:
            p = train_tune_phased(s[tr_m], y[tr_m], ev_tr, dt, seed=SEED,
                                  K_on=K_on, K_off=K_off, phase_th=th)
            alarm = hard_pipeline(s[te_m], p["theta_hi"], p["theta_lo"],
                                  p["T_on"], p["T_off"])
            r = kpi_panel(alarm, y[te_m], cfg=KPIConfig(dt=dt))
            r.update({"dataset": ds, "scorer": sc, "phase": name})
            rows.append(r)
            print(f"{ds:9s} phase={name:12s} F1={r['EventF1']:.3f} "
                  f"AAR={r['AAR_per_hour']:.3f} det={r['event_detection_rate']:.2f}",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("written", out)


if __name__ == "__main__":
    main()
