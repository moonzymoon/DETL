"""R7 反馈实验: T_on 扫描一致性探针 (软目标 vs 硬指标在 T_on 空间的排序一致性).

背景 (R7-任务二[1]): tau_w 固定导致低温下软 on-delay 趋近 cap 语义, DeepSeek
要求给出"代理能引导正确 T_on 梯度方向"的证据. 本探针: 冻结其它参数于收敛值,
扫 T_on = 1..K_on, 在同一调优段上比较 (a) 末温软目标 miss+0.5*lag 与
(b) 硬 event-F1, 报告 Spearman 秩相关 (含调优测试段作参考).

输出: results/audit/r7_ton_sweep.csv + 控制台汇总
运行: cd src && python experiments/e_r7_ton_sweep.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import RESULTS_DIR, SEED
from scorers.adapter import get_scores
from kpi.panel import (KPIConfig, events_from_binary, kpi_panel)
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from tuning.soft_ops import soft_pipeline
from tuning.losses import soft_kpi
from experiments.e3_baselines import _kmax
from experiments.e_pilot_tuning import interleave_split

COMBOS = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]
LAMBDA_LAG = 0.5


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r7_ton_sweep.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    for scorer, ds in COMBOS:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)
        K_on, K_off = _kmax(dt)
        params, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0, F_max=0.02,
                                 epochs=300, seed=SEED, K_on=K_on, K_off=K_off)
        std_tr = float(np.std(s_tr)) + 1e-6
        s_t = torch.as_tensor(s_tr, dtype=torch.float32)
        y_t = torch.as_tensor(y_tr)
        for T in range(1, K_on + 1):
            a = soft_pipeline(s_t, params["theta_hi"], params["theta_lo"],
                              float(T), float(params["T_off"]),
                              0.05 * std_tr, 0.02, K_on=K_on, K_off=K_off)
            k = soft_kpi(a, y_t, ev_tr, dt)
            obj = float(k["miss"]) + LAMBDA_LAG * float(k["lag"])
            alarm_te = hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                                     T, params["T_off"])
            p_te = kpi_panel(alarm_te, y_te, cfg=KPIConfig(dt=dt))
            alarm_tr = hard_pipeline(s_tr, params["theta_hi"], params["theta_lo"],
                                     T, params["T_off"])
            p_tr = kpi_panel(alarm_tr, y_tr, cfg=KPIConfig(dt=dt))
            rows.append({"dataset": ds, "T_on": T, "soft_obj": obj,
                         "soft_miss": float(k["miss"]), "soft_lag": float(k["lag"]),
                         "soft_aar": float(k["aar"]), "soft_far": float(k["far"]),
                         "hard_F1_train": p_tr["EventF1"],
                         "hard_AAR_train": p_tr["AAR_per_hour"],
                         "hard_F1_test": p_te["EventF1"],
                         "hard_AAR_test": p_te["AAR_per_hour"],
                         "hard_MAD_test": p_te["mean_alarm_delay_s"]})
        df = pd.DataFrame([r for r in rows if r["dataset"] == ds])
        rho_tr = spearmanr(-df["soft_obj"], df["hard_F1_train"]).statistic
        rho_te = spearmanr(-df["soft_obj"], df["hard_F1_test"]).statistic
        best_soft = int(df.loc[df["soft_obj"].idxmin(), "T_on"])
        best_hard_tr = int(df.loc[df["hard_F1_train"].idxmax(), "T_on"])
        best_hard_te = int(df.loc[df["hard_F1_test"].idxmax(), "T_on"])
        tuned = params["T_on"]
        print(f"[r7-sweep] {ds}: tuned T_on={tuned} | argmin soft={best_soft} "
              f"argmax hard(train)={best_hard_tr} argmax hard(test)={best_hard_te} "
              f"| Spearman rho(train)={rho_tr:.3f} rho(test)={rho_te:.3f}",
              flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
