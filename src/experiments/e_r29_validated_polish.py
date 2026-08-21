"""R30b: 验证式抛光 (validated warm-polish) —— 用整定段内部验证挡过拟合.

r28 发现: warm20 在 TE-ext 0.207→0.560 且 SWaT/TE 不动, 但 MetroPT3
被带偏 (0.067→0.031, 整定段过拟合). 修复: 整定-训练流按时序对半分为
拟合半/验证半 —— TPE 于拟合半优化 (温启动同 r28), 仅当验证半硬 F1
超过代理解的验证半 F1 时采纳抛光解. 全部部署侧可算, 无测试泄漏.

输出: results/audit/r29_validated_polish.csv
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
from experiments.e3_baselines import _kmax, eval_params
from experiments.e_pilot_tuning import interleave_split
from experiments.e_r28_polish import tpe_search

PANEL = KPIConfig  # 占位引用避免误删 import


def hard_f1(s, y, dt, th, tl, ton, toff):
    a = hard_pipeline(s, th, tl, int(ton), int(toff))
    return kpi_panel(a, y, cfg=KPIConfig(dt=dt))["EventF1"]


def run_stream(s_tr, y_tr, s_te, y_te, dt, ours_prm, ds, rows):
    ev_tr = events_from_binary(y_tr)
    sd = float(np.std(s_tr))
    n = len(s_tr)
    half = n // 2
    s_fit, y_fit = s_tr[:half], y_tr[:half]
    s_val, y_val = s_tr[half:], y_tr[half:]
    ev_fit = events_from_binary(y_fit)
    if not ev_fit:
        # 拟合半段无事件: 硬 F1 目标无定义, 跳过抛光 (保持代理解)
        p = kpi_panel(hard_pipeline(s_te, ours_prm["theta_hi"],
                                    ours_prm["theta_lo"],
                                    int(ours_prm["T_on"]),
                                    int(ours_prm["T_off"])),
                      y_te, cfg=KPIConfig(dt=dt))
        p.update({"method": "validated-polish", "dataset": ds,
                  "adopted": False, "val_F_ours": float("nan"),
                  "val_F_polished": float("nan"),
                  "note": "fit-half event-free, polish skipped"})
        rows.append(p)
        print(f"[r29] {ds:9s} 拟合半无事件 -> 跳过抛光 | test "
              f"F1={p['EventF1']:.3f}", flush=True)
        return
    q_ours = float((s_tr < ours_prm["theta_hi"]).mean())
    warm = {"q": min(max(q_ours, 0.50), 0.999),
            "margin": (ours_prm["theta_hi"] - ours_prm["theta_lo"]) / sd,
            "t_on": int(ours_prm["T_on"]), "t_off": int(ours_prm["T_off"])}
    pr2 = tpe_search(s_fit, y_fit, ev_fit, dt, SEED, n_trials=20, warm=warm)
    th2 = np.quantile(s_tr, pr2["q"])
    f_val_ours = hard_f1(s_val, y_val, dt, ours_prm["theta_hi"],
                         ours_prm["theta_lo"], ours_prm["T_on"],
                         ours_prm["T_off"])
    f_val_pol = hard_f1(s_val, y_val, dt, th2, th2 - pr2["margin"],
                        pr2["t_on"], pr2["t_off"])
    adopt = f_val_pol > f_val_ours
    if adopt:
        final = (th2, th2 - pr2["margin"], pr2["t_on"], pr2["t_off"])
    else:
        final = (ours_prm["theta_hi"], ours_prm["theta_lo"],
                 ours_prm["T_on"], ours_prm["T_off"])
    a = hard_pipeline(s_te, final[0], final[1], int(final[2]), int(final[3]))
    p = kpi_panel(a, y_te, cfg=KPIConfig(dt=dt))
    p.update({"method": "validated-polish", "dataset": ds,
              "adopted": adopt, "val_F_ours": f_val_ours,
              "val_F_polished": f_val_pol})
    rows.append(p)
    print(f"[r29] {ds:9s} val: ours={f_val_ours:.3f} polished={f_val_pol:.3f} "
          f"-> {'采纳' if adopt else '拒绝'} | test F1={p['EventF1']:.3f} "
          f"det={p['event_detection_rate']:.2f} AAR={p['AAR_per_hour']:.3f}",
          flush=True)


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r29_validated_polish.csv")
    prm_tab = pd.read_csv(os.path.join(RESULTS_DIR, "e3",
                                       "e3_selected_params.csv"))
    rows = []
    for scorer, ds in [("cmhmil", "SWaT"), ("iforest", "TE"),
                       ("iforest", "MetroPT3")]:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        pp = prm_tab[(prm_tab.dataset == ds)
                     & (prm_tab.method == "diff-tune")].iloc[0]
        ours = {"theta_hi": pp.theta_hi, "theta_lo": pp.theta_lo,
                "T_on": int(pp.T_on), "T_off": int(pp.T_off)}
        run_stream(s[tr_m], y[tr_m], s[te_m], y[te_m], dt, ours, ds, rows)
    # TE-ext
    from experiments.e_r25b_diag import build_stream
    s, y, dt = build_stream()
    tr_m, te_m = interleave_split(len(s), dt)
    K_on, K_off = _kmax(dt)
    prm, _ = train_tuning(s[tr_m], y[tr_m], events_from_binary(y[tr_m]), dt,
                          R_max=10.0, F_max=0.02, epochs=300, seed=SEED,
                          K_on=K_on, K_off=K_off)
    run_stream(s[tr_m], y[tr_m], s[te_m], y[te_m], dt, prm, "TE-ext", rows)

    # SWaT 表盘校准扫描 (M2' 转移映射)
    s, y, dt = get_scores("cmhmil", "SWaT", seed=SEED)
    tr_m, te_m = interleave_split(len(s), dt)
    K_on, K_off = _kmax(dt)
    for r_max in (0.005, 0.01, 0.015, 0.02):
        prm, _ = train_tuning(s[tr_m], y[tr_m],
                              events_from_binary(y[tr_m]), dt,
                              R_max=r_max, F_max=0.02, epochs=300, seed=SEED,
                              K_on=K_on, K_off=K_off)
        a = hard_pipeline(s[te_m], prm["theta_hi"], prm["theta_lo"],
                          prm["T_on"], prm["T_off"])
        p = kpi_panel(a, y[te_m], cfg=KPIConfig(dt=dt))
        rows.append({"method": f"dial-{r_max}", "dataset": "SWaT",
                     "EventF1": p["EventF1"], "AAR_per_hour": p["AAR_per_hour"],
                     "event_detection_rate": p["event_detection_rate"],
                     "n_alarm_events": p["n_alarm_events"]})
        print(f"[r29] dial SWaT R_max={r_max:<6} -> hard AAR="
              f"{p['AAR_per_hour']:.4f}  F1={p['EventF1']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
