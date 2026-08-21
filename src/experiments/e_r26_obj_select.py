"""R29-A1: 目标自动选择策略 —— 用整定段(训练侧)指标选目标, 无测试泄漏.

§5 已给出人工决策规则(事件丰富→事件目标; 稀疏→点级+约束). 本脚本把规则
自动化并检验: 对每个数据流, 用两目标各自收敛配置在 **整定-训练流** 上的
硬事件 F1 (in-sample, 部署时可算) 做 argmax 选择, 再报告该选择隐含的
tuning-test F1. 若 in-sample 选择在所有流上恢复 test-更优目标, 则
"自适应目标选择"使框架在全部流上取到最优或持平 —— A1 攻击线闭合.

数据流: SWaT(cmhmil), TE(iforest), TE-ext(若 r25 缓存存在), MetroPT3(iforest).
输出: results/audit/r26_obj_select.csv
运行: cd src && python -u experiments/e_r26_obj_select.py
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


def eval_params_on(alarm_stream_params, s, y, dt):
    alarm = hard_pipeline(s, *alarm_stream_params)
    return kpi_panel(alarm, y, cfg=KPIConfig(dt=dt))


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r26_obj_select.csv")
    prm_tab = pd.read_csv(os.path.join(RESULTS_DIR, "e3",
                                       "e3_selected_params.csv"))
    streams = [("cmhmil", "SWaT"), ("iforest", "TE"), ("iforest", "MetroPT3")]
    rows = []
    for scorer, ds in streams:
        s, y, dt = get_scores(scorer, ds, seed=SEED)
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)
        K_on, K_off = _kmax(dt)
        # ours 参数取缓存 (与主表一致); pw+constr 重训 (确定性)
        pp = prm_tab[(prm_tab.dataset == ds)
                     & (prm_tab.method == "diff-tune")].iloc[0]
        p_ours = (pp.theta_hi, pp.theta_lo, int(pp.T_on), int(pp.T_off))
        prm, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0, F_max=0.02,
                              epochs=300, seed=SEED, K_on=K_on, K_off=K_off,
                              loss_mode="relaxed_f1_constrained")
        p_pw = (prm["theta_hi"], prm["theta_lo"], prm["T_on"], prm["T_off"])
        f_in = {}
        f_te = {}
        for tag, p in [("event", p_ours), ("pointwise", p_pw)]:
            f_in[tag] = eval_params_on(p, s_tr, y_tr, dt)["EventF1"]
            f_te[tag] = eval_params_on(p, s_te, y_te, dt)["EventF1"]
        pick = max(f_in, key=f_in.get)
        rows.append({"dataset": ds, "n_train_events": len(ev_tr),
                     "F_in_event": f_in["event"],
                     "F_in_pointwise": f_in["pointwise"],
                     "selector_pick": pick,
                     "F_test_event": f_te["event"],
                     "F_test_pointwise": f_te["pointwise"],
                     "F_test_selected": f_te[pick]})
        print(f"[r26] {ds:9s} train_ev={len(ev_tr)} in-sample "
              f"event={f_in['event']:.3f} pw={f_in['pointwise']:.3f} "
              f"-> 选 {pick:9s} | test event={f_te['event']:.3f} "
              f"pw={f_te['pointwise']:.3f} 选中={f_te[pick]:.3f}",
              flush=True)
    # 扩展 TE (若 r25 数据缓存存在): 重训两目标
    ext = os.path.join(RESULTS_DIR, "audit", "r25_te_ext_data.npz")
    if os.path.exists(ext):
        from experiments.e_r25_te_ext import load_ext
        from sklearn.ensemble import IsolationForest
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
        tr_m, te_m = interleave_split(len(s), dt)
        s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
        ev_tr = events_from_binary(y_tr)
        K_on, K_off = _kmax(dt)
        f_in, f_te = {}, {}
        for mode, tag in [("constrained", "event"),
                          ("relaxed_f1_constrained", "pointwise")]:
            prm, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0,
                                  F_max=0.02, epochs=300, seed=SEED,
                                  K_on=K_on, K_off=K_off, loss_mode=mode)
            p = (prm["theta_hi"], prm["theta_lo"], prm["T_on"], prm["T_off"])
            f_in[tag] = eval_params_on(p, s_tr, y_tr, dt)["EventF1"]
            f_te[tag] = eval_params_on(p, s_te, y_te, dt)["EventF1"]
        pick = max(f_in, key=f_in.get)
        rows.append({"dataset": "TE-ext", "n_train_events": len(ev_tr),
                     "F_in_event": f_in["event"],
                     "F_in_pointwise": f_in["pointwise"],
                     "selector_pick": pick,
                     "F_test_event": f_te["event"],
                     "F_test_pointwise": f_te["pointwise"],
                     "F_test_selected": f_te[pick]})
        print(f"[r26] TE-ext     train_ev={len(ev_tr)} in-sample "
              f"event={f_in['event']:.3f} pw={f_in['pointwise']:.3f} "
              f"-> 选 {pick:9s} | test event={f_te['event']:.3f} "
              f"pw={f_te['pointwise']:.3f} 选中={f_te[pick]:.3f}",
              flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
