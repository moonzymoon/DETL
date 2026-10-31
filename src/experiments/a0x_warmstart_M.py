# -*- coding: utf-8 -*-
"""A0x: T2 — warm-start 协议 M 重跑 (冻结登记 §T2).

配置: warm-20 (trial0=层参数 enqueue + 19 TPE) / cold-20 / cold-60, 种子{7,8,9}.
目标: 整定侧硬 event-F1 (与 search_unified_M 逐位同语义: 同参数域/同约束/同罚).
选点: R0 规则 (可行中 EF1 最大, tie→低 AAR). 测试侧只报不选.
层参数: a0_params.csv layer_r0 → q=theta_hi 的整定侧经验分位, mf=margin/std.
输出: results/audit/a0x_warmstart_M.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from a0_lib import (load_te_segments, interleave_mask, eval_params_M,  # noqa
                    R_MAX, F_MAX, M_SIGMA)

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))
K_ON, K_OFF = 24, 48


def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    s, y, dt, bounds, meta = load_te_segments()
    tune_m, test_m = interleave_mask(len(s), dt, bounds)
    sd = float(s[tune_m].std())
    m_sigma = M_SIGMA

    prm = pd.read_csv(os.path.join(OUT, "a0_params.csv"))
    lay = prm[prm["method"] == "layer_r0"].iloc[0]
    st = np.sort(s[tune_m])
    q_lay = float(np.searchsorted(st, lay["theta_hi"]) / len(st))
    mf_lay = float((lay["theta_hi"] - lay["theta_lo"]) / sd)
    warm0 = {"q": float(min(max(q_lay, 0.50), 0.999)),
             "mf": float(min(max(mf_lay, 0.0), m_sigma)),
             "t_on": int(lay["T_on"]), "t_off": int(lay["T_off"])}
    print(f"layer warm start: q={warm0['q']:.4f} mf={warm0['mf']:.4f} "
          f"T_on={warm0['t_on']} T_off={warm0['t_off']}")

    def objective(trial):
        q = trial.suggest_float("q", 0.50, 0.999)
        mf = trial.suggest_float("mf", 0.0, m_sigma)
        t_on = trial.suggest_int("t_on", 1, K_ON)
        t_off = trial.suggest_int("t_off", 1, K_OFF)
        p = {"theta_hi": float(np.quantile(s[tune_m], q)),
             "theta_lo": float(np.quantile(s[tune_m], q)) - mf * sd,
             "T_on": int(t_on), "T_off": int(t_off)}
        m = eval_params_M(p, s, y, tune_m, dt, bounds)
        trial.set_user_attr("viol", (
            max(0.0, m["AAR_side"] - R_MAX) / R_MAX,
            max(0.0, m["FAR_side"] - F_MAX) / F_MAX))
        trial.set_user_attr("row", {"q": q, "mf": mf, "t_on": t_on,
                                    "t_off": t_off, "EF1_ov": m["EF1_ov"],
                                    "AAR": m["AAR_side"], "FAR": m["FAR_side"],
                                    "feas": m["feasible"]})
        return m["EF1_ov"]

    def constraints_fun(trial):
        return np.array(trial.user_attrs["viol"])

    def run_cfg(cfg, seed, budget):
        sampler = optuna.samplers.TPESampler(seed=seed,
                                             constraints_func=constraints_fun)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        if cfg == "warm-20":
            study.enqueue_trial(warm0)
        study.optimize(objective, n_trials=budget, show_progress_bar=False)
        rows = []
        for t in study.trials:
            r = t.user_attrs.get("row")
            if r:
                r = dict(r)
                r["trial"] = t.number
                rows.append(r)
        rows.sort(key=lambda r: (0 if r["feas"] else 1,
                                 -r["EF1_ov"] if r["feas"] else 0,
                                 r["AAR"] if r["feas"] else 0, r["trial"]))
        return rows[0]

    out = []
    for seed in (7, 8, 9):
        for cfg, budget in [("warm-20", 20), ("cold-20", 20), ("cold-60", 60)]:
            t0 = time.time()
            best = run_cfg(cfg, seed, budget)
            p = {"theta_hi": float(np.quantile(s[tune_m], best["q"])),
                 "theta_lo": float(np.quantile(s[tune_m], best["q"]))
                 - best["mf"] * sd,
                 "T_on": int(best["t_on"]), "T_off": int(best["t_off"])}
            me = eval_params_M(p, s, y, test_m, dt, bounds)
            out.append({"stream": "TE", "seed": seed, "config": cfg,
                        "tune_EF1": round(best["EF1_ov"], 4),
                        "tune_feas": best["feas"],
                        "test_EF1": round(me["EF1_ov"], 4),
                        "test_det": round(me["det_ov"], 4),
                        "test_AAR": round(me["AAR_side"], 4),
                        "test_feas": me["feasible"],
                        "q": round(best["q"], 4), "mf": round(best["mf"], 4),
                        "t_on": best["t_on"], "t_off": best["t_off"],
                        "first_q": warm0["q"] if cfg == "warm-20" else None,
                        "wall_s": round(time.time() - t0, 1)})
            print(f"[s{seed} {cfg}] tuneEF1={best['EF1_ov']:.3f} "
                  f"testEF1={me['EF1_ov']:.3f} det={me['det_ov']:.2f} "
                  f"feas={me['feasible']} ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUT, "a0x_warmstart_M.csv"), index=False,
              encoding="utf-8-sig")
    print("\n== means over seeds ==")
    print(df.groupby("config")[["tune_EF1", "test_EF1", "test_det"]]
          .mean().round(4).to_string())


if __name__ == "__main__":
    main()
