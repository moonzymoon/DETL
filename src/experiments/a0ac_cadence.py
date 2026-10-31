# -*- coding: utf-8 -*-
"""A0ac: 决策周期敏感性 — SWaT(cmhmil, 原生5s) 抽稀 x1/x2/x6/x12, 冻结参数评估.

协议: 分数/标签按窗尾对齐后整点抽稀(取每k个样本第1个, 无插值);
      层与 optuna B240 冻结参数在抽稀流上按协议 M(单段, interleave) 重评.
输出: results/audit/a0ac_cadence_sensitivity.csv
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from a0_lib import interleave_mask, eval_params_M, hard_chain_segmented  # noqa
from kpi.panel import KPIConfig, events_from_binary, kpi_panel  # noqa

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))


def main():
    from scorers.adapter import get_scores
    s, y, dt0 = get_scores("cmhmil", "SWaT")
    s = np.asarray(s, np.float64)
    y = np.asarray(y)
    prm = pd.read_csv(os.path.join(OUT, "a0_cmhmil_SWaT_params.csv"))
    lay = prm[prm["method"] == "layer"].iloc[0]
    opts = prm[prm["method"].str.startswith("optuna")]
    rows = []
    for k, tag in [(1, "5s"), (2, "10s"), (6, "30s"), (12, "60s")]:
        sk, yk, dtk = s[::k], y[::k], dt0 * k
        n = len(sk)
        bounds = [(0, n)]
        tune_m, test_m = interleave_mask(n, dtk, bounds)
        pl = {"theta_hi": float(lay["theta_hi"]),
              "theta_lo": float(lay["theta_lo"]),
              "T_on": int(lay["T_on"]), "T_off": int(lay["T_off"])}
        ml = eval_params_M(pl, sk, yk, test_m, dtk, bounds)
        # search seed-mean
        ef, det, aar = [], [], []
        for _, r in opts.iterrows():
            p = {"theta_hi": float(r["theta_hi"]),
                 "theta_lo": float(r["theta_lo"]),
                 "T_on": int(r["T_on"]), "T_off": int(r["T_off"])}
            m = eval_params_M(p, sk, yk, test_m, dtk, bounds)
            ef.append(m["EF1_ov"]); det.append(m["det_ov"])
            aar.append(m["AAR_side"])
        rows.append({
            "cadence_s": dtk, "decim": k,
            "layer_EF1": round(ml["EF1_ov"], 3),
            "layer_det": round(ml["det_ov"], 3),
            "layer_AAR": round(ml["AAR_side"], 4),
            "layer_feas": bool(ml["feasible"]),
            "search_EF1": round(float(np.mean(ef)), 3),
            "search_det": round(float(np.mean(det)), 3),
            "search_AAR": round(float(np.mean(aar)), 4),
        })
        print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "a0ac_cadence_sensitivity.csv"), index=False,
              encoding="utf-8-sig")
    print("saved")


if __name__ == "__main__":
    main()
