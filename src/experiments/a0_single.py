# -*- coding: utf-8 -*-
"""A0.6b: SWaT(cmhmil) 与 MetroPT3(iforest) 的协议 M 重跑 (单段连续流).

层 + Optuna/Random×3seeds + grid + 固定窗, 统一域 M̄=2σ, float64.
输出: results/audit/a0_{scorer}_{ds}.csv
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from a0_lib import (interleave_mask, eval_params_M, train_layer_M,
                    search_unified_M)

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))


def run(scorer, ds):
    from scorers.adapter import get_scores
    s, y, dt = get_scores(scorer, ds)
    s = np.asarray(s, np.float64)
    y = np.asarray(y)
    n = len(s)
    bounds = [(0, n)]                     # 单段连续时间轴 (加载器拼接为一流)
    tune_m, test_m = interleave_mask(n, dt, bounds)
    rows, prms = [], []
    prm, _, m_max = train_layer_M(s, y, tune_m, bounds, dt, epochs=300)
    me = eval_params_M(prm, s, y, test_m, dt, bounds)
    me["method"] = "layer"
    rows.append(me); prms.append({"method": "layer", **prm})
    print(f"[{ds}-{scorer} layer] EF1={me['EF1_ov']:.3f} det={me['det_ov']:.2f} "
          f"AAR={me['AAR_side']:.2f} margin={prm['theta_hi']-prm['theta_lo']:.4f}")
    for meth, tpe in [("optuna", True), ("random", False)]:
        for seed in (1, 2, 3):
            picks, _ = search_unified_M(s, y, tune_m, bounds, dt, seed,
                                        budget=240, use_tpe=tpe)
            me = eval_params_M(picks[240]["prm"], s, y, test_m, dt, bounds)
            me["method"] = f"{meth}_s{seed}"
            rows.append(me); prms.append({"method": me["method"],
                                          **picks[240]["prm"]})
            print(f"[{ds}-{scorer} {meth} s{seed}] EF1={me['EF1_ov']:.3f} "
                  f"det={me['det_ov']:.2f}")
    sd = float(s[tune_m].std())
    best = None
    for q in [0.90, 0.95, 0.99]:
        for w_s in [0, 60, 180, 300, 600, 900]:
            p2 = {"theta_hi": float(np.quantile(s[tune_m], q)),
                  "theta_lo": float(np.quantile(s[tune_m], q)),
                  "T_on": 1, "T_off": max(1, int(round(w_s / dt)))}
            mt = eval_params_M(p2, s, y, tune_m, dt, bounds)
            key = (0 if mt["feasible"] else 1, -mt["EF1_ov"])
            if best is None or key < best[0]:
                best = (key, p2)
    me = eval_params_M(best[1], s, y, test_m, dt, bounds)
    me["method"] = "fixed-window"
    rows.append(me); prms.append({"method": "fixed-window", **best[1]})
    print(f"[{ds}-{scorer} fixed-window] EF1={me['EF1_ov']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"a0_{scorer}_{ds}.csv"),
              index=False, encoding="utf-8-sig")
    pd.DataFrame(prms).to_csv(os.path.join(OUT, f"a0_{scorer}_{ds}_params.csv"),
                              index=False, encoding="utf-8-sig")
    print(df[["method", "EF1_ov", "EF1_iou01", "det_ov", "AAR_side",
              "TIA_side"]].to_string(index=False))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
