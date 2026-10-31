# -*- coding: utf-8 -*-
"""A0.6a: 宽域敏感性 (M̄' = 分位箱宽 q99.9-q50, 设计文档 §2 预固定的次级范围).
层 + Optuna(s1,s2) 在 M̄' 下重训/重搜, 主表并报; 排名随范围改变则写入结论.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import a0_lib
from a0_lib import (load_te_segments, interleave_mask, eval_params_M,
                    search_unified_M)

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))


def main():
    s, y, dt, bounds, meta = load_te_segments()
    tune_m, test_m = interleave_mask(len(s), dt, bounds)
    sd = float(s[tune_m].std())
    m_wide = float(np.quantile(s[tune_m], 0.999) - np.quantile(s[tune_m], 0.5))
    print(f"M̄'=箱宽={m_wide:.4f} = {m_wide/sd:.2f}σ (主域 M̄=2σ={2*sd:.4f})")

    rows = []
    # 层 (M̄')
    import a0_lib as L
    L.M_SIGMA = m_wide / sd            # 复用常数换算: m_max = M_SIGMA*sd
    prm, hist, m_max = L.train_layer_M(s, y, tune_m, bounds, dt, epochs=300)
    me = eval_params_M(prm, s, y, test_m, dt, bounds)
    me["method"] = "layer_wide"
    rows.append(me)
    print(f"[layer M̄'] {prm} -> EF1={me['EF1_ov']:.3f} det={me['det_ov']:.2f} "
          f"AAR={me['AAR_side']:.2f}")
    # 搜索 (M̄') seeds 1-2
    for seed in (1, 2):
        picks, _ = search_unified_M(s, y, tune_m, bounds, dt, seed, budget=240,
                                    use_tpe=True, m_sigma=m_wide / sd)
        for B in (60, 240):
            me = eval_params_M(picks[B]["prm"], s, y, test_m, dt, bounds)
            me["method"] = f"optuna_wide_s{seed}_B{B}"
            rows.append(me)
            print(f"[optuna M̄' s{seed} B{B}] EF1={me['EF1_ov']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "a0_wide.csv"), index=False,
              encoding="utf-8-sig")
    print(df[["method", "EF1_ov", "EF1_iou01", "det_ov", "AAR_side",
              "TIA_side"]].to_string(index=False))


if __name__ == "__main__":
    main()
