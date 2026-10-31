# -*- coding: utf-8 -*-
"""A0p: 选解规则审计 — 按冻结登记 (00返修/冻结登记_补实验_20261025.md) 执行.

数据: 全部已有缓存 (a0_trials_optuna_s*.csv 整定侧 + a0j_matched_aar.csv 测试侧),
不新跑仿真。规则族 R0-R4 与预测子清单在运行前冻结; 无拟合, 故无拆分需求。
输出: results/audit/a0p_selection_audit.{json,csv}
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "audit"))

# ── 载入 ────────────────────────────────────────────────────────────
tune_rows = []
for seed in range(1, 6):
    df = pd.read_csv(os.path.join(OUT, f"a0_trials_optuna_s{seed}.csv"))
    df["seed"] = seed
    tune_rows.append(df)
tune = pd.concat(tune_rows, ignore_index=True)
test = pd.read_csv(os.path.join(OUT, "a0j_matched_aar.csv"))
d = tune.merge(test, on=["seed", "trial"], suffixes=("_tune", "_test"))
assert len(d) == 1158, len(d)

feas = d[d["feas"] == True].copy()          # 整定侧可行 (规则定义域)
feas_test_ok = feas[feas["feasible"] == True]     # 其中测试侧也可行 (n=1155)

# ── 1) 冻结预测子 vs 测试侧 Spearman ────────────────────────────────
preds = ["EF1_ov_tune", "AAR_tune", "FAR_tune", "q_tune", "mf_tune", "t_on", "t_off"]
corr = {}
for p in preds:
    r_det, p_det = spearmanr(feas_test_ok[p], feas_test_ok["det_ov"])
    r_ef1, p_ef1 = spearmanr(feas_test_ok[p], feas_test_ok["EF1_ov_test"]
                             if "EF1_ov_test" in feas_test_ok else
                             feas_test_ok["EF1_ov"])
    corr[p] = {"rho_vs_test_det": round(float(r_det), 4),
               "p_det": float(p_det),
               "rho_vs_test_EF1": round(float(r_ef1, ), 4) if False else
               round(float(r_ef1), 4),
               "p_EF1": float(p_ef1)}

# ── 2) 冻结规则族逐 seed 选择 ────────────────────────────────────────
LAYER_AAR = 0.1072  # a0j layer_point.AAR (主表冻结值)

def pick_pos(gg, rule):
    g = gg[gg["feas"] == True]
    if rule in ("R0", "R4"):
        key = g if rule == "R0" else (
            g[g["AAR_tune"] <= LAYER_AAR] if (g["AAR_tune"] <= LAYER_AAR).any()
            else g)
        return key.sort_values(["EF1_ov_tune", "AAR_tune"],
                               ascending=[False, True]).index[0]
    if rule == "R1":
        return g.sort_values(["AAR_tune", "EF1_ov_tune"],
                             ascending=[True, False]).index[0]
    if rule == "R2":
        return g.sort_values(["FAR_tune", "EF1_ov_tune"],
                             ascending=[True, False]).index[0]
    if rule == "R3":
        return g.sort_values(["EF1_ov_tune", "AAR_tune", "FAR_tune"],
                             ascending=[False, True, True]).index[0]
    if rule == "ORACLE":
        return g.sort_values("det_ov", ascending=False).index[0]
    raise ValueError(rule)

rules = ["R0", "R1", "R2", "R3", "R4", "ORACLE"]
sel = []
for seed in range(1, 6):
    gg = d[d["seed"] == seed].reset_index(drop=True)
    for r in rules:
        row = gg.iloc[pick_pos(gg, r)]
        sel.append({"seed": seed, "rule": r, "trial": int(row["trial"]),
                    "q": round(float(row["q_tune"]), 3),
                    "mf": round(float(row["mf_tune"]), 3),
                    "t_on": int(row["t_on"]), "t_off": int(row["t_off"]),
                    "tune_EF1": round(float(row["EF1_ov_tune"]), 3),
                    "test_det": round(float(row["det_ov"]), 3),
                    "test_EF1": round(float(row["EF1_ov_test"]), 3),
                    "test_AAR": round(float(row["AAR_test"]), 4),
                    "test_TIA": round(float(row["TIA_side"]), 4)})
sel_df = pd.DataFrame(sel)
summary = (sel_df.groupby("rule")[["test_det", "test_EF1", "test_AAR",
                                   "test_TIA"]]
           .mean().round(4).reset_index())

out = {
    "protocol": "frozen plan 00返修/冻结登记_补实验_20261025.md; descriptive, no fitting",
    "n_tune_feasible": int(len(feas)),
    "n_test_feasible": int(len(feas_test_ok)),
    "predictor_spearman": corr,
    "per_seed_selection": sel,
    "rule_summary_mean_over_5_seeds": summary.to_dict("records"),
}
with open(os.path.join(OUT, "a0p_selection_audit.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
sel_df.to_csv(os.path.join(OUT, "a0p_selection_audit.csv"),
              index=False, encoding="utf-8-sig")

print("== predictor vs test-side (n=1155 feasible) ==")
for k, v in corr.items():
    print(f"{k:12s} rho(det)={v['rho_vs_test_det']:+.3f} p={v['p_det']:.2e}"
          f"  rho(EF1)={v['rho_vs_test_EF1']:+.3f} p={v['p_EF1']:.2e}")
print("\n== rule means over 5 seeds ==")
print(summary.to_string(index=False))
