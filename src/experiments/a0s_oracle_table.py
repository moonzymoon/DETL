# -*- coding: utf-8 -*-
"""A0s: oracle 同工作点对照表 + 参数差距分解 (DS P0, 纯缓存).

行: layer / R0 规则选出的 search (逐 seed 均值) / ORACLE-det / ORACLE-EF1 /
     ORACLE-EF1 在 layer AAR 预算内 (AAR<=0.1072) 的最优.
列: det_ov, EF1_ov, AAR, TIA_side, MAD_s, 参数 (q, mf, t_on, t_off).
参数分解: ORACLE-det vs layer 的四算子参数距离 (分位数单位/绝对).
输出: results/audit/a0s_oracle_table.{csv,json}
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "audit"))

test = pd.read_csv(os.path.join(OUT, "a0j_matched_aar.csv"))
j = json.load(open(os.path.join(OUT, "a0j_matched_aar.json"),
                   encoding="utf-8"))
LAYER = j["layer_point"]
LAYER_AAR = LAYER["AAR"]
feast = test[test["feasible"] == True].copy()

# layer 参数: a0_params.csv layer_r0 (协议 M 主表)
prm = pd.read_csv(os.path.join(OUT, "a0_params.csv"))
lay = prm[prm["method"] == "layer_r0"].iloc[0]

# a0p: R0 规则逐 seed 选择
sel = json.load(open(os.path.join(OUT, "a0p_selection_audit.json"),
                     encoding="utf-8"))
r0 = [r for r in sel["per_seed_selection"] if r["rule"] == "R0"]

rows = [{
    "config": "layer (deployed)", "det": LAYER["det_ov"],
    "EF1": LAYER["EF1_ov"], "AAR": LAYER["AAR"], "TIA": LAYER["TIA_side"],
    "MAD_s": LAYER["MAD_s"],
    "q": None, "mf": None, "t_on": int(lay["T_on"]), "t_off": int(lay["T_off"]),
    "theta_hi": round(float(lay["theta_hi"]), 4),
}]
rows.append({
    "config": "search, R0 rule (seed-mean)", "det":
        float(np.mean([r["test_det"] for r in r0])),
    "EF1": float(np.mean([r["test_EF1"] for r in r0])),
    "AAR": float(np.mean([r["test_AAR"] for r in r0])),
    "TIA": float(np.mean([r["test_TIA"] for r in r0])),
    "MAD_s": None, "q": float(np.mean([r["q"] for r in r0])),
    "mf": float(np.mean([r["mf"] for r in r0])),
    "t_on": int(np.mean([r["t_on"] for r in r0])),
    "t_off": int(np.mean([r["t_off"] for r in r0])), "theta_hi": None,
})
for tag, col, sub in [
        ("ORACLE max-det (global)", "det_ov", feast),
        ("ORACLE max-EF1 (global)", "EF1_ov", feast),
        ("ORACLE max-EF1 at layer AAR", "EF1_ov",
         feast[feast["AAR"] <= LAYER_AAR])]:
    r = sub.loc[sub[col].idxmax()]
    rows.append({
        "config": tag, "det": round(float(r["det_ov"]), 3),
        "EF1": round(float(r["EF1_ov"]), 3), "AAR": round(float(r["AAR"]), 4),
        "TIA": round(float(r["TIA_side"]), 4),
        "MAD_s": float(r["MAD_s"]),
        "q": round(float(r["q"]), 3), "mf": round(float(r["mf"]), 3),
        "t_on": int(r["T_on"]), "t_off": int(r["T_off"]),
        "theta_hi": None,
    })
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "a0s_oracle_table.csv"), index=False,
          encoding="utf-8-sig")

# 参数差距分解: ORACLE max-det vs layer
od = feast.loc[feast["det_ov"].idxmax()]
# layer 的等效 q (theta_hi 对应整定侧分位) — 由 a0j 段落口径, layer 用分位初始化
# 但 theta 是绝对值; 差距按 std 单位报
lay_theta = float(lay["theta_hi"])
lay_margin = float(lay["theta_hi"] - lay(lay["theta_lo"])) \
    if False else float(lay["theta_hi"] - lay["theta_lo"])
dec = {
    "oracle_maxdet_params": {"q": round(float(od["q"]), 3),
                              "mf": round(float(od["mf"]), 3),
                              "T_on": int(od["T_on"]),
                              "T_off": int(od["T_off"])},
    "layer_params": {"theta_hi": round(lay_theta, 4),
                     "theta_lo": round(float(lay["theta_lo"]), 4),
                     "T_on": int(lay["T_on"]), "T_off": int(lay["T_off"])},
    "notes": "oracle 以整定侧分位数 q 表示阈值; layer 为绝对 theta。"
             "deadband: oracle mf*sd vs layer (hi-lo)。延迟差: T_on/T_off。",
}
with open(os.path.join(OUT, "a0s_oracle_table.json"), "w",
          encoding="utf-8") as f:
    json.dump({"table": df.to_dict("records"), "decomposition": dec,
               "oracle_det_row": {k: (round(float(v), 4) if isinstance(
                   v, (int, float, np.floating)) else v)
                   for k, v in od.items() if k in
                   ("q", "mf", "T_on", "T_off", "det_ov", "EF1_ov", "AAR",
                    "TIA_side", "MAD_s", "seed", "trial")}},
              f, indent=1, ensure_ascii=False, default=str)
print(df.to_string(index=False))
