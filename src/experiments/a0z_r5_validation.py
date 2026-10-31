# -*- coding: utf-8 -*-
"""A0z-3: R1 step2 — R5a/R5b/R5c vs R0 跨流验证 (冻结登记 §R1).

输入: a0z_trials_{SWaT_cmh,Metro_if,TE_cmh,TE_usad}.csv (seed 1-3, B240)
规则 (整定侧可行域内, 运行前已冻结):
  R0: max tune EF1 (tie->lower AAR)   [现行]
  R5a: max tune AAR (tie->lower FAR)
  R5b: min q (tie->higher tune EF1)
  R5c: tune AAR >= 0.5*layer_tune_AAR 中 max tune EF1
判定: mean test det 提升>=0.1 且 test EF1 >= R0-0.05, >=3/4 流成立=迁移成功.
输出: results/audit/a0z_r5_validation.{csv,json}
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "audit"))
TAGS = ["SWaT_cmh", "Metro_if", "TE_cmh", "TE_usad"]


def pick(g, rule, layer_aar):
    tu = g[(g["side"] == "tune") & (~g["layer"])]
    te = g[(g["side"] == "test") & (~g["layer"])]
    te = te[["seed", "trial", "test_det", "test_EF1", "test_AAR",
             "test_TIA", "test_feas"]]
    f = te.merge(tu[["seed", "trial", "q", "tune_EF1", "tune_AAR",
                     "tune_FAR", "tune_feas"]],
                 on=["seed", "trial"], how="inner")
    f = f[f["tune_feas"] == True]
    if rule == "R0":
        k = f.sort_values(["tune_EF1", "tune_AAR"], ascending=[False, True])
    elif rule == "R5a":
        k = f.sort_values(["tune_AAR", "tune_FAR"], ascending=[False, True])
    elif rule == "R5b":
        k = f.sort_values(["q", "tune_EF1"], ascending=[True, False])
    elif rule == "R5c":
        sub = f[f["tune_AAR"] >= 0.5 * layer_aar]
        if len(sub) == 0:
            sub = f
        k = sub.sort_values(["tune_EF1", "tune_AAR"],
                            ascending=[False, True])
    else:
        raise ValueError(rule)
    return k.iloc[0]


def main():
    res = []
    for tag in TAGS:
        df = pd.read_csv(os.path.join(OUT, f"a0z_trials_{tag}.csv"))
        lay = df[(df["layer"] == True) & (df["side"] == "tune")]
        layer_aar = float(lay["AAR_side"].iloc[0])
        per_rule = {}
        for rule in ["R0", "R5a", "R5b", "R5c"]:
            picks = []
            for seed in (1, 2, 3):
                g = df[(df["seed"] == seed) & (~df["layer"])]
                if len(g) == 0:
                    continue
                r = pick(g, rule, layer_aar)
                picks.append({"test_det": r.get("test_det"),
                              "test_EF1": r.get("test_EF1")})
            p = pd.DataFrame(picks)
            per_rule[rule] = {"det": round(float(p.test_det.mean()), 3),
                              "ef1": round(float(p.test_EF1.mean()), 3)}
        res.append({"stream": tag, "layer_tune_AAR": round(layer_aar, 4),
                    **{f"{k}_{m}": v[m] for k, v in per_rule.items()
                       for m in ("det", "ef1")}})
        print(tag, per_rule)
    t = pd.DataFrame(res)
    t.to_csv(os.path.join(OUT, "a0z_r5_validation.csv"), index=False,
             encoding="utf-8-sig")
    # 预注册判定
    wins = []
    for _, r in t.iterrows():
        best_r5 = max(["R5a", "R5b", "R5c"],
                      key=lambda k: r[f"{k}_det"])
        ok = (r[f"{best_r5}_det"] - r["R0_det"] >= 0.1
              and r[f"{best_r5}_ef1"] >= r["R0_ef1"] - 0.05)
        wins.append(ok)
    verdict = {"n_streams_pass": int(sum(wins)),
               "transfer_success": bool(sum(wins) >= 3),
               "per_stream": t.to_dict("records")}
    json.dump(verdict, open(os.path.join(OUT, "a0z_r5_validation.json"),
                            "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in verdict.items()
                      if k != "per_stream"}))


if __name__ == "__main__":
    main()
