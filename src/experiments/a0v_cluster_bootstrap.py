# -*- coding: utf-8 -*-
"""A0v: D — 聚类 bootstrap + fault-stratified 表 + leave-one-IDV-out.

cohort-1 = a0_perrun_table.csv (n=10, 5 IDV); cohort-2 = a0u_newruns_ext.csv.
比较对象 = layer (r0) vs optuna seed-mean (与正文 seed-averaged 口径一致).
统计: run-level bootstrap + 按 IDV 聚类 bootstrap (B=10000, seed=0);
fault-stratified 逐故障表; leave-one-IDV-out 配对差.
输出: results/audit/a0v_cluster_bootstrap.{json,csv}
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "audit"))
B, SEED = 10000, 0


def load_runs():
    # cohort-1
    c1 = pd.read_csv(os.path.join(OUT, "a0_perrun_table.csv"))
    runs1 = pd.DataFrame({
        "idv": c1["idv"], "run": c1["run"], "cohort": 1,
        "layer_EF1": c1["EventF1"], "layer_det": c1["det"],
        "opt_EF1": c1["best_opt_EF1"],
        "opt_det": None})          # cohort-1 表无 seed-mean det 列 → 后补
    # cohort-1 的 opt det: 从 a0_newruns.csv 重算 seed-mean
    nr = pd.read_csv(os.path.join(OUT, "a0_newruns.csv"))
    od = (nr[nr["method"].str.startswith("optuna")]
          .groupby(["idv", "run"]).det.mean().rename("opt_det").reset_index())
    runs1 = runs1.merge(od, on=["idv", "run"], suffixes=("", "_y"))
    runs1["opt_det"] = runs1["opt_det_y"]
    runs1 = runs1.drop(columns=["opt_det_y"])
    # cohort-2
    nr2 = pd.read_csv(os.path.join(OUT, "a0u_newruns_ext.csv"))
    lay = (nr2[nr2["method"] == "layer_r0"]
           .set_index(["idv", "run"]))
    opt = (nr2[nr2["method"].str.startswith("optuna")]
           .groupby(["idv", "run"]).agg(opt_EF1=("EventF1", "mean"),
                                        opt_det=("det", "mean")))
    runs2 = lay[["EventF1", "det"]].join(opt).reset_index()
    runs2 = pd.DataFrame({
        "idv": runs2["idv"], "run": runs2["run"], "cohort": 2,
        "layer_EF1": runs2["EventF1"], "layer_det": runs2["det"],
        "opt_EF1": runs2["opt_EF1"], "opt_det": runs2["opt_det"]})
    return pd.concat([runs1, runs2], ignore_index=True)


def boot_ci(x, clusters=None):
    rng = np.random.default_rng(SEED)
    x = np.asarray(x, float)
    if clusters is None:
        idx = rng.integers(0, len(x), size=(B, len(x)))
        st = x[idx].mean(1)
    else:
        cl = pd.Series(clusters)
        uniq = cl.unique()
        pool = {u: np.flatnonzero((cl == u).values) for u in uniq}
        st = np.empty(B)
        for b in range(B):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            rows = np.concatenate([pool[u] for u in pick])
            st[b] = x[rows].mean()
    return {"mean": round(float(x.mean()), 4),
            "ci95": [round(float(np.percentile(st, 2.5)), 4),
                     round(float(np.percentile(st, 97.5)), 4)]}


def main():
    df = load_runs()
    df["d_EF1"] = df["layer_EF1"] - df["opt_EF1"]
    df["d_det"] = df["layer_det"] - df["opt_det"]

    strat = (df.groupby(["cohort", "idv"])
             .agg(n=("run", "count"), layer_EF1=("layer_EF1", "mean"),
                  opt_EF1=("opt_EF1", "mean"), d_EF1=("d_EF1", "mean"),
                  layer_det=("layer_det", "mean"), opt_det=("opt_det", "mean"),
                  d_det=("d_det", "mean")).round(4).reset_index())
    strat.to_csv(os.path.join(OUT, "a0v_fault_stratified.csv"), index=False,
                 encoding="utf-8-sig")

    loo = []
    for u in sorted(df["idv"].unique()):
        sub = df[df["idv"] != u]
        loo.append({"drop_idv": int(u),
                    "mean_d_EF1": round(float(sub["d_EF1"].mean()), 4),
                    "mean_d_det": round(float(sub["d_det"].mean()), 4)})
    res = {
        "protocol": "cohort1=a0_perrun_table n=10; cohort2=a0u ext; "
                    "comparator=optuna seed-mean; B=10000 seed=0",
        "n_runs_total": len(df),
        "n_idv": int(df["idv"].nunique()),
        "run_level": {
            "d_EF1": boot_ci(df["d_EF1"]),
            "d_det": boot_ci(df["d_det"]),
            "layer_det": boot_ci(df["layer_det"]),
            "opt_det": boot_ci(df["opt_det"]),
        },
        "idv_clustered": {
            "d_EF1": boot_ci(df["d_EF1"], df["idv"]),
            "d_det": boot_ci(df["d_det"], df["idv"]),
        },
        "per_cohort": {
            f"cohort{c}": {
                "n": int((df["cohort"] == c).sum()),
                "d_EF1": boot_ci(df[df["cohort"] == c]["d_EF1"]),
                "d_det": boot_ci(df[df["cohort"] == c]["d_det"]),
            } for c in (1, 2)
        },
        "leave_one_idv_out": loo,
        "stratified_table": strat.to_dict("records"),
    }
    with open(os.path.join(OUT, "a0v_cluster_bootstrap.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(strat.to_string(index=False))
    print("\nrun-level d_EF1:", res["run_level"]["d_EF1"])
    print("clustered  d_EF1:", res["idv_clustered"]["d_EF1"])
    print("run-level d_det:", res["run_level"]["d_det"])
    print("clustered  d_det:", res["idv_clustered"]["d_det"])
    for c in (1, 2):
        print(f"cohort{c}:", res["per_cohort"][f"cohort{c}"]["d_EF1"],
              res["per_cohort"][f"cohort{c}"]["d_det"])


if __name__ == "__main__":
    main()
