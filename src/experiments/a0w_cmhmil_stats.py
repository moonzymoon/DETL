# -*- coding: utf-8 -*-
"""A0w-2: T1 统计 — clone a0v (聚类 bootstrap/分层/LOO/最优种子), CMH-MIL 口径.
比较对象 = cmhmil layer vs cmhmil optuna seed-mean (3 seeds, 旧协议整定参数).
输出: a0w_cmhmil_stats.{json,csv}
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "audit"))
B, SEED = 10000, 0


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
        for b_ in range(B):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            rows = np.concatenate([pool[u] for u in pick])
            st[b_] = x[rows].mean()
    return {"mean": round(float(x.mean()), 4),
            "ci95": [round(float(np.percentile(st, 2.5)), 4),
                     round(float(np.percentile(st, 97.5)), 4)]}


def main():
    nr = pd.read_csv(os.path.join(OUT, "a0w_cmhmil_newruns.csv"))
    lay = nr[nr["method"] == "layer"].set_index(["idv", "run"])
    opt = (nr[nr["method"].str.startswith("optuna")]
           .groupby(["idv", "run"])
           .agg(opt_EF1=("EventF1", "mean"), opt_det=("det", "mean"),
                opt_AAR=("AAR", "mean"), best_EF1=("EventF1", "max"),
                best_det=("det", "max")))
    df = lay[["EventF1", "det", "AAR"]].join(opt).reset_index()
    df["cohort"] = np.where(df["idv"].isin([2, 5, 7, 10, 13]), 1, 2)
    df["d_EF1"] = df["EventF1"] - df["opt_EF1"]
    df["d_det"] = df["det"] - df["opt_det"]

    strat = (df.groupby(["cohort", "idv"])
             .agg(n=("run", "count"), layer_EF1=("EventF1", "mean"),
                  opt_EF1=("opt_EF1", "mean"), d_EF1=("d_EF1", "mean"),
                  layer_det=("det", "mean"), opt_det=("opt_det", "mean"),
                  d_det=("d_det", "mean"),
                  layer_AAR=("AAR", "mean"), opt_AAR=("opt_AAR", "mean"))
             .round(4).reset_index())
    strat.to_csv(os.path.join(OUT, "a0w_cmhmil_stratified.csv"), index=False,
                 encoding="utf-8-sig")
    loo = [{"drop_idv": int(u),
            "mean_d_EF1": round(float(df[df.idv != u]["d_EF1"].mean()), 4),
            "mean_d_det": round(float(df[df.idv != u]["d_det"].mean()), 4)}
           for u in sorted(df["idv"].unique())]
    res = {
        "protocol": "cmhmil TE_seed7 frozen ckpt; params a0_cmhmil_TE_params.csv "
                    "(legacy-tuned, 3 optuna seeds); comparator=optuna seed-mean; "
                    "B=10000 seed=0",
        "n_runs": len(df), "n_idv": int(df["idv"].nunique()),
        "run_level": {"d_EF1": boot_ci(df["d_EF1"]),
                      "d_det": boot_ci(df["d_det"]),
                      "layer_det": boot_ci(df["det"]),
                      "opt_det": boot_ci(df["opt_det"]),
                      "layer_AAR": boot_ci(df["AAR"]),
                      "opt_AAR": boot_ci(df["opt_AAR"])},
        "idv_clustered": {"d_EF1": boot_ci(df["d_EF1"], df["idv"]),
                          "d_det": boot_ci(df["d_det"], df["idv"])},
        "best_seed": {"layer_EF1": round(float(df["EventF1"].mean()), 4),
                      "best_EF1": round(float(df["best_EF1"].mean()), 4),
                      "layer_det": round(float(df["det"].mean()), 4),
                      "best_det": round(float(df["best_det"].mean()), 4)},
        "leave_one_idv_out": loo,
        "stratified": strat.to_dict("records"),
    }
    json.dump(res, open(os.path.join(OUT, "a0w_cmhmil_stats.json"), "w",
                        encoding="utf-8"), indent=1)
    print(strat.to_string(index=False))
    print("\nclustered d_EF1:", res["idv_clustered"]["d_EF1"])
    print("clustered d_det:", res["idv_clustered"]["d_det"])
    print("best-seed:", res["best_seed"])
    loos = [x["mean_d_det"] for x in loo]
    print("LOO d_det range:", min(loos), "to", max(loos))


if __name__ == "__main__":
    main()
