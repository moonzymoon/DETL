# -*- coding: utf-8 -*-
"""B-1b: dilate+10% 整定的层在 10 个未见 run 上的验证."""
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from a0_newruns import rebuild_scorer, score_run, eval_run, UNSEEN, TE_DIR, OUT  # noqa


def main():
    mu, sd, ifo = rebuild_scorer()
    prm_row = pd.read_csv(os.path.join(OUT, "a0_params_dilate.csv")).iloc[0]
    prm = {"theta_hi": prm_row["theta_hi"], "theta_lo": prm_row["theta_lo"],
           "T_on": int(prm_row["T_on"]), "T_off": int(prm_row["T_off"])}
    rows = []
    for idv, run in UNSEEN:
        s_, y_, dt_ = score_run(os.path.join(TE_DIR, f"mode1_{idv}_{run}.xlsx"),
                                mu, sd, ifo)
        m = eval_run(prm, s_, y_, dt_)
        rows.append({"idv": idv, "run": run, "method": "layer_dilate10",
                     "EventF1": m["EventF1"],
                     "det": m["event_detection_rate"],
                     "AAR": m["AAR_per_hour"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "a0_newruns_dilate.csv"),
              index=False, encoding="utf-8-sig")
    print(df.groupby("method")[["EventF1", "det", "AAR"]].mean().round(3))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
