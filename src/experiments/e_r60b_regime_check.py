"""R60b: NoBOOM 事件尺度 vs 路由规则 (469 点 / ~8 min 阈值)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.noboom import noboom_stream
from kpi.panel import events_from_binary
from experiments.e_r60_noboom import interleave_local


def runlen(ev):
    return [e - b for b, e in ev]


for which in ["acetone", "butanol"]:
    d = noboom_stream(which)
    y = d["Y"]
    tr_m, te_m = interleave_local(len(y), d["dt"])
    ev_tr = events_from_binary(y[tr_m])
    Ls = runlen(ev_tr)
    med = np.median(Ls)
    below = sum(1 for L in Ls if L < 469)
    print(f"{which}: tuning episodes n={len(Ls)} median={med:.0f}s "
          f"({med/60:.1f} min) q25={np.percentile(Ls,25):.0f}s "
          f"q75={np.percentile(Ls,75):.0f}s | <469s: {below}/{len(Ls)}")
