"""R52: 四数据集 tuning-test 流时长 (复审 #21: AAR/事件数可复核性)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SEED
from scorers.adapter import get_scores
from experiments.e_pilot_tuning import interleave_split
from experiments.e_r36_mcap_sweep import batadal_stream

for ds, sc in [("SWaT", "cmhmil"), ("TE", "iforest"), ("MetroPT3", "iforest")]:
    s, y, dt = get_scores(sc, ds, seed=SEED)
    tr, te = interleave_split(len(s), dt)
    h = te.sum() * dt / 3600.0
    print(f"{ds:8s} tuning-test {h:.1f} h (n={te.sum()}, dt={dt}s)")
sb, yb, dtb = batadal_stream()
tr, te = interleave_split(len(sb), dtb)
print(f"BATADAL  tuning-test {te.sum() * dtb / 3600.0:.1f} h (n={te.sum()}, dt={dtb}s)")
