# -*- coding: utf-8 -*-
"""A0.6 扩展: cmhmil-TE 与 usad-TE 两条流的新协议重跑 (回 R1.1/R2.5).

用法: python a0_scorers.py cmhmil | python a0_scorers.py usad
层(1 次, float64 确定性) + Optuna/Random×3seeds×B240 + grid-delay + 固定窗.
输出: results/audit/a0_{scorer}_TE.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from a0_lib import (interleave_mask, eval_params_M, train_layer_M,
                    search_unified_M)
from datasets.tep import FAULT_FILES
from kpi.panel import events_from_binary  # noqa

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))
W = 16


def te_bounds(n_scores):
    """TE 分数流段边界 (与 a0_lib.load_te_segments 相同映射)."""
    off = int(48007 * 0.50) + W - 1
    bounds, meta, p = [], [], 30001
    e0 = 30001 - off
    if e0 > 0:
        bounds.append((0, e0)); meta.append(("normal_tail", 0))
    for f in FAULT_FILES:
        idv = int(f.split("_")[1])
        bounds.append((p - off, p + 3001 - off)); meta.append((f, idv))
        p += 3001
    assert bounds[-1][1] == n_scores
    return bounds, meta


def load_cmhmil():
    from scorers.adapter import get_scores
    s, y, dt = get_scores("cmhmil", "TE")
    return np.asarray(s, np.float64), np.asarray(y), float(dt)


def load_usad():
    """复现 e_r48_usad 的 USAD 训练+打分 (固定 seed, 确定性)."""
    import torch
    import torch.nn as nn
    from datasets.tep import load_tep
    from common import SEED
    torch.manual_seed(SEED); np.random.seed(SEED)

    class AE(nn.Module):
        def __init__(self, d, h=64):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(),
                                     nn.Linear(h, h // 2))
            self.dec = nn.Sequential(nn.Linear(h // 2, h), nn.ReLU(),
                                     nn.Linear(h, d))

        def forward(self, x):
            return self.dec(self.enc(x))

    d = load_tep()
    X, Y, dt = d["X"], d["Y"], d["dt"]
    a, b = int(len(X) * 0.35), int(len(X) * 0.50)
    n = len(X)
    idx = np.arange(W)[None, :] + np.arange(n - W + 1)[:, None]
    feats = torch.as_tensor(X[idx].reshape(len(idx), -1)).to(torch.float64)
    Xw_tr = feats[a - W + 1:b]                     # e_r48 同义训练窗段
    ae1, ae2 = AE(feats.shape[1]), AE(feats.shape[1])
    opt1 = torch.optim.Adam(ae1.parameters(), lr=1e-3)
    opt2 = torch.optim.Adam(ae2.parameters(), lr=1e-3)
    A_TR, B_TE, EPOCHS = 5.0, 5.0, 30
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xw_tr))[:512 * 20]
        xb = Xw_tr[perm]
        # 阶段1+2: 训 AE2
        z = xb
        l2 = ((z - ae2(xb)) ** 2).mean(1) + A_TR * (
            (z - ae2(ae1(xb))) ** 2).mean(1)
        opt2.zero_grad(); l2.mean().backward(); opt2.step()
        # 阶段3: 训 AE1
        l1 = ((z - ae1(xb)) ** 2).mean(1) + A_TR * (
            (z - ae1(ae2(xb))) ** 2).mean(1)
        opt1.zero_grad(); l1.mean().backward(); opt1.step()
    with torch.no_grad():
        r11 = ((feats - ae1(feats)) ** 2).mean(1)
        r22 = ((feats - ae2(feats)) ** 2).mean(1)
        r12 = ((feats - ae2(ae1(feats))) ** 2).mean(1)
        r21 = ((feats - ae1(ae2(feats))) ** 2).mean(1)
        scores = (r11 + r22 + B_TE * (r12 + r21)).numpy()
    # 与 local_iforest 对齐: 测试段窗尾从 b+W-1 起 (分数 i <-> 全轴 b+W-1+i)
    s = scores[b:]
    y = Y[b + W - 1:]
    return np.asarray(s, np.float64), np.asarray(y), float(dt)


def main(scorer):
    loader = {"cmhmil": load_cmhmil, "usad": load_usad}[scorer]
    s, y, dt = loader()
    bounds, meta = te_bounds(len(s))
    tune_m, test_m = interleave_mask(len(s), dt, bounds)
    rows, prms = [], []
    prm, hist, m_max = train_layer_M(s, y, tune_m, bounds, dt, epochs=300)
    me = eval_params_M(prm, s, y, test_m, dt, bounds)
    me["method"] = "layer"
    rows.append(me); prms.append({"method": "layer", **prm})
    print(f"[{scorer} layer] EF1={me['EF1_ov']:.3f} det={me['det_ov']:.2f} "
          f"AAR={me['AAR_side']:.2f}")
    for meth, tpe in [("optuna", True), ("random", False)]:
        for seed in (1, 2, 3):
            picks, _ = search_unified_M(s, y, tune_m, bounds, dt, seed,
                                        budget=240, use_tpe=tpe)
            me = eval_params_M(picks[240]["prm"], s, y, test_m, dt, bounds)
            me["method"] = f"{meth}_s{seed}"
            rows.append(me); prms.append({"method": me["method"],
                                          **picks[240]["prm"]})
            print(f"[{scorer} {meth} s{seed}] EF1={me['EF1_ov']:.3f} "
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
    print(f"[{scorer} fixed-window] EF1={me['EF1_ov']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"a0_{scorer}_TE.csv"),
              index=False, encoding="utf-8-sig")
    pd.DataFrame(prms).to_csv(os.path.join(OUT, f"a0_{scorer}_TE_params.csv"),
                              index=False, encoding="utf-8-sig")
    print(df[["method", "EF1_ov", "EF1_iou01", "det_ov", "AAR_side",
              "TIA_side"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
