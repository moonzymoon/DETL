"""R48 (意见5): USAD 深度打分器 (忠实自实现, Audibert et al. KDD 2020).

三阶段对抗双自编码器 (MLP 窗口版, 按原文损失):
  阶段1+2 (训 AE2): ||x-D2(E2(x))||^2 + a||x - D2(E2(D1(E1(x))))||^2
  阶段3 (训 AE1):   ||x-D1(E1(x))||^2 + a||x - D1(E1(D2(E2(x))))||^2
测试分数: 两条重构链 + b 加权交叉链的误差和.
数据: TE / MetroPT3 (npz 缓存 X,Y). 管线: 窗口16 -> 分数 ->
交错切分 -> AUROC + ours + optuna. 输出: results/audit/r48_usad.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import CACHE_DIR, RESULTS_DIR, SEED
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import train_tuning
from tuning.hard_ops import hard_pipeline
from experiments.e3_baselines import _kmax, run_optuna
from experiments.e_pilot_tuning import interleave_split
from sklearn.metrics import roc_auc_score

torch.manual_seed(SEED)
np.random.seed(SEED)
W = 16
EPOCHS = 30
H = 64
BATCH = 512
A_TR = 5.0    # 训练阶段对抗权重 (原文渐增, 取中值常数)
B_TE = 5.0    # 测试分数交叉链权重


class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, H), nn.ReLU(),
                                 nn.Linear(H, H // 2))
        self.dec = nn.Sequential(nn.Linear(H // 2, H), nn.ReLU(),
                                 nn.Linear(H, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_usad(Xw, epochs=EPOCHS):
    ae1, ae2 = AE(Xw.shape[1]), AE(Xw.shape[1])
    opt1 = torch.optim.Adam(ae1.parameters(), lr=1e-3)
    opt2 = torch.optim.Adam(ae2.parameters(), lr=1e-3)
    X = torch.tensor(Xw, dtype=torch.float32)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            xb = X[perm[i:i + BATCH]]
            l1 = (xb - ae1(xb)).pow(2).mean()
            l2 = (xb - ae2(xb)).pow(2).mean()
            # 阶段 1+2: 训 AE2 (含对抗链, ae1 侧停梯度)
            opt2.zero_grad()
            la2 = l2 + A_TR * (xb - ae2(ae1(xb).detach())).pow(2).mean()
            la2.backward()
            opt2.step()
            # 阶段 3: 训 AE1 (含对抗链, ae2 侧停梯度)
            opt1.zero_grad()
            la1 = l1 + A_TR * (xb - ae1(ae2(xb).detach())).pow(2).mean()
            la1.backward()
            opt1.step()
    return ae1, ae2


def score_stream(ae1, ae2, Xw):
    X = torch.tensor(Xw, dtype=torch.float32)
    with torch.no_grad():
        e1 = (X - ae1(X)).pow(2).mean(1)
        e2 = (X - ae2(X)).pow(2).mean(1)
        c1 = (X - ae2(ae1(X))).pow(2).mean(1)
        c2 = (X - ae1(ae2(X))).pow(2).mean(1)
        return (e1 + e2 + B_TE * (c1 + c2)).numpy().astype(np.float64)


def windows(X, W=W):
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(X, W, axis=0).transpose(0, 2, 1).reshape(
        len(X) - W + 1, -1)


def run_ds(ds, X, Y, dt, rows):
    n = len(X)
    a, b = int(n * 0.35), int(n * 0.50)
    Xw_fit = windows(X[a:b])[:100000]
    ae1, ae2 = train_usad(Xw_fit)
    s = score_stream(ae1, ae2, windows(X[b:]))
    y = Y[b:][W - 1:]
    tr_m, te_m = interleave_split(len(s), dt)
    s_tr, y_tr, s_te, y_te = s[tr_m], y[tr_m], s[te_m], y[te_m]
    ev_tr = events_from_binary(y_tr)
    gt = events_from_binary(y_te)
    K_on, K_off = _kmax(dt)
    cfg = KPIConfig(dt=dt)
    au = roc_auc_score(y_te, s_te)
    print(f"[r48] {ds:8s} USAD AUROC={au:.3f} 整定段事件 {len(ev_tr)} "
          f"测试事件 {len(gt)}", flush=True)

    def panel(alarm, method):
        p = kpi_panel(alarm, y_te, cfg=cfg)
        rows.append({"dataset": ds, "scorer": "USAD", "method": method,
                     "AUROC": au, "F1": p["EventF1"],
                     "det": p["event_detection_rate"],
                     "AAR": p["AAR_per_hour"],
                     "n_ev": p["n_alarm_events"]})
        print(f"[r48] {ds:8s} USAD {method:8s}: F1={p['EventF1']:.3f} "
              f"det={p['event_detection_rate']:.2f} "
              f"AAR={p['AAR_per_hour']:.4f} n_ev={p['n_alarm_events']}",
              flush=True)

    prm = run_optuna(s_tr, y_tr, ev_tr, dt, SEED, True)
    th = np.quantile(s_tr, prm["q"])
    panel(hard_pipeline(s_te, th, th - prm["margin"], int(prm["t_on"]),
                        int(prm["t_off"])), "optuna")
    params, _ = train_tuning(s_tr, y_tr, ev_tr, dt, R_max=10.0, F_max=0.02,
                             epochs=300, seed=SEED, K_on=K_on, K_off=K_off)
    panel(hard_pipeline(s_te, params["theta_hi"], params["theta_lo"],
                        params["T_on"], params["T_off"]), "ours")


def main():
    out = os.path.join(RESULTS_DIR, "audit", "r48_usad.csv")
    rows = []
    for ds, f, dt in [("TE", "tep_data.npz", 60.0),
                      ("MetroPT3", "metropt3_data.npz", 60.0)]:
        d = np.load(os.path.join(CACHE_DIR, f))
        X, Y = d["X"].astype(np.float32), d["Y"].astype(np.int8)
        nfit = int(len(X) * .35)
        mu = X[:nfit].mean(0, keepdims=True)
        sd = X[:nfit].std(0, keepdims=True) + 1e-8
        run_ds(ds, (X - mu) / sd, Y, dt, rows)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
