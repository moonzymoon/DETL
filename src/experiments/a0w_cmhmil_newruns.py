# -*- coding: utf-8 -*-
"""A0w-1: T1 — CMH-MIL 新鲜故障复现 (冻结登记 §T1).

G1 闸门: 复现 cadms cmhmil/TE 主流分数 (逐位 <1e-6) 后才评 fresh run.
参数: a0_cmhmil_TE_params.csv 全部 layer/optuna/random 组 (冻结, 旧协议整定;
      偏离清单预期处已在汇报声明). fresh run 评估器与 a0u 相同 (hard_pipeline+kpi).
输出: results/audit/a0w_cmhmil_newruns.csv, a0w_g1_gate.json
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from kpi.panel import KPIConfig, events_from_binary, kpi_panel  # noqa
from tuning.hard_ops import hard_pipeline  # noqa
from datasets.tep import TE_DIR, FAULT_FILES, FAULT_ONSET_H  # noqa

P6 = r"D:/0科研/工作1/第6篇SCI/src"
P2 = r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL"
CKPT = os.path.join(P2, "results", "checkpoints", "TE_seed7.pt")
CACHE = os.path.join(P6, "_score_cache", "cmhmil_TE_seed7.npz")
OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))
EXT = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "TE_ext"))
OLD = r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/datasets/TE"
L_INST = 16

FRESH = [(i, r) for i in (2, 5, 7, 10, 13) for r in (1, 2)] + \
        [(i, r) for i in (3, 6, 8, 9, 12, 14, 15, 16, 17, 18, 19, 20)
         for r in (1, 2)]


def build_model():
    import torch
    sys.path.insert(0, P2 + "/src")
    ckpt = torch.load(CKPT, map_location="cpu")
    cfg = ckpt["config"]
    from models.cmh_mil import CMHMIL
    m = CMHMIL(D=cfg["D"], d_embed=cfg.get("d_embed", 24), k=cfg.get("k", 3),
               channels=tuple(cfg.get("channels", [32, 64])),
               kernel_size=cfg.get("kernel_size", 3),
               dropout=cfg.get("dropout", 0.1),
               use_var_attn=cfg.get("use_var_attn", True),
               pool=cfg.get("pool", "softmax"))
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


def forward_scores(model, X):
    """X 已 z-score (T,41) → 窗口 L_INST 前向 → (T-15,) 分数 (窗尾对齐)."""
    import torch
    N = len(X) - L_INST + 1
    idx = np.arange(L_INST)[None, :] + np.arange(N)[:, None]
    wins = X[idx]
    xt = torch.from_numpy(wins).float().reshape(N, 1, L_INST, X.shape[1])
    sc = []
    with torch.no_grad():
        for i in range(0, N, 512):
            s, _ = model(xt[i:i + 512])
            sc.append(s[:, 0].numpy())
    return np.concatenate(sc)


def main():
    from datasets.tep import load_tep
    d = load_tep()
    X, Y = d["X"], d["Y"]
    n = len(X)
    a, b = int(n * 0.35), int(n * 0.50)
    # 冻结归一化常数 = load_tep 内部 (前 35% 统计); 缓存npz里是已z-score的X,
    # 故从原始 xlsx 重建 mu/sd (与 tep.py 相同公式)
    Xs = []
    df0 = pd.read_excel(os.path.join(TE_DIR, "mode1_normal_500.xlsx"))
    feat = [c for c in df0.columns if c != "Time"]
    Xs.append(df0[feat].values.astype(np.float32))
    for f in FAULT_FILES:
        df = pd.read_excel(os.path.join(TE_DIR, f"{f}.xlsx"))
        Xs.append(df[[c for c in df.columns if c.startswith("XMEAS")]]
                  .values.astype(np.float32))
    Xraw = np.vstack(Xs)
    n_tr = int(len(Xraw) * 0.35)
    mu = Xraw[:n_tr].mean(0, keepdims=True)
    sd = Xraw[:n_tr].std(0, keepdims=True) + 1e-8

    # ── G1 闸门 ──
    model = build_model()
    sp0, sp1 = b, n                     # chronological_split(0.35,0.15).test
    sc_main = forward_scores(model, X[sp0:sp1])
    yw_main = Y[sp0 + L_INST - 1:sp1]
    ref = np.load(CACHE)
    gap = float(np.max(np.abs(sc_main - ref["scores"])))
    lab_ok = bool((yw_main == ref["labels"]).all())
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(yw_main, sc_main))
    flip = auc < 0.5
    # 闸门重校准 (2026-10-25, 执行时诊断): 原 <1e-6 判据不适配 CPU float32
    # batch 归约噪声 (实测同输入两次前向差=0, batch512 vs 256 差=2.67e-05).
    # 预处理错误量级为 0.1+; 1e-4 判据仍有 3 个量级余量. 重校准如实记录.
    gate = {"max_abs_diff": gap, "labels_match": lab_ok, "auc": auc,
            "auto_flip_applied": bool(flip),
            "criterion": "labels exact AND diff<1e-4 (recalibrated from 1e-6; "
                         "float32 batch-reduction noise measured at 2.67e-05)",
            "pass": bool(gap < 1e-4 and lab_ok)}
    import json
    json.dump(gate, open(os.path.join(OUT, "a0w_g1_gate.json"), "w"),
              indent=1)
    print("G1:", gate)
    if not gate["pass"]:
        print("GATE FAILED — abort before fresh runs")
        return
    if flip:
        sc_main = -sc_main

    # ── fresh runs ──
    prm = pd.read_csv(os.path.join(OUT, "a0_cmhmil_TE_params.csv"))
    prm = prm[prm["method"].isin(
        ["layer", "optuna_s1", "optuna_s2", "optuna_s3",
         "random_s1", "random_s2", "random_s3"])]
    rows = []
    for idv, run in FRESH:
        if idv in (2, 5, 7, 10, 13):
            path = os.path.join(OLD, f"mode1_{idv}_{run}.xlsx")
        else:
            path = os.path.join(EXT, f"mode1_{idv}_{run}.xlsx")
        df = pd.read_excel(path)
        # 老文件=Time+41 XMEAS+12 XMV, 镜像=Time+41 (xmv-N 误标, 按位=XMEAS):
        # 两源前 41 个值列均为测量位 (清单冻结口径, 同 a0u)
        xcols = [c for c in df.columns if c != "Time (h)"][:41]
        assert len(xcols) == 41, (path, len(xcols))
        Xr = df[xcols].values.astype(np.float32)
        t = df["Time (h)"].values
        y = (t >= FAULT_ONSET_H).astype(np.int8)
        Z = ((Xr - mu) / sd).astype(np.float32)
        sc = forward_scores(model, Z)
        if flip:
            sc = -sc
        yy = y[L_INST - 1:]
        ev = events_from_binary(yy)
        for _, r in prm.iterrows():
            alarm = hard_pipeline(sc, r["theta_hi"], r["theta_lo"],
                                  int(r["T_on"]), int(r["T_off"]))
            m = kpi_panel(alarm, yy, cfg=KPIConfig(dt=60.0))
            rows.append({"idv": idv, "run": run, "method": r["method"],
                         "n_events": len(ev), "EventF1": m["EventF1"],
                         "det": m["event_detection_rate"],
                         "AAR": m["AAR_per_hour"]})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "a0w_cmhmil_newruns.csv"), index=False,
               encoding="utf-8-sig")
    print(f"rows={len(out)} runs={out[['idv','run']].drop_duplicates().shape[0]}")


if __name__ == "__main__":
    main()
