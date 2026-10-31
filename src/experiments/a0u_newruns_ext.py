# -*- coding: utf-8 -*-
"""A0u: D — 新未见故障 run 前向评估 (冻结登记 §D, cohort-2).

- 打分器/常数/窗口全部冻结 (与 a0_newruns 完全一致), 不重拟合
- 输入: data/TE_ext/mode1_{3,6,8,9,12,14..20}_{1,2}.xlsx (xmv-N 列名重命名为 XMEAS-N)
- 参数集: a0_params.csv 同一过滤器 (layer_r*/optuna_s*_B240/random_s*_B240/grid-delay)
- 输出: results/audit/a0u_newruns_ext.csv (+ 每文件 SHA256 记入 a0t_new_files.csv 合并)
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from a0_lib import load_te_segments  # noqa
from kpi.panel import KPIConfig, events_from_binary, kpi_panel  # noqa
from tuning.hard_ops import hard_pipeline  # noqa
from datasets.tep import FAULT_ONSET_H, N_XMEAS  # noqa
from sklearn.ensemble import IsolationForest  # noqa

OUT = os.path.normpath(os.path.join(_HERE, "..", "..", "results", "audit"))
EXT = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "TE_ext"))
UNSEEN_EXT = [(idv, run) for idv in (3, 6, 8, 9, 12, 14, 15, 16, 17, 18, 19, 20)
              for run in (1, 2)]
WINDOW = 16


def rebuild_scorer():
    from datasets.tep import TE_DIR, FAULT_FILES
    Xs = []
    df0 = pd.read_excel(os.path.join(TE_DIR, "mode1_normal_500.xlsx"))
    feat = [c for c in df0.columns if c != "Time"]
    Xs.append(df0[feat].values.astype(np.float32))
    for f in FAULT_FILES:
        df = pd.read_excel(os.path.join(TE_DIR, f"{f}.xlsx"))
        Xs.append(df[[c for c in df.columns if c.startswith("XMEAS")]]
                  .values.astype(np.float32))
    X = np.vstack(Xs)
    n_tr = int(len(X) * 0.35)
    mu = X[:n_tr].mean(0, keepdims=True)
    sd = X[:n_tr].std(0, keepdims=True) + 1e-8
    Z = (X - mu) / sd
    a, b = int(len(Z) * 0.35), int(len(Z) * 0.50)
    idx_w = (np.arange(WINDOW)[None, :]
             + np.arange(len(Z[a:b]) - WINDOW + 1)[:, None])
    featW = Z[a:b][idx_w].reshape(len(idx_w), -1)
    ifo = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=0, n_jobs=-1).fit(featW)
    return mu, sd, ifo


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_ext_run(path, mu, sd, ifo):
    df = pd.read_excel(path)
    # 镜像源列名 xmv-N (按位 = XMEAS-N) → 统一重命名
    rename = {}
    xcols = [c for c in df.columns if c != "Time (h)"]
    assert len(xcols) == N_XMEAS, (path, len(xcols))
    for i, c in enumerate(xcols):
        rename[c] = f"XMEAS_{i+1}"
    df = df.rename(columns=rename)
    X = df[[c for c in df.columns if c.startswith("XMEAS")]].values
    t = df["Time (h)"].values
    y = (t >= FAULT_ONSET_H).astype(np.int8)
    Z = (X - mu) / sd
    N = len(Z) - WINDOW + 1
    idx = np.arange(WINDOW)[None, :] + np.arange(N)[:, None]
    feats = Z[idx].reshape(N, -1)
    s = -ifo.decision_function(feats)
    return s.astype(np.float64), y[WINDOW - 1:], 60.0


def main():
    mu, sd, ifo = rebuild_scorer()
    prm_df = pd.read_csv(os.path.join(OUT, "a0_params.csv"))
    prm_df = prm_df[prm_df["method"].str.match(
        r"layer_r\d|optuna_s\d_B240|random_s\d_B240|grid-delay")]
    rows, hashes = [], []
    for idv, run in [(i, r) for i in (3, 6, 8, 9, 12, 14, 15, 16, 17, 18, 19, 20)
                     for r in (1, 2)]:
        path = os.path.join(EXT, f"mode1_{idv}_{run}.xlsx")
        if not os.path.exists(path):
            print(f"MISSING {path}")
            continue
        hashes.append({"file": path, "idv": idv, "run": run,
                       "sha256": sha256(path)})
        s, y, dt = score_ext_run(path, mu, sd, ifo)
        ev = events_from_binary(y)
        for _, r in prm_df.iterrows():
            prm = {"theta_hi": r["theta_hi"], "theta_lo": r["theta_lo"],
                   "T_on": int(r["T_on"]), "T_off": int(r["T_off"])}
            alarm = hard_pipeline(s, prm["theta_hi"], prm["theta_lo"],
                                  int(prm["T_on"]), int(prm["T_off"]))
            m = kpi_panel(alarm, y, cfg=KPIConfig(dt=dt))
            rows.append({"idv": idv, "run": run, "method": r["method"],
                         "n_events": len(ev), "EventF1": m["EventF1"],
                         "det": m["event_detection_rate"],
                         "AAR": m["AAR_per_hour"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "a0u_newruns_ext.csv"), index=False,
              encoding="utf-8-sig")
    pd.DataFrame(hashes).to_csv(os.path.join(OUT, "a0u_file_hashes.csv"),
                                index=False, encoding="utf-8-sig")
    print(f"rows={len(df)} runs={df[['idv','run']].drop_duplicates().shape[0]}")
    # 快速摘要: layer vs optuna seed-mean
    piv = df.pivot_table(index=["idv", "run"], columns="method",
                         values=["EventF1", "det"])
    print(piv.iloc[:8, :6].round(3).to_string())


if __name__ == "__main__":
    main()
