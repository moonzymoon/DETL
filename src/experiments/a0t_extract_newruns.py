# -*- coding: utf-8 -*-
"""A0t-1: D — Rieth RData 结构探测 + 新未见故障 run 提取.

步骤:
  1) 探测 TEP_Faulty_Testing.RData 结构 (长表/宽表, run 长, 采样, 故障起始)
  2) 若结构匹配本仓库 TE 口径 (50h@1min, 故障 8h 起, XMEAS 列), 提取
     IDV {3,6,8,9,12,14,15,16,17,18,19,20} x run{1,2} → data/TE_ext/mode1_{idv}_{run}.xlsx
     (列名/时间轴与现有 17 文件同构); 记 SHA256.
若结构不匹配 → 只打印结构, 不写任何文件 (回 DTU 方案再议).
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "audit"))
DST = os.path.normpath(os.path.join(HERE, "..", "..", "data", "TE_ext"))

TARGET_IDV = [3, 6, 8, 9, 12, 14, 15, 16, 17, 18, 19, 20]
RUNS = [1, 2]


def probe(path):
    import pyreadr
    r = pyreadr.read_r(path)
    print("objects:", list(r.keys()))
    for k, v in r.items():
        print(f"  {k}: shape={v.shape}")
        print("  cols:", list(v.columns)[:16], "..." if v.shape[1] > 16 else "")
        print(v.head(3).iloc[:, :6])
        for cand in ("faultNumber", "fault", "Fault", "attack", "sim",
                     "simulationRun", "run", "sample", "time", "Time"):
            if cand in v.columns:
                u = pd.unique(v[cand])
                print(f"  {cand}: n_unique={len(u)}, sample={u[:12]}")
        return v
    return None


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_long(df, idv_col, run_col, sample_col, xmeas_cols, xmv_cols=None):
    """Rieth 2018 长表: faultNumber/simulationRun/sample + xmeas_*/xmv_*."""
    recs = []
    for idv in TARGET_IDV:
        sub = df[df[idv_col] == idv]
        if len(sub) == 0:
            print(f"  IDV {idv}: absent")
            continue
        runs = sorted(pd.unique(sub[run_col]))
        print(f"  IDV {idv}: {len(runs)} runs available, len/run="
              f"{sub.groupby(run_col).size().iloc[0]}")
        for run in RUNS:
            if run > len(runs):
                continue
            g = sub[sub[run_col] == runs[run - 1]].sort_values(sample_col)
            data = {f"XMEAS-{i+1}": g[c].values
                    for i, c in enumerate(xmeas_cols)}
            if xmv_cols:
                for i, c in enumerate(xmv_cols):
                    data[f"XMV-{i+1}"] = g[c].values
            n = len(g)
            out = pd.DataFrame({"Time (h)": np.arange(n) / 60.0, **data})
            fp = os.path.join(DST, f"mode1_{idv}_{run}.xlsx")
            out.to_excel(fp, index=False)
            recs.append({"file": fp, "idv": idv, "run": run,
                         "rows": n, "sha256": sha256(fp)})
    pd.DataFrame(recs).to_csv(os.path.join(OUT, "a0t_new_files.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"extracted {len(recs)} files -> a0t_new_files.csv")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        DST, "TEP_Faulty_Testing.RData")
    df = probe(src)
    if df is None or "--probe-only" in sys.argv:
        sys.exit(0)
    idv_col = next((c for c in ("faultNumber", "fault", "Fault", "attack")
                    if c in df.columns), None)
    run_col = next((c for c in ("simulationRun", "run", "sim")
                    if c in df.columns), None)
    smp_col = next((c for c in ("sample", "time", "Time")
                    if c in df.columns), None)
    xm = [c for c in df.columns if str(c).lower().startswith("xmeas")]
    xv = [c for c in df.columns if str(c).lower().startswith("xmv")]
    print(f"idv_col={idv_col} run_col={run_col} sample_col={smp_col} "
          f"n_xmeas={len(xm)} n_xmv={len(xv)}")
    if idv_col and run_col and smp_col and len(xm) >= 41:
        extract_long(df, idv_col, run_col, smp_col, xm[:41],
                     xv[:12] if len(xv) >= 12 else (xv or None))
    else:
        print("STRUCTURE MISMATCH — no files written")
