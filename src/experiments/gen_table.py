# -*- coding: utf-8 -*-
"""从 a0_*.csv 生成协议 M 主表 LaTeX 行 (防转录错误)."""
import os

import pandas as pd

ROOT = r"D:\0科研\工作1\第9篇SCI\results\audit"


def fmt(x, nd=3):
    if pd.isna(x):
        return "--"
    return f"{x:.{nd}f}"


def span(vals, nd=3):
    return f"{min(vals):.{nd}f}--{max(vals):.{nd}f}"


rows = []
files = [
    ("SWaT (CMH-MIL)", "a0_cmhmil_SWaT.csv", "optuna_s", "random_s"),
    ("TE (iForest)", "a0_te_iforest.csv", "optuna_s", "random_s"),
    ("TE (CMH-MIL)", "a0_cmhmil_TE.csv", "optuna_s", "random_s"),
    ("TE (USAD)", "a0_usad_TE.csv", "optuna_s", "random_s"),
    ("MetroPT3 (iForest)", "a0_iforest_MetroPT3.csv", "optuna_s", "random_s"),
]
out = []
for name, f, *_ in files:
    df = pd.read_csv(os.path.join(ROOT, f))
    df["method"] = df["method"].str.replace(r"_r\d+$", "", regex=True)
    lay = df[df.method == "layer"].iloc[0]
    opt = df[df.method.str.startswith("optuna_s")]
    rnd = df[df.method.str.startswith("random_s")]
    grd = df[df.method == "grid-delay"]
    fw = df[df.method == "fixed-window"]
    out.append(f"% ----- {name} -----")
    def emit(label, r, bold=()):
        r = r.iloc[0]
        b = lambda k: (r"\\textbf{" + fmt(r[k]) + "}") if k in bold else fmt(r[k])
        out.append(f"{label} & {b('EF1_ov')} & {fmt(r['EF1_iou01'])} & "
                   f"{fmt(r['det_ov'],2)} & {fmt(r['AAR_side'],2)} & "
                   f"{fmt(r['TIA_side'],2)} & {int(r['n_stale_1h'])} \\\\")
    emit(r"\\textbf{Layer (ours)}", df[df.method == 'layer'],
         bold=("EF1_ov",) if False else ())
    emit("Optuna (3 seeds)", opt)
    out[-1] = out[-1].replace(fmt(opt.iloc[0]['EF1_ov']),
                              span(opt['EF1_ov'].tolist()))
    emit("Random (3 seeds)", rnd)
    out[-1] = out[-1].replace(fmt(rnd.iloc[0]['EF1_ov']),
                              span(rnd['EF1_ov'].tolist()))
    if len(grd):
        emit("Grid delay", grd)
    if len(fw):
        emit("Fixed-window", fw)
    # det/AAR 的 seed 范围 (供正文)
    print(f"{name}: optuna det {span(opt['det_ov'].tolist(),2)} "
          f"AAR {span(opt['AAR_side'].tolist(),2)} | "
          f"random EF1 {span(rnd['EF1_ov'].tolist())}")

with open(os.path.join(ROOT, "a0_table_main.tex"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(out) + "\n")
print("\n".join(out))
