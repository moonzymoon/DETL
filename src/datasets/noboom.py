"""NoBOOM 批式精馏加载器 (R60: 第五基准, 真实化工厂).

两个子数据集, 各含多个 operating_point; 每点 train_normal 实验 +
test_anormal 实验 (1s 采样). 标签列 'Label (common/all)'.
构造与 TE 同构: test_anormal 实验按 operating_point 序号拼接成一条流,
train_normal 实验用于 scorer 拟合 (与 35/15/50 检测器侧划分的约定一致:
不与打分区重叠). 返回 dict(X, Y, dt).
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from common import DATA_ROOT

LABEL_COL = "Label (common/all)"
SUBDIRS = {
    "acetone": "noboom/batch_dist_ternary_acetone+butan-1-ol+methanol",
    "butanol": "noboom/batch_dist_ternary_butan-1-ol+propan-2-ol+water",
}
_FEAT_CACHE: dict = {}


def _op_key(p):
    return int(re.search(r"operating_point_(\d+)", p).group(1))


def _exp_key(p):
    return int(re.search(r"experiment_(\d+)", p).group(1))


def _load_csv(path):
    df = pd.read_csv(path)
    y = df[LABEL_COL].fillna(0).astype(int).values
    feats = [c for c in df.columns
             if not c.startswith("Label") and c != "Time"]
    return df[feats].values.astype(np.float32), y


def noboom_stream(which="acetone"):
    if which in _FEAT_CACHE:
        return _FEAT_CACHE[which]
    _here = os.path.dirname(os.path.abspath(__file__))
    cand = [os.environ.get("NOBOOM_DIR", ""),
            os.path.join(_here, "..", "..", "data"),
            DATA_ROOT]
    root = next((os.path.join(d, SUBDIRS[which])
                 for d in cand
                 if d and os.path.isdir(os.path.join(d, SUBDIRS[which]))),
                None)
    if root is None:
        raise FileNotFoundError(
            "NoBOOM not found; set NOBOOM_DIR or place under <repo>/data/")
    tr_files = sorted(glob.glob(os.path.join(root, "operating_point_*",
                                             "train_normal_*.csv")))
    te_files = sorted(glob.glob(os.path.join(root, "operating_point_*",
                                             "test_anormal_*.csv")))
    te_files.sort(key=lambda p: (_op_key(p), _exp_key(p)))
    Xs, Ys = [], []
    for p in te_files:
        x, y = _load_csv(p)
        Xs.append(x)
        Ys.append(y)
    X = np.vstack(Xs)
    Y = np.concatenate(Ys)
    Xtr = np.vstack([_load_csv(p)[0] for p in tr_files]) \
        if tr_files else X[:100]
    out = {"X": X, "Y": Y, "X_train": Xtr, "dt": 1.0,
           "n_runs": len(te_files)}
    _FEAT_CACHE[which] = out
    return out


if __name__ == "__main__":
    for w in SUBDIRS:
        d = noboom_stream(w)
        ev = np.sum(np.diff(np.concatenate([[0], d["Y"], [0]])) != 0)
        print(f"{w}: X{d['X'].shape} events~{ev//2} runs={d['n_runs']} "
              f"anom%={d['Y'].mean():.3f} train={d['X_train'].shape}")
