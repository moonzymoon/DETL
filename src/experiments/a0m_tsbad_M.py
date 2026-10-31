# -*- coding: utf-8 -*-
"""A0m: TSB-AD-MV 协议 M 重跑 (方案 A, 版本 9d).

继承 r43 的预注册纳入规则与自适应 24 块交织 (12h 块的流长自适应变体,
单段流无接缝), 替换为修正协议核心: 统一域 margin<=2sigma + 约束感知
TPE 可行优先 + 全套指标 (IoU 并报/检出/MAD/TIA/stale).
方法: layer(300ep, 3 重复) + optuna(种子 7/8/9, 预算 240 前缀).
用法: python a0m_tsbad_M.py [start] [end]   # 断点续跑
输出: results/audit/a0m_tsbad_M.csv (追加)
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))

from a0_lib import eval_params_M, train_layer_M, search_unified_M  # noqa

ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
from common import DATA_ROOT  # noqa: E402

TSB_DIR = os.path.join(DATA_ROOT, "TSB-AD-MV", "TSB-AD-M")
OUT = os.path.join(ROOT, 'results', 'audit', 'a0m_tsbad_M.csv')
DT = 1.0
W = 16
SEEDS = [7, 8, 9]


def interleave_blocks(n, nblk=24):
    idx = np.arange(n)
    blocks = np.array_split(idx, nblk)
    tr_m = np.zeros(n, dtype=bool)
    for i, b in enumerate(blocks):
        if i % 2 == 0:
            tr_m[b] = True
    return tr_m, ~tr_m


def load_stream(path):
    from sklearn.ensemble import IsolationForest
    d = pd.read_csv(path)
    X = d.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X.values.astype(np.float32)
    y = pd.to_numeric(d.iloc[:, -1], errors="coerce").fillna(0).values
    y = (y == 1).astype(np.int8)
    n_fit = int(len(X) * 0.35)
    mu = X[:n_fit].mean(0, keepdims=True)
    sd = X[:n_fit].std(0, keepdims=True) + 1e-8
    Xz = (X - mu) / sd
    ifo = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=0, n_jobs=-1).fit(Xz[:n_fit])
    s = -ifo.decision_function(Xz[n_fit:]).astype(np.float64)
    return s, y[n_fit:]


def n_events(mask):
    yy = np.asarray(mask, dtype=np.int8)
    if len(yy) == 0:
        return 0
    return int(np.sum(np.diff(np.concatenate([[0], yy, [0]])) == 1))


def run_stream(f, s, y):
    n = len(s)
    tune_m, test_m = interleave_blocks(n)
    bounds = [(0, n)]
    rows = []
    for rep in range(3):
        prm, _, _ = train_layer_M(s, y, tune_m, bounds, DT)
        m = eval_params_M(prm, s, y, test_m, DT, bounds)
        rows.append({'file': f, 'method': f'layer_r{rep}', **m})
    for seed in SEEDS:
        picks, _ = search_unified_M(s, y, tune_m, bounds, DT, seed=seed,
                                    budget=240)
        for B, r in picks.items():
            m = eval_params_M(r['prm'], s, y, test_m, DT, bounds)
            rows.append({'file': f, 'method': f'optuna_s{seed}_B{B}', **m})
    return rows


def main():
    args = [int(a) for a in sys.argv[1:]]
    start, end = (args if len(args) == 2 else (0, 10 ** 9))
    files = sorted(glob.glob(os.path.join(TSB_DIR, "*.csv")))[start:end]
    done = set(pd.read_csv(OUT)['file']) if os.path.exists(OUT) else set()
    n_excl = {"short": 0, "test_ev": 0, "train_ev": 0}
    t0 = time.time()
    for k, f in enumerate(files):
        if os.path.basename(f) in done:
            continue
        try:
            s, y = load_stream(f)
        except Exception as ex:
            print(f'[skip] {os.path.basename(f)}: {ex}', flush=True)
            continue
        if len(s) < 1000:
            n_excl["short"] += 1
            continue
        tune_m, test_m = interleave_blocks(len(s))
        if n_events(y[test_m]) < 2:
            n_excl["test_ev"] += 1
            continue
        if n_events(y[tune_m]) < 1:
            n_excl["train_ev"] += 1
            continue
        rows = run_stream(f, s, y)
        pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT),
                                  index=False)
        lay = np.mean([r['EF1_ov'] for r in rows if r['method'].startswith('layer')])
        opt = np.mean([r['EF1_ov'] for r in rows
                       if r['method'] == 'optuna_s7_B240'])
        print(f'[{k+1}/{len(files)}] {os.path.basename(f)[:36]}: '
              f'layer={lay:.3f} opt={opt:.3f} ({time.time()-t0:.0f}s)',
              flush=True)
    print('EXCLUDED:', n_excl)
    print('DONE')


if __name__ == '__main__':
    main()
