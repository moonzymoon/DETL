# -*- coding: utf-8 -*-
"""A0k: 旧协议三节的协议 M 重跑 (方案 A, 版本 9d).

流: BATADAL (r33 缓存+确定性打分) + NoBOOM acetone/butanol (分数缓存+批界).
协议: 与主表完全一致 —— 12h 块交替掩码 + 段界状态复位 + 统一域
margin<=2sigma + 约束感知 TPE (种子 7/8/9, 预算前缀 60/120/240) + layer 300ep.
输出: results/audit/a0k_regen_M.csv (+params)
冻结数字零接触; 本脚本全部为新增 9d 数字.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', 'datasets'))

from a0_lib import (interleave_mask, eval_params_M, train_layer_M,  # noqa
                    search_unified_M)

ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
W = 16
SEEDS = [7, 8, 9]


def score_from_cached_data(X, Y):
    """与 r33 相同的确定性打分: iForest 拟合 [35,50)% 滑窗, 打分后 50%."""
    from sklearn.ensemble import IsolationForest
    n = len(X)
    a, b = int(n * 0.35), int(n * 0.50)
    idx_w = np.arange(W)[None, :] + np.arange(b - a - W + 1)[:, None]
    featW = X[a:b][idx_w].reshape(len(idx_w), -1)
    ifo = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=0, n_jobs=-1).fit(featW)
    Xt = X[b:]
    N = len(Xt) - W + 1
    idx = np.arange(W)[None, :] + np.arange(N)[:, None]
    s = -ifo.decision_function(Xt[idx].reshape(N, -1)).astype(np.float64)
    y = Y[b:][W - 1:]
    return s, y


def batadal_stream():
    npz = os.path.join(ROOT, 'results', 'audit', 'r33_batadal_data.npz')
    d = np.load(npz)
    s, y = score_from_cached_data(d['X'], d['Y'])
    return s, y, float(d['dt']), [(0, len(s))]   # 单一连续真实时间线


def noboom_bounds(which):
    """从加载器文件序列推批界 (分数流坐标, 首窗偏移 W-1)."""
    from noboom import SUBDIRS, _op_key, _exp_key
    from common import DATA_ROOT
    root = os.path.join(DATA_ROOT, SUBDIRS[which])
    if not os.path.isdir(root):
        cand = [os.path.join(ROOT, 'data', SUBDIRS[which])]
        root = next((c for c in cand if os.path.isdir(c)), None)
    te_files = sorted(glob.glob(os.path.join(root, 'operating_point_*',
                                             'test_anormal_*.csv')))
    te_files.sort(key=lambda p: (_op_key(p), _exp_key(p)))
    lens = []
    for p in te_files:
        df = pd.read_csv(p, usecols=['Time'])
        lens.append(len(df))
    # 分数流: 每批贡献 len-W+1 个窗 (拼接后连续滑窗跨批!), 纠正: 缓存分数
    # 来自拼接 X 的全局滑窗 -> 批界在分数坐标 = cumsum(lens) - (W-1)*k
    bounds, off = [], 0
    for k, L in enumerate(lens):
        off += L
        b = off - (W - 1) * (k + 1)      # 批 k+1 的首窗起点
        bounds.append(b)
    # 转成 [(s0,e0),...]
    segs, start = [], 0
    for b in bounds[:-1]:
        segs.append((start, b))
        start = b
    segs.append((start, None))
    return [(a, (b if b is not None else 10**9)) for a, b in segs], lens


def noboom_stream(which):
    npz = os.path.join(ROOT, 'src', '_score_cache', f'noboom_{which}.npz')
    d = np.load(npz)
    bounds, lens = noboom_bounds(which)
    return d['scores'].astype(np.float64), d['labels'], float(d['dt']), bounds



def interleave_blocks(n, nblk=24):
    """TSB 先例的流长自适应交织 (12h 块的变体, 1s 采样短流用)."""
    idx = np.arange(n)
    blocks = np.array_split(idx, nblk)
    tr_m = np.zeros(n, dtype=bool)
    for i, b in enumerate(blocks):
        if i % 2 == 0:
            tr_m[b] = True
    return tr_m, ~tr_m

def run_stream(name, s, y, dt, bounds, adaptive=False):
    n = len(s)
    bounds = [(a, min(b, n)) for a, b in bounds if a < n]
    if adaptive:
        tune_m, test_m = interleave_blocks(n)
    else:
        tune_m, test_m = interleave_mask(n, dt, bounds)
    rows, prows = [], []

    for rep in range(3):
        prm, _, _ = train_layer_M(s, y, tune_m, bounds, dt)
        m = eval_params_M(prm, s, y, test_m, dt, bounds)
        rows.append({'stream': name, 'method': f'layer_r{rep}', **m})
        prows.append({'stream': name, 'method': f'layer_r{rep}', **prm})
        print(f'[{name}] layer_r{rep}: EF1={m["EF1_ov"]:.3f} '
              f'det={m["det_ov"]:.2f} AAR={m["AAR_side"]:.4f}', flush=True)

    for seed in SEEDS:
        picks, _ = search_unified_M(s, y, tune_m, bounds, dt, seed=seed,
                                    budget=240)
        for B, r in picks.items():
            m = eval_params_M(r['prm'], s, y, test_m, dt, bounds)
            rows.append({'stream': name, 'method': f'optuna_s{seed}_B{B}',
                         **m})
            prows.append({'stream': name, 'method': f'optuna_s{seed}_B{B}',
                          **r['prm']})
            print(f'[{name}] optuna_s{seed}_B{B}: '
                  f'EF1={m["EF1_ov"]:.3f} det={m["det_ov"]:.2f} '
                  f'AAR={m["AAR_side"]:.4f}', flush=True)
    return rows, prows


def main():
    all_rows, all_prows = [], []
    for name, fn, adaptive in [# ('BATADAL', batadal_stream, False),
                     # ('NoBOOM_acetone', lambda: noboom_stream('acetone'), True),
                     ('NoBOOM_butanol', lambda: noboom_stream('butanol'), False)]:
        print(f'===== {name} =====', flush=True)
        s, y, dt, bounds = fn()
        print(f'stream: n={len(s)}, dt={dt}s, segments={len(bounds)}', flush=True)
        r, p = run_stream(name, s, y, dt, bounds, adaptive)
        all_rows += r
        all_prows += p
        pd.DataFrame(all_rows).to_csv(
            os.path.join(ROOT, 'results', 'audit', 'a0k_regen_M.csv'),
            index=False)
        pd.DataFrame(all_prows).to_csv(
            os.path.join(ROOT, 'results', 'audit', 'a0k_regen_M_params.csv'),
            index=False)
    print('DONE', len(all_rows), 'rows')


if __name__ == '__main__':
    main()
