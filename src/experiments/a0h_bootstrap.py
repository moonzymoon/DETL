# -*- coding: utf-8 -*-
"""A0h: 新鲜未见故障 run 的事件级 bootstrap 置信区间 (CERD 转投新增统计).

数据源: results/audit/a0_perrun_table.csv (十个 IDV 2/5/7/10/13 各两 run).
输出:   results/audit/a0h_bootstrap.json
协议:   run 级 bootstrap (B=10000, seed=0), 配对差同 resample;
        符号检验与 Fisher 精确检验如实报告 (n 小, 仅方向性处标注).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact

rng = np.random.default_rng(0)
B = 10000

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
df = pd.read_csv(os.path.join(ROOT, 'results', 'audit', 'a0_perrun_table.csv'))


def boot_ci(x):
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), (B, len(x)))
    means = x[idx].mean(axis=1)
    return {'mean': round(float(x.mean()), 3),
            'ci95': [round(float(np.percentile(means, 2.5)), 3),
                     round(float(np.percentile(means, 97.5)), 3)]}


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    idx = rng.integers(0, len(d), (B, len(d)))
    means = d[idx].mean(axis=1)
    return {'mean_diff': round(float(d.mean()), 3),
            'ci95': [round(float(np.percentile(means, 2.5)), 3),
                     round(float(np.percentile(means, 97.5)), 3)]}


out = {
    'protocol': 'run-level bootstrap, B=10000, seed=0, per-run table a0_perrun_table.csv',
    'layer_eventF1': boot_ci(df['EventF1']),
    'layer_detection': boot_ci(df['det']),
    'best_search_eventF1': boot_ci(df['best_opt_EF1']),
    'paired_layer_minus_best_search_EF1': paired(df['EventF1'], df['best_opt_EF1']),
    'paired_layer_det_minus_040': paired(df['det'], [0.4] * 10),
    'wide_label_detection': boot_ci(df['dil_det']),
    'paired_wide_label_minus_best_search_EF1': paired(df['dil_EF1'], df['best_opt_EF1']),
    'sign_test_det_gt_040': {'wins': int((df['det'] > 0.4).sum()), 'n': 10,
                             'p': round(float(binomtest(
                                 int((df['det'] > 0.4).sum()), 10, 0.5).pvalue), 4)},
    'fisher_te_iforest_det_3of5_vs_1of5': {
        'note': 'n=5 events, directional only',
        'p': round(float(fisher_exact([[3, 2], [1, 4]])[1]), 3)},
}
json.dump(out, open(os.path.join(ROOT, 'results', 'audit', 'a0h_bootstrap.json'), 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
