# -*- coding: utf-8 -*-
"""R3 数据补充: IoU 敏感性 + 报警时长 + fresh-run 逐 run 表."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
R = r'D:\0科研\工作1\第9篇SCI\results\audit'

# ---- 1) IoU 0.3/0.5 敏感性 (TE-iForest 主比较) ----
from a0_lib import load_te_segments, interleave_mask, hard_chain_segmented, \
    protocol_M_metrics
s, y, dt, bounds, meta = load_te_segments()
tune_m, test_m = interleave_mask(len(s), dt, bounds)
prm_df = pd.read_csv(os.path.join(R, 'a0_params.csv'))
prm_df['method'] = prm_df['method'].str.replace(r'_r\d+$', '', regex=True)
rows = []
for _, r in prm_df.iterrows():
    p = {'theta_hi': r['theta_hi'], 'theta_lo': r['theta_lo'],
         'T_on': int(r['T_on']), 'T_off': int(r['T_off'])}
    alarm = hard_chain_segmented(s, p, bounds)
    for thr in (0.3, 0.5):
        m = protocol_M_metrics(alarm, y, test_m, dt, iou_thr=thr)
        key = f'EF1_iou{int(thr*10):02d}'
        if not rows or rows[-1]['method'] != r['method']:
            rows.append({'method': r['method']})
        rows[-1][key] = m['EF1_iou01']
df3 = pd.DataFrame(rows)
df3.to_csv(os.path.join(R, 'a0_iou_sensitivity.csv'), index=False,
           encoding='utf-8-sig')
print('=== IoU 敏感性 (TE-iForest) ===')
lay = df3[df3.method == 'layer'].iloc[0]
opt = df3[df3.method.str.startswith('optuna')]
print(f'layer: IoU≥.3={lay.EF1_iou03:.3f} IoU≥.5={lay.EF1_iou05:.3f}')
print(f'optuna: IoU≥.3={opt.EF1_iou03.min():.3f}-{opt.EF1_iou03.max():.3f}'
      f' IoU≥.5={opt.EF1_iou05.min():.3f}-{opt.EF1_iou05.max():.3f}')

# ---- 2) 报警时长 (主表每流层/搜索) ----
print('\n=== 报警中位时长 (h) ===')
for f, name in [('a0_cmhmil_SWaT.csv', 'SWaT'), ('a0_te_iforest.csv', 'TE'),
                ('a0_cmhmil_TE.csv', 'TE-cm'), ('a0_usad_TE.csv', 'TE-us'),
                ('a0_iforest_MetroPT3.csv', 'Metro')]:
    df = pd.read_csv(os.path.join(R, f))
    df['method'] = df['method'].str.replace(r'_r\d+$', '', regex=True)
    lay = df[df.method == 'layer']
    opt = df[df.method.str.startswith('optuna')]
    if 'alarm_dur_med_h' in df.columns:
        l_d = lay['alarm_dur_med_h'].iloc[0] if len(lay) else float('nan')
        o_d = opt['alarm_dur_med_h'].iloc[0] if len(opt) else float('nan')
        print(f'{name:6s} layer={l_d:.1f} optuna={o_d:.1f}')

# ---- 3) fresh-run 逐 run 表 ----
dfn = pd.read_csv(os.path.join(R, 'a0_newruns.csv'))
dfd = pd.read_csv(os.path.join(R, 'a0_newruns_dilate.csv'))
# 层 vs 最强搜索 seed (整定数据选定: B240 中最优)
best_opt = dfn[dfn.method.str.startswith('optuna')].groupby(
    ['idv', 'run']).EventF1.mean().reset_index()
best_opt.columns = ['idv', 'run', 'best_opt_EF1']
lay = dfn[dfn.method == 'layer'][['idv', 'run', 'EventF1', 'det']]
tbl = lay.merge(best_opt, on=['idv', 'run']).merge(
    dfd[['idv', 'run', 'EventF1', 'det']], on=['idv', 'run'],
    suffixes=('', '_dil'))
tbl = tbl.rename(columns={'EventF1': 'layer_EF1', 'det': 'layer_det',
                          'EventF1_dil': 'dil_EF1', 'det_dil': 'dil_det'})
print('\n=== 逐 run 表 (层 vs 最强搜索均值 vs dilate) ===')
print(tbl.round(3).to_string(index=False))
tbl.to_csv(os.path.join(R, 'a0_perrun_table.csv'), index=False,
           encoding='utf-8-sig')
