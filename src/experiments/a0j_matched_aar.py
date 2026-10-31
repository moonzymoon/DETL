# -*- coding: utf-8 -*-
"""A0j: matched-AAR 运行包线 (TE-iForest, 5 种子 trial 日志重评估).

科学问题 (预注册):
  layer 运行点 AAR≈0.107/h, 检出 0.60; search 自选点 AAR≈0.010/h 检出 0.20.
  同负荷 (AAR∈[0.10,0.15]) 下 search 能否达到 0.60?
  - 若能 → layer 检出优势表述收窄
  - 若不能 → 优势声明维持并引用本实验

方法: 5 种子 trial 日志的每个可行 trial, 按 a0_lib.search_unified_M 的
官方映射 (theta_hi=quantile(s_tune,q), theta_lo=theta_hi-mf*sd_tune)
恢复硬参数, 用 eval_params_M 在测试侧重算全套指标; 按 AAR 分箱 (0.02宽)
造 search 前沿. 判据箱 [0.10,0.15].
输出: results/audit/a0j_matched_aar.csv + .json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))

from a0_lib import (load_te_segments, interleave_mask, eval_params_M)  # noqa: E402

ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
SEEDS = [1, 2, 3, 4, 5]
BIN_W = 0.02
VERDICT_LO, VERDICT_HI = 0.10, 0.15


def main():
    s, y, dt, seg_bounds, seg_meta = load_te_segments()
    tune_m, test_m = interleave_mask(len(s), dt, seg_bounds)
    sd = float(s[tune_m].std())

    rows = []
    n_feas = 0
    for seed in SEEDS:
        tr = pd.read_csv(os.path.join(ROOT, 'results', 'audit',
                                      f'a0_trials_optuna_s{seed}.csv'))
        tr = tr[tr.feas == True]  # noqa: E712
        n_feas += len(tr)
        for r in tr.itertuples():
            theta_hi = float(np.quantile(s[tune_m], r.q))
            prm = {"theta_hi": theta_hi,
                   "theta_lo": theta_hi - r.mf * sd,
                   "T_on": int(r.t_on), "T_off": int(r.t_off)}
            m = eval_params_M(prm, s, y, test_m, dt, seg_bounds)
            rows.append({
                'seed': seed, 'trial': int(r.trial),
                'q': float(r.q), 'mf': float(r.mf),
                'T_on': int(r.t_on), 'T_off': int(r.t_off),
                'AAR': m['AAR_side'], 'FAR': m['FAR_side'],
                'feasible': m['feasible'],
                'EF1_ov': m['EF1_ov'], 'det_ov': m['det_ov'],
                'EF1_iou01': m['EF1_iou01'],
                'MAD_s': m['MAD_s'], 'TIA_side': m['TIA_side'],
                'n_stale_1h': m['n_stale_1h'],
            })
    df = pd.DataFrame(rows)
    print(f'feasible trials re-evaluated: {len(df)} (from {len(SEEDS)} seeds)')

    # layer 运行点 (主表)
    lp = pd.read_csv(os.path.join(ROOT, 'results', 'audit',
                                  'a0_te_iforest.csv'))
    layer = lp[lp.method == 'layer_r0'].iloc[0]
    layer_point = {'AAR': round(float(layer.AAR_side), 4),
                   'det_ov': round(float(layer.det_ov), 3),
                   'EF1_ov': round(float(layer.EF1_ov), 3),
                   'MAD_s': round(float(layer.MAD_s), 1),
                   'TIA_side': round(float(layer.TIA_side), 3),
                   'n_stale_1h': int(layer.n_stale_1h)}
    print('layer point:', layer_point)

    # AAR 分箱前沿 (全部可行 trial, 按 AAR 分箱取最好检出)
    df['aar_bin'] = (df.AAR // BIN_W) * BIN_W
    frontier = (df.sort_values('det_ov', ascending=False)
                  .groupby('aar_bin')
                  .agg(best_det=('det_ov', 'max'),
                       best_EF1=('EF1_ov', 'max'),
                       n=('det_ov', 'size'),
                       seeds=('seed', lambda x: sorted(set(x))))
                  .reset_index())

    # 判据箱
    v = frontier[(frontier.aar_bin >= VERDICT_LO - 1e-9) &
                 (frontier.aar_bin <= VERDICT_HI + 1e-9)]
    verdict = {
        'bin_range': [VERDICT_LO, VERDICT_HI],
        'best_det_in_bins': round(float(v.best_det.max()), 3) if len(v) else None,
        'layer_det': layer_point['det_ov'],
        'layer_AAR': layer_point['AAR'],
        'conclusion': None,
    }
    if verdict['best_det_in_bins'] is None:
        verdict['conclusion'] = 'no feasible search point in verdict bins'
    elif verdict['best_det_in_bins'] >= layer_point['det_ov']:
        verdict['conclusion'] = ('matched-AAR search reaches layer detection; '
                                 'detection advantage narrows')
    else:
        verdict['conclusion'] = ('matched-AAR search stays below layer '
                                 'detection; advantage holds')

    # 判据箱逐种子明细
    detail = v if len(v) else df[(df.AAR >= VERDICT_LO) & (df.AAR <= VERDICT_HI)]

    out_csv = os.path.join(ROOT, 'results', 'audit', 'a0j_matched_aar.csv')
    df.sort_values(['aar_bin', 'det_ov'], ascending=[True, False]) \
      .to_csv(out_csv, index=False)

    out = {'n_feasible_evaluated': len(df),
           'bin_width': BIN_W,
           'layer_point': layer_point,
           'frontier': frontier.to_dict('records'),
           'verdict': verdict,
           'verdict_bin_rows': detail.to_dict('records')}
    json.dump(out, open(os.path.join(ROOT, 'results', 'audit',
                                     'a0j_matched_aar.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({'frontier': out['frontier'], 'verdict': verdict},
                     ensure_ascii=False, indent=1))
    print('written', out_csv)


if __name__ == '__main__':
    main()
