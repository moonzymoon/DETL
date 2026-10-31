# -*- coding: utf-8 -*-
"""A0i: 机理一致性标注 (lite 版, 9c).

数据: results/audit/a0f_attribution_full.csv (5 事件 x top-3 |z| 变量)
依据: Downs & Vogel (1993) 故障机理表 (MECH 字典, 三档分类)
输出: results/audit/a0i_mechconsistency.csv + .json
"""
import json
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))

# ---- D&V 1993 机理表 (xmeas 编号 1-41; DIRECT=故障直接作用变量,
#      CASCADE=文献记载的下游级联变量, 经气相回路/分离链传播) ----
MECH = {
    1: {   # A/C 进料比阶跃 (stream 4), B 组分恒定
        'name': 'A/C feed ratio step',
        'direct':      [1, 4, 23, 24, 25],
        'cascade':     [5, 6, 10, 12, 13, 14, 15, 16, 17, 18, 20,
                        29, 30, 31, 37, 38],
    },
    4: {   # 反应器冷却水入口温度阶跃
        'name': 'Reactor cooling-water inlet temp step',
        'direct':      [9, 21],
        'cascade':     [8, 11, 13, 16, 18, 19, 20, 23, 24, 25, 26, 27,
                        28, 29, 30, 31, 32, 33, 34, 35, 36, 41],
    },
    11: {  # 反应器冷却水入口温度随机波动
        'name': 'Reactor cooling-water inlet temp random variation',
        'direct':      [9, 21],
        'cascade':     [8, 11, 18, 20, 23, 24, 25, 26, 27, 28, 29, 30,
                        31, 41],
    },
}

def tier(idv, var):
    m = MECH[int(idv)]
    if var in m['direct']:
        return 'direct'
    if var in m['cascade']:
        return 'cascade'
    return 'unexplained'

def main():
    df = pd.read_csv(os.path.join(ROOT, 'results', 'audit',
                                  'a0f_attribution_full.csv'),
                        encoding='utf-8-sig')
    df['tier'] = [tier(int(r['idv']), int(r['var']))
                  for _, r in df.iterrows()]

    # ---- 经验佐证: 该 IDV 事件窗内全体 41 变量的实测 top-5 偏移 ----
    d = np.load(os.path.join(ROOT, 'src', '_score_cache', 'tep_data.npz'))
    X, Y = d['X'], d['Y']          # X z-scored, Y int8
    segs, p = [], 30001
    idvs = [1, 4, 11, 1, 4, 11]    # FAULT_FILES 顺序
    for idv in idvs:
        segs.append((p, p + 3001, idv)); p += 3001
    emp = {}
    for a0, b0, idv in segs:
        top5 = (np.argsort(-np.abs(X[a0:b0]).mean(axis=0))[:5] + 1).tolist()
        emp.setdefault(idv, []).append(top5)

    out = {'mech_table': {k: v['name'] for k, v in MECH.items()}}
    rates = {}
    for ev, g in df.groupby('event'):
        r = (g['tier'] != 'unexplained').mean()
        rates[int(ev)] = {'idv': int(g['idv'].iloc[0]),
                          'detected': bool(g['detected'].iloc[0]),
                          'consistent_frac': round(float(r), 3),
                          'tiers': g['tier'].tolist()}
    out['per_event'] = rates
    out['overall_consistent_frac'] = round(
        float((df['tier'] != 'unexplained').mean()), 3)
    det = df[df['detected']]
    mis = df[~df['detected']]
    out['consistent_frac_detected'] = round(
        float((det['tier'] != 'unexplained').mean()), 3) if len(det) else None
    out['consistent_frac_missed'] = round(
        float((mis['tier'] != 'unexplained').mean()), 3) if len(mis) else None
    out['empirical_top5_per_idv'] = {k: v for k, v in emp.items()}
    cross = {}
    for idv, runs in emp.items():
        inter = set().union(*[set(r) for r in runs]) & set(MECH[idv]['direct'])
        cross[idv] = sorted(inter)
    out['empirical_direct_overlap'] = cross

    df.to_csv(os.path.join(ROOT, 'results', 'audit',
                           'a0i_mechconsistency.csv'), index=False)
    json.dump(out, open(os.path.join(ROOT, 'results', 'audit',
                                     'a0i_mechconsistency.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    print(df.to_string())
    print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == '__main__':
    main()
