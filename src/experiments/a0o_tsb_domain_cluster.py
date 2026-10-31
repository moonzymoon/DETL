# -*- coding: utf-8 -*-
"""A0o: TSB 域聚类 bootstrap + 域级分解 (9e).
输入: results/audit/a0m_tsbad_M.csv (120 流逐流结果)
输出: results/audit/a0o_tsb_domain.json
"""
import json, os, re
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import wilcoxon, binomtest

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

def domain_of(fname):
    base = os.path.basename(fname)
    m = re.match(r'\d+_([A-Za-z0-9]+)_', base)
    return m.group(1) if m else 'unknown'

def main():
    df = pd.read_csv(os.path.join(ROOT, 'results', 'audit', 'a0m_tsbad_M.csv'))
    rows = []
    for f, g in df.groupby('file'):
        lay_det = g[g.method.str.startswith('layer')].det_ov.mean()
        lay_f1 = g[g.method.str.startswith('layer')].EF1_ov.mean()
        o = g[g.method.str.startswith('optuna') & g.method.str.contains('B240')]
        opt_det = o.det_ov.mean(); opt_f1 = o.EF1_ov.mean()
        rows.append({'file': f, 'domain': domain_of(f),
                     'lay_det': lay_det, 'opt_det': opt_det,
                     'det_d': lay_det - opt_det,
                     'lay_f1': lay_f1, 'opt_f1': opt_f1,
                     'f1_d': lay_f1 - opt_f1})
    r = pd.DataFrame(rows)
    det = r.dropna(subset=['det_d'])
    f1 = r.dropna(subset=['f1_d'])

    rng = np.random.default_rng(0)
    dd = det.det_d.values
    stream_boots = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(10000)]
    stream_ci = np.percentile(stream_boots, [2.5, 97.5])

    domains = det.domain.unique()
    dom_means = det.groupby('domain').det_d.mean()
    dom_boots = []
    for _ in range(10000):
        picked = rng.choice(domains, size=len(domains), replace=True)
        dom_boots.append(np.mean([dom_means[d] for d in picked]))
    dom_ci = np.percentile(dom_boots, [2.5, 97.5])

    dom_table = []
    for dom, g in det.groupby('domain'):
        dom_table.append({
            'domain': dom, 'n_streams': len(g),
            'det_diff_mean': round(float(g.det_d.mean()), 3),
            'layer_wins': int((g.det_d > 0).sum()),
            'layer_loses': int((g.det_d < 0).sum()),
            'ties': int((g.det_d == 0).sum()),
        })

    w_det = wilcoxon(det.lay_det, det.opt_det).pvalue
    nw = int((det.det_d > 0).sum()); nl = int((det.det_d < 0).sum())
    from scipy.stats import binomtest
    sign_p = binomtest(nw, nw + nl, 0.5).pvalue
    w_f1 = wilcoxon(f1.lay_f1, f1.opt_f1).pvalue

    out = {
        'n_total': len(r), 'n_det_evaluable': len(det), 'n_f1': len(f1),
        'n_domains': len(domains),
        'det_diff': {
            'mean': round(float(dd.mean()), 4),
            'stream_bootstrap_ci95': [round(float(x), 4) for x in stream_ci],
            'domain_clustered_ci95': [round(float(x), 4) for x in dom_ci],
            'wilcoxon_p': round(float(w_det), 4),
            'sign_test_p': round(float(sign_p), 4),
            'layer_wins': nw, 'layer_loses': nl, 'ties': int((dd == 0).sum()),
        },
        'f1_diff': {'mean': round(float(f1.f1_d.mean()), 4),
                    'wilcoxon_p': round(float(w_f1), 6)},
        'domain_table': dom_table,
    }
    json.dump(out, open(os.path.join(ROOT, 'results', 'audit',
                                     'a0o_tsb_domain.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(out['det_diff'], ensure_ascii=False, indent=1))
    print('domains:')
    for d in sorted(dom_table, key=lambda x: -abs(x['det_diff_mean'])):
        print(f"  {d['domain']:12} n={d['n_streams']:3} det_d={d['det_diff_mean']:+.3f}")

if __name__ == '__main__':
    main()
