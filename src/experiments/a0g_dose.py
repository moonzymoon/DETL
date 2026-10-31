# -*- coding: utf-8 -*-
"""方案二增强版: 事件数剂量-响应实验.
6事件chronological tuning, 对每个k∈{1,2,3,4,5,6}, 随机抽3个子集,
每个子集独立训练+10个未见run测试. 分离事件数作为唯一变量."""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from a0_lib import load_te_segments
from a0_newruns import rebuild_scorer, score_run, UNSEEN, TE_DIR, OUT
from kpi.panel import KPIConfig, events_from_binary, kpi_panel
from tuning.constrained import TuningParams
from tuning.losses import soft_kpi
from tuning.hard_ops import hard_pipeline

torch.set_default_dtype(torch.float64)
R = r'D:\0科研\工作1\第9篇SCI\results\audit'


def train_subset(s, y, ev_subset, dt, bounds, epochs=300):
    """在给定事件子集上训练 (其余事件标签清零)."""
    y2 = y.copy()
    ev_set = set(ev_subset)
    for s0, e0 in events_from_binary(y):
        if (s0, e0) not in ev_set:
            y2[s0:e0] = 0
    ev2 = events_from_binary(y2)

    s_t = torch.as_tensor(s)
    y_t = torch.as_tensor(y2)
    sd = float(np.std(s))
    model = TuningParams(s_t, K_on=24, K_off=48, m_max=2.0 * sd)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=0.05)
    mu = np.array([1.0, 1.0])
    for ep in range(epochs):
        tf = max(0.05, 1.0 * (0.985 ** ep))
        tv = max(0.02, 0.15 * (0.985 ** ep))
        opt.zero_grad()
        a_full = torch.cat([model(s_t[s0:e0], tf, tv) for s0, e0 in bounds])
        k = soft_kpi(a_full, y_t, ev2, dt)
        constraints_on = tf <= 0.3 or not ev2
        with torch.no_grad():
            viol = [max(0, float(k['aar']) - 10.0) / 10.0,
                    max(0, float(k['far']) - 0.02) / 0.02]
        pen = mu[0] * torch.relu(k['aar'] - 10.0) / 10.0 + \
              mu[1] * torch.relu(k['far'] - 0.02) / 0.02
        loss = k['miss'] + 0.5 * k['lag'] + (pen if constraints_on else 0.0)
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        else:
            opt.zero_grad()
        if constraints_on:
            mu = np.minimum(mu * np.exp(0.05 * np.array(viol)), 100.0)
    return model.hard_params()


def main():
    s, y, dt, bounds, meta = load_te_segments()
    ev_all = events_from_binary(y)
    print(f'主流: {len(ev_all)} 个事件, {len(bounds)} 段')

    mu_s, sd_s, ifo = rebuild_scorer()
    test_runs = []
    for idv, run in UNSEEN:
        s_, y_, dt_ = score_run(os.path.join(TE_DIR, f'mode1_{idv}_{run}.xlsx'),
                                mu_s, sd_s, ifo)
        test_runs.append((idv, run, s_, y_, dt_))

    rng = np.random.default_rng(42)
    rows = []

    # k=6 就是全部事件 (1 个子集)
    # k=1..5 每个抽 3 个随机子集
    for k in range(1, 7):
        if k == 6:
            subsets = [tuple(range(6))]
            n_sub = 1
        elif k == 5:
            # C(6,5)=6 个子集, 全跑
            subsets = [tuple(i for i in range(6) if i != j) for j in range(6)]
            n_sub = 6
        else:
            # 每个k抽3个子集
            all_idx = list(range(6))
            subsets = set()
            while len(subsets) < 3:
                sub = tuple(sorted(rng.choice(all_idx, size=k, replace=False)))
                subsets.add(sub)
            subsets = list(subsets)
            n_sub = len(subsets)

        for si, subset in enumerate(subsets):
            ev_subset = [ev_all[i] for i in subset]
            torch.manual_seed(7)
            prm = train_subset(s, y, ev_subset, dt, bounds)
            f1s, dets = [], []
            for idv, run, s_, y_, dt_ in test_runs:
                alarm = hard_pipeline(s_, prm['theta_hi'], prm['theta_lo'],
                                      int(prm['T_on']), int(prm['T_off']))
                p = kpi_panel(alarm, y_, cfg=KPIConfig(dt=dt_))
                f1s.append(p['EventF1'])
                dets.append(p['event_detection_rate'])
            rows.append({
                'k': k, 'subset_id': si, 'subset': str(subset),
                'unseen_EF1': np.mean(f1s), 'unseen_det': np.mean(dets),
                'margin_std': (prm['theta_hi']-prm['theta_lo'])/np.std(s)
            })
            print(f'  k={k} sub{si} {subset}: EF1={np.mean(f1s):.3f} '
                  f'det={np.mean(dets):.2f}')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R, 'a0g_dose_response.csv'), index=False,
              encoding='utf-8-sig')

    # 按 k 汇总
    summary = df.groupby('k').agg(
        EF1_mean=('unseen_EF1', 'mean'),
        EF1_min=('unseen_EF1', 'min'),
        EF1_max=('unseen_EF1', 'max'),
        det_mean=('unseen_det', 'mean'),
        det_min=('unseen_det', 'min'),
        det_max=('unseen_det', 'max'),
        n_subsets=('EF1_mean', 'count'))
    print('\n===== 剂量-响应汇总 =====')
    print(summary.round(3).to_string())


if __name__ == '__main__':
    main()
