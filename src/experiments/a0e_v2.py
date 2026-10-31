# -*- coding: utf-8 -*-
"""修复事件丰富实验两个 bug + 重跑.
Bug1: drop 逻辑先清空全部事件再删——导致 drop 25%/50% 都是全删
Bug2: 训练没用段边界——跨独立仿真 run 传播状态"""
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
torch.manual_seed(7)
np.random.seed(7)

R = r'D:\0科研\工作1\第9篇SCI\results\audit'


def perturb_events_fixed(y, events, kind, frac, rng):
    """修复版: drop 不先清空全部, 只删指定比例的事件."""
    y2 = y.copy()
    if kind == 'drop':
        k = max(1, int(round(len(events) * abs(frac))))
        idx = rng.choice(len(events), size=min(k, len(events)), replace=False)
        for i in idx:
            y2[events[i][0]:events[i][1]] = 0
        return y2
    for s0, e0 in events:
        L = e0 - s0
        d = int(round(L * abs(frac)))
        y2[s0:e0] = 0
        if kind == 'shift':
            d2 = int(round(L * frac))
            a = max(0, s0 + d2)
            b = min(len(y2), a + L)
            y2[a:b] = 1
        elif kind == 'dilate':
            a, b = max(0, s0 - d), min(len(y2), e0 + d)
            y2[a:b] = 1
        elif kind == 'erode':
            a, b = s0 + d, e0 - d
            if b > a:
                y2[a:b] = 1
    return y2


def train_segmented(s, y, events, dt, bounds, epochs=300):
    """段边界复位训练 (修复 Bug2)."""
    s_t = torch.as_tensor(s)
    y_t = torch.as_tensor(y)
    sd = float(np.std(s))
    model = TuningParams(s_t, K_on=24, K_off=48, m_max=2.0 * sd)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=0.05)
    mu = np.array([1.0, 1.0])
    normal = y == 0
    nm = torch.as_tensor(normal)
    for ep in range(epochs):
        tf = max(0.05, 1.0 * (0.985 ** ep))
        tv = max(0.02, 0.15 * (0.985 ** ep))
        opt.zero_grad()
        # 段边界复位软扫描 (与 a0_lib 协议 M 一致)
        a_full = torch.cat([
            model(s_t[s0:e0], tf, tv) for s0, e0 in bounds])
        k = soft_kpi(a_full, y_t, events, dt)
        constraints_on = tf <= 0.3 or not events
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
    print(f'主流: {len(s)} 样本, {len(ev_all)} 个 GT 事件, '
          f'{len(bounds)} 段 (段界复位)')

    mu_s, sd_s, ifo = rebuild_scorer()
    test_runs = []
    for idv, run in UNSEEN:
        s_, y_, dt_ = score_run(os.path.join(TE_DIR, f'mode1_{idv}_{run}.xlsx'),
                                mu_s, sd_s, ifo)
        test_runs.append((idv, run, s_, y_, dt_))

    rng = np.random.default_rng(11)
    configs = [('none', 0.0)] + \
              [('shift', f) for f in (0.05, -0.05, 0.10, -0.10)] + \
              [('dilate', f) for f in (0.10, 0.20)] + \
              [('erode', f) for f in (0.10, 0.20)] + \
              [('drop', f) for f in (0.25, 0.50)]

    rows = []
    for kind, frac in configs:
        if kind == 'none':
            y_tune = y
        else:
            y_tune = perturb_events_fixed(y, ev_all, kind, frac, rng)
        ev_tune = events_from_binary(y_tune)
        print(f'[{kind} {frac:+.2f}] tune events={len(ev_tune)}', end=' ')

        prm = train_segmented(s, y_tune, ev_tune, dt, bounds)
        m = prm['theta_hi'] - prm['theta_lo']

        f1s, dets, aars, fars = [], [], [], []
        for idv, run, s_, y_, dt_ in test_runs:
            alarm = hard_pipeline(s_, prm['theta_hi'], prm['theta_lo'],
                                  int(prm['T_on']), int(prm['T_off']))
            p = kpi_panel(alarm, y_, cfg=KPIConfig(dt=dt_))
            f1s.append(p['EventF1'])
            dets.append(p['event_detection_rate'])
            aars.append(p['AAR_per_hour'])
            fars.append(float(alarm[y_ == 0].mean()))
        row = {'kind': kind, 'frac': frac, 'n_tune_events': len(ev_tune),
               'margin_std': m / float(np.std(s)),
               'unseen_EF1_mean': np.mean(f1s),
               'unseen_det_mean': np.mean(dets),
               'unseen_AAR_mean': np.mean(aars),
               'unseen_FAR_mean': np.mean(fars),
               'feasible': bool(np.mean(aars) <= 10 and np.mean(fars) <= 0.02)}
        rows.append(row)
        print(f'margin={m/np.std(s):.2f}σ unseen: EF1={np.mean(f1s):.3f} '
              f'det={np.mean(dets):.2f} AAR={np.mean(aars):.3f} '
              f'FAR={np.mean(fars):.4f} feas={row["feasible"]}')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R, 'a0e_eventrich_labels.csv'), index=False,
              encoding='utf-8-sig')
    print(f'\narchived (fixed): {R}/a0e_eventrich_labels.csv')
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
