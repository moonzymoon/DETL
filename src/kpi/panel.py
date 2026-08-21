"""ISA-18.2 / EEMUA 191 告警 KPI 面板 (硬规则版).

输入: 二值告警序列 alarm (T,) 0/1, 事件级真值 gt_events [(s,e),...], 采样周期 dt (秒/点).
所有指标只依赖告警流的时间结构, 与打分器无关, 供 E1-E9 与可微整定层共用.

指标与公式 (对照 ISA-18.2 / EEMUA 191 公开文献定义):
  1. AAR (Average Alarm Rate)   = N_alarm / T_hour
     N_alarm = 告警激活次数 (rising edges), T_hour = 总时长(小时).
     EEMUA 191 设计目标 ~10 条/小时/操作员; >30 为过载.
  2. Flood time fraction (洪泛时间占比)
     EEMUA 191: 10 分钟内激活 >10 条告警即为洪泛时段.
     flood(t) = 1{ #alarms in (t-10min, t] > 10 }; 指标 = mean_t flood(t).
  3. Chatter rate (震颤率)
     ISA-18.2 常用定义: 告警在短时间窗内反复触发. 取可配置窗口 T_chatter (默认 60s):
     chatter(e_i) = 1{ t(e_i) - t(e_{i-1}) <= T_chatter }; 指标 = mean_i chatter(e_i).
     注: 本文分数流为单一聚合告警源(检测器级), 按流内重复触发统计; 逐变量告警源
     的 chatter 定义相同, 仅按 tag 分组后套用本式.
  4. MAD (Mean Alarm Delay, 平均检测滞后)
     对每场真值事件 [s,e): lag_i = (首条 t>=s 的告警时刻) - s, 搜索截至事件结束;
     未检出事件不计入 lag, 单独报 detection_rate.
  5. Event-F1: 事件级 F1, 一对一匹配 (IoU>0 算匹配, 漏检零信用) — 时间重叠匹配协议.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class KPIConfig:
    dt: float = 5.0                 # 采样周期 (秒/点)
    flood_window_s: float = 600.0   # EEMUA 191 洪泛窗 10 min
    flood_count: int = 10           # 10 min 内 >10 条即洪泛
    chatter_window_s: float = 60.0  # chatter 判定窗


def rising_edges(alarm):
    """告警激活时刻索引 (0->1 的跳变)."""
    a = np.asarray(alarm)
    return np.flatnonzero((a[1:] == 1) & (a[:-1] == 0)) + 1


def average_alarm_rate(alarm, dt):
    """AAR = rising_edges / 小时数."""
    a = np.asarray(alarm)
    hours = len(a) * dt / 3600.0
    return float(len(rising_edges(a)) / hours) if hours > 0 else float("nan")


def flood_time_fraction(alarm, cfg: KPIConfig):
    """EEMUA 191 洪泛: 过去 flood_window_s 内激活数 > flood_count 的时间点占比."""
    a = np.asarray(alarm)
    edges = rising_edges(a)
    if len(edges) == 0:
        return 0.0
    w = max(1, int(round(cfg.flood_window_s / cfg.dt)))   # 窗口点数
    # 滑窗计数: 以二值边沿序列 c (激活=1) 做前缀和, 窗口 (i-w, i]
    c = np.zeros(len(a)); c[edges] = 1.0
    cs = np.concatenate([[0.0], np.cumsum(c)])
    idx = np.arange(len(a))
    lo = np.maximum(0, idx - w)
    cnt = cs[idx] - cs[lo]
    return float((cnt > cfg.flood_count).mean())


def chatter_rate(alarm, cfg: KPIConfig):
    """chatter 率 = 与上一条告警间隔 <= chatter_window_s 的告警占比."""
    edges = rising_edges(np.asarray(alarm))
    if len(edges) <= 1:
        return 0.0
    gap = np.diff(edges) * cfg.dt
    return float((gap <= cfg.chatter_window_s).mean())


def events_from_binary(b):
    """连续 1 段 -> 事件区间 [(start, end), ...], end exclusive."""
    b = np.asarray(b)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], b, [0]])))
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2])]


def event_f1_one2one(gt_events, pred_events):
    """事件级 F1: 一对一指派, IoU>0 算匹配, 漏检零信用."""
    if not gt_events:
        return float("nan")
    if not pred_events:
        return 0.0
    IoU = np.zeros((len(gt_events), len(pred_events)))
    for i, (gs, ge) in enumerate(gt_events):
        for j, (ps, pe) in enumerate(pred_events):
            inter = max(0, min(ge, pe) - max(gs, ps))
            union = max(ge, pe) - min(gs, ps)
            IoU[i, j] = inter / union if union else 0.0
    rows, cols = linear_sum_assignment(-IoU)
    tp = sum(1 for r, c in zip(rows, cols) if IoU[r, c] > 0)
    fp = len(pred_events) - tp
    fn = len(gt_events) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return float(2 * p * r / (p + r) if p + r else 0.0)


def mean_alarm_delay(gt_events, alarm, cfg: KPIConfig):
    """平均检测滞后 (秒) + 事件检出率. 搜索首条告警的范围为 [s, e)."""
    a = np.asarray(alarm)
    lags, detected = [], 0
    for s, e in gt_events:
        seg = a[s:e]
        hit = np.flatnonzero(seg == 1)
        if len(hit):
            detected += 1
            lags.append((hit[0]) * cfg.dt)   # 事件内首条告警相对事件起点的滞后
    mad = float(np.mean(lags)) if lags else float("nan")
    rate = detected / len(gt_events) if gt_events else float("nan")
    return mad, rate


def kpi_panel(alarm, y=None, gt_events=None, cfg: KPIConfig = None):
    """汇总面板. alarm: (T,) 0/1; y: 点级真值(可选, 或直接给 gt_events)."""
    cfg = cfg or KPIConfig()
    a = np.asarray(alarm)
    if gt_events is None:
        gt_events = events_from_binary(np.asarray(y)) if y is not None else []
    mad, det_rate = mean_alarm_delay(gt_events, a, cfg)
    return {
        "AAR_per_hour": average_alarm_rate(a, cfg.dt),
        "flood_time_frac": flood_time_fraction(a, cfg),
        "chatter_rate": chatter_rate(a, cfg),
        "mean_alarm_delay_s": mad,
        "event_detection_rate": det_rate,
        "n_gt_events": len(gt_events),
        "n_alarm_events": len(events_from_binary(a)),
        "EventF1": event_f1_one2one(gt_events, events_from_binary(a)),
    }
