"""kpi.panel 单元测试: 每个指标用手工构造的小序列验证公式."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kpi.panel import (KPIConfig, average_alarm_rate, chatter_rate,
                       event_f1_one2one, events_from_binary,
                       flood_time_fraction, kpi_panel, mean_alarm_delay,
                       rising_edges)


def test_rising_edges():
    a = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1])
    assert list(rising_edges(a)) == [1, 4, 7]
    # 序列开头即为 1 不算 rising edge (无 0->1 跳变); 面板约定: 开头告警视为已在告警态
    assert list(rising_edges(np.array([1, 1, 0]))) == []


def test_aar():
    # dt=1s, 100 点 = 1/36 h; 2 次 rising edge -> AAR = 72
    a = np.zeros(100); a[10] = 1; a[50:52] = 1
    assert abs(average_alarm_rate(a, dt=1.0) - 2 / (100 / 3600)) < 1e-9


def test_flood():
    # 窗 [t-w, t), dt=1s, w=10 点; 洪泛 = 窗内激活数 >3. 激活=单点告警(独立 rising edge)
    cfg = KPIConfig(dt=1.0, flood_window_s=10.0, flood_count=3)
    a = np.zeros(20)
    for t in [1, 3, 5, 7]:
        a[t] = 1                # 4 次独立激活
    frac = flood_time_fraction(a, cfg)
    # cnt(i)=#[i-10,i) 内边沿: i=8..11 窗含全部 4 条 -> 4 个洪泛点 / 20
    assert abs(frac - 4 / 20) < 1e-9
    # 恰好 3 条 (不超过) -> 无洪泛
    a2 = np.zeros(20)
    for t in [1, 3, 5]:
        a2[t] = 1
    assert flood_time_fraction(a2, cfg) == 0.0
    # 持续告警段只算 1 次激活, 不构成洪泛
    a3 = np.zeros(20)
    a3[1:7] = 1
    assert flood_time_fraction(a3, cfg) == 0.0
    assert flood_time_fraction(np.zeros(20), cfg) == 0.0


def test_chatter():
    cfg = KPIConfig(dt=1.0, chatter_window_s=60.0)
    # 4 条告警间隔: [10, 15, 100, 130] -> gap [5, 85, 30] 中 <=60 的占 2/3
    a = np.zeros(200)
    for t in [10, 15, 100, 130]:
        a[t] = 1
    assert abs(chatter_rate(a, cfg) - 2 / 3) < 1e-9
    assert chatter_rate(np.zeros(10), cfg) == 0.0


def test_events_from_binary():
    ev = events_from_binary(np.array([0, 1, 1, 0, 0, 1, 1, 1]))
    assert ev == [(1, 3), (5, 8)]
    assert events_from_binary(np.array([0, 0])) == []


def test_event_f1():
    gt = [(10, 20), (50, 60)]
    # 完美匹配
    assert abs(event_f1_one2one(gt, [(11, 19), (55, 61)]) - 1.0) < 1e-9
    # 全漏
    assert event_f1_one2one(gt, []) == 0.0
    # 1 对 1 命中 + 1 漏 + 1 误报 -> P=1/2 R=1/2 F1=1/2
    assert abs(event_f1_one2one(gt, [(10, 20), (90, 95)]) - 0.5) < 1e-9
    # 无重叠不算匹配
    assert event_f1_one2one(gt, [(30, 40)]) == 0.0


def test_delay():
    cfg = KPIConfig(dt=2.0)
    gt = [(10, 20)]
    a = np.zeros(30); a[13] = 1
    mad, rate = mean_alarm_delay(gt, a, cfg)
    assert mad == 3 * 2.0 and rate == 1.0
    # 事件结束之后才告警 -> 未检出
    a2 = np.zeros(30); a2[25] = 1
    mad2, rate2 = mean_alarm_delay(gt, a2, cfg)
    assert np.isnan(mad2) and rate2 == 0.0


def test_panel_smoke():
    rng = np.random.default_rng(0)
    y = np.zeros(2000); y[100:300] = 1; y[1000:1100] = 1
    a = ((rng.random(2000) < 0.05) | (y == 1)).astype(int)
    p = kpi_panel(a, y, cfg=KPIConfig(dt=5.0))
    for k in ["AAR_per_hour", "flood_time_frac", "chatter_rate",
              "mean_alarm_delay_s", "event_detection_rate", "EventF1"]:
        assert k in p
    assert 0 <= p["chatter_rate"] <= 1 and p["n_gt_events"] == 2


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all kpi tests passed")
