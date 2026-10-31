"""tuning 模块单元测试: 软→硬一致性 / 硬状态机 / 梯度存在 / 训练烟测."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tuning.soft_ops import (soft_hysteresis, soft_on_delay, soft_off_delay,
                             soft_pipeline, soft_threshold)
from tuning.hard_ops import hard_delay_timer, hard_hysteresis, hard_pipeline, merge_gaps
from tuning.losses import soft_kpi
from tuning.constrained import train_tuning


def test_soft_threshold_low_temp():
    s = torch.tensor([0.0, 1.0, 2.0, 3.0])
    a = soft_threshold(s, theta=2.0, temp_s=0.01)
    assert torch.allclose(a, torch.tensor([0.0, 0.0, 0.5, 1.0]), atol=1e-2)


def test_soft_on_delay():
    # T_on=5: 持续高 -> 输出高; 单点尖峰 -> 被吞
    h_cont = torch.ones(30)
    u = soft_on_delay(h_cont, torch.tensor(5.0), temp_v=0.05, K_on=10)
    assert float(u[20]) > 0.9
    h_spike = torch.zeros(30); h_spike[10] = 1.0
    u2 = soft_on_delay(h_spike, torch.tensor(5.0), temp_v=0.05, K_on=10)
    assert float(u2[15]) < 0.2, "短于 T_on 的激活应被 on-delay 吞掉"


def test_soft_off_delay():
    # T_off=10: 单次确认后保持 ~10 点
    u = torch.zeros(40); u[5] = 1.0
    a = soft_off_delay(u, torch.tensor(10.0), temp_v=0.05, K_off=20)
    assert float(a[8]) > 0.7, "确认后应保持"
    assert float(a[25]) < 0.3, "超过 T_off 后应撤销"
    # 梯度存在性
    T = torch.tensor(10.0, requires_grad=True)
    a = soft_off_delay(u, T, temp_v=0.05, K_off=20)
    a.sum().backward()
    assert T.grad is not None and abs(float(T.grad)) > 0


def test_soft_vs_hard_hysteresis():
    rng = np.random.default_rng(0)
    s = np.sort(rng.normal(0, 1, 300))
    hi_t, lo_t = 0.5, 0.0
    soft = soft_hysteresis(torch.as_tensor(s, dtype=torch.float32), hi_t, lo_t, temp_s=0.01)
    hard = hard_hysteresis(s, hi_t, lo_t)
    assert np.mean(np.abs(soft.numpy() - hard) < 0.5) > 0.95, "低温下软硬迟滞应一致"


def test_hard_delay_timer():
    b = np.zeros(20, dtype=int)
    b[3:5] = 1      # 2 点激活 < T_on=3 -> 不通报
    b[10:16] = 1    # 6 点激活 -> t=12 通报 (持续3点后), T_off=4 -> t=19 撤销
    y = hard_delay_timer(b, T_on=3, T_off=4)
    assert y[3:12].sum() == 0, "短激活不通报"
    assert y[12] == 1 and y[16] == 1 and y[17] == 1 and y[18] == 1
    assert len(y) <= 20 and y[10] == 0  # 激活前2点未通报


def test_hard_pipeline_matches_timer():
    s = np.zeros(50); s[10:20] = 2.0; s[30] = 2.0   # 迟滞阈 1.0/0.5
    y = hard_pipeline(s, 1.0, 0.5, T_on=3, T_off=5)
    y_ref = hard_delay_timer(hard_hysteresis(s, 1.0, 0.5), 3, 5)
    assert (y == y_ref).all()


def test_merge_gaps():
    b = np.zeros(30, dtype=int)
    b[2:5] = 1; b[8:10] = 1          # 间隔 3 点 (idx5-7)
    m5 = merge_gaps(b, W=3)
    assert m5[2:10].all(), "间隔<=W 应合并"
    m1 = merge_gaps(b, W=1)
    assert m1[5:8].sum() == 0 and m1[2:5].all() and m1[8:10].all()


def test_soft_kpi_gradients():
    a = torch.rand(100, requires_grad=True)
    y = np.zeros(100, dtype=int); y[40:60] = 1
    k = soft_kpi(a, y, [(40, 60)], dt=5.0)
    total = k["miss"] + k["lag"] + k["aar"] + k["far"]
    total.backward()
    assert a.grad is not None and float(a.grad.abs().sum()) > 0


def test_train_tuning_smoke():
    rng = np.random.default_rng(7)
    s = rng.normal(0, 1, 4000)
    y = np.zeros(4000, dtype=int)
    s[1000:1100] += 3.0; y[1000:1100] = 1     # 一场强异常
    params, hist = train_tuning(s, y, [(1000, 1100)], dt=5.0, epochs=40, verbose=False)
    assert all(np.isfinite(v) for v in params.values())
    assert params["T_on"] >= 1 and params["T_off"] >= 1
    # 训练后硬链路应能检出该事件且不洪泛
    alarm = hard_pipeline(s, params["theta_hi"], params["theta_lo"],
                          params["T_on"], params["T_off"])
    assert alarm[1000:1100].any(), "训练后应能检出事件"
    from kpi.panel import KPIConfig, kpi_panel
    p = kpi_panel(alarm, y, cfg=KPIConfig(dt=5.0))
    assert p["event_detection_rate"] == 1.0
    # 损失整体应下降 (首末均值比较)
    head = np.mean([h["loss"] for h in hist[:5]])
    tail = np.mean([h["loss"] for h in hist[-5:]])
    assert tail < head




def test_on_delay_limit_semantics():
    """tau_v->0 且 tau_w 固定时, 加权 softmin 收敛到 cap-K 最小值 (非 T 窗最小值);
    这是论文 §3.2 声明的极限语义, 与硬层 (T 窗) 的差异是记录在案的松弛间隙."""
    torch = pytest.importorskip("torch") if False else None  # noqa: E501
    import torch
    from tuning.soft_ops import soft_on_delay
    K, T = 8, 3
    h = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # cap-K 最小值 = 0 (超出 T 窗的滞后存在 0) —— 极限行为
    u_lim = float(soft_on_delay(h, torch.tensor(float(T)), 1e-3, K)[-1])
    assert u_lim < 1e-2, "tau_v->0 极限应为 cap-K 最小值 (0), 而非 T 窗最小值"
    # 窗内全高 (整条 h=1) 时, 任何窗口语义都应输出 1
    h2 = torch.ones(K)
    assert abs(float(soft_on_delay(h2, torch.tensor(float(T)), 1e-3, K)[-1]) - 1.0) < 1e-3


def test_soft_hard_pipeline_agreement():
    """T_on=K_on 且 T_off=K_off (无窗口尾部) 时, 软链在低温下逐点逼近硬链."""
    import torch
    from tuning.soft_ops import soft_pipeline
    from tuning.hard_ops import hard_pipeline
    rng = np.random.default_rng(11)
    s = rng.normal(0, 1, 2000)
    s[500:560] += 3.0
    st = torch.tensor(s, dtype=torch.float32)
    th, m, Ton, Toff = 1.5, 0.5, 8, 8
    temp_s = 1e-4 * float(st.std())
    a_soft = soft_pipeline(st, th, th - m, float(Ton), float(Toff),
                           temp_s, 1e-3, K_on=Ton, K_off=Toff).numpy()
    a_hard = hard_pipeline(s, th, th - m, Ton, Toff).astype(float)
    assert np.abs(a_soft - a_hard).mean() < 0.05
    assert np.abs(a_soft - a_hard).max() < 0.35


def test_miss_surrogate_bound():
    """smax 过冲有界: 零覆盖时 d_i = ln(L)/kappa (过冲), miss = 1-ln(L)/kappa;
    lag 代理在零覆盖时恰为 1 (论文声明); miss 有下界 -ln(L)/kappa."""
    import math
    import torch
    from tuning.losses import soft_kpi
    L = 200
    a = torch.zeros(300)
    y = torch.zeros(300, dtype=torch.bool); y[50:250] = True
    out = soft_kpi(a, y, [(50, 250)], dt=5.0)
    assert abs(float(out["miss"]) - (1.0 - math.log(L) / 10.0)) < 1e-5
    assert abs(float(out["lag"]) - 1.0) < 1e-6, "零覆盖时 lag 应为最大值 1"
    a2 = torch.ones(300) * 0.5
    out2 = soft_kpi(a2, y, [(50, 250)], dt=5.0)
    # d_i = 0.5 + ln(L)/10 -> miss 为负但有下界
    assert float(out2["miss"]) > -0.7 and float(out2["miss"]) < 0.0


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tuning tests passed")
