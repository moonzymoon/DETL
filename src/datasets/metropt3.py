"""MetroPT3 加载器 (第9篇适配版, 重写自第4篇 metropt_loader 思路, 不 import 老项目).

MetroPT3: 真实地铁空压机预测维护数据 (UCI dataset 791), 15 传感器 (7 模拟+8 数字),
原始 CSV ~1517 万行, 采样约 1.4s/点. 4 个 Air Leak 故障事件 (failure report):
  Leak 1: 2020-04-18,  Leak 2: 2020-05-29~30,  Leak 3: 2020-06-05~07,  Leak 4: 2020-07-15.

处理: 降采样 (默认 5x) 后 z-score (统计只用前 35% 训练段, 防泄漏),
标签按 Air Leak 时间区间标记. 返回 dict(X, Y, dt).
"""
import os

import numpy as np
import pandas as pd

from common import DATA_ROOT

METRO_DIR = os.path.join(DATA_ROOT, "MetroPT3")

# failure report 给出的 Air Leak 事件区间 (闭区间, 精确到天)
AIR_LEAK_EVENTS = [
    ("2020-04-18 00:00", "2020-04-18 23:59"),
    ("2020-05-29 00:00", "2020-05-30 23:59"),
    ("2020-06-05 00:00", "2020-06-07 23:59"),
    ("2020-07-15 00:00", "2020-07-15 23:59"),
]


def load_metropt3(downsample=5, cache=True):
    """返回 dict: X (T,D) float32 z-scored, Y (T,) int8, dt 采样周期(秒)."""
    npz = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_score_cache",
                       "metropt3_data.npz")
    npz = os.path.normpath(npz)
    if cache and os.path.exists(npz):
        d = np.load(npz)
        return {"X": d["X"], "Y": d["Y"], "dt": float(d["dt"])}

    csvs = [f for f in os.listdir(METRO_DIR) if f.endswith(".csv")]
    assert csvs, f"MetroPT3 CSV 不存在: {METRO_DIR}"
    df = pd.read_csv(os.path.join(METRO_DIR, csvs[0]))
    ts = pd.to_datetime(df["timestamp"])
    feat_cols = [c for c in df.columns if c not in ("timestamp", df.columns[0])]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    Y = np.zeros(len(df), dtype=np.int8)
    for start, end in AIR_LEAK_EVENTS:
        Y[((ts >= start) & (ts <= end)).values] = 1
    dt_raw = float(np.median(np.diff(ts.values).astype("timedelta64[ns]").astype(np.int64)) / 1e9)

    if downsample and downsample > 1:
        X, Y, ts = X[::downsample], Y[::downsample], ts.iloc[::downsample]
    # z-score 只用前 35% (训练段) 统计
    n_tr = int(len(X) * 0.35)
    mu = X[:n_tr].mean(0, keepdims=True)
    sd = X[:n_tr].std(0, keepdims=True) + 1e-8
    X = (X - mu) / sd
    dt = dt_raw * downsample
    print(f"MetroPT3: X{X.shape} anomaly_rate={Y.mean():.4f} dt={dt:.2f}s "
          f"({ts.iloc[0]} ~ {ts.iloc[-1]})")
    np.savez_compressed(npz, X=X, Y=Y, dt=dt)
    return {"X": X, "Y": Y, "dt": dt}
