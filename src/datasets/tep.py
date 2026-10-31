"""TE (Tennessee Eastman) 加载器 — Rieth et al. 扩展版 TE 仿真数据.

数据来源: github.com/mv-per/tennessee-eastman-dataset (MIT), MATLAB 扩展 TE 仿真器
(ADCHEM 2015 / Rieth et al. 2017). 已下载至 DATA_ROOT/TE/:
  mode1_normal_500.xlsx : 500h 正常运行, 30001 样本, 1 min 采样 (dt=60s)
  mode1_{1,4,11}_{1,2}.xlsx : IDV(1)/IDV(4)/IDV(11) 各 2 批次, 每批 50h/3001 样本

故障注入约定 (Rieth et al. 源论文): 每个故障运行在第 8 小时注入故障并持续到结束,
故真值标签 = 故障文件内 t >= 8h 的样本. IDV(1)=A/C 进料比值阶跃, IDV(4)=反应器
冷却水入口温度阶跃, IDV(11)=反应器冷却水入口温度随机变化 — 文献常用故障.

特征: 两类文件共同变量为 41 个过程测量 XMEAS (正常文件列名误标为 xmv-N, 按位置
对齐重命名; 数值已核对一致). XMV 操纵变量仅故障文件含, 取交集弃用.

流构造: normal_500 在前 (提供 35% 训练 + 15% 验证 的纯正常段), 6 个故障文件
顺序拼接在后, 35/15/50 时间切分使测试段含正常尾段 + 全部故障段.
"""
import os
import re

import numpy as np
import pandas as pd

from common import DATA_ROOT

TE_DIR = os.path.join(DATA_ROOT, "TE")
FAULT_FILES = ["mode1_1_1", "mode1_1_2", "mode1_4_1",
               "mode1_4_2", "mode1_11_1", "mode1_11_2"]
FAULT_ONSET_H = 8.0   # Rieth et al. 约定: 第 8 小时注入
N_XMEAS = 41
XMEAS_COLS = [f"XMEAS_{i}" for i in range(1, N_XMEAS + 1)]


def _load_normal():
    df = pd.read_excel(os.path.join(TE_DIR, "mode1_normal_500.xlsx"))
    feat = [c for c in df.columns if c != "Time"]
    assert len(feat) == N_XMEAS, f"正常文件变量数异常: {len(feat)}"
    df = df.rename(columns={c: XMEAS_COLS[i] for i, c in enumerate(feat)})
    return df[XMEAS_COLS].values.astype(np.float32)


def _load_fault(name):
    df = pd.read_excel(os.path.join(TE_DIR, f"{name}.xlsx"))
    t = df["Time (h)"].values
    X = df[[c for c in df.columns if c.startswith("XMEAS")]].values.astype(np.float32)
    y = (t >= FAULT_ONSET_H).astype(np.int8)
    return X, y


def load_tep(cache=True):
    """返回 dict: X (T,41) z-scored, Y (T,) int8, dt=60s."""
    npz = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "_score_cache", "tep_data.npz"))
    if cache and os.path.exists(npz):
        d = np.load(npz)
        return {"X": d["X"], "Y": d["Y"], "dt": float(d["dt"])}

    Xs, Ys = [_load_normal()], [np.zeros(30001, dtype=np.int8)]
    for f in FAULT_FILES:
        X, y = _load_fault(f)
        Xs.append(X); Ys.append(y)
    X = np.vstack(Xs); Y = np.concatenate(Ys)
    n_tr = int(len(X) * 0.35)
    mu = X[:n_tr].mean(0, keepdims=True)
    sd = X[:n_tr].std(0, keepdims=True) + 1e-8
    X = ((X - mu) / sd).astype(np.float32)
    print(f"TE: X{X.shape} anomaly_rate={Y.mean():.4f} dt=60s "
          f"({len(FAULT_FILES)} 个故障运行, 注入于 t={FAULT_ONSET_H}h)")
    np.savez_compressed(npz, X=X, Y=Y, dt=60.0)
    return {"X": X, "Y": Y, "dt": 60.0}


if __name__ == "__main__":
    d = load_tep(cache=False)
    # 事件数与长度统计 (诚实记录用)
    from kpi.panel import events_from_binary
    ev = events_from_binary(d["Y"])
    print(f"GT 事件数: {len(ev)}, 长度(h): {[round((e - s) * 60 / 3600, 1) for s, e in ev]}")
