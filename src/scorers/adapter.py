"""打分器适配层: 统一入口 get_scores(scorer, dataset) -> (scores, labels, dt).

两类来源:
  1. cadms 打分器 (第6篇工具复用): SWaT/WADI/SMAP/MSL 的 cmhmil / AT / iforest,
     直接调 cadms.scorers.get_scorer, 缓存在第6篇 _score_cache.
     注意: cmhmil 需第2篇 checkpoint, AT 需 Anomaly-Transformer 落盘分数 —
     没有的组合会在 AVAILABILITY 里标 False 并在 E1 中跳过 (诚实记录).
  2. 本地 iforest (第9篇自实现, 配方对齐 cadms iforest_scores):
     MetroPT3 / TE — 滑窗 flatten + IsolationForest, 拟合用前 35% 弱训练段,
     打分后 50% 测试段, 分数 = -decision_function (越大越异常).

采样周期 dt (秒/点, 降采样后有效值) 用于 ISA-KPI 换算:
  SWaT: 原始 1s x downsample=5 -> 5s (第2篇 load_swat)
  WADI: 原始 1s x downsample=10 -> 10s (第2篇 load_wadi)
  SMAP/MSL: NASA 遥测 1 点/min -> 60s
  MetroPT3 / TE: 加载时从时间戳计算.
"""
import os
import sys

import numpy as np

from common import CACHE_DIR, DATA_ROOT, PAPER6_SRC

# 采样周期表 (秒/点): cmhmil/iforest 走第2篇 loaders (SWaT 降采样5x, WADI 10x);
# AT 走 Anomaly-Transformer 仓库自身的数据缓存 (原始 1s 采样, 无降采样).
DT_TABLE = {
    ("cmhmil", "SWaT"): 5.0, ("iforest", "SWaT"): 5.0, ("AT", "SWaT"): 1.0,
    ("cmhmil", "WADI"): 10.0, ("iforest", "WADI"): 10.0,
    ("cmhmil", "SMAP"): 60.0, ("iforest", "SMAP"): 60.0,
    ("cmhmil", "MSL"): 60.0, ("iforest", "MSL"): 60.0,
    ("cmhmil", "TE"): 60.0, ("cmhmil", "MetroPT3"): 5.0,   # 2026-08-15 补训 checkpoint
}


def _ensure_paper2_loaders():
    """把第9篇新数据集 (TE/MetroPT3) 注册进第2篇 LOADERS (cadms 打分需要).
    已有缓存时不触发; 注册不改第2篇文件, 仅运行时注入."""
    try:
        import data.loaders as p2l
    except ImportError:
        return
    if "TE" not in p2l.LOADERS:
        from datasets.tep import load_tep
        from datasets.metropt3 import load_metropt3
        p2l.LOADERS["TE"] = lambda: (lambda d: (d["X"], d["Y"]))(load_tep())
        p2l.LOADERS["MetroPT3"] = lambda: (lambda d: (d["X"], d["Y"]))(load_metropt3())


def cadms_scores(scorer, dataset, seed=7):
    """包装 cadms 打分器. scorer in {cmhmil, AT, iforest}."""
    if PAPER6_SRC not in sys.path:
        sys.path.insert(0, PAPER6_SRC)
    from cadms.scorers import get_scorer
    _ensure_paper2_loaders()
    kw = {"seed": seed} if scorer == "cmhmil" else {}
    s, y = get_scorer(scorer, dataset, **kw)
    return s.astype(np.float64), y.astype(np.int8), DT_TABLE[(scorer, dataset)]


def local_iforest(X, Y, window=16, seed=0, cache_key=None):
    """滑窗 flatten + IsolationForest (配方同 cadms iforest_scores).
    拟合: 前 35% 弱训练段; 打分: 后 50% 测试段 (35/15/50 协议).
    窗口标签对齐窗尾 (与 cadms 一致: yw = Yt[window-1:])."""
    if cache_key is not None:
        npz = os.path.join(CACHE_DIR, f"{cache_key}.npz")
        if os.path.exists(npz):
            d = np.load(npz)
            return d["scores"], d["labels"], float(d["dt"])
    from sklearn.ensemble import IsolationForest

    n = len(X)
    a = int(n * 0.35)
    b = int(n * 0.50)
    Xt, Yt = X[b:], Y[b:]
    N = len(Xt) - window + 1
    idx = np.arange(window)[None, :] + np.arange(N)[:, None]      # (N, window)
    feats = Xt[idx].reshape(N, -1)                                 # (N, window*D)
    idx_w = np.arange(window)[None, :] + np.arange(len(X[a:b]) - window + 1)[:, None]
    featW = X[a:b][idx_w].reshape(len(idx_w), -1)
    ifo = IsolationForest(n_estimators=100, contamination="auto",
                          random_state=seed, n_jobs=-1)
    ifo.fit(featW)
    scores = -ifo.decision_function(feats)                         # 越大越异常
    labels = Yt[window - 1:]
    if cache_key is not None:
        np.savez(npz, scores=scores, labels=labels, dt=0.0)        # dt 由调用方补
    return scores, labels, None


def get_scores(scorer, dataset, seed=7):
    """统一入口. 返回 (scores:(T,), labels:(T,), dt:float 秒/点)."""
    if (scorer, dataset) in DT_TABLE:
        return cadms_scores(scorer, dataset, seed=seed)
    if scorer == "iforest" and dataset == "MetroPT3":
        from datasets.metropt3 import load_metropt3
        d = load_metropt3()
        s, y, _ = local_iforest(d["X"], d["Y"], cache_key="iforest_MetroPT3")
        _patch_dt("iforest_MetroPT3", d["dt"])
        return s.astype(np.float64), y.astype(np.int8), d["dt"]
    if dataset == "TE":
        from datasets.tep import load_tep
        d = load_tep()
        s, y, _ = local_iforest(d["X"], d["Y"], cache_key="iforest_TE")
        _patch_dt("iforest_TE", d["dt"])
        return s.astype(np.float64), y.astype(np.int8), d["dt"]
    raise ValueError(f"未知数据集 {dataset}")


def _patch_dt(cache_key, dt):
    """把加载器算出的 dt 写回分数缓存, 保证下次读缓存 dt 不丢."""
    npz = os.path.join(CACHE_DIR, f"{cache_key}.npz")
    d = dict(np.load(npz))
    if d.get("dt", 0.0) == 0.0:
        d["dt"] = dt
        np.savez(npz, **d)


# E1 可用性矩阵: (scorer, dataset) -> 是否可跑 (缺 checkpoint/分数即 False, 跳过并记录)
AVAILABILITY = {
    ("cmhmil", "SWaT"): True, ("AT", "SWaT"): True, ("iforest", "SWaT"): True,
    ("cmhmil", "WADI"): os.path.exists(r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/results/checkpoints/WADI_seed7.pt"),
    ("AT", "WADI"): False, ("iforest", "WADI"): True,
    ("cmhmil", "SMAP"): True, ("AT", "SMAP"): False, ("iforest", "SMAP"): True,
    ("cmhmil", "MSL"): True, ("AT", "MSL"): False, ("iforest", "MSL"): True,
    ("cmhmil", "MetroPT3"): True, ("AT", "MetroPT3"): False, ("iforest", "MetroPT3"): True,
    ("cmhmil", "TE"): True, ("AT", "TE"): False, ("iforest", "TE"): True,
}
