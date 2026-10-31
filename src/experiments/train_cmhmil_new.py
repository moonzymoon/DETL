"""补训 cmhmil TE / MetroPT3 checkpoint (填充 E1 主表深打分器空格).

做法 (不修改第2篇任何文件, 全部外部 monkeypatch):
  1. 在 paper2 的 data.loaders.LOADERS 注册 TE/MetroPT3 (返回 (X,Y) 二元组,
     与其加载器协议一致);
  2. patch train.make_synthetic_data 指向我们的数据 (train.py 对未知名回退合成,
     回退函数被我们替换即等价于真实加载);
  3. 调 train_one_seed(cfg, 7, dataset) — 第2篇完整训练循环 (12 epoch, CPU);
  4. 训完用 cadms 接口对测试段打分并落盘缓存, 报告 AUROC.

运行: cd src && python experiments/train_cmhmil_new.py
"""
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
from common import PAPER6_SRC  # noqa: E402

PAPER2_SRC = r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/src"

sys.path.insert(0, PAPER2_SRC)
sys.path.insert(0, PAPER6_SRC)

from datasets.metropt3 import load_metropt3  # noqa: E402
from datasets.tep import load_tep  # noqa: E402
import data.loaders as p2loaders  # noqa: E402


def _xy(loader):
    d = loader()
    return d["X"], d["Y"]


p2loaders.LOADERS["TE"] = lambda: _xy(load_tep)
p2loaders.LOADERS["MetroPT3"] = lambda: _xy(load_metropt3)


def main():
    import train as p2train
    from utils.repro import load_config
    cfg = load_config(os.path.join(os.path.dirname(PAPER2_SRC), "configs", "default.yaml"))

    for ds, loader in [("TE", load_tep), ("MetroPT3", load_metropt3)]:
        print(f"\n[train] cmhmil x {ds}: 开始训练 (12 epochs, CPU)", flush=True)
        p2train.make_synthetic_data = lambda **kw: _xy(loader)
        p2train.train_one_seed(cfg, 7, ds)
        # 训完立刻打分 (cadms 接口, 分数落第6篇缓存)
        from cadms.scorers import cmh_mil_scores
        s, y = cmh_mil_scores(ds, seed=7, use_cache=False)
        from sklearn.metrics import roc_auc_score
        print(f"[train] cmhmil x {ds}: AUROC={roc_auc_score(y, s):.4f} "
              f"n={len(s)} anomaly_rate={y.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
