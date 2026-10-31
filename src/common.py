"""第9篇公共配置: 路径 / 数据根 / 35-15-50 时间切分.

数据集统一从第2篇主仓库读, 环境变量 DATA_ROOT 可覆盖.
"""
import os

# 数据主仓库 (第2篇存放). 可用环境变量 DATA_ROOT 覆盖.
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    r"D:/0科研/工作1/第2篇SCI/Contrastive_TopK_MIL/datasets")

# 第6篇 cadms 包 (打分器工具复用, 只用 scorers/eval_utils, 不碰 conformal/thresholds)
PAPER6_SRC = r"D:/0科研/工作1/第6篇SCI/src"

# 第9篇自己的分数缓存 (TE / MetroPT3 等新数据集的 iforest 分数)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_score_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 实验产出目录
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

SEED = 7


def chronological_split(n, train_frac=0.35, val_frac=0.15):
    """35/15/50 时间顺序切分 (第1篇协议, 工具性使用). 返回 (train, val, test) 索引区间."""
    n_tr = int(n * train_frac)
    n_va = int(n * val_frac)
    return slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, n)
