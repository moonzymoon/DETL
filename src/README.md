# 第9篇 src：E1/E2 实验代码

## 结构
- `common.py` — 路径 / DATA_ROOT / 35-15-50 切分
- `datasets/` — metropt3.py（MetroPT3 加载）、tep.py（TE 加载，Rieth 扩展版）
- `scorers/adapter.py` — 打分器统一入口 `get_scores(scorer, dataset) -> (scores, labels, dt)`；
  SWaT/WADI/SMAP/MSL 走第6篇 cadms（缓存于第6篇 `_score_cache/`），
  MetroPT3/TE 走本地 iforest（缓存于本目录 `_score_cache/`）
- `kpi/panel.py` — ISA-18.2/EEMUA 191 硬规则 KPI 面板（公式注释见文件头）
- `tuning/` — 可微整定层（核心）：`soft_ops.py`（软阈值/迟滞/on-off 延迟）、
  `hard_ops.py`（硬状态机，推理侧+基线）、`losses.py`（软事件损失+约束代理）、
  `constrained.py`（拉格朗日乘子训练器，温度退火）
- `experiments/` — e1_point_quality.py、e2_alarm_amplification.py、e_pilot_tuning.py、
  e3_baselines.py、e4_ablation.py、e5_soft_hard_gap.py、e6_loss_comparison.py、e8_transfer.py
- `tests/` — test_kpi.py（8 项）、test_tuning.py（9 项）

## 运行
```bash
cd src
python tests/test_kpi.py                 # KPI 单测
python tests/test_tuning.py              # 可微整定层单测
python experiments/e1_point_quality.py   # -> results/e1/
python experiments/e2_alarm_amplification.py  # -> results/e2/
python experiments/e_pilot_tuning.py     # -> results/e_pilot/ (第5周 Go/No-Go)
python experiments/e3_baselines.py       # -> results/e3/    (黑盒+经典+Gyasi基线)
python experiments/e4_ablation.py        # -> results/e4/    (消融+参数耦合轨迹)
python experiments/e5_soft_hard_gap.py   # -> results/e5/    (软硬间隙)
python experiments/e6_loss_comparison.py # -> results/e6/    (损失对比)
python experiments/e8_transfer.py        # -> results/e8/    (参数迁移)
```
依赖见 `requirements.txt`。环境变量 `DATA_ROOT` 指向数据集主仓库（默认第2篇 datasets/）。

## 红线
- 不使用/不引用 cadms 的 conformal.py、thresholds.py（与第3/6篇切割）。
- 本层不做阈值校准、不改进检测器——只做"分数流之后"的整定层研究。
