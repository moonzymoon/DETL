"""约束优化训练器: min 漏报 + 检测滞后  s.t.  AAR <= R_max, FAR <= F_max.

方案 §4.1 ④: 乘子法 (非固定加权和) —— 拉格朗日
  L = miss + λ_lag·lag + μ_AAR·relu(AAR - R_max)/R_max + μ_FAR·relu(FAR - F_max)/F_max
乘子对偶上升: μ ← μ·exp(η_c·violation) (乘性更新, 上限 100, 违约越大乘子越重).

防跑飞设计 (方案 §4.2 预案: "合理初始化 + 温度调度"):
  - θ 用分位数边界 sigmoid 参数化, 锁在 [q50, q99.9] 内 —— 分数远离阈值时
    sigmoid 饱和、漏报梯度消失, θ 无约束漂移会把事件全吞 (实测踩过);
  - 分数域温度 temp_s = temp_frac·std(s), temp_frac 从 1.0 退火到 0.05
    (温度与分数尺度解耦); 值域温度 temp_v 独立从 0.15 退火到 0.02.

参数 (全部可微学习): θ_hi, margin (θ_lo = θ_hi - margin, softplus 正定),
T_on, T_off (连续窗长, sigmoid 压到 [1, K]); 推理时取整走硬算子.
"""
import math

import numpy as np
import torch

from .losses import soft_kpi
from .soft_ops import soft_pipeline


def _logit(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def _inv_softplus(x):
    return math.log(math.expm1(max(x, 1e-6)))


class TuningParams(torch.nn.Module):
    """可微整定参数组. K_on/K_off 为窗长上限(点数), 按数据集 dt 配置."""

    def __init__(self, s_train, init_q=0.90, K_on=24, K_off=48,
                 init_T_on=4.0, init_T_off=8.0, tia_box=None, m_max=None):
        super().__init__()
        s = torch.as_tensor(s_train, dtype=torch.float32)
        # tia_box: 占空比下界 —— θ 箱下端从 q50 抬到 q_{1-tia_box}, 使整定段
        # 点级占空比 <= tia_box 按构造成立 (常设告警不可行域变体).
        q_lo_prob = 0.50 if tia_box is None else min(max(1.0 - tia_box, 0.50), 0.999)
        self.q_lo = float(torch.quantile(s, q_lo_prob))
        self.q_hi = float(torch.quantile(s, 0.999))
        self.m_max = m_max          # 死区上限 (分数单位); 防迟滞锁存常设告警
        self.std = float(s.std()) + 1e-6
        init = float(torch.quantile(s, init_q))
        frac = (init - self.q_lo) / max(self.q_hi - self.q_lo, 1e-6)
        self._theta_raw = torch.nn.Parameter(torch.tensor(_logit(frac)))
        self._margin_raw = torch.nn.Parameter(
            torch.tensor(_inv_softplus(0.2 * self.std)))
        if m_max is not None:
            # 初始化映入箱内: m = m_max·σ(ω_m), ω 初值取原 0.2·std 截断到箱内
            frac = min(0.2 * self.std / max(m_max, 1e-9), 0.95)
            self._margin_raw = torch.nn.Parameter(
                torch.tensor(_logit(frac)))
        self.K_on, self.K_off = K_on, K_off
        self._t_on = torch.nn.Parameter(
            torch.tensor(_logit((init_T_on - 1) / (K_on - 1))))
        self._t_off = torch.nn.Parameter(
            torch.tensor(_logit((init_T_off - 1) / (K_off - 1))))

    @property
    def theta_hi(self):
        return self.q_lo + (self.q_hi - self.q_lo) * torch.sigmoid(self._theta_raw)

    def margin(self):
        if self.m_max is not None:
            return self.m_max * torch.sigmoid(self._margin_raw) + 1e-6
        return torch.nn.functional.softplus(self._margin_raw) + 1e-6

    def T_on(self):
        return 1.0 + (self.K_on - 1) * torch.sigmoid(self._t_on)

    def T_off(self):
        return 1.0 + (self.K_off - 1) * torch.sigmoid(self._t_off)

    def forward(self, s, temp_frac, temp_v):
        temp_s = max(temp_frac, 0.01) * self.std
        return soft_pipeline(s, self.theta_hi, self.theta_hi - self.margin(),
                             self.T_on(), self.T_off(), temp_s, temp_v,
                             K_on=self.K_on, K_off=self.K_off)

    def hard_params(self):
        """推理侧硬参数 (取整 + 转 python float)."""
        with torch.no_grad():
            return {
                "theta_hi": float(self.theta_hi),
                "theta_lo": float(self.theta_hi - self.margin()),
                "T_on": max(1, int(round(float(self.T_on())))),
                "T_off": max(1, int(round(float(self.T_off())))),
            }


def _soft_point_f1(a_hat, y):
    """点级 relaxed F1 (VAD/TSAD 常用替代损失): 软 TP/FP/FN 计数."""
    tp = (a_hat * y).sum()
    fp = (a_hat * (1 - y)).sum()
    fn = ((1 - a_hat) * y).sum()
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    return 2 * prec * rec / (prec + rec + 1e-6)


def train_tuning(s, y, events, dt, R_max=10.0, F_max=0.02, epochs=300,
                 lr=0.05, eta_c=0.05, lambda_lag=0.5, seed=7, verbose=False,
                 K_on=24, K_off=48, loss_mode="constrained", disable=(),
                 hard_eval_fn=None, hard_eval_every=10, C_max=None,
                 init_t_on=4.0, init_t_off=8.0, tia_max=None, tia_box=None,
                 m_max=None):
    """在 (s, y, events) 上训练整定参数. 返回 (hard_params, history).

    loss_mode:
      'constrained' — 本文: miss + lag + 乘子约束 (AAR/FAR[, chatter]);
      'relaxed_f1'  — 对照 (E6): 点级 relaxed F1 替代损失, 不带 ISA 约束项.
      C_max: 可选 chatter 约束上限 (启用第三个乘子, 窗口 = ISA 常用 60s).
      tia_max: 可选时间占比约束上限 (第四乘子): mean(ã) <= tia_max,
              可微 time-in-alarm 代理 —— 针对常设告警角点的结构性封堵.
    disable: E4 消融 — {'hysteresis','on_delay','off_delay'} 子集, 对应参数
      冻结为算子退化值 (margin→0 / T_on→1 / T_off→1), 不参与梯度.
    hard_eval_fn: E5 用 — 每 hard_eval_every 个 epoch 调 model 做硬规则评估,
      返回 dict 并入该 epoch 的 history (键加 hard_ 前缀).
    """
    torch.manual_seed(seed)
    s_t = torch.as_tensor(s, dtype=torch.float32)
    y_t = torch.as_tensor(y)
    model = TuningParams(s_t, K_on=K_on, K_off=K_off,
                         init_T_on=init_t_on, init_T_off=init_t_off,
                         tia_box=tia_box, m_max=m_max)
    with torch.no_grad():
        if "hysteresis" in disable:                    # margin -> ~0
            model._margin_raw.fill_(-20.0)
            model._margin_raw.requires_grad_(False)
        if "on_delay" in disable:                      # T_on -> 1
            model._t_on.fill_(-20.0)
            model._t_on.requires_grad_(False)
        if "off_delay" in disable:                     # T_off -> 1
            model._t_off.fill_(-20.0)
            model._t_off.requires_grad_(False)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    mu = np.array([1.0, 1.0, 1.0, 1.0])
    K_chatter = max(1, int(round(60.0 / dt))) if C_max is not None else None
    hist = []
    for ep in range(epochs):
        temp_frac = max(0.05, 1.0 * (0.985 ** ep))      # 分数域: 1.0·std → 0.05·std
        temp_v = max(0.02, 0.15 * (0.985 ** ep))        # 值域: 0.15 → 0.02
        opt.zero_grad()
        a = model(s_t, temp_frac, temp_v)
        k = soft_kpi(a, y_t, events, dt, K_chatter=K_chatter)
        tia_soft = a.mean()                             # 可微 time-in-alarm
        # 高温阶段软 FAR 系统性高估 (sigmoid 平尾): 先纯目标退火 (拉住 θ 靠近
        # 事件分数, 保住漏报梯度), 温度降到位后再启用约束项与乘子对偶更新.
        # 无事件段 (如纯时序划分的正常段整定) 无目标梯度, 约束立即启用.
        constraints_on = (temp_frac <= 0.3) or (not events)
        with torch.no_grad():
            viol = [max(0.0, float(k["aar"]) - R_max) / R_max,
                    max(0.0, float(k["far"]) - F_max) / F_max]
            viol.append(max(0.0, float(k["chatter"]) - C_max) / max(C_max, 1e-6)
                        if C_max is not None else 0.0)
            viol.append(max(0.0, float(tia_soft) - tia_max) / max(tia_max, 1e-6)
                        if tia_max is not None else 0.0)
        pen = (mu[0] * torch.relu(k["aar"] - R_max) / R_max
               + mu[1] * torch.relu(k["far"] - F_max) / F_max)
        if C_max is not None:
            pen = pen + mu[2] * torch.relu(k["chatter"] - C_max) / max(C_max, 1e-6)
        if tia_max is not None:
            pen = pen + mu[3] * torch.relu(tia_soft - tia_max) / max(tia_max, 1e-6)
        if loss_mode == "relaxed_f1":
            loss = 1.0 - _soft_point_f1(a, y_t)
        elif loss_mode == "relaxed_f1_constrained":
            loss = (1.0 - _soft_point_f1(a, y_t)
                    + (pen if constraints_on else 0.0))
        else:
            loss = k["miss"] + lambda_lag * k["lag"] + (pen if constraints_on else 0.0)
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        else:
            opt.zero_grad()          # 无梯度 (无事件且约束满足) 时跳过该步
        if constraints_on:
            mu = np.minimum(mu * np.exp(eta_c * np.array(viol)), 100.0)
        rec = {"epoch": ep, "temp_frac": temp_frac, "temp_v": temp_v,
               "loss": float(loss.detach()),
               "miss": float(k["miss"].detach()),
               "lag": float(k["lag"].detach()),
               "aar": float(k["aar"].detach()),
               "far": float(k["far"].detach()),
               "tia_soft": float(tia_soft.detach()),
               "mu_aar": mu[0], "mu_far": mu[1], "mu_tia": mu[3]}
        if C_max is not None:
            rec["chatter"] = float(k["chatter"].detach())
            rec["mu_chatter"] = mu[2]
        with torch.no_grad():                            # E4 参数耦合轨迹
            rec.update({"theta_hi": float(model.theta_hi),
                        "margin": float(model.margin()),
                        "T_on": float(model.T_on()),
                        "T_off": float(model.T_off())})
        if hard_eval_fn is not None and (ep % hard_eval_every == 0 or ep == epochs - 1):
            hard = hard_eval_fn(model) or {}
            rec.update({f"hard_{kk}": vv for kk, vv in hard.items()})
        hist.append(rec)
        if verbose and ep % 50 == 0:
            print(f"  ep{ep} miss={k['miss']:.3f} lag={k['lag']:.3f} "
                  f"AAR={k['aar']:.2f} FAR={k['far']:.4f} mu={mu.round(2)}")
    return model.hard_params(), hist
