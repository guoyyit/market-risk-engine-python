"""历史模拟 VaR / ES 计算（等权滚动 & 指数衰减加权）。

口径摘要（完整说明见 docs/risk-methodology.md）：

1. 情景损益：对窗口内相邻两交易日 t-1, t，使用简单收益率
   r_i(t) = P_i(t)/P_i(t-1) - 1，组合损益
   L(t) = -sum_i q_i * P_i(t-1) * r_i(t)（损失为正）。
   窗口 W 个价格点 => W-1 个损益情景。
2. ROLLING：W-1 个情景等权。
3. DECAYED：按情景年龄指数加权，w(a) = (1-lambda)*lambda^a，a=0 最新；
   仅取窗口内最近 W-1 个情景。
4. VaR：损益升序排列后累计权重首次达到 1-c 的那个情景损失；
   ES：该位置及更坏情景（权重归一化后）的加权平均损失。
5. 持有期缩放：VaR_h = VaR_1 * sqrt(h)，ES 同理（i.i.d. 近似，文档注明）。
结果均为非负损失值；最坏情景仍为盈利时 VaR 取 0。
"""

from __future__ import annotations

import math
from datetime import date

from market_risk.models import (
    ErrorCode,
    Portfolio,
    RiskConfig,
    RiskEngineError,
    RiskResult,
)
from market_risk.series import WindowView


def portfolio_scenario_pnls(
    positions: tuple, window: WindowView
) -> list[float]:
    """由窗口价格构造组合逐日损失情景（损失为正），按时间升序。"""
    n = window.scenario_count
    pnls = [0.0] * n
    # positions 与 window.prices 的列顺序一致（engine 负责按标的对齐）
    for pos, prices in zip(positions, window.prices, strict=True):
        for t in range(1, len(prices)):
            prev = prices[t - 1]
            r = prices[t] / prev - 1.0
            pnls[t - 1] += -(pos.quantity * prev * r)
    return pnls


def scenario_weights(count: int, config: RiskConfig) -> list[float]:
    """返回按时间升序（旧->新）的归一化权重。"""
    if count <= 0:
        return []
    if config.method.value == "DECAYED":
        lam = config.decay_lambda
        # age=0 对最新情景；升序序列中 age = count-1-index
        raw = [
            (1.0 - lam) * lam ** (count - 1 - i) for i in range(count)
        ]
    else:
        raw = [1.0] * count
    total = sum(raw)
    return [w / total for w in raw]


def historical_var_es(
    pnls: list[float], weights: list[float], confidence: float
) -> tuple[float, float]:
    """加权历史模拟 VaR 与 ES。

    pnls 为损失（正=亏损），按升序遍历累计权重。定义（经验分布、不做
    线性插值，保证结果必为某个真实情景且完全可复现）：

    - VaR：累计权重首次**严格超过** alpha=1-c 时所在情景的损失；
      alpha 恰好等于某个样本累计权重（等权样本下常见）时，该分界样本
      自身就是分位点，VaR 取其损失。
    - ES：VaR 分位样本及其后更坏情景（损益更小）的加权平均，权重在
      尾部内重新归一化。

    返回 (VaR, ES)，均截断为 >= 0（最坏情景仍为净盈利时风险为 0）。
    """
    if not pnls:
        raise RiskEngineError(
            ErrorCode.INSUFFICIENT_HISTORY, "无损益情景可供计算"
        )
    order = sorted(range(len(pnls)), key=lambda i: pnls[i])
    alpha = 1.0 - confidence
    cum = 0.0
    cut_rank = len(order) - 1
    for rank, idx in enumerate(order):
        cum += weights[idx]
        if cum > alpha + 1e-12 or rank == len(order) - 1:
            cut_rank = rank
            break
    cut = order[cut_rank]
    var = max(0.0, pnls[cut])
    tail = order[cut_rank:]
    tail_w = sum(weights[i] for i in tail)
    es = sum(weights[i] * pnls[i] for i in tail) / tail_w
    return var, max(0.0, es)


def compute_result(
    portfolio: Portfolio,
    window: WindowView,
    config: RiskConfig,
    data_version: int,
    as_of: date | None,
) -> RiskResult:
    """按配置完成单个组合的风险计算。"""
    pnls = portfolio_scenario_pnls(portfolio.positions, window)
    weights = scenario_weights(len(pnls), config)
    var, es = historical_var_es(pnls, weights, config.confidence)
    scale = math.sqrt(config.horizon)
    result_as_of = as_of or window.days[-1]
    return RiskResult(
        portfolio_id=portfolio.portfolio_id,
        data_version=data_version,
        config_version=config.version_key(),
        var=var * scale,
        es=es * scale,
        sample_size=len(pnls),
        as_of=result_as_of,
        method=config.method,
    )
