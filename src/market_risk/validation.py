"""输入校验：缺失 / 非正价格 / 重复 / 乱序 / 迟到。

既定规则（见 docs/risk-methodology.md）：

- 批次内日期乱序：**自动修正**，按 (标的, 日期) 升序排序后入库；
- 缺失价格（None / NaN / ±Inf）：整批拒绝，错误码 MISSING_PRICE；
- 非正价格（<= 0）：整批拒绝，错误码 NON_POSITIVE_PRICE；
- (标的, 日期) 重复：整批拒绝，错误码 DUPLICATE_OBSERVATION；
- 迟到（观测日期晚于批次 as_of 截止线）：整批拒绝，错误码 LATE_OBSERVATION；
- 补录早于已提交数据最新日期的观测：整批拒绝，错误码 OUT_OF_ORDER，
  以保证已提交数据版本不可变。

校验为全有或全无：任一观测非法即抛 ValidationError，调用方不得应用该批数据，
因此不会产生部分结果。
"""

from __future__ import annotations

import math

from market_risk.models import (
    ErrorCode,
    HistoryBatch,
    Portfolio,
    RiskConfig,
    SimulationMethod,
    ValidationError,
)


def _bad_price(price: float) -> bool:
    return price is None or not math.isfinite(price)


def validate_batch(batch: HistoryBatch) -> None:
    """校验一个行情批次；非法时抛 ValidationError，不产生任何副作用。"""
    if not batch.observations:
        raise ValidationError(ErrorCode.EMPTY_HISTORY, "行情批次为空")

    seen: set[tuple[str, object]] = set()
    for obs in batch.observations:
        if not obs.instrument:
            raise ValidationError(
                ErrorCode.INVALID_PARAMETER,
                f"存在空标的名（日期={obs.day}）",
            )
        if obs.day is None:
            raise ValidationError(
                ErrorCode.MISSING_PRICE,
                f"标的 {obs.instrument} 缺少日期",
            )
        if _bad_price(obs.price):
            raise ValidationError(
                ErrorCode.MISSING_PRICE,
                f"标的 {obs.instrument} 在 {obs.day} 的价格缺失或非有限值",
            )
        if obs.price <= 0:
            raise ValidationError(
                ErrorCode.NON_POSITIVE_PRICE,
                f"标的 {obs.instrument} 在 {obs.day} 的价格非正: {obs.price}",
            )
        key = (obs.instrument, obs.day)
        if key in seen:
            raise ValidationError(
                ErrorCode.DUPLICATE_OBSERVATION,
                f"标的 {obs.instrument} 在 {obs.day} 出现重复观测",
            )
        seen.add(key)
        if batch.as_of is not None and obs.day > batch.as_of:
            raise ValidationError(
                ErrorCode.LATE_OBSERVATION,
                f"标的 {obs.instrument} 在 {obs.day} 的观测晚于截止线 "
                f"{batch.as_of}",
            )


def validate_no_backfill(
    new_observations: list, latest_day_by_instrument: dict[str, object]
) -> None:
    """拒绝向已提交序列中间补录旧日期（迟到补录）。

    已提交数据版本不可变；如需修订历史必须显式重建。
    """
    for obs in new_observations:
        latest = latest_day_by_instrument.get(obs.instrument)
        if latest is not None and obs.day < latest:
            raise ValidationError(
                ErrorCode.OUT_OF_ORDER,
                f"标的 {obs.instrument} 的观测日期 {obs.day} 早于已提交最新 "
                f"日期 {latest}，迟到补录被拒绝",
            )


def validate_portfolio(portfolio: Portfolio) -> None:
    """校验组合非空、数量有限非零、标的不重复。"""
    if not portfolio.positions:
        raise ValidationError(
            ErrorCode.EMPTY_PORTFOLIO,
            f"组合 {portfolio.portfolio_id} 持仓为空",
        )
    seen: set[str] = set()
    for pos in portfolio.positions:
        if not pos.instrument:
            raise ValidationError(
                ErrorCode.INVALID_PARAMETER,
                f"组合 {portfolio.portfolio_id} 含空标的名持仓",
            )
        if pos.instrument in seen:
            raise ValidationError(
                ErrorCode.DUPLICATE_OBSERVATION,
                f"组合 {portfolio.portfolio_id} 中标的 {pos.instrument} 持仓"
                f"重复，请先合并数量",
            )
        seen.add(pos.instrument)
        if not math.isfinite(pos.quantity) or pos.quantity == 0:
            raise ValidationError(
                ErrorCode.INVALID_PARAMETER,
                f"组合 {portfolio.portfolio_id} 标的 {pos.instrument} 数量"
                f"非法或为零: {pos.quantity}",
            )


def validate_config(config: RiskConfig) -> None:
    """校验置信水平 / 持有期 / 窗口 / 衰减因子取值。"""
    if not (0.0 < config.confidence < 1.0):
        raise ValidationError(
            ErrorCode.INVALID_PARAMETER,
            f"置信水平必须在开区间 (0,1) 内: {config.confidence}",
        )
    if not isinstance(config.horizon, int) or config.horizon < 1:
        raise ValidationError(
            ErrorCode.INVALID_PARAMETER,
            f"持有期必须为 >=1 的整数交易日: {config.horizon}",
        )
    if not isinstance(config.window, int) or config.window < 2:
        raise ValidationError(
            ErrorCode.INVALID_PARAMETER,
            f"滚动窗口样本数必须为 >=2 的整数: {config.window}",
        )
    if config.method is SimulationMethod.DECAYED:
        if not (0.0 < config.decay_lambda < 1.0):
            raise ValidationError(
                ErrorCode.INVALID_PARAMETER,
                f"衰减因子必须在开区间 (0,1) 内: {config.decay_lambda}",
            )
