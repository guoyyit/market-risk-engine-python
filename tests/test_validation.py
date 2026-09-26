"""异常输入：缺失/非正/重复/乱序/迟到，全部可区分且不留部分结果。"""

from __future__ import annotations

import math
from datetime import date

import pytest

from market_risk.models import (
    ErrorCode,
    HistoryBatch,
    Portfolio,
    Position,
    PriceObservation,
    RiskConfig,
    SimulationMethod,
    ValidationError,
)
from market_risk.validation import (
    validate_batch,
    validate_config,
    validate_portfolio,
)

D = date(2025, 1, 1)


def obs(inst="A", day=D, price=100.0):
    return PriceObservation(inst, day, price)


@pytest.mark.parametrize(
    "price,code",
    [(0.0, ErrorCode.NON_POSITIVE_PRICE),
     (-1.0, ErrorCode.NON_POSITIVE_PRICE),
     (math.nan, ErrorCode.MISSING_PRICE),
     (math.inf, ErrorCode.MISSING_PRICE),
     (-math.inf, ErrorCode.MISSING_PRICE)],
)
def test_bad_prices_rejected(price, code):
    with pytest.raises(ValidationError) as ei:
        validate_batch(HistoryBatch((obs(price=price),), as_of=D))
    assert ei.value.code is code


def test_empty_batch_rejected():
    with pytest.raises(ValidationError) as ei:
        validate_batch(HistoryBatch(()))
    assert ei.value.code is ErrorCode.EMPTY_HISTORY


def test_duplicate_within_batch_rejected():
    b = HistoryBatch((obs(day=D), obs(day=D, price=101)), as_of=D)
    with pytest.raises(ValidationError) as ei:
        validate_batch(b)
    assert ei.value.code is ErrorCode.DUPLICATE_OBSERVATION


def test_late_observation_rejected():
    b = HistoryBatch((obs(day=date(2025, 2, 1)),), as_of=D)
    with pytest.raises(ValidationError) as ei:
        validate_batch(b)
    assert ei.value.code is ErrorCode.LATE_OBSERVATION


def test_out_of_order_is_accepted_and_sorted(seeded_engine):
    """批次内乱序自动排序修正，不拒绝。"""
    engine, prices = seeded_engine
    good = [
        PriceObservation("B", date(2025, 1, 3), 10.0),
        PriceObservation("B", date(2025, 1, 1), 10.0),
        PriceObservation("B", date(2025, 1, 2), 10.0),
    ]
    v = engine.submit_history(HistoryBatch(tuple(good)))
    assert v == engine.store.version
    days = engine.store.get("B").days
    assert days == tuple(sorted(days))


def test_portfolio_validation():
    with pytest.raises(ValidationError) as ei:
        validate_portfolio(Portfolio("p", ()))
    assert ei.value.code is ErrorCode.EMPTY_PORTFOLIO

    bad_dup = Portfolio("p", (Position("A", 1), Position("A", 2)))
    with pytest.raises(ValidationError) as ei:
        validate_portfolio(bad_dup)
    assert ei.value.code is ErrorCode.DUPLICATE_OBSERVATION

    bad_qty = Portfolio("p", (Position("A", 0),))
    with pytest.raises(ValidationError) as ei:
        validate_portfolio(bad_qty)
    assert ei.value.code is ErrorCode.INVALID_PARAMETER


@pytest.mark.parametrize("kwargs", [
    {"confidence": 0.0}, {"confidence": 1.0}, {"confidence": 1.5},
    {"horizon": 0}, {"window": 1},
])
def test_config_validation_basic(kwargs):
    with pytest.raises(ValidationError) as ei:
        validate_config(RiskConfig(**kwargs))
    assert ei.value.code is ErrorCode.INVALID_PARAMETER


def test_decay_lambda_validation():
    with pytest.raises(ValidationError):
        validate_config(RiskConfig(
            method=SimulationMethod.DECAYED, decay_lambda=1.0))
    validate_config(RiskConfig(
        method=SimulationMethod.DECAYED, decay_lambda=0.9))


def test_failed_batch_leaves_no_partial_result(seeded_engine):
    engine, _ = seeded_engine
    v_before = engine.store.version
    # 批次中先合法后非法：整批拒绝，版本与序列都不变
    bad = (
        PriceObservation("C", date(2025, 1, 1), 50.0),
        PriceObservation("C", date(2025, 1, 2), 0.0),
    )
    with pytest.raises(ValidationError):
        engine.submit_history(HistoryBatch(bad))
    assert engine.store.version == v_before
    assert engine.store.get("C") is None


def test_backfill_after_commit_rejected(seeded_engine):
    engine, _ = seeded_engine
    old = (PriceObservation("A", date(2024, 12, 31), 90.0),)
    with pytest.raises(ValidationError) as ei:
        engine.submit_history(HistoryBatch(old))
    assert ei.value.code is ErrorCode.OUT_OF_ORDER
