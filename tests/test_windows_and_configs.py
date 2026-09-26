"""滚动窗口边界、衰减口径端到端、配置版本缓存隔离。"""

from __future__ import annotations

import pytest

from market_risk.models import (
    ErrorCode,
    RiskEngineError,
    SimulationMethod,
)

from conftest import default_config
from market_risk.models import HistoryBatch


def _calc(engine, pf="p1", **cfg_kw):
    tid = engine.calculate(pf, default_config(**cfg_kw))
    return engine.get_result(tid, timeout=10)


def test_window_sample_size(seeded_engine):
    engine, _ = seeded_engine
    r = _calc(engine, window=6)
    # window 个价格点 -> window-1 个情景
    assert r.sample_size == 5
    r = _calc(engine, window=30)
    assert r.sample_size == 29


def test_window_only_uses_recent_data(seeded_engine):
    """窗口外的旧数据不影响结果：用更长窗口结果必须可能不同。"""
    engine, prices = seeded_engine
    short = _calc(engine, window=5)
    long_ = _calc(engine, window=30)
    assert short.sample_size == 4 and long_.sample_size == 29
    # 两个任务 id 不同（配置版本隔离）
    assert short.config_version != long_.config_version


def test_config_change_creates_new_result_not_reuse(seeded_engine):
    engine, _ = seeded_engine
    r1 = _calc(engine, confidence=0.90)
    r2 = _calc(engine, confidence=0.99)
    r3 = _calc(engine, horizon=1)
    r4 = _calc(engine, horizon=2)
    r5 = _calc(engine, method=SimulationMethod.DECAYED, decay_lambda=0.95)
    versions = {r1.config_version, r2.config_version, r3.config_version,
                r4.config_version, r5.config_version}
    assert len(versions) == 5
    # horizon=2 的 VaR 为 horizon=1 的 sqrt(2) 倍
    assert r4.var == pytest.approx(r3.var * (2 ** 0.5))


def test_same_config_reuses_task(seeded_engine):
    engine, _ = seeded_engine
    t1 = engine.calculate("p1", default_config())
    t2 = engine.calculate("p1", default_config())
    assert t1 == t2


def test_decay_end_to_end(seeded_engine):
    engine, _ = seeded_engine
    r = _calc(engine, method=SimulationMethod.DECAYED,
              decay_lambda=0.97, window=20)
    assert r.method is SimulationMethod.DECAYED
    assert r.sample_size == 19
    assert r.var >= 0 and r.es >= r.var - 1e-9


def test_insufficient_history(engine):
    from datetime import date
    from market_risk.models import PriceObservation, Portfolio, Position
    engine.submit_history(HistoryBatch((
        PriceObservation("A", date(2025, 1, 1), 100.0),),
        as_of=date(2025, 1, 1)))
    engine.register_portfolio(Portfolio("p", (Position("A", 1.0),)))
    tid = engine.calculate("p", default_config(window=10))
    with pytest.raises(RiskEngineError) as ei:
        engine.get_result(tid, timeout=10)
    assert ei.value.code is ErrorCode.INSUFFICIENT_HISTORY
    # 失败任务状态可见
    assert engine.get_task_status(tid).state.value == "FAILED"


def test_unknown_instrument_in_portfolio(seeded_engine):
    from market_risk.models import Portfolio, Position
    engine, _ = seeded_engine
    engine.register_portfolio(
        Portfolio("pX", (Position("NOPE", 1.0),)))
    tid = engine.calculate("pX", default_config())
    with pytest.raises(RiskEngineError) as ei:
        engine.get_result(tid, timeout=10)
    assert ei.value.code is ErrorCode.UNKNOWN_INSTRUMENT


def test_multi_portfolio_isolation(engine):
    from datetime import date
    from market_risk.models import (HistoryBatch, Portfolio, Position,
                                    PriceObservation)
    obs = []
    p_a = p_b = 100.0
    for d in range(1, 21):
        p_a *= 1 + (0.001 if d % 2 else -0.001)
        p_b *= 1 + (-0.05 if d == 20 else 0.001)
        obs.append(PriceObservation("A", date(2025, 1, d), p_a))
        obs.append(PriceObservation("B", date(2025, 1, d), p_b))
    engine.submit_history(HistoryBatch(tuple(obs), as_of=date(2025, 1, 20)))
    engine.register_portfolio(Portfolio("pa", (Position("A", 1.0),)))
    engine.register_portfolio(Portfolio("pb", (Position("B", 1.0),)))
    ra = _calc(engine, "pa", window=20, confidence=0.95)
    rb = _calc(engine, "pb", window=20, confidence=0.95)
    assert ra.portfolio_id == "pa" and rb.portfolio_id == "pb"
    # B 在末日大跌，其 ES 应显著大于 A
    assert rb.es > ra.es
