"""测试公共夹具：临时数据目录、引擎、合成数据构造器。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import pytest

from market_risk.engine import RiskEngine
from market_risk.models import (
    HistoryBatch,
    Portfolio,
    Position,
    PriceObservation,
    RiskConfig,
)


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "risk-data"


@pytest.fixture
def engine(store_path: Path) -> RiskEngine:
    eng = RiskEngine(store_path, max_concurrency=4)
    yield eng
    eng.close()


def gen_prices(
    days: int,
    *,
    start: float = 100.0,
    returns: Sequence[float] | None = None,
    seed: int = 7,
) -> list[float]:
    """生成确定性价格序列；returns 给定则逐日使用，否则用固定伪随机。"""
    prices = [start]
    if returns is None:
        # 简单确定性 LCG，不依赖第三方库
        state = seed
        for _ in range(days - 1):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            r = (state % 2000 - 1000) / 10000.0  # [-0.1, 0.1)
            prices.append(max(prices[-1] * (1 + r), 0.01))
    else:
        assert len(returns) >= days - 1
        for r in returns[: days - 1]:
            prices.append(prices[-1] * (1 + r))
    return prices


def make_observations(
    instrument: str,
    prices: Sequence[float],
    *,
    start_day: date = date(2025, 1, 1),
    shuffle: bool = False,
) -> list[PriceObservation]:
    obs = [
        PriceObservation(instrument, start_day + timedelta(days=i), p)
        for i, p in enumerate(prices)
    ]
    if shuffle:
        obs = list(reversed(obs))
    return obs


@pytest.fixture
def seeded_engine(engine: RiskEngine):
    """填充 30 天单标的历史 + 单标的组合，返回 (engine, 价格列表)。"""
    prices = gen_prices(30)
    obs = make_observations("A", prices)
    engine.submit_history(
        HistoryBatch(tuple(obs), as_of=obs[-1].day))
    engine.register_portfolio(
        Portfolio("p1", (Position("A", 1.0),)))
    return engine, prices


def default_config(**kw) -> RiskConfig:
    base = dict(confidence=0.95, window=10, horizon=1)
    base.update(kw)
    return RiskConfig(**base)
