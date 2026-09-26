"""Domain types: market data, positions, configuration, results and jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from uuid import uuid4


class SimulationMethod(str, Enum):
    """Historical-simulation variant."""

    ROLLING = "rolling"  # equal-weight rolling window
    DECAY = "decay"  # exponentially decayed weights


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Quote:
    """One price observation for one instrument."""

    quote_date: date
    instrument_id: str
    price: float


@dataclass(frozen=True, slots=True)
class Position:
    instrument_id: str
    quantity: float


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Computation configuration.

    confidence: quantile level in (0.5, 1), e.g. 0.99.
    holding_period_days: scaling horizon, integer >= 1.
    method: rolling (equal weight) or decay (exponential weights).
    window: number of historical return scenarios, integer >= 2.
    decay_lambda: required for method=decay, 0 < lambda < 1.
    """

    confidence: float = 0.99
    holding_period_days: int = 1
    method: SimulationMethod = SimulationMethod.ROLLING
    window: int = 250
    decay_lambda: float | None = None


@dataclass(frozen=True, slots=True)
class RiskResult:
    portfolio_id: str
    config_fingerprint: str
    as_of: date
    observations_used: int
    var: float
    expected_shortfall: float
    method: SimulationMethod


@dataclass(slots=True)
class Job:
    id: str = field(default_factory=lambda: uuid4().hex)
    portfolio_id: str = ""
    config: RiskConfig | None = None
    status: JobStatus = JobStatus.PENDING
    result: RiskResult | None = None
    error_code: str | None = None
    error_detail: str | None = None

    @classmethod
    def new(cls, portfolio_id: str, config: RiskConfig) -> Job:
        return cls(portfolio_id=portfolio_id, config=config)
