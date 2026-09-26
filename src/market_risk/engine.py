"""Concurrency, caching, incremental updates, cancellation and timeouts."""

from __future__ import annotations

from market_risk.models import Job, Position, Quote, RiskConfig, RiskResult
from market_risk.store import DataStore


class RiskEngine:
    def __init__(
        self,
        store: DataStore | None = None,
        *,
        max_workers: int = 4,
        journal_path: str | None = None,
    ) -> None:  # pragma: no cover - stub
        self.store = store or DataStore()
        self.max_workers = max_workers

    # data management
    def upsert_quotes(self, quotes: list[Quote]) -> None:
        raise NotImplementedError

    def register_portfolio(self, portfolio_id: str, positions: list[Position]) -> None:
        raise NotImplementedError

    # compute
    def submit(self, portfolio_id: str, config: RiskConfig) -> Job:
        raise NotImplementedError

    def compute(self, portfolio_id: str, config: RiskConfig) -> RiskResult:
        raise NotImplementedError

    def get_result(self, job_id: str) -> Job:
        raise NotImplementedError

    def cancel(self, job_id: str) -> None:
        raise NotImplementedError

    def wait_for(self, job_id: str, timeout: float | None = None) -> RiskResult:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError
