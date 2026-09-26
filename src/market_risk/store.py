"""Bounded in-memory market-data store.

* Quotes are stored per instrument as a sorted mapping ``date -> price``;
  a late or duplicate-date quote is an **upsert** (the engine invalidates
  cached results via :meth:`DataStore.version`).
* History is bounded by ``max_history`` observations per instrument.  If the
  store holds more than ``max_history`` dates the oldest dates are evicted, so
  long runs cannot grow memory without limit.
* Every mutating call is atomic with respect to other threads and bumps a
  monotonic version counter.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import date
from typing import Iterable

from market_risk.models import Position, Quote
from market_risk.validation import validate_positions, validate_quotes


class DataStore:
    def __init__(self, max_history: int = 10_000) -> None:
        if not isinstance(max_history, int) or max_history < 2:
            raise ValueError("max_history must be an integer >= 2")
        self.max_history = max_history
        self._lock = threading.RLock()
        self._prices: dict[str, dict[date, float]] = {}
        self._order: dict[str, deque[date]] = {}
        self._portfolios: dict[str, tuple[Position, ...]] = {}
        self._version = 0

    @property
    def version(self) -> int:
        """Monotonic data version; changes whenever any input data changes."""
        return self._version

    def upsert_quotes(self, quotes: Iterable[Quote]) -> int:
        """Validate then atomically apply a quote batch.

        Returns the new store version.  Either the whole batch is applied or
        none of it is (validation happens before any mutation).
        """
        normalized = validate_quotes(list(quotes))
        with self._lock:
            changed = False
            for q in normalized:
                series = self._prices.get(q.instrument_id)
                if series is None:
                    series = {}
                    self._prices[q.instrument_id] = series
                    self._order[q.instrument_id] = deque()
                existing = series.get(q.quote_date)
                if existing != q.price:
                    if existing is None:
                        self._order[q.instrument_id].append(q.quote_date)
                    series[q.quote_date] = q.price
                    changed = True
            self._enforce_bounds()
            if changed:
                self._version += 1
            return self._version

    def register_portfolio(self, portfolio_id: str, positions: Iterable[Position]) -> int:
        if not portfolio_id or not isinstance(portfolio_id, str):
            raise ValueError("portfolio_id must be a non-empty string")
        normalized = tuple(validate_positions(list(positions)))
        with self._lock:
            if self._portfolios.get(portfolio_id) != normalized:
                self._portfolios[portfolio_id] = normalized
                self._version += 1
            return self._version

    def _enforce_bounds(self) -> None:
        for instrument_id, order in self._order.items():
            series = self._prices[instrument_id]
            while len(order) > self.max_history:
                oldest = order.popleft()
                series.pop(oldest, None)

    def prices(self, instrument_id: str) -> list[tuple[date, float]]:
        with self._lock:
            series = self._prices.get(instrument_id)
            if series is None:
                return []
            return [(d, series[d]) for d in sorted(series)]

    def positions(self, portfolio_id: str) -> list[Position]:
        with self._lock:
            ps = self._portfolios.get(portfolio_id)
            if ps is None:
                raise KeyError(portfolio_id)
            return list(ps)

    def has_portfolio(self, portfolio_id: str) -> bool:
        with self._lock:
            return portfolio_id in self._portfolios

    def known_instruments(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._prices)

    def price_map(self, instrument_id: str) -> dict[date, float]:
        """Return a defensive copy of one instrument's date->price series."""
        with self._lock:
            series = self._prices.get(instrument_id)
            return dict(series) if series is not None else {}

    def snapshot_version(self) -> int:
        with self._lock:
            return self._version
