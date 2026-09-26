"""Input validation and normalization.

Failure reasons are distinguished via :class:`~market_risk.errors.ErrorCode`:

* ``missing_data``         -- null/NaN price, missing identifier, missing date.
* ``non_positive_price``   -- price <= 0 or non-finite.
* ``duplicate_observation``-- two quotes for the same (instrument, date) with
  conflicting prices inside one batch.
* ``duplicate_position``   -- two positions for the same instrument in one
  portfolio definition.
* ``invalid_quantity``     -- non-finite position quantity (zero is allowed and
  simply dropped).
* ``invalid_config``       -- bad confidence / window / horizon / lambda.

Out-of-order quotes inside a batch are **corrected** by sorting; late quotes
are accepted at the store layer (upsert). Neither condition is an error on its
own.  Validation is transactional: a batch either normalizes in full or the
caller applies nothing, so no partial state is ever produced.
"""

from __future__ import annotations

import math
from datetime import date

from market_risk.errors import ErrorCode, InputValidationError
from market_risk.models import Position, Quote, RiskConfig, SimulationMethod


def _is_real_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_quotes(quotes: list[Quote]) -> list[Quote]:
    """Validate and normalize one quote batch.

    Returns the quotes sorted by (instrument_id, quote_date) with exact
    (instrument, date) duplicates carrying the same price deduplicated.
    Raises :class:`InputValidationError` on any hard violation.
    """
    seen: dict[tuple[str, date], float] = {}
    normalized: list[Quote] = []
    for i, q in enumerate(quotes):
        if not isinstance(q, Quote):
            raise InputValidationError(ErrorCode.INVALID_CONFIG, f"item {i} is not a Quote")
        if not q.instrument_id or not isinstance(q.instrument_id, str):
            raise InputValidationError(
                ErrorCode.MISSING_DATA, f"quote #{i}: missing instrument_id"
            )
        if not isinstance(q.quote_date, date):
            raise InputValidationError(
                ErrorCode.MISSING_DATA, f"quote #{i}: missing/invalid quote_date"
            )
        price = q.price
        if not _is_real_number(price):
            if price is None or (isinstance(price, float) and math.isnan(float(price))):
                raise InputValidationError(
                    ErrorCode.MISSING_DATA,
                    f"quote #{i} ({q.instrument_id} @ {q.quote_date}): price missing/NaN",
                )
            raise InputValidationError(
                ErrorCode.NON_POSITIVE_PRICE,
                f"quote #{i} ({q.instrument_id} @ {q.quote_date}): non-finite price",
            )
        price = float(price)
        if price <= 0.0:
            raise InputValidationError(
                ErrorCode.NON_POSITIVE_PRICE,
                f"quote #{i} ({q.instrument_id} @ {q.quote_date}): price {price} is not positive",
            )
        key = (q.instrument_id, q.quote_date)
        if key in seen:
            if seen[key] != price:
                raise InputValidationError(
                    ErrorCode.DUPLICATE_OBSERVATION,
                    f"{q.instrument_id} @ {q.quote_date}: conflicting prices "
                    f"{seen[key]} vs {price}",
                )
            # exact duplicate: silently deduplicated
            continue
        seen[key] = price
        normalized.append(Quote(q.quote_date, q.instrument_id, price))
    normalized.sort(key=lambda q: (q.instrument_id, q.quote_date))
    return normalized


def validate_positions(positions: list[Position]) -> list[Position]:
    """Validate a portfolio definition; zero-quantity positions are dropped."""
    seen: set[str] = set()
    normalized: list[Position] = []
    for i, p in enumerate(positions):
        if not isinstance(p, Position):
            raise InputValidationError(ErrorCode.INVALID_CONFIG, f"item {i} is not a Position")
        if not p.instrument_id or not isinstance(p.instrument_id, str):
            raise InputValidationError(
                ErrorCode.MISSING_DATA, f"position #{i}: missing instrument_id"
            )
        if not _is_real_number(p.quantity):
            raise InputValidationError(
                ErrorCode.INVALID_QUANTITY,
                f"position #{i} ({p.instrument_id}): quantity must be finite",
            )
        quantity = float(p.quantity)
        if quantity == 0.0:
            continue
        if p.instrument_id in seen:
            raise InputValidationError(
                ErrorCode.DUPLICATE_POSITION,
                f"position #{i} ({p.instrument_id}): duplicate instrument in portfolio",
            )
        seen.add(p.instrument_id)
        normalized.append(Position(p.instrument_id, quantity))
    return normalized


def validate_config(config: RiskConfig) -> RiskConfig:
    """Validate a risk configuration and return it unchanged when valid."""
    if not isinstance(config, RiskConfig):
        raise InputValidationError(ErrorCode.INVALID_CONFIG, "not a RiskConfig")
    if not _is_real_number(config.confidence) or not 0.5 < config.confidence < 1.0:
        raise InputValidationError(
            ErrorCode.INVALID_CONFIG,
            f"confidence must be in (0.5, 1), got {config.confidence!r}",
        )
    if not isinstance(config.holding_period_days, int) or isinstance(config.holding_period_days, bool):
        raise InputValidationError(
            ErrorCode.INVALID_CONFIG, "holding_period_days must be an integer"
        )
    if config.holding_period_days < 1:
        raise InputValidationError(
            ErrorCode.INVALID_CONFIG,
            f"holding_period_days must be >= 1, got {config.holding_period_days}",
        )
    if not isinstance(config.window, int) or isinstance(config.window, bool) or config.window < 2:
        raise InputValidationError(
            ErrorCode.INVALID_CONFIG, f"window must be an integer >= 2, got {config.window!r}"
        )
    if config.method == SimulationMethod.DECAY:
        lam = config.decay_lambda
        if not _is_real_number(lam) or not 0.0 < lam < 1.0:
            raise InputValidationError(
                ErrorCode.INVALID_CONFIG,
                f"decay method requires decay_lambda in (0, 1), got {lam!r}",
            )
    elif config.decay_lambda is not None:
        raise InputValidationError(
            ErrorCode.INVALID_CONFIG,
            "decay_lambda must be None for the rolling method",
        )
    return config
