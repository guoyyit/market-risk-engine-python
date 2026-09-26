"""Historical-simulation VaR and expected shortfall.

Reproducible valuation conventions
-----------------------------------
1. **Returns** are simple daily returns from adjacent observations of every
   instrument: ``r_t = P_t / P_{t-1} - 1``.  No interpolation is performed; a
   return date exists only where *every* instrument in the portfolio has a
   price for both adjacent dates (inner join on observation dates).
2. **Scenarios** are the ``window`` most recent such joint return vectors.
3. **Portfolio P&L** for scenario *t* is
   ``PL_t = sum_i quantity_i * P_i(as_of) * r_{i,t}`` (short positions use a
   negative quantity).  Loss is ``L_t = -PL_t``.
4. **Weights** come from :func:`market_risk.window.build_weights` (equal or
   exponential-decay), oldest scenario first, normalized to sum to 1.
5. **VaR** at confidence *p* is the right-continuous inverse empirical CDF:
   the smallest loss whose cumulative weight is ``>= p``.
6. **Expected shortfall** uses the Rockafellar–Uryasev completion.  Let
   ``W_above`` be the total weight of scenarios strictly worse than VaR.  The
   remaining tail weight

       w_extra = max(0, (1 - p) - W_above)

   is taken from the mass at the VaR point itself, so the tail probability is
   exactly ``1-p``:

       ES = ( sum_{L > VaR} w*L + w_extra*VaR ) / (1 - p)

   For equal weights with n=250, p=0.99 this is the average of the 2 worst
   scenarios plus half the 3rd worst (the VaR scenario).  Consequently
   ``ES >= VaR`` always, even with tied losses.
7. **Holding period** scales both measures by ``sqrt(h)`` (square-root-of-time
   rule): ``VaR_h = VaR_1 * sqrt(h)``, same for ES.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Iterable

from market_risk.errors import ErrorCode, InsufficientHistoryError
from market_risk.models import Position, RiskConfig, RiskResult, SimulationMethod
from market_risk.validation import validate_config
from market_risk.window import build_weights, config_fingerprint

_EPS = 1e-12


def compute_returns(prices: list[float]) -> list[float]:
    """Simple returns ``P_t/P_{t-1} - 1`` from an ascending price series."""
    if len(prices) < 2:
        return []
    return [prices[t] / prices[t - 1] - 1.0 for t in range(1, len(prices))]


def aligned_return_dates(
    price_history: dict[str, Iterable[tuple[date, float]]],
    as_of: date | None,
) -> tuple[list[date], dict[str, list[float]]]:
    """Inner-join instrument price series on dates, optionally truncate at as_of.

    Returns the common (sorted) dates and per-instrument prices on those dates.
    """
    per_instrument_dates: dict[str, list[date]] = {}
    per_instrument_prices: dict[str, list[float]] = {}
    for instrument_id, series in price_history.items():
        ordered = sorted((d for d, _ in series if as_of is None or d <= as_of))
        if not ordered:
            return [], {}
        lookup = {d: p for d, p in series}
        per_instrument_dates[instrument_id] = ordered
        per_instrument_prices[instrument_id] = [lookup[d] for d in ordered]
    common: list[date] | None = None
    for dates in per_instrument_dates.values():
        s = set(dates)
        common = s if common is None else (common & s)
    assert common is not None
    if not per_instrument_dates:
        return [], {}
    common_dates = sorted(common)
    aligned = {
        inst: _project(per_instrument_dates[inst], per_instrument_prices[inst], common_dates)
        for inst in price_history
    }
    return common_dates, aligned


def _project(dates: list[date], prices: list[float], common: list[date]) -> list[float]:
    lookup = dict(zip(dates, prices))
    return [lookup[d] for d in common]


def portfolio_losses(
    positions: list[Position],
    common_dates: list[date],
    aligned_prices: dict[str, list[float]],
    window: int,
) -> tuple[list[float], date]:
    """Build the rolling-window portfolio loss scenarios.

    Returns ``(losses, as_of)`` where losses are ordered oldest->newest and
    there are exactly ``window`` scenarios.
    """
    if not positions:
        raise InsufficientHistoryError(
            ErrorCode.INSUFFICIENT_HISTORY, "portfolio has no positions"
        )
    quantities = {p.instrument_id: p.quantity for p in positions}
    reference_prices = {inst: aligned_prices[inst][-1] for inst in quantities}
    # Returns per instrument over common dates
    returns: dict[str, list[float]] = {
        inst: compute_returns(aligned_prices[inst]) for inst in quantities
    }
    all_returns = list(returns.values())
    total_scenarios = min((len(r) for r in all_returns), default=0)
    if total_scenarios < window:
        raise InsufficientHistoryError(
            ErrorCode.INSUFFICIENT_HISTORY,
            f"need {window} joint return scenarios, have {total_scenarios}",
        )
    start = total_scenarios - window
    losses: list[float] = []
    for t in range(start, total_scenarios):
        pl = 0.0
        for inst, q in quantities.items():
            pl += q * reference_prices[inst] * returns[inst][t]
        losses.append(-pl)
    return losses, common_dates[-1]


def compute_var_es(
    losses: list[float], weights: list[float], config: RiskConfig
) -> tuple[float, float]:
    """Weighted empirical VaR and ES (conventions in the module docstring)."""
    validate_config(config)
    n = len(losses)
    if n == 0 or n != len(weights):
        raise ValueError("losses and weights must be non-empty and equally sized")
    p = config.confidence
    ordered = sorted(zip(losses, weights), key=lambda x: x[0])
    cumulative = 0.0
    var: float | None = None
    for loss, w in ordered:
        if cumulative + w >= p - _EPS:
            var = loss
            break
        cumulative += w
    assert var is not None
    above = [(loss, w) for loss, w in ordered if loss > var + _EPS]
    w_above = sum(w for _, w in above)
    w_extra = max(0.0, (1.0 - p) - w_above)
    tail_sum = sum(w * loss for loss, w in above)
    es = (tail_sum + w_extra * var) / (1.0 - p)
    scale = math.sqrt(config.holding_period_days)
    return var * scale, es * scale


def compute_full(
    portfolio_id: str,
    positions: list[Position],
    price_history: dict[str, list[tuple[date, float]]],
    as_of: date | None,
    config: RiskConfig,
) -> RiskResult:
    """Full recomputation of one portfolio result from raw aligned history."""
    validate_config(config)
    common_dates, aligned_prices = aligned_return_dates(price_history, as_of)
    if len(common_dates) == 0:
        raise InsufficientHistoryError(
            ErrorCode.INSUFFICIENT_HISTORY, "no common observation dates"
        )
    losses, effective_as_of = portfolio_losses(
        positions, common_dates, aligned_prices, config.window
    )
    weights = build_weights(len(losses), config)
    var, es = compute_var_es(losses, weights, config)
    return RiskResult(
        portfolio_id=portfolio_id,
        config_fingerprint=config_fingerprint(config),
        as_of=effective_as_of,
        observations_used=len(losses),
        var=var,
        expected_shortfall=es,
        method=config.method,
    )
