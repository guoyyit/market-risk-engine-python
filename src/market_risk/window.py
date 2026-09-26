"""Scenario weighting and configuration fingerprints.

Two historical-simulation schemes are supported:

* ``rolling`` -- equal weights ``1/n`` over the ``window`` most recent
  scenarios.  Moving the window boundary changes the scenario set, which
  changes the *data version*, so cached results never mix with a new boundary.
* ``decay`` -- exponentially decayed weights

      w_i = lambda**(n - 1 - i) / sum_k lambda**(n - 1 - k),   i = 0..n-1

  where ``i`` is chronological order (i = 0 oldest, i = n-1 newest).  The most
  recent scenario therefore carries weight ``(1 - lambda) / (1 - lambda**n)``.

Results are cached by a key combining the portfolio id, the config fingerprint
and the data version.  Changing window/confidence/horizon/lambda changes the
fingerprint; adding or correcting observations changes the data version.  The
two mechanisms together guarantee stale cached results are never reused.
"""

from __future__ import annotations

import hashlib
import json

from market_risk.models import RiskConfig
from market_risk.validation import validate_config


def build_weights(n: int, config: RiskConfig) -> list[float]:
    """Return ``n`` normalized scenario weights, oldest scenario first."""
    validate_config(config)
    if n <= 0:
        raise ValueError("n must be positive")
    if config.method.value == "rolling":
        return [1.0 / n] * n
    lam = config.decay_lambda
    assert lam is not None
    raw = [lam ** (n - 1 - i) for i in range(n)]
    total = sum(raw)
    return [w / total for w in raw]


def config_fingerprint(config: RiskConfig) -> str:
    """Stable hash of every parameter that affects numerical results."""
    validate_config(config)
    payload = {
        "confidence": config.confidence,
        "holding_period_days": config.holding_period_days,
        "method": config.method.value,
        "window": config.window,
        "decay_lambda": config.decay_lambda,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
