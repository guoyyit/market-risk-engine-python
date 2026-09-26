"""Temporary skeleton smoke test."""


def test_imports():
    import market_risk  # noqa: F401
    from market_risk import api, engine, metrics, models, store, validation, window  # noqa: F401
    from market_risk import errors, recovery  # noqa: F401
