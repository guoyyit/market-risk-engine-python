"""Distinguishable failure reasons for rejected input and failed jobs."""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    MISSING_DATA = "missing_data"
    DUPLICATE_OBSERVATION = "duplicate_observation"
    DUPLICATE_POSITION = "duplicate_position"
    OUT_OF_ORDER = "out_of_order"
    LATE_OBSERVATION = "late_observation"
    NON_POSITIVE_PRICE = "non_positive_price"
    INVALID_QUANTITY = "invalid_quantity"
    UNKNOWN_INSTRUMENT = "unknown_instrument"
    UNKNOWN_PORTFOLIO = "unknown_portfolio"
    INVALID_CONFIG = "invalid_config"
    INSUFFICIENT_HISTORY = "insufficient_history"
    JOB_NOT_FOUND = "job_not_found"
    JOB_CANCELLED = "job_cancelled"
    JOB_TIMEOUT = "job_timeout"
    JOB_FAILED = "job_failed"


class RiskEngineError(Exception):
    def __init__(self, code: ErrorCode, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


class InputValidationError(RiskEngineError):
    """An input batch violates the data contract; no partial results are produced."""


class ConfigurationError(RiskEngineError):
    """Confidence level, window or weight configuration is invalid."""


class InsufficientHistoryError(RiskEngineError):
    """Not enough price observations to form the required scenario set."""


class JobError(RiskEngineError):
    """A compute job is missing, cancelled, timed out or failed."""
