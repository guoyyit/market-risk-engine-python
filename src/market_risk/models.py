"""核心领域类型：数据契约、配置、结果与错误类型。

本文件只定义类型与常量，不含计算逻辑，便于其它模块共享且无循环依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Final


# ---------------------------------------------------------------------------
# 错误码：异常输入必须给出可区分的失败原因
# ---------------------------------------------------------------------------
class ErrorCode(str, Enum):
    MISSING_PRICE = "MISSING_PRICE"          # 缺失价格（null/NaN）
    NON_POSITIVE_PRICE = "NON_POSITIVE_PRICE"  # 价格 <= 0
    DUPLICATE_OBSERVATION = "DUPLICATE_OBSERVATION"  # (标的,日期) 重复
    OUT_OF_ORDER = "OUT_OF_ORDER"            # 历史序列乱序且无法排序修正
    LATE_OBSERVATION = "LATE_OBSERVATION"    # 迟到观测违反版本截止线
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"  # 有效样本不足
    EMPTY_PORTFOLIO = "EMPTY_PORTFOLIO"      # 空持仓
    EMPTY_HISTORY = "EMPTY_HISTORY"          # 空历史
    UNKNOWN_INSTRUMENT = "UNKNOWN_INSTRUMENT"  # 持仓引用了无行情的标的
    INVALID_PARAMETER = "INVALID_PARAMETER"  # 置信水平/窗口/持有期非法
    VERSION_CONFLICT = "VERSION_CONFLICT"    # 配置版本与缓存不匹配
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    TASK_CANCELLED = "TASK_CANCELLED"


class RiskEngineError(Exception):
    """所有引擎层错误的基类，携带稳定错误码与可读原因。"""

    def __init__(self, code: ErrorCode, reason: str = "") -> None:
        self.code = code
        self.reason = reason or code.value
        super().__init__(f"[{code.value}] {self.reason}")


class ValidationError(RiskEngineError):
    """输入校验失败：整个批次必须被拒绝，不产生任何部分结果。"""


class TaskCancelledError(RiskEngineError):
    def __init__(self, reason: str = "task cancelled") -> None:
        super().__init__(ErrorCode.TASK_CANCELLED, reason)


# ---------------------------------------------------------------------------
# 输入类型
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PriceObservation:
    """单条行情观测：某标的在某交易日的收盘价。"""
    instrument: str
    day: date
    price: float


@dataclass(frozen=True, slots=True)
class Position:
    """持仓：标的 + 数量（正=多头，负=空头）。"""
    instrument: str
    quantity: float


@dataclass(frozen=True, slots=True)
class Portfolio:
    """一组持仓（一个组合）。"""
    portfolio_id: str
    positions: tuple[Position, ...]


@dataclass(frozen=True, slots=True)
class HistoryBatch:
    """一次提交的行情批次（可能多标的、乱序、迟到）。"""
    observations: tuple[PriceObservation, ...]
    as_of: date | None = None  # 批次截止线；晚于此日期的观测视为迟到


# ---------------------------------------------------------------------------
# 计算口径配置
# ---------------------------------------------------------------------------
class SimulationMethod(str, Enum):
    """历史模拟口径。"""
    ROLLING = "ROLLING"        # 等权滚动窗口
    DECAYED = "DECAYED"        # 指数衰减加权


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """风险计算配置（值对象，变更即产生新版本，旧缓存不得混用）。

    - confidence: 置信水平，开区间 (0,1)，例如 0.99
    - horizon: 持有期（交易日，>=1）；采用 sqrt(h) 时间缩放
    - method: ROLLING / DECAYED
    - window: 滚动窗口样本数（>=2）
    - decay_lambda: 衰减因子，开区间 (0,1)，仅 DECAYED 使用；
      权重 w_i = (1-lambda)*lambda^(age)，age=0 最新
    """
    confidence: float = 0.99
    horizon: int = 1
    method: SimulationMethod = SimulationMethod.ROLLING
    window: int = 250
    decay_lambda: float = 0.99

    def version_key(self) -> str:
        if self.method is SimulationMethod.DECAYED:
            return (
                f"c={self.confidence}:h={self.horizon}:d:{self.window}"
                f":l={self.decay_lambda!r}"
            )
        return f"c={self.confidence}:h={self.horizon}:r:{self.window}"


# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RiskResult:
    """单个组合在某数据版本+配置版本下的已提交结果。"""
    portfolio_id: str
    data_version: int
    config_version: str
    var: float          # 历史模拟 VaR（损失，>=0）
    es: float           # 预期缺口（尾部平均损失，>=0）
    sample_size: int    # 使用的有效情景样本数
    as_of: date         # 结果对应的数据截止日
    method: SimulationMethod


class TaskState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class TaskStatus:
    task_id: str
    state: TaskState
    portfolio_id: str
    error_code: str | None = None
    error_reason: str | None = None


# ---------------------------------------------------------------------------
# 引擎全局约束（有界资源）
# ---------------------------------------------------------------------------
DEFAULT_MAX_INSTRUMENTS: Final[int] = 10_000
DEFAULT_MAX_HISTORY_DAYS: Final[int] = 10_000
DEFAULT_MAX_CONCURRENCY: Final[int] = 4
DEFAULT_QUERY_TIMEOUT_S: Final[float] = 30.0
