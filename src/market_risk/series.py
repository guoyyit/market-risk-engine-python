"""按标的维护的有序历史价格序列与滚动窗口视图。

职责：去重合并、排序、迟到/补录裁决、按长度上限淘汰旧数据、多标的按
共同交易日对齐；不含任何风险计算。

数据版本：MarketDataStore.apply_batch 成功（全部标的合并均通过）后版本号
才 +1；任一冲突抛异常且内部状态不变（原子失败）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from market_risk.models import (
    DEFAULT_MAX_HISTORY_DAYS,
    ErrorCode,
    PriceObservation,
    RiskEngineError,
    ValidationError,
)


@dataclass(frozen=True, slots=True)
class InstrumentSeries:
    """单个标的的有序历史（日期严格升序，与价格一一对应）。"""
    instrument: str
    days: tuple[date, ...] = ()
    prices: tuple[float, ...] = ()

    @property
    def latest_day(self) -> date | None:
        return self.days[-1] if self.days else None

    def merge(
        self,
        observations: list[PriceObservation],
        max_history_days: int,
    ) -> "InstrumentSeries":
        """合并本标的的一批新观测，返回新序列；冲突时抛 ValidationError。

        - 与已有日期重复 -> DUPLICATE_OBSERVATION
        - 早于已有最新日期 -> OUT_OF_ORDER（迟到补录被拒）
        合并成功后按 max_history_days 淘汰最旧数据以保证内存有界。
        """
        incoming = sorted(observations, key=lambda o: o.day)
        existing_days = set(self.days)
        for obs in incoming:
            if obs.day in existing_days:
                raise ValidationError(
                    ErrorCode.DUPLICATE_OBSERVATION,
                    f"标的 {self.instrument} 在 {obs.day} 已存在观测，"
                    f"重复提交被拒绝",
                )
            if self.latest_day is not None and obs.day < self.latest_day:
                raise ValidationError(
                    ErrorCode.OUT_OF_ORDER,
                    f"标的 {self.instrument} 的观测 {obs.day} 早于已提交"
                    f"最新日期 {self.latest_day}，迟到补录被拒绝",
                )
        merged_pairs = list(zip(self.days, self.prices, strict=True))
        merged_pairs.extend((o.day, o.price) for o in incoming)
        merged_pairs.sort(key=lambda p: p[0])
        if len(merged_pairs) > max_history_days:
            merged_pairs = merged_pairs[-max_history_days:]
        return InstrumentSeries(
            instrument=self.instrument,
            days=tuple(d for d, _ in merged_pairs),
            prices=tuple(p for _, p in merged_pairs),
        )


@dataclass(frozen=True, slots=True)
class WindowView:
    """对齐后的窗口。

    days: 窗口内日期（升序，长度 W = 价格点数）
    prices[i]: 第 i 个标的在 days 各日上的价格；
    损益情景数 = W - 1（相邻两日价格之比）。
    """
    days: tuple[date, ...]
    prices: tuple[tuple[float, ...], ...]

    @property
    def price_points(self) -> int:
        return len(self.days)

    @property
    def scenario_count(self) -> int:
        return max(0, len(self.days) - 1)


@dataclass(slots=True)
class MarketDataStore:
    """全部标的历史数据的内存态；apply_batch 成功后整体版本号 +1。"""
    version: int = 0
    as_of: date | None = None
    max_history_days: int = DEFAULT_MAX_HISTORY_DAYS
    _series: dict[str, InstrumentSeries] = field(default_factory=dict)

    def apply_batch(
        self, observations: list[PriceObservation], as_of: date | None
    ) -> int:
        """原子应用一个已通过批次内校验的批次，返回新版本号。

        先在副本上完成全部标的合并，全部成功后才替换状态，因此失败不留
        中间状态、版本号不增加。
        """
        grouped: dict[str, list[PriceObservation]] = {}
        for obs in observations:
            grouped.setdefault(obs.instrument, []).append(obs)

        new_series = dict(self._series)
        for instrument, obs_list in grouped.items():
            current = new_series.get(
                instrument, InstrumentSeries(instrument)
            )
            new_series[instrument] = current.merge(
                obs_list, self.max_history_days
            )

        new_as_of = self.as_of
        if as_of is not None:
            if self.as_of is not None and as_of < self.as_of:
                raise ValidationError(
                    ErrorCode.OUT_OF_ORDER,
                    f"批次截止线 {as_of} 早于已提交截止线 {self.as_of}",
                )
            new_as_of = as_of
        if new_as_of is None and observations:
            new_as_of = max(o.day for o in observations)

        self._series = new_series
        self.as_of = new_as_of
        self.version += 1
        return self.version

    def get(self, instrument: str) -> InstrumentSeries | None:
        return self._series.get(instrument)

    @property
    def instruments(self) -> tuple[str, ...]:
        return tuple(sorted(self._series))

    def common_window(self, instruments: list[str], window: int) -> WindowView:
        """取多标的对齐到共同交易日的最近 window 个价格点。

        - 任一标的无行情 -> UNKNOWN_INSTRUMENT
        - 共同交易日价格点 < 2 -> INSUFFICIENT_HISTORY
        """
        series_list: list[InstrumentSeries] = []
        for inst in instruments:
            s = self._series.get(inst)
            if s is None or not s.days:
                raise RiskEngineError(
                    ErrorCode.UNKNOWN_INSTRUMENT,
                    f"标的 {inst} 没有可用历史行情",
                )
            series_list.append(s)

        common = set(series_list[0].days)
        for s in series_list[1:]:
            common &= set(s.days)
        common_days = sorted(common)[-window:]
        if len(common_days) < 2:
            raise RiskEngineError(
                ErrorCode.INSUFFICIENT_HISTORY,
                "共同交易日价格点不足 2 个，无法构造损益情景",
            )

        day_index_list = [
            {d: i for i, d in enumerate(s.days)} for s in series_list
        ]
        columns: list[tuple[float, ...]] = []
        for s, idx in zip(series_list, day_index_list, strict=True):
            columns.append(tuple(s.prices[idx[d]] for d in common_days))
        return WindowView(days=tuple(common_days), prices=tuple(columns))
