"""引擎门面：数据提交、组合注册、配置版本隔离、增量重算、并发查询与恢复。

关键不变量
==========
1. 数据版本 ``data_version`` 随成功批次单调递增；配置版本由
   ``RiskConfig.version_key()`` 派生。结果键为
   ``(portfolio_id, data_version, config_version)``，窗口/权重/置信水平
   任何变化都会产生新 config_version，已缓存结果绝不与旧配置混用。
2. 任务 id 确定性派生自结果键 ``task(portfolio, dataver, cfgver)``：
   - 同一键重复提交直接复用任务，不重复计算；
   - 崩溃恢复后重放日志得到相同 id，不重复提交、不丢任务。
3. 新增观测后只重算被请求的结果；底层序列只对尾部做合并，窗口外旧数据
   自然淘汰，因此增量结果与全量重算一致（见测试
   ``test_incremental_matches_full_rebuild``）。
4. 任务结束才写结果文件（原子 rename），查询只能读到完整版本。
"""

from __future__ import annotations

import threading
from pathlib import Path

from market_risk.models import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_QUERY_TIMEOUT_S,
    ErrorCode,
    HistoryBatch,
    Portfolio,
    RiskConfig,
    RiskEngineError,
    RiskResult,
    TaskState,
    TaskStatus,
)
from market_risk.series import MarketDataStore
from market_risk.simulation import compute_result
from market_risk.storage import Journal, TaskRecord
from market_risk.tasks import CancelToken, TaskManager
from market_risk.validation import (
    validate_batch,
    validate_config,
    validate_no_backfill,
    validate_portfolio,
)
from market_risk.models import ValidationError

_TASK_PREFIX = "task"


def _task_id(portfolio_id: str, data_version: int,
             config_version: str) -> str:
    return f"{_TASK_PREFIX}:{portfolio_id}:{data_version}:{config_version}"


def _config_to_dict(config: RiskConfig) -> dict:
    return {
        "confidence": config.confidence,
        "horizon": config.horizon,
        "method": config.method.value,
        "window": config.window,
        "decay_lambda": config.decay_lambda,
    }


def _config_from_dict(d: dict) -> RiskConfig:
    from market_risk.models import SimulationMethod as _M
    return RiskConfig(
        confidence=float(d["confidence"]),
        horizon=int(d["horizon"]),
        method=_M(d["method"]),
        window=int(d["window"]),
        decay_lambda=float(d["decay_lambda"]),
    )


class RiskEngine:
    def __init__(
        self,
        store_path: str | Path = "./.risk-data",
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        query_timeout: float = DEFAULT_QUERY_TIMEOUT_S,
        max_history_days: int | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency 必须 >=1")
        self.journal = Journal(store_path)
        self.max_concurrency = max_concurrency
        self.query_timeout = query_timeout
        kwargs = {} if max_history_days is None else {
            "max_history_days": max_history_days}
        self.store = MarketDataStore(**kwargs)
        self.portfolios: dict[str, Portfolio] = {}
        self.tasks = TaskManager[RiskResult](max_workers=max_concurrency)
        self.results: dict[str, RiskResult] = {}
        self._submit_lock = threading.Lock()
        self._wal_lock = threading.Lock()
        self._wal_events = 0
        self._wal_checkpoint_interval = 10_000
        self._batch_events = 0
        self._snapshot_interval = 1_000
        self._closed = False
        self._replay()

    def _count_wal_events(self) -> int:
        path = self.journal.tasks_path
        if not path.exists():
            return 0
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def _journal_event(self, record: TaskRecord) -> None:
        """串行化追加任务事件；超阈值时原子压缩 WAL。"""
        with self._wal_lock:
            self.journal.append_task(record)
            self._wal_events += 1
            self._maybe_checkpoint_locked()

    def _note_commit_event(self) -> None:
        """commit_result 内部已追加 COMMITTED 事件后的计数/压缩钩子。"""
        with self._wal_lock:
            self._wal_events += 1
            self._maybe_checkpoint_locked()

    def _maybe_checkpoint_locked(self) -> None:
        """事件过多时把 WAL 压缩为每任务一条最新记录（调用方持锁）。"""
        if self._wal_events < self._wal_checkpoint_interval:
            return
        records: dict[str, TaskRecord] = {}
        from market_risk.storage import _read_jsonl
        for d in _read_jsonl(self.journal.tasks_path):
            records[d["task_id"]] = TaskRecord(
                d["task_id"], d["portfolio_id"], d["config_version"],
                TaskState(d["state"]), d.get("error_code"),
                d.get("error_reason"), d.get("config"))
        self.journal.checkpoint_tasks(records)
        self._wal_events = len(records)

    # ------------------------------------------------------------------
    # 恢复 / 重放
    # ------------------------------------------------------------------
    def _replay(self) -> None:
        """启动重放：行情/组合重建内存，任务区分已提交与未完成。"""
        batches, snap_as_of, snap_version = self.journal.load_market()
        for batch in batches:
            # 已落盘数据此前已通过校验；冲突不应再出现，直接应用。
            self.store.apply_batch(list(batch.observations), batch.as_of)
        if snap_version:
            # 快照批次本身让版本 +1；对齐到快照记录的真实版本，
            # 随后增量批次继续在此基础上递增。
            self.store.version = snap_version + (len(batches) - 1)
        self.portfolios = self.journal.load_portfolios()
        records, results = self.journal.load()
        self.results = dict(results)
        self._pending_recovery: list[TaskRecord] = []
        for rec in records.values():
            if rec.state in (TaskState.PENDING, TaskState.RUNNING):
                self._pending_recovery.append(rec)
            elif rec.state is TaskState.COMMITTED:
                res = self.results.get(rec.task_id)
                self.tasks.register_terminal(
                    rec.task_id, rec.portfolio_id, TaskState.COMMITTED,
                    result=res)
            elif rec.state is TaskState.FAILED:
                self.tasks.register_terminal(
                    rec.task_id, rec.portfolio_id, TaskState.FAILED,
                    error_code=rec.error_code, error_reason=rec.error_reason)
            elif rec.state is TaskState.CANCELLED:
                self.tasks.register_terminal(
                    rec.task_id, rec.portfolio_id, TaskState.CANCELLED)
        # RUNNING 是崩溃中断：落盘重置为 PENDING（结果文件缺失，无歧义）。
        dirty = {r.task_id: TaskRecord(
            task_id=r.task_id, portfolio_id=r.portfolio_id,
            config_version=r.config_version, state=TaskState.PENDING,
            config=r.config)
            for r in self._pending_recovery
            if r.state is TaskState.RUNNING}
        if dirty:
            for rec in dirty.values():
                self._journal_event(rec)
            self._pending_recovery = [
                dirty[r.task_id] if r.task_id in dirty else r
                for r in self._pending_recovery
            ]
        # 以当前 WAL 行数初始化计数（load() 可能幂等补登 COMMITTED）
        self._wal_events = self._count_wal_events()

    def recover(self) -> int:
        """恢复后继续处理：把未完成任务重新入队，返回重新排队的任务数。

        幂等：已 COMMITTED 任务不会重新计算；重复调用安全。
        """
        pending = getattr(self, "_pending_recovery", [])
        count = 0
        for rec in pending:
            if self.tasks.has(rec.task_id):
                continue
            if rec.task_id in self.results:
                continue
            if rec.config is None:
                # 旧版本日志或损坏：无法重算，标记失败，不静默丢任务。
                self._journal_event(TaskRecord(
                    rec.task_id, rec.portfolio_id, rec.config_version,
                    TaskState.FAILED, ErrorCode.VERSION_CONFLICT.value,
                    "恢复任务缺少配置快照，无法重算"))
                continue
            data_version = int(rec.task_id.split(":")[2])
            cfg = _config_from_dict(rec.config)
            self._enqueue(rec.task_id, rec.portfolio_id,
                          rec.config_version, config=cfg,
                          data_version=data_version, recovered=True)
            count += 1
        self._pending_recovery = []
        return count

    # ------------------------------------------------------------------
    # 数据写入（全部先校验，持久化成功后才改内存版本）
    # ------------------------------------------------------------------
    def submit_history(self, batch: HistoryBatch) -> int:
        with self._submit_lock:
            validate_batch(batch)
            observations = list(batch.observations)
            # 轻量跨批次校验（O(批次大小)，不复制全量序列）：
            # 先查重复键（同日=重复 DUPLICATE），再查迟到补录
            # （早于最新且不重复=OUT_OF_ORDER）。apply_batch 为唯一修改
            # 状态处，内部纯函数式合并失败则状态不变。
            existing_days = {inst: set(s.days)
                             for inst, s in self.store._series.items()}
            for o in observations:
                if o.day in existing_days.get(o.instrument, set()):
                    raise ValidationError(
                        ErrorCode.DUPLICATE_OBSERVATION,
                        f"标的 {o.instrument} 在 {o.day} 已存在观测，"
                        f"重复提交被拒绝")
            latest = {inst: s.latest_day
                      for inst, s in self.store._series.items()}
            validate_no_backfill(observations, latest)
            self.journal.append_batch(batch)
            version = self.store.apply_batch(observations, batch.as_of)
            self._batch_events += 1
            self._maybe_snapshot()
            return version

    def _maybe_snapshot(self) -> None:
        """增量批次日志超过阈值时，原子写市场快照并截断日志（有界持久化）。"""
        if self._batch_events < self._snapshot_interval:
            return
        series_map = {
            inst: (s.days, s.prices)
            for inst, s in self.store._series.items()}
        self.journal.write_market_snapshot(
            series_map, self.store.as_of, self.store.version)
        self._batch_events = 0

    def register_portfolio(self, portfolio: Portfolio) -> None:
        with self._submit_lock:
            validate_portfolio(portfolio)
            if portfolio.portfolio_id in self.portfolios:
                raise RiskEngineError(
                    ErrorCode.DUPLICATE_OBSERVATION,
                    f"组合 {portfolio.portfolio_id} 已注册",
                )
            self.journal.append_portfolio(portfolio)
            self.portfolios[portfolio.portfolio_id] = portfolio

    # ------------------------------------------------------------------
    # 计算
    # ------------------------------------------------------------------
    def calculate(
        self, portfolio_id: str, config: RiskConfig,
        data_version: int | None = None,
    ) -> str:
        """提交一次风险计算，返回 task_id；相同键复用已有任务。"""
        validate_config(config)
        if portfolio_id not in self.portfolios:
            raise RiskEngineError(
                ErrorCode.UNKNOWN_INSTRUMENT,
                f"组合未注册: {portfolio_id}",
            )
        dv = self.store.version if data_version is None else data_version
        if dv > self.store.version:
            raise RiskEngineError(
                ErrorCode.VERSION_CONFLICT,
                f"请求数据版本 {dv} 大于当前版本 {self.store.version}",
            )
        cfg_ver = config.version_key()
        task_id = _task_id(portfolio_id, dv, cfg_ver)
        return self._enqueue(task_id, portfolio_id, cfg_ver, config=config,
                             data_version=dv)

    def _enqueue(self, task_id: str, portfolio_id: str,
                 config_version: str, *, config: RiskConfig | None = None,
                 data_version: int | None = None,
                 recovered: bool = False) -> str:
        with self._submit_lock:
            if self.tasks.has(task_id) or task_id in self.results:
                return task_id
            # 恢复任务需要从 task_id 反解 data_version
            if data_version is None:
                data_version = int(task_id.split(":")[2])
            cfg = config
            portfolio = self.portfolios[portfolio_id]
            record = TaskRecord(task_id, portfolio_id, config_version,
                                TaskState.PENDING,
                                config=_config_to_dict(cfg)
                                if cfg is not None else None)

            def work(token: CancelToken) -> RiskResult:
                result = self._run_calculation(
                    portfolio, data_version, cfg, token)
                return result

            def on_failure(_tid: str, code: ErrorCode,
                           reason: str) -> None:
                self._journal_event(TaskRecord(
                    task_id, portfolio_id, config_version,
                    TaskState.FAILED, code.value, reason,
                    config=_config_to_dict(cfg)
                    if cfg is not None else None))

            def on_cancel(_tid: str) -> None:
                self._journal_event(TaskRecord(
                    task_id, portfolio_id, config_version,
                    TaskState.CANCELLED,
                    config=_config_to_dict(cfg)
                    if cfg is not None else None))

            self._journal_event(record)
            self.tasks.submit(
                task_id, portfolio_id, work,
                on_commit=self._on_commit,
                on_failure=on_failure,
                on_cancel=on_cancel,
            )
            return task_id

    def _run_calculation(
        self, portfolio: Portfolio, data_version: int,
        config: RiskConfig, token: CancelToken,
    ) -> RiskResult:
        token.raise_if_cancelled()
        instruments = [p.instrument for p in portfolio.positions]
        window = self.store.common_window(instruments, config.window)
        token.raise_if_cancelled()
        # 列顺序与持仓顺序一致
        aligned = Portfolio(portfolio.portfolio_id, portfolio.positions)
        result = compute_result(
            aligned, window, config, data_version, self.store.as_of)
        token.raise_if_cancelled()
        # 持久化：原子提交，崩溃恢复以结果文件为准
        rec = TaskRecord(
            task_id=_task_id(portfolio.portfolio_id, data_version,
                             config.version_key()),
            portfolio_id=portfolio.portfolio_id,
            config_version=config.version_key(),
            state=TaskState.COMMITTED)
        self.journal.commit_result(rec, result)
        # commit_result 内部追加了 COMMITTED 事件，计入并按需压缩 WAL
        self._note_commit_event()
        return result

    # ------------------------------------------------------------------
    # 任务回调
    # ------------------------------------------------------------------
    def _on_commit(self, task_id: str, result: RiskResult) -> None:
        self.results[task_id] = result

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_result(self, task_id: str,
                   timeout: float | None = None) -> RiskResult:
        return self.tasks.get(
            task_id,
            self.query_timeout if timeout is None else timeout,
        )

    def get_committed_result(self, task_id: str) -> RiskResult | None:
        """直接读取已提交（可能来自上次进程）结果，不等待。"""
        if task_id in self.results:
            return self.results[task_id]
        result = self.journal.get_result(task_id)
        if result is not None:
            self.results[task_id] = result
        return result

    def get_task_status(self, task_id: str) -> TaskStatus:
        return self.tasks.status(task_id)

    def cancel_task(self, task_id: str) -> bool:
        return self.tasks.cancel(task_id)

    def recalculate(self, task_id: str) -> str:
        """对已终止（FAILED/CANCELLED）的同键任务重新入队。

        COMMITTED 任务直接返回原 id（结果不变，不重复计算）；
        未知任务抛 KeyError。
        """
        st = self.tasks.status(task_id)
        if st.state is TaskState.COMMITTED:
            return task_id
        parts = task_id.split(":", 3)
        if len(parts) != 4 or parts[0] != _TASK_PREFIX:
            raise ValueError(f"非法任务 id: {task_id}")
        _, portfolio_id, dv_str, cfg_ver = parts
        data_version = int(dv_str)
        # 从 WAL 最后一条含 config 的记录恢复配置
        config = self._load_task_config(task_id)
        # 移除旧的终态条目，允许同 id 重新调度
        self.tasks.replace_entry(task_id)
        return self._enqueue(task_id, portfolio_id, cfg_ver,
                             config=config, data_version=data_version)

    def _load_task_config(self, task_id: str) -> RiskConfig:
        from market_risk.storage import _read_jsonl
        config = None
        for d in _read_jsonl(self.journal.tasks_path):
            if d.get("task_id") == task_id and d.get("config"):
                config = d["config"]
        if config is None:
            raise RiskEngineError(
                ErrorCode.VERSION_CONFLICT,
                f"任务 {task_id} 缺少配置快照，无法重新计算")
        return _config_from_dict(config)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.tasks.shutdown(wait=True)

    def __enter__(self) -> "RiskEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
