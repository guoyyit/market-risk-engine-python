"""并发任务调度：有界线程池、每任务取消令牌、状态机与结果读取。

并发安全保证：

- 每个 task 有独立 Future 与 threading.Event 取消令牌，任务之间状态
  完全隔离（同一数据集的多个任务互不干扰）；
- 终态（COMMITTED / FAILED / CANCELLED）只落一次，由条件更新保护；
- 取消只在任务尚未取得终态时生效；已提交结果的任务不可取消；
- 工作函数通过取消令牌协作式检查，取消后不会调用提交回调，因此不留
  中间状态；
- 查询超时只影响调用方等待，不影响后台任务，结果在完成后可再次读取。
"""

from __future__ import annotations

import threading
import concurrent.futures
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from market_risk.models import (
    ErrorCode,
    TaskCancelledError,
    TaskState,
    TaskStatus,
)

T = TypeVar("T")


@dataclass(slots=True)
class CancelToken:
    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise TaskCancelledError()

    def wait(self, timeout: float | None) -> bool:
        """等待 timeout 秒且未被取消返回 True。"""
        return not self._event.wait(timeout)


@dataclass(slots=True)
class _Entry(Generic[T]):
    task_id: str
    portfolio_id: str
    token: CancelToken
    lock: threading.Lock = field(default_factory=threading.Lock)
    done: threading.Event = field(default_factory=threading.Event)
    state: TaskState = TaskState.PENDING
    result: T | None = None
    error_code: str | None = None
    error_reason: str | None = None
    future: concurrent.futures.Future | None = None


class TaskManager(Generic[T]):
    """通用任务管理器；T 为提交结果类型（通常为 RiskResult）。"""

    def __init__(self, max_workers: int = 4) -> None:
        if max_workers < 1:
            raise ValueError("max_workers 必须 >=1")
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="risk-task"
        )
        self._tasks: dict[str, _Entry[T]] = {}
        self._guard = threading.Lock()

    # ------------------------------------------------------------------
    def submit(
        self,
        task_id: str,
        portfolio_id: str,
        fn: Callable[[CancelToken], T],
        on_commit: Callable[[str, T], None] | None = None,
        on_failure: Callable[[str, ErrorCode, str], None] | None = None,
        on_cancel: Callable[[str], None] | None = None,
    ) -> None:
        """登记并提交一个任务。fn 接收 CancelToken，返回完整结果。"""
        with self._guard:
            if task_id in self._tasks:
                raise KeyError(f"任务已存在: {task_id}")
            entry = _Entry[T](task_id=task_id, portfolio_id=portfolio_id,
                              token=CancelToken())
            self._tasks[task_id] = entry

        def work() -> None:
            with entry.lock:
                if entry.token.cancelled:
                    entry.state = TaskState.CANCELLED
                    entry.done.set()
                    if on_cancel:
                        on_cancel(task_id)
                    return
                entry.state = TaskState.RUNNING
            try:
                result = fn(entry.token)
            except TaskCancelledError:
                with entry.lock:
                    if entry.state not in (
                        TaskState.COMMITTED, TaskState.FAILED
                    ):
                        entry.state = TaskState.CANCELLED
                if on_cancel:
                    on_cancel(task_id)
            except Exception as exc:  # noqa: BLE001 - 记录一切任务失败
                code = getattr(exc, "code", ErrorCode.INVALID_PARAMETER)
                reason = getattr(exc, "reason", str(exc))
                with entry.lock:
                    if entry.state not in (
                        TaskState.COMMITTED, TaskState.CANCELLED
                    ):
                        entry.state = TaskState.FAILED
                        entry.error_code = (
                            code.value if hasattr(code, "value") else str(code)
                        )
                        entry.error_reason = reason
                if on_failure:
                    on_failure(task_id, code, reason)
            else:
                with entry.lock:
                    if entry.token.cancelled:
                        entry.state = TaskState.CANCELLED
                        entry.done.set()
                        if on_cancel:
                            on_cancel(task_id)
                        return
                    entry.result = result
                    entry.state = TaskState.COMMITTED
                    entry.done.set()
                if on_commit:
                    on_commit(task_id, result)
            finally:
                entry.done.set()

        entry.future = self._pool.submit(work)

    # ------------------------------------------------------------------
    def cancel(self, task_id: str) -> bool:
        """请求取消；返回是否成功转为 CANCELLED。

        已 COMMITTED / FAILED / CANCELLED 的任务返回 False。
        """
        entry = self._tasks.get(task_id)
        if entry is None:
            return False
        with entry.lock:
            if entry.state in (
                TaskState.COMMITTED, TaskState.FAILED, TaskState.CANCELLED
            ):
                return False
            entry.token.cancel()
            if entry.state in (TaskState.PENDING,):
                entry.state = TaskState.CANCELLED
        # 若仍在运行，由工作函数观察令牌后收尾；若已排队将在 work() 开头收尾
        if entry.future is not None:
            entry.future.cancel()
        return True

    def get(self, task_id: str, timeout: float | None = None) -> T:
        """读取已提交的完整结果；未完成则等待 timeout 秒。"""
        entry = self._tasks.get(task_id)
        if entry is None:
            raise KeyError(task_id)
        if not entry.done.wait(timeout):
            raise TimeoutError(f"等待任务结果超时: {task_id}")
        with entry.lock:
            if entry.state is TaskState.COMMITTED:
                return entry.result  # type: ignore[return-value]
            if entry.state is TaskState.CANCELLED:
                raise TaskCancelledError(entry.error_reason or "任务已取消")
            from market_risk.models import RiskEngineError
            raise RiskEngineError(
                ErrorCode(entry.error_code) if entry.error_code
                else ErrorCode.INVALID_PARAMETER,
                entry.error_reason or "任务失败",
            )

    def status(self, task_id: str) -> TaskStatus:
        entry = self._tasks.get(task_id)
        if entry is None:
            raise KeyError(task_id)
        with entry.lock:
            return TaskStatus(
                task_id=entry.task_id,
                state=entry.state,
                portfolio_id=entry.portfolio_id,
                error_code=entry.error_code,
                error_reason=entry.error_reason,
            )

    def has(self, task_id: str) -> bool:
        return task_id in self._tasks

    def replace_entry(self, task_id: str) -> None:
        """移除一个已终止任务条目，使同 id 可被重新提交（重算/重试用）。

        仅允许在任务已完成（done）后调用，运行中条目拒绝替换。
        """
        with self._guard:
            entry = self._tasks.get(task_id)
            if entry is None:
                return
            if not entry.done.is_set():
                raise RuntimeError(f"任务仍在运行，不能替换: {task_id}")
            del self._tasks[task_id]

    def register_terminal(
        self,
        task_id: str,
        portfolio_id: str,
        state: TaskState,
        result: T | None = None,
        error_code: str | None = None,
        error_reason: str | None = None,
    ) -> None:
        """登记一个已有终态的任务（崩溃恢复用），不触发任何执行/回调。"""
        with self._guard:
            if task_id in self._tasks:
                return
            entry = _Entry[T](task_id=task_id, portfolio_id=portfolio_id,
                              token=CancelToken(), state=state, result=result,
                              error_code=error_code, error_reason=error_reason)
            entry.token.cancel()
            entry.done.set()
            self._tasks[task_id] = entry

    def task_ids(self) -> list[str]:
        with self._guard:
            return list(self._tasks)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=not wait)
