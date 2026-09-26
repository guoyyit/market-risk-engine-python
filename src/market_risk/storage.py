"""崩溃恢复持久化：append-only 事件日志 + 原子结果文件。

目录布局::

    <path>/
      batches.jsonl      行情批次（追加，恢复时重放重建内存数据）
      portfolios.jsonl   组合注册（追加，去重）
      tasks.wal          任务状态事件流（追加，启动时 checkpoint 压缩）
      results/<id>.json  已提交结果（tmp+fsync+rename，只可能完整出现）

崩溃恢复语义：

- 结果文件是「已提交」的权威标志：results/<id>.json 经原子 rename 落地，
  外部永远只能读到完整文件；
- 任务在结果文件落地后、COMMITTED 日志写入前崩溃 => 恢复时补登
  COMMITTED（幂等），不重复计算提交；
- WAL 末态为 PENDING/RUNNING 且无结果文件 => 恢复为 PENDING 重新入队，
  不丢任务。
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from market_risk.models import (
    HistoryBatch,
    Portfolio,
    Position,
    PriceObservation,
    RiskResult,
    SimulationMethod,
    TaskState,
)


@dataclass(slots=True)
class TaskRecord:
    task_id: str
    portfolio_id: str
    config_version: str
    state: TaskState
    error_code: str | None = None
    error_reason: str | None = None
    config: dict | None = None  # RiskConfig 参数（恢复未完成任务用）


# ---------------------------------------------------------------------------
# JSON 编解码（日期/枚举）
# ---------------------------------------------------------------------------
def _obs_to_dict(o: PriceObservation) -> dict:
    return {"instrument": o.instrument, "day": o.day.isoformat(),
            "price": o.price}


def _obs_from_dict(d: dict) -> PriceObservation:
    from datetime import date
    return PriceObservation(d["instrument"], date.fromisoformat(d["day"]),
                            float(d["price"]))


def _result_to_dict(r: RiskResult) -> dict:
    return {
        "portfolio_id": r.portfolio_id,
        "data_version": r.data_version,
        "config_version": r.config_version,
        "var": r.var,
        "es": r.es,
        "sample_size": r.sample_size,
        "as_of": r.as_of.isoformat(),
        "method": r.method.value,
    }


def _result_from_dict(d: dict) -> RiskResult:
    from datetime import date
    return RiskResult(
        portfolio_id=d["portfolio_id"],
        data_version=int(d["data_version"]),
        config_version=d["config_version"],
        var=float(d["var"]),
        es=float(d["es"]),
        sample_size=int(d["sample_size"]),
        as_of=date.fromisoformat(d["as_of"]),
        method=SimulationMethod(d["method"]),
    )


def _record_to_dict(rec: TaskRecord) -> dict:
    out = {
        "task_id": rec.task_id,
        "portfolio_id": rec.portfolio_id,
        "config_version": rec.config_version,
        "state": rec.state.value,
        "error_code": rec.error_code,
        "error_reason": rec.error_reason,
    }
    if rec.config is not None:
        out["config"] = rec.config
    return out


def _record_from_dict(d: dict) -> TaskRecord:
    return TaskRecord(
        task_id=d["task_id"],
        portfolio_id=d["portfolio_id"],
        config_version=d["config_version"],
        state=TaskState(d["state"]),
        error_code=d.get("error_code"),
        error_reason=d.get("error_reason"),
        config=d.get("config"),
    )


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _append_jsonl(path: Path, obj: dict, fsync_file: bool = True) -> None:
    """单行 JSON 追加；O_APPEND 保证事件不交错，fsync 保证落盘。"""
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        if fsync_file:
            os.fsync(f.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # 末尾可能是崩溃撕裂行（未 fsync 的半行）：停止重放，
                # 此前的行均为已确认事件。
                break
    return out


class Journal:
    """基于目录的任务日志、行情/组合日志与结果存储。"""

    def __init__(self, path: str | Path) -> None:
        self.root = Path(path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.root / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.batches_path = self.root / "batches.jsonl"
        self.snapshot_path = self.root / "market.snapshot.json"
        self.portfolios_path = self.root / "portfolios.jsonl"
        self.tasks_path = self.root / "tasks.wal"

    # -- 行情批次 --------------------------------------------------------
    def append_batch(self, batch: HistoryBatch) -> None:
        _append_jsonl(self.batches_path, {
            "as_of": batch.as_of.isoformat() if batch.as_of else None,
            "observations": [_obs_to_dict(o) for o in batch.observations],
        })

    def write_market_snapshot(
        self,
        series_by_instrument: dict[str, tuple[tuple, tuple]],
        as_of,
        version: int,
    ) -> None:
        """原子写出当前全部序列快照，并清空增量批次日志。

        快照布局：每标的仅保留（已在内存裁剪过的）有序 (days, prices)，
        因此持久化体积与 max_history_days 同阶，长期运行有界。
        """
        payload = {
            "version": version,
            "as_of": as_of.isoformat() if as_of is not None else None,
            "instruments": [
                {"instrument": inst,
                 "days": [d.isoformat() for d in days],
                 "prices": list(prices)}
                for inst, (days, prices) in
                sorted(series_by_instrument.items())
            ],
        }
        fd, tmp_name = tempfile.mkstemp(prefix=".snapshot.", suffix=".tmp",
                                        dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self.snapshot_path)
            _fsync_dir(self.root)
            # 快照已涵盖此前所有批次，截断增量日志
            with open(self.batches_path, "w", encoding="utf-8") as f:
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    def load_market(self) -> tuple[list[HistoryBatch], object, int]:
        """返回 (快照之后的增量批次, 快照 as_of, 快照版本)。"""
        snapshot_as_of = None
        snapshot_version = 0
        if self.snapshot_path.exists():
            from datetime import date
            try:
                with open(self.snapshot_path, encoding="utf-8") as f:
                    payload = json.load(f)
                obs = []
                for inst_block in payload.get("instruments", []):
                    for day_s, price in zip(inst_block["days"],
                                           inst_block["prices"],
                                           strict=True):
                        obs.append(PriceObservation(
                            inst_block["instrument"],
                            date.fromisoformat(day_s), float(price)))
                snap_batch = HistoryBatch(
                    observations=tuple(obs),
                    as_of=(date.fromisoformat(payload["as_of"])
                           if payload.get("as_of") else None))
                snapshot_as_of = snap_batch.as_of
                snapshot_version = int(payload.get("version", 0))
                # 快照作为单一重建批次
                batches = [snap_batch]
            except (json.JSONDecodeError, KeyError, ValueError):
                batches = []
        else:
            batches = []
        batches.extend(self.load_batches())
        return batches, snapshot_as_of, snapshot_version

    def load_batches(self) -> list[HistoryBatch]:
        from datetime import date
        batches: list[HistoryBatch] = []
        for d in _read_jsonl(self.batches_path):
            as_of = date.fromisoformat(d["as_of"]) if d.get("as_of") else None
            batches.append(HistoryBatch(
                observations=tuple(_obs_from_dict(o)
                                   for o in d["observations"]),
                as_of=as_of,
            ))
        return batches

    # -- 组合 ------------------------------------------------------------
    def append_portfolio(self, portfolio: Portfolio) -> None:
        _append_jsonl(self.portfolios_path, {
            "portfolio_id": portfolio.portfolio_id,
            "positions": [
                {"instrument": p.instrument, "quantity": p.quantity}
                for p in portfolio.positions
            ],
        })

    def load_portfolios(self) -> dict[str, Portfolio]:
        out: dict[str, Portfolio] = {}
        for d in _read_jsonl(self.portfolios_path):
            out[d["portfolio_id"]] = Portfolio(
                portfolio_id=d["portfolio_id"],
                positions=tuple(
                    Position(p["instrument"], float(p["quantity"]))
                    for p in d["positions"]
                ),
            )
        return out

    # -- 任务事件 --------------------------------------------------------
    def append_task(self, record: TaskRecord) -> None:
        _append_jsonl(self.tasks_path, _record_to_dict(record))

    update_task = append_task

    def commit_result(self, record: TaskRecord, result: RiskResult) -> None:
        """原子提交：结果文件 tmp->rename->fsync 目录，再写 COMMITTED 事件。

        两步之间崩溃也安全：恢复时以结果文件为准补登 COMMITTED。
        """
        payload = _result_to_dict(result)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{record.task_id}.", suffix=".tmp",
            dir=self.results_dir,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self.results_dir / f"{record.task_id}.json")
            _fsync_dir(self.results_dir)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise
        committed = TaskRecord(
            task_id=record.task_id,
            portfolio_id=record.portfolio_id,
            config_version=record.config_version,
            state=TaskState.COMMITTED,
        )
        _append_jsonl(self.tasks_path, _record_to_dict(committed))
        return True

    def checkpoint_tasks(self, records: dict[str, TaskRecord]) -> None:
        """把压缩后的任务终态/当前态重写为新 WAL（有界日志）。"""
        fd, tmp_name = tempfile.mkstemp(prefix=".tasks.", suffix=".tmp",
                                        dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for rec in records.values():
                    f.write(json.dumps(
                        _record_to_dict(rec), ensure_ascii=False,
                        separators=(",", ":")) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self.tasks_path)
            _fsync_dir(self.root)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

    # -- 恢复 ------------------------------------------------------------
    def load(self) -> tuple[dict[str, TaskRecord], dict[str, RiskResult]]:
        """回放任务事件 + 扫描结果目录，返回 (任务记录, 已提交结果)。

        结果文件存在即视为 COMMITTED，并幂等补登日志；无结果文件的
        RUNNING 视为崩溃中断，由调用方重置为 PENDING。
        """
        records: dict[str, TaskRecord] = {}
        for d in _read_jsonl(self.tasks_path):
            rec = _record_from_dict(d)
            records[rec.task_id] = rec

        results: dict[str, RiskResult] = {}
        for path in sorted(self.results_dir.glob("*.json")):
            task_id = path.stem
            try:
                with open(path, encoding="utf-8") as f:
                    result = _result_from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError, ValueError):
                # 正常情况下原子 rename 保证不会出现；损坏文件直接忽略，
                # 对应任务按未完成处理。
                continue
            results[task_id] = result
            old = records.get(task_id)
            if old is None or old.state is not TaskState.COMMITTED:
                records[task_id] = TaskRecord(
                    task_id=task_id,
                    portfolio_id=(old.portfolio_id
                                  if old is not None else
                                  result.portfolio_id),
                    config_version=(old.config_version if old is not None
                                    else result.config_version),
                    state=TaskState.COMMITTED,
                )
                _append_jsonl(self.tasks_path,
                              _record_to_dict(records[task_id]))
        return records, results

    def get_result(self, task_id: str) -> RiskResult | None:
        path = self.results_dir / f"{task_id}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return _result_from_dict(json.load(f))
