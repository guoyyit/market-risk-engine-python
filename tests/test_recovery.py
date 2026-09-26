"""崩溃恢复：已提交与未完成状态可区分，恢复后不重复提交/不丢任务。"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from market_risk.engine import (
    RiskEngine,
    _config_to_dict,
    _task_id,
)
from market_risk.models import (
    HistoryBatch,
    Portfolio,
    Position,
    PriceObservation,
    TaskState,
)
from market_risk.storage import TaskRecord

from conftest import default_config, gen_prices


def _seed(eng: RiskEngine, n=30, inst="A"):
    prices = gen_prices(n)
    obs = [PriceObservation(inst, date(2025, 1, 1) + timedelta(days=i), p)
           for i, p in enumerate(prices)]
    eng.submit_history(HistoryBatch(tuple(obs), as_of=obs[-1].day))
    eng.register_portfolio(Portfolio("p1", (Position(inst, 1.0),)))
    return prices


def test_replay_rebuilds_data_and_portfolios(tmp_path):
    eng = RiskEngine(tmp_path, max_concurrency=2)
    _seed(eng, 25)
    cfg = default_config(window=10)
    tid = eng.calculate("p1", cfg)
    r = eng.get_result(tid, 10)
    eng.close()

    eng2 = RiskEngine(tmp_path, max_concurrency=2)
    assert eng2.store.version == 1
    assert len(eng2.store.get("A").days) == 25
    assert "p1" in eng2.portfolios
    # 已提交结果无需重新计算即可读
    r2 = eng2.get_committed_result(tid)
    assert r2 is not None and (r2.var, r2.es) == (r.var, r.es)
    assert eng2.get_task_status(tid).state is TaskState.COMMITTED
    assert eng2.recover() == 0
    eng2.close()


def test_pending_and_running_tasks_requeued(tmp_path):
    eng = RiskEngine(tmp_path, max_concurrency=1)
    _seed(eng, 30)
    c1 = default_config(confidence=.90, window=8)
    c2 = default_config(confidence=.99, window=20)
    # 已完成的
    done_tid = eng.calculate("p1", c1)
    eng.get_result(done_tid, 10)
    # 模拟崩溃时分别处于 PENDING / RUNNING 的两个任务
    pending_tid = _task_id("p1", 1, c2.version_key())
    eng.journal.append_task(TaskRecord(
        pending_tid, "p1", c2.version_key(), TaskState.PENDING,
        config=_config_to_dict(c2)))
    c3 = default_config(confidence=.95, window=12)
    running_tid = _task_id("p1", 1, c3.version_key())
    eng.journal.append_task(TaskRecord(
        running_tid, "p1", c3.version_key(), TaskState.RUNNING,
        config=_config_to_dict(c3)))
    eng.close()

    eng2 = RiskEngine(tmp_path, max_concurrency=4)
    # 已提交与未完成可区分
    assert eng2.get_task_status(done_tid).state is TaskState.COMMITTED
    # PENDING/RUNNING 在恢复调用后重新入队
    n = eng2.recover()
    assert n == 2
    rp = eng2.get_result(pending_tid, 10)
    rr = eng2.get_result(running_tid, 10)
    assert rp.sample_size == 19 and rr.sample_size == 11
    eng2.close()

    # 再次重启：全部已提交，无重复任务
    eng3 = RiskEngine(tmp_path)
    assert eng3.recover() == 0
    assert eng3.get_committed_result(pending_tid).var == rp.var
    assert eng3.get_committed_result(running_tid).var == rr.var
    eng3.close()


def test_orphan_result_file_heals_wal(tmp_path):
    """结果文件已落地但 COMMITTED 日志缺失 => 恢复时补登，不重复计算。"""
    eng = RiskEngine(tmp_path, max_concurrency=2)
    _seed(eng, 20)
    cfg = default_config(window=10)
    tid = eng.calculate("p1", cfg)
    r = eng.get_result(tid, 10)
    eng.close()
    # 删除 tasks.wal（模拟「结果已 rename、日志事件丢失」），保留结果文件
    (tmp_path / "tasks.wal").unlink()
    eng2 = RiskEngine(tmp_path)
    assert eng2.recover() == 0  # 不重算
    healed = eng2.get_committed_result(tid)
    assert healed is not None and healed.var == r.var
    assert eng2.get_task_status(tid).state is TaskState.COMMITTED
    eng2.close()


def test_torn_wal_line_ignored(tmp_path):
    """WAL 末尾撕裂半行不影响此前已确认事件。"""
    eng = RiskEngine(tmp_path)
    _seed(eng, 20)
    cfg = default_config(window=10)
    tid = eng.calculate("p1", cfg)
    eng.get_result(tid, 10)
    eng.close()
    with open(tmp_path / "tasks.wal", "a") as f:
        f.write('{"task_id": "task:x", "state": "RUNNIN')  # 半行
    eng2 = RiskEngine(tmp_path)
    assert eng2.get_task_status(tid).state is TaskState.COMMITTED
    eng2.close()


def test_recovery_does_not_duplicate_commit(tmp_path):
    """恢复后重新计算的任务，结果文件内容与手工提交一致且只一个文件。"""
    eng = RiskEngine(tmp_path)
    _seed(eng, 24)
    c = default_config(window=9)
    crash = _task_id("p1", 1, c.version_key())
    eng.journal.append_task(TaskRecord(
        crash, "p1", c.version_key(), TaskState.PENDING,
        config=_config_to_dict(c)))
    eng.close()

    eng2 = RiskEngine(tmp_path)
    eng2.recover()
    r = eng2.get_result(crash, 10)
    eng2.close()
    files = list((tmp_path / "results").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["var"] == r.var


def test_failed_task_state_persists_and_recalculate_works(tmp_path):
    eng = RiskEngine(tmp_path)
    # 只有 1 个价格点的标的 -> 计算必失败
    eng.submit_history(HistoryBatch(
        (PriceObservation("A", date(2025, 1, 1), 100.0),),
        as_of=date(2025, 1, 1)))
    eng.register_portfolio(Portfolio("p1", (Position("A", 1.0),)))
    cfg = default_config(window=10)
    tid = eng.calculate("p1", cfg)
    with pytest.raises(Exception):
        eng.get_result(tid, 10)
    eng.close()

    eng2 = RiskEngine(tmp_path)
    st = eng2.get_task_status(tid)
    assert st.state is TaskState.FAILED
    assert st.error_code == "INSUFFICIENT_HISTORY"
    eng2.close()


def test_wal_is_bounded_by_checkpoint(tmp_path):
    """大量任务事件后 WAL 被压缩为每任务一条，恢复语义不变。"""
    from market_risk.storage import _read_jsonl
    eng = RiskEngine(tmp_path, max_concurrency=4)
    eng._wal_checkpoint_interval = 50
    prices = gen_prices(60)
    obs = [PriceObservation("A", date(2025, 1, 1) + timedelta(days=i), p)
           for i, p in enumerate(prices)]
    eng.submit_history(HistoryBatch(tuple(obs), as_of=obs[-1].day))
    eng.register_portfolio(Portfolio("p1", (Position("A", 1.0),)))
    # 制造 > 50 个事件：不同窗口/置信度各提交一次（每个至少 PENDING+COMMITTED）
    tids = []
    for w in range(5, 30):
        cfg = default_config(confidence=0.90, window=w)
        t = eng.calculate("p1", cfg)
        tids.append(t)
        eng.get_result(t, 10)
    raw_lines = len(_read_jsonl(eng.journal.tasks_path))
    # 25 个任务生命周期产生 50+ 事件；checkpoint 把 WAL 压缩为每任务一条。
    assert raw_lines == 25, raw_lines
    eng.close()

    eng2 = RiskEngine(tmp_path)
    assert eng2.recover() == 0
    for t in tids:
        assert eng2.get_task_status(t).state is TaskState.COMMITTED
    eng2.close()
