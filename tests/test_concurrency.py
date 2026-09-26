"""并发计算/查询：任务隔离、并发度有界、取消与超时不留中间状态。"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pytest

from market_risk.models import (
    HistoryBatch,
    Portfolio,
    Position,
    PriceObservation,
    TaskCancelledError,
    TaskState,
)

from conftest import default_config, gen_prices


def _seed(engine, n_inst=3, n_days=30):
    obs = []
    prices = gen_prices(n_days)
    last_day = date(2025, 1, 1) + timedelta(days=n_days - 1)
    for j in range(n_inst):
        inst = f"I{j}"
        for i, p in enumerate(prices):
            obs.append(PriceObservation(
                inst, date(2025, 1, 1) + timedelta(days=i),
                p * (1 + 0.1 * j)))
    engine.submit_history(
        HistoryBatch(tuple(obs), as_of=last_day))
    for j in range(n_inst):
        engine.register_portfolio(
            Portfolio(f"p{j}", (Position(f"I{j}", 1.0 + j),)))


def test_many_concurrent_tasks_isolation(tmp_path):
    from market_risk.engine import RiskEngine
    eng = RiskEngine(tmp_path, max_concurrency=4)
    _seed(eng, n_inst=5, n_days=40)
    cfgs = [default_config(confidence=c, window=w)
            for c in (0.90, 0.95, 0.99) for w in (5, 10, 20, 30)]
    ids: list[str] = []
    lock = threading.Lock()

    def submit(pf, cfg):
        tid = eng.calculate(pf, cfg)
        with lock:
            ids.append(tid)
        return eng.get_result(tid, timeout=30)

    jobs = [(f"p{j}", cfg) for j in range(5) for cfg in cfgs]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda j: submit(*j), jobs))
    assert len(results) == len(jobs)
    assert len(set(ids)) == len(jobs)  # 每键一任务
    for r in results:
        assert r.var >= 0 and r.es >= 0
    eng.close()


def test_duplicate_submit_collapses_to_one_task(tmp_path):
    from market_risk.engine import RiskEngine
    eng = RiskEngine(tmp_path, max_concurrency=4)
    _seed(eng)
    cfg = default_config()
    tids = set()
    def submit():
        tids.add(eng.calculate("p0", cfg))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: submit(), range(20)))
    assert len(tids) == 1
    eng.close()


def test_cancel_running_task_no_partial_state(tmp_path, monkeypatch):
    """取消运行中任务：状态 CANCELLED，无结果文件，可重新提交。"""
    from market_risk.engine import RiskEngine

    eng = RiskEngine(tmp_path, max_concurrency=1)
    _seed(eng)
    cfg = default_config(window=30)

    # 让计算在窗口读取后停顿，给取消留出窗口
    from market_risk.series import MarketDataStore
    orig = MarketDataStore.common_window
    started = threading.Event()
    release = threading.Event()

    def slow_window(self, instruments, window):
        v = orig(self, instruments, window)
        started.set()
        release.wait(3)
        return v
    monkeypatch.setattr(MarketDataStore, "common_window", slow_window)

    tid = eng.calculate("p0", cfg)
    assert started.wait(2)
    assert eng.cancel_task(tid) is True
    # 释放后工作函数应观察到取消；不得写出结果
    release.set()
    with pytest.raises(TaskCancelledError):
        eng.get_result(tid, timeout=5)
    assert eng.get_task_status(tid).state is TaskState.CANCELLED
    assert eng.get_committed_result(tid) is None
    monkeypatch.undo()
    # 取消后用相同键重新入队计算
    tid2 = eng.recalculate(tid)
    r = eng.get_result(tid2, timeout=10)
    assert r is not None and r.sample_size == 29
    assert eng.get_task_status(tid).state is TaskState.COMMITTED
    eng.close()


def test_cancel_committed_returns_false(tmp_path):
    from market_risk.engine import RiskEngine
    eng = RiskEngine(tmp_path, max_concurrency=2)
    _seed(eng)
    tid = eng.calculate("p0", default_config())
    eng.get_result(tid, 10)
    assert eng.cancel_task(tid) is False
    assert eng.get_task_status(tid).state is TaskState.COMMITTED
    eng.close()


def test_query_timeout_keeps_result_readable(tmp_path, monkeypatch):
    from market_risk.engine import RiskEngine
    from market_risk.series import MarketDataStore
    eng = RiskEngine(tmp_path, max_concurrency=1)
    _seed(eng)
    cfg = default_config(window=30)
    orig = MarketDataStore.common_window
    blocker = threading.Event()

    def slow(self, i, w):
        blocker.wait(2)
        return orig(self, i, w)
    monkeypatch.setattr(MarketDataStore, "common_window", slow)
    tid = eng.calculate("p0", cfg)
    with pytest.raises(TimeoutError):
        eng.get_result(tid, timeout=0.05)
    blocker.set()
    r = eng.get_result(tid, timeout=10)
    assert r.sample_size == 29
    eng.close()


def test_concurrency_bound_is_configurable(tmp_path):
    """有界线程池：大量任务时活跃线程数不超过 max_concurrency。"""
    from market_risk.engine import RiskEngine
    eng = RiskEngine(tmp_path, max_concurrency=2)
    _seed(eng, n_inst=5, n_days=30)
    active = {"n": 0, "max": 0}
    guard = threading.Lock()
    from market_risk.series import MarketDataStore
    orig = MarketDataStore.common_window

    def tracked(self, i, w):
        with guard:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        time.sleep(0.02)
        v = orig(self, i, w)
        with guard:
            active["n"] -= 1
        return v
    mp = pytest.MonkeyPatch()
    mp.setattr(MarketDataStore, "common_window", tracked)
    tids = [eng.calculate(f"p{j % 5}", default_config(window=6 + (j % 20)))
            for j, k in enumerate(range(30))]
    try:
        for t in set(tids):
            eng.get_result(t, 30)
        assert active["max"] <= 2
    finally:
        mp.undo()
    eng.close()
