"""有界内存、并发度配置、大批量场景无漂移。"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from market_risk.engine import RiskEngine
from market_risk.models import (
    HistoryBatch,
    Portfolio,
    Position,
    PriceObservation,
    RiskConfig,
    SimulationMethod,
)

from conftest import default_config, gen_prices


def _prices(n, seed=1):
    return gen_prices(n, seed=seed)


def test_history_is_bounded_by_max_history_days(tmp_path):
    """每个标的保留长度 <= max_history_days，旧数据被淘汰。"""
    cap = 100
    eng = RiskEngine(tmp_path, max_history_days=cap)
    prices = _prices(500)
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(500)]
    for start in (0, 250):
        obs = tuple(PriceObservation("A", days[i], prices[i])
                    for i in range(start, start + 250))
        eng.submit_history(HistoryBatch(obs, as_of=days[start + 249]))
    assert len(eng.store.get("A").days) == cap
    # 保留的是最新 cap 天
    assert eng.store.get("A").days[0] == days[-cap]
    eng.close()


def test_many_instruments_bounded(tmp_path):
    """多标的场景：每标的均受 cap 限制，计算只取窗口。"""
    cap, n_inst, n_days = 60, 50, 120
    eng = RiskEngine(tmp_path, max_concurrency=4,
                     max_history_days=cap)
    all_obs = []
    for j in range(n_inst):
        prices = _prices(n_days, seed=j + 1)
        for i, p in enumerate(prices):
            all_obs.append(PriceObservation(
                f"I{j:03d}", date(2024, 1, 1) + timedelta(days=i), p))
    eng.submit_history(HistoryBatch(
        tuple(all_obs), as_of=date(2024, 1, 1) + timedelta(n_days - 1)))
    for j in range(n_inst):
        assert len(eng.store.get(f"I{j:03d}").days) == cap
    # 注册并并发计算 20 个组合
    for j in range(20):
        eng.register_portfolio(
            Portfolio(f"p{j}", (Position(f"I{j:03d}", 1.0),)))
    cfg = default_config(window=40, confidence=0.95)
    tids = [eng.calculate(f"p{j}", cfg) for j in range(20)]
    results = [eng.get_result(t, 30) for t in tids]
    assert all(r.sample_size == 39 for r in results)
    eng.close()


def test_long_running_no_unbounded_growth(tmp_path):
    """持续追加观测 + 计算：内存增长受 cap 与结果集合约束。

    采用 tracemalloc 比较「早期稳态」与「追加大量批次后」的引擎相关
    分配，序列侧增量应趋近 0（仅被淘汰替换）。
    """
    cap = 50
    eng = RiskEngine(tmp_path, max_concurrency=2,
                     max_history_days=cap)
    eng.submit_history(HistoryBatch(tuple(
        PriceObservation("A", date(2024, 1, 1) + timedelta(days=i),
                         100.0 + i)
        for i in range(cap)),
        as_of=date(2024, 1, 1) + timedelta(days=cap - 1)))
    eng.register_portfolio(Portfolio("p", (Position("A", 1.0),)))
    cfg = default_config(window=20)

    def series_bytes():
        s = eng.store.get("A")
        return sum(sys.getsizeof(x) for x in s.days) + \
               sum(sys.getsizeof(x) for x in s.prices)

    base = series_bytes()
    # 再追加 400 天（必然淘汰），序列长度与占用保持有界
    for k in range(cap, cap + 400):
        eng.submit_history(HistoryBatch((
            PriceObservation("A", date(2024, 1, 1) + timedelta(days=k),
                             100.0 + k),),
            as_of=date(2024, 1, 1) + timedelta(days=k)))
    assert len(eng.store.get("A").days) == cap
    after = series_bytes()
    # 元素数量固定，占用差应极小（date 与 float 为常量级对象）
    assert after <= base * 1.5 + 4096
    # 只计算最新版本一次，验证结果文件数量与结果缓存受控
    tid = eng.calculate("p", cfg)
    eng.get_result(tid, 10)
    assert len(list((tmp_path / "results").glob("*.json"))) == 1
    eng.close()


def test_no_result_drift_on_repeat_runs(tmp_path):
    """同一数据/配置多次计算结果完全一致（无随机性/无漂移）。"""
    eng = RiskEngine(tmp_path, max_concurrency=4)
    prices = _prices(120, seed=42)
    days = [date(2024, 1, 1) + timedelta(days=i)
            for i in range(120)]
    eng.submit_history(HistoryBatch(tuple(
        PriceObservation("A", days[i], prices[i]) for i in range(120)),
        as_of=days[-1]))
    eng.register_portfolio(Portfolio("p", (Position("A", 2.0),)))
    cfgs = [
        default_config(confidence=.99, window=50),
        RiskConfig(confidence=.95, window=80,
                   method=SimulationMethod.DECAYED, decay_lambda=.98),
        RiskConfig(confidence=.90, window=30, horizon=5),
    ]
    # 同键任务只计算一次；为验证可复现性，另建独立引擎重算
    first = {}
    for c in cfgs:
        t = eng.calculate("p", c)
        first[c.version_key()] = eng.get_result(t, 30)
    eng.close()

    eng2 = RiskEngine(tmp_path)
    for c in cfgs:
        t = eng2.calculate("p", c)
        r = eng2.get_result(t, 30)
        f = first[c.version_key()]
        assert (r.var, r.es) == (f.var, f.es)
    eng2.close()


def test_large_batch_performance_bound(tmp_path):
    """大批量（5 万观测）端到端在合理时间内完成，窗口计算不扫全历史。"""
    import time
    n_days, n_inst = 1000, 50
    eng = RiskEngine(tmp_path, max_concurrency=4,
                     max_history_days=300)
    obs = []
    base_day = date(2022, 1, 1)
    for j in range(n_inst):
        prices = _prices(n_days, seed=100 + j)
        for i, p in enumerate(prices):
            obs.append(PriceObservation(
                f"I{j:02d}", base_day + timedelta(days=i), p))
    t0 = time.perf_counter()
    eng.submit_history(HistoryBatch(
        tuple(obs), as_of=base_day + timedelta(days=n_days - 1)))
    for j in range(n_inst):
        eng.register_portfolio(
            Portfolio(f"p{j}", (Position(f"I{j:02d}", 1.0),)))
    cfg = default_config(window=100, confidence=0.99)
    t0 = time.perf_counter()
    tids = [eng.calculate(f"p{j}", cfg) for j in range(n_inst)]
    results = [eng.get_result(t, 60) for t in tids]
    t_calc = time.perf_counter() - t0
    assert all(r.sample_size == 99 for r in results)
    # 宽松的 CI 友好上界，主要防止全历史 O(n^2) 退化
    assert t_calc < 10.0, t_calc
    eng.close()


def test_market_snapshot_keeps_persistence_bounded(tmp_path):
    """快照压缩：批次日志被截断，重启后数据与结果与压缩前一致。"""
    from market_risk.storage import _read_jsonl
    eng = RiskEngine(tmp_path, max_concurrency=2, max_history_days=40)
    eng._snapshot_interval = 5
    prices = _prices(60, seed=21)
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(60)]
    # 分 12 批提交，触发两次快照
    for start in range(0, 60, 5):
        eng.submit_history(HistoryBatch(
            tuple(PriceObservation("A", days[i], prices[i])
                  for i in range(start, start + 5)),
            as_of=days[start + 4]))
    eng.register_portfolio(Portfolio("p", (Position("A", 1.0),)))
    cfg = default_config(window=30, confidence=0.95)
    tid = eng.calculate("p", cfg)
    r = eng.get_result(tid, 10)
    # 快照后增量日志行数 < 间隔阈值（被截断过）
    batch_lines = len(_read_jsonl(eng.journal.batches_path))
    assert batch_lines < 5
    assert (tmp_path / "market.snapshot.json").exists()
    # 内存序列受 cap 约束
    assert len(eng.store.get("A").days) == 40
    assert eng.store.version == 12
    eng.close()

    # 重启：快照 + 剩余增量重建，版本/序列/结果一致
    eng2 = RiskEngine(tmp_path, max_history_days=40)
    assert eng2.store.version == 12
    assert len(eng2.store.get("A").days) == 40
    assert eng2.get_committed_result(tid).var == r.var
    cfg2 = default_config(window=30, confidence=0.95)
    t2 = eng2.calculate("p", cfg2)
    r2 = eng2.get_result(t2, 10)
    assert r2.var == r.var and r2.es == r.es
    eng2.close()
