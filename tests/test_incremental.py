"""增量更新：新增观测只重算受影响结果，且与全量重算逐位一致。

一致性策略：增量路径（数据分多批提交）与全量路径（一次性提交全部数据）
在相同 data_version 与相同 config 下，VaR/ES 必须相等；滚动窗口淘汰旧
数据（max_history_days）后同样一致；权重配置变化产生独立结果，不与旧
缓存混用。
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from market_risk.engine import RiskEngine, _task_id
from market_risk.models import (
    HistoryBatch,
    Portfolio,
    Position,
    PriceObservation,
    RiskConfig,
    SimulationMethod,
)

from conftest import default_config, gen_prices


def _portfolio():
    return Portfolio("p1", (Position("A", 1.0),))


def _result(engine, pf, cfg, version=None):
    tid = engine.calculate(pf.portfolio_id, cfg, data_version=version)
    return engine.get_result(tid, timeout=10)


def _build(path, batch_sizes, prices, *, max_history_days=None,
           as_of_final=True):
    kwargs = {"max_concurrency": 2}
    if max_history_days is not None:
        kwargs["max_history_days"] = max_history_days
    eng = RiskEngine(path, **kwargs)
    days = [date(2025, 1, 1) + timedelta(days=i)
            for i in range(len(prices))]
    start = 0
    versions = []
    for size in batch_sizes:
        chunk = [PriceObservation("A", days[i], prices[i])
                 for i in range(start, start + size)]
        as_of = days[start + size - 1] if as_of_final else None
        eng.submit_history(
            HistoryBatch(tuple(chunk), as_of=as_of))
        versions.append(eng.store.version)
        start += size
    eng.register_portfolio(_portfolio())
    return eng


@pytest.mark.parametrize("batches", [
    [30],            # 单批（基准）
    [10, 10, 10],    # 均匀三批
    [1, 1, 28],      # 逐日后大批
    [25, 1, 1, 1, 2],
    [29, 1],
])
def test_incremental_matches_full_rebuild(tmp_path, batches):
    prices = gen_prices(30, seed=11)
    inc = _build(tmp_path / "inc", batches, prices)
    full = _build(tmp_path / "full", [30], prices)
    for cfg in [
        default_config(confidence=0.90, window=5),
        default_config(confidence=0.95, window=10),
        default_config(confidence=0.99, window=20),
        RiskConfig(confidence=0.95, window=10,
                   method=SimulationMethod.DECAYED, decay_lambda=0.97),
        RiskConfig(confidence=0.90, window=8, horizon=3),
    ]:
        r_inc = _result(inc, _portfolio(), cfg)
        r_full = _result(full, _portfolio(), cfg)
        assert r_inc.var == pytest.approx(r_full.var, abs=1e-12), cfg
        assert r_inc.es == pytest.approx(r_full.es, abs=1e-12), cfg
        assert r_inc.sample_size == r_full.sample_size


def test_incremental_versions_stepwise_match(tmp_path):
    """每个中间版本的增量结果都与该时点的全量重算一致。"""
    prices = gen_prices(20, seed=3)
    days = [date(2025, 1, 1) + timedelta(days=i)
            for i in range(len(prices))]
    inc = RiskEngine(tmp_path / "inc", max_concurrency=2)
    inc.register_portfolio(_portfolio())
    cfg = default_config(window=8, confidence=0.90)
    for k in range(1, 21):
        chunk = [PriceObservation("A", days[k - 1], prices[k - 1])]
        inc.submit_history(HistoryBatch(tuple(chunk), as_of=days[k - 1]))
        if k < 9:
            continue
        r_inc = _result(inc, _portfolio(), cfg)
        # 全量：另建引擎一次性喂入前 k 天
        full = RiskEngine(tmp_path / f"full{k}", max_concurrency=2)
        full.register_portfolio(_portfolio())
        full.submit_history(HistoryBatch(
            tuple(PriceObservation("A", days[i], prices[i])
                  for i in range(k)), as_of=days[k - 1]))
        r_full = _result(full, _portfolio(), cfg)
        assert r_inc.var == pytest.approx(r_full.var, abs=1e-12), k
        assert r_inc.es == pytest.approx(r_full.es, abs=1e-12), k
        full.close()
    inc.close()


def test_eviction_by_max_history_matches_full(tmp_path):
    """有界内存淘汰旧数据后，增量结果与仅含保留窗口的全量结果一致。"""
    prices = gen_prices(30, seed=5)
    # 增量引擎保留最近 12 天；分 3 批提交触发淘汰
    inc = _build(tmp_path / "inc", [10, 10, 10], prices,
                 max_history_days=12)
    assert len(inc.store.get("A").days) == 12
    # 全量基准：只喂最后 12 天
    full = RiskEngine(tmp_path / "full", max_concurrency=2,
                      max_history_days=12)
    days = [date(2025, 1, 1) + timedelta(days=i)
            for i in range(len(prices))]
    full.submit_history(HistoryBatch(tuple(
        PriceObservation("A", days[i], prices[i])
        for i in range(18, 30)), as_of=days[29]))
    full.register_portfolio(_portfolio())
    cfg = default_config(window=12, confidence=0.95)
    r_inc = _result(inc, _portfolio(), cfg)
    r_full = _result(full, _portfolio(), cfg)
    assert r_inc.var == pytest.approx(r_full.var, abs=1e-12)
    assert r_inc.es == pytest.approx(r_full.es, abs=1e-12)


def test_old_version_results_remain_after_new_observations(tmp_path):
    prices = gen_prices(15, seed=2)
    days = [date(2025, 1, 1) + timedelta(days=i)
            for i in range(len(prices))]
    eng = RiskEngine(tmp_path / "e", max_concurrency=2)
    eng.submit_history(HistoryBatch(tuple(
        PriceObservation("A", days[i], prices[i]) for i in range(10)),
        as_of=days[9]))
    eng.register_portfolio(_portfolio())
    cfg = default_config(window=8)
    r_v1 = _result(eng, _portfolio(), cfg, version=1)
    eng.submit_history(HistoryBatch(tuple(
        PriceObservation("A", days[i], prices[i]) for i in range(10, 15)),
        as_of=days[14]))
    r_v2 = _result(eng, _portfolio(), cfg)
    assert r_v1.data_version == 1 and r_v2.data_version == 2
    # 旧结果仍可读取且不变
    tid_v1 = _task_id("p1", 1, cfg.version_key())
    cached = eng.get_committed_result(tid_v1)
    assert cached.var == r_v1.var and cached.es == r_v1.es


def test_weight_change_does_not_reuse_old_cache(tmp_path):
    """窗口/权重变化后已缓存结果不得与旧配置混用。"""
    prices = gen_prices(25, seed=9)
    days = [date(2025, 1, 1) + timedelta(days=i)
            for i in range(len(prices))]
    eng = RiskEngine(tmp_path / "e", max_concurrency=2)
    eng.submit_history(HistoryBatch(tuple(
        PriceObservation("A", days[i], prices[i])
        for i in range(25)), as_of=days[24]))
    eng.register_portfolio(_portfolio())
    c1 = RiskConfig(confidence=.95, window=10,
                    method=SimulationMethod.DECAYED, decay_lambda=.90)
    c2 = RiskConfig(confidence=.95, window=10,
                    method=SimulationMethod.DECAYED, decay_lambda=.99)
    r1 = _result(eng, _portfolio(), c1)
    r2 = _result(eng, _portfolio(), c2)
    t1 = _task_id("p1", 1, c1.version_key())
    t2 = _task_id("p1", 1, c2.version_key())
    assert t1 != t2 and r1.config_version != r2.config_version
    assert eng.get_committed_result(t1).config_version == c1.version_key()
    assert eng.get_committed_result(t2).config_version == c2.version_key()


def test_incremental_only_affected_tasks_recompute(tmp_path):
    """同一版本结果复用、新版本才产生新任务。"""
    prices = gen_prices(20, seed=4)
    days = [date(2025, 1, 1) + timedelta(days=i)
            for i in range(len(prices))]
    eng = RiskEngine(tmp_path / "e", max_concurrency=2)
    eng.submit_history(HistoryBatch(tuple(
        PriceObservation("A", days[i], prices[i]) for i in range(15)),
        as_of=days[14]))
    eng.register_portfolio(_portfolio())
    cfg = default_config(window=8)
    t1 = eng.calculate("p1", cfg)
    eng.get_result(t1, 10)
    # 相同请求复用任务，不重算
    assert eng.calculate("p1", cfg) == t1
    eng.submit_history(HistoryBatch(
        (PriceObservation("A", days[15], prices[15]),), as_of=days[15]))
    t2 = eng.calculate("p1", cfg)
    assert t2 != t1
    eng.get_result(t2, 10)
    assert eng.tasks.status(t1).state.value == "COMMITTED"
    assert eng.tasks.status(t2).state.value == "COMMITTED"
