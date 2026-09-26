"""滚动窗口与衰减加权历史模拟的口径测试（可复现的手工算例）。"""

from __future__ import annotations

from datetime import date

import pytest

from market_risk.models import (
    Portfolio,
    Position,
    RiskConfig,
    SimulationMethod,
)
from market_risk.series import WindowView
from market_risk.simulation import (
    historical_var_es,
    portfolio_scenario_pnls,
    scenario_weights,
    compute_result,
)

DAYS = tuple(date(2025, 1, d) for d in range(1, 13))


def window_of(prices: tuple[float, ...]) -> WindowView:
    return WindowView(days=DAYS[: len(prices)], prices=(prices,))


def test_scenario_pnl_long_short_signs():
    # 100 -> 110 -> 99: 多/空头损益相反
    w = window_of((100.0, 110.0, 99.0))
    long_pnl = portfolio_scenario_pnls((Position("A", 1.0),), w)
    short_pnl = portfolio_scenario_pnls((Position("A", -1.0),), w)
    assert long_pnl == pytest.approx([-10.0, 11.0])
    assert short_pnl == pytest.approx([10.0, -11.0])


def test_equal_weight_quantiles_without_interpolation():
    # 损失序列（已排序）：-20,-5,5,10
    L = [-20.0, -5.0, 5.0, 10.0]
    w = [0.25] * 4
    # c=.75 alpha=.25：rank0 累计 .25 未严格超过 -> 分界 rank1(-5)
    var, es = historical_var_es(L, w, 0.75)
    assert var == 0.0
    assert es == pytest.approx((-5 + 5 + 10) / 3)
    # c=.5：分界 rank2(5)
    var, es = historical_var_es(L, w, 0.50)
    assert var == pytest.approx(5.0)
    assert es == pytest.approx(7.5)
    # c=.25：最坏点
    var, es = historical_var_es(L, w, 0.25)
    assert var == 10.0 and es == 10.0


def test_decay_weights_monotone_and_normalized():
    cfg = RiskConfig(method=SimulationMethod.DECAYED, decay_lambda=0.9)
    wts = scenario_weights(5, cfg)
    assert sum(wts) == pytest.approx(1.0)
    assert all(a < b for a, b in zip(wts, wts[1:]))  # 越新权重越高
    # w(age 0) = (1-lambda)（归一化前）
    assert wts[-1] > wts[0]


def test_decay_vs_equal_weight_differ():
    # 唯一大损失发生在最新情景：强衰减(lambda=.01)在 c=.90 下，
    # 前三个旧情景累计权重 ~0.092 <= alpha=.10，分位点落在最新的 100；
    # 等权下 rank0 累计 .25 即超 .10，分位点为 -5，VaR=0。
    losses = [-5.0, -5.0, -5.0, 100.0]
    eq = [0.25] * 4
    cfg = RiskConfig(method=SimulationMethod.DECAYED, decay_lambda=0.01)
    dec = scenario_weights(4, cfg)
    v_eq, _ = historical_var_es(losses, eq, 0.90)
    v_dec, e_dec = historical_var_es(losses, dec, 0.90)
    assert v_dec == 100.0
    assert e_dec == pytest.approx(100.0)
    assert v_eq == 0.0


def test_decay_older_loss_downweighted():
    # 大损失发生在最旧情景，衰减后其权重极低 -> VaR 可能为 0
    losses = [100.0, -5.0, -5.0, -5.0]
    cfg = RiskConfig(method=SimulationMethod.DECAYED, decay_lambda=0.2)
    dec = scenario_weights(4, cfg)
    var, es = historical_var_es(losses, dec, 0.90)
    # 最旧权重 0.8^? 实际 w_old 极小，累计在盈利区超过 0.1
    assert var == 0.0


def test_horizon_sqrt_scaling():
    w = window_of((100.0, 80.0))  # 单一情景损失 20
    pf = Portfolio("p", (Position("A", 1.0),))
    r1 = compute_result(pf, w, RiskConfig(confidence=0.99, window=2, horizon=1),
                        1, DAYS[1])
    r4 = compute_result(pf, w, RiskConfig(confidence=0.99, window=2, horizon=4),
                        1, DAYS[1])
    assert r1.var == pytest.approx(20.0)
    assert r4.var == pytest.approx(40.0)
    assert r4.es == pytest.approx(r1.es * 2.0)


def test_multi_instrument_aggregation():
    # 两标的等数量，同日相反方向波动 -> 组合损失为两部分之和
    w = WindowView(
        days=(date(2025, 1, 1), date(2025, 1, 2)),
        prices=((100.0, 90.0), (50.0, 55.0)),
    )
    pf = Portfolio("p", (Position("A", 1.0), Position("B", 2.0)))
    pnls = portfolio_scenario_pnls(pf.positions, w)
    # A: -(100*-0.1)=+10 损失；B: -(2*50*0.1)=-10 盈利 => 0
    assert pnls == pytest.approx([0.0])


def test_results_are_deterministic():
    w = window_of(tuple(100.0 + i * 0.37 for i in range(10)))
    pf = Portfolio("p", (Position("A", 3.0),))
    cfg = RiskConfig(confidence=0.97, window=9, decay_lambda=0.93)
    r1 = compute_result(pf, w, cfg, 3, DAYS[-1])
    r2 = compute_result(pf, w, cfg, 3, DAYS[-1])
    assert (r1.var, r1.es) == (r2.var, r2.es)
