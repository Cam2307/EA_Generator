"""Edge-first discovery: metrics, probes, and execution expansion."""
from __future__ import annotations

import random

import pytest

from factory.edge import (
    DEFAULT_EXECUTION_MECHANICS,
    break_even_win_rate,
    edge_probe_trade_mgmt,
    edge_score,
    expand_execution_variants,
    has_signal_edge,
    is_edge_probe,
    is_execution_variant,
    payoff_ratio,
)
from factory.generator import random_strategy
from factory.models import (
    BacktestMetrics, ExecutionMechanicType, TakeProfitMode,
)


def test_break_even_win_rate_at_1_to_1():
    assert break_even_win_rate(1.0) == 0.5
    assert break_even_win_rate(2.0) == pytest.approx(1.0 / 3.0)


def test_has_signal_edge_requires_wr_vs_rr():
    # 40% WR at 2:1 RR clears break-even (~33%) with cushion.
    good = BacktestMetrics(
        trade_count=20, net_profit=100.0, profit_factor=1.3,
        win_rate=0.40, avg_win=20.0, avg_loss=10.0, expectancy=2.0,
    )
    assert has_signal_edge(good)
    assert edge_score(good) > 0

    # High WR but negative expectancy fails.
    bad = BacktestMetrics(
        trade_count=20, net_profit=-50.0, profit_factor=0.8,
        win_rate=0.70, avg_win=5.0, avg_loss=20.0, expectancy=-2.5,
    )
    assert not has_signal_edge(bad)
    assert edge_score(bad) < 0


def test_payoff_ratio():
    m = BacktestMetrics(avg_win=30.0, avg_loss=10.0)
    assert payoff_ratio(m) == 3.0


def test_random_strategy_edge_phase_is_rr_probe():
    rng = random.Random(7)
    strat = random_strategy(
        "EURUSD", "M15", rng=rng, search_phase="edge",
        allowed_mechanics=[
            ExecutionMechanicType.DCA_GRID,
            ExecutionMechanicType.PARTIAL_CLOSE,
        ],
    )
    assert strat.mechanic.type == ExecutionMechanicType.STANDARD_SLTP
    assert strat.lineage.role == "edge"
    assert strat.profile.search_phase == "edge"
    assert is_edge_probe(strat)
    assert not is_execution_variant(strat)
    # FX edge probe uses R:R take-profit (or percent on non-FX).
    assert strat.trade_mgmt.tp_mode in (
        TakeProfitMode.RR, TakeProfitMode.PERCENT,
    )


def test_edge_probe_trade_mgmt_non_fx_uses_percent():
    tm = edge_probe_trade_mgmt("BTCUSD", random.Random(1))
    assert tm.sl_mode.value == "PERCENT"
    assert tm.tp_mode.value == "PERCENT"


def test_edge_probe_rr_biased_to_break_even_band():
    """Most FX edge probes should land tp_rr in [1.5, 3.0]."""
    from factory.models import StopLossMode

    rrs = []
    atr_count = 0
    for i in range(200):
        tm = edge_probe_trade_mgmt("EURUSD", random.Random(i))
        if tm.tp_mode == TakeProfitMode.RR:
            rrs.append(float(tm.params.get("tp_rr", 0)))
        if tm.sl_mode == StopLossMode.ATR:
            atr_count += 1
    assert rrs
    in_band = sum(1 for r in rrs if 1.5 <= r <= 3.0)
    assert in_band / len(rrs) >= 0.55
    # ATR preferred (~80%) over FIXED.
    assert atr_count / 200 >= 0.70


def test_expand_execution_variants_reuses_filters():
    rng = random.Random(11)
    edge = random_strategy("EURUSD", "H1", rng=rng, search_phase="edge")
    variants = expand_execution_variants(
        edge,
        mechanics=[
            ExecutionMechanicType.STANDARD_SLTP,
            ExecutionMechanicType.PARTIAL_CLOSE,
        ],
        tm_features=["trailing", "breakeven", "risk_reward_tp", "adaptive_sl"],
        rng=rng,
        max_variants=4,
    )
    assert 1 <= len(variants) <= 4
    edge_filters = {f.type for f in edge.entry_filters}
    for v in variants:
        assert is_execution_variant(v)
        assert v.lineage.edge_id == edge.id
        assert edge.id in v.lineage.parents
        assert {f.type for f in v.entry_filters}.issubset(edge_filters)
        assert v.profile.search_phase == "execution"


def test_default_execution_mechanics_are_defined_risk():
    assert ExecutionMechanicType.STANDARD_SLTP.value in DEFAULT_EXECUTION_MECHANICS
    assert ExecutionMechanicType.PARTIAL_CLOSE.value in DEFAULT_EXECUTION_MECHANICS
    assert ExecutionMechanicType.DCA_GRID.value not in DEFAULT_EXECUTION_MECHANICS
