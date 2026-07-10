"""Composite signal logic: all / any / majority (end-to-end)."""
import random

import numpy as np
import pandas as pd

from factory.backtest.simulator import SymbolSpec, compute_signals
from factory.generator import describe_rules, mutate, random_strategy
from factory.models import (
    EntryFilter, EntryFilterType, ExecutionMechanic, ExecutionMechanicType,
    StrategyDefinition,
)
from factory.mql5.renderer import render_ea

SPEC = SymbolSpec(point=0.0001, spread_points=0.0, slippage_points=0.0)


def _df(n=400, seed=2):
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0004, n))
    open_ = np.roll(close, 1)
    open_[0] = 1.10
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="15min", tz="utc"),
        "open": open_, "high": np.maximum(open_, close) + 0.0002,
        "low": np.minimum(open_, close) - 0.0002, "close": close,
        "volume": np.full(n, 100.0)})


def _two_filter_strategy(logic):
    return StrategyDefinition(
        symbol="TEST", timeframe="M15", signal_logic=logic,
        entry_filters=[
            EntryFilter(type=EntryFilterType.RSI_REVERSION,
                        params={"rsi_period": 5, "oversold": 45,
                                "overbought": 55}),
            EntryFilter(type=EntryFilterType.STOCHASTIC,
                        params={"k_period": 5, "oversold": 45,
                                "overbought": 55}),
        ],
        mechanic=ExecutionMechanic(type=ExecutionMechanicType.STANDARD_SLTP,
                                   params={"sl_points": 300, "tp_points": 300}))


def test_or_is_superset_and_is_subset():
    df = _df()
    l_and, s_and, _ = compute_signals(df, _two_filter_strategy("all"), SPEC)
    l_or, s_or, _ = compute_signals(df, _two_filter_strategy("any"), SPEC)
    l_maj, s_maj, _ = compute_signals(df, _two_filter_strategy("majority"), SPEC)

    # AND implies OR on every bar; OR must fire strictly more often here
    assert np.all(l_or[l_and])
    assert np.all(s_or[s_and])
    assert l_or.sum() > l_and.sum()
    # with 2 filters, majority needs 2 votes -> identical to AND
    assert np.array_equal(l_maj, l_and)
    assert np.array_equal(s_maj, s_and)


def test_single_filter_logic_is_irrelevant():
    df = _df()
    base = _two_filter_strategy("all")
    base.entry_filters = base.entry_filters[:1]
    for logic in ("all", "any", "majority"):
        strat = base.model_copy(deep=True)
        strat.signal_logic = logic
        l, s, _ = compute_signals(df, strat, SPEC)
        if logic == "all":
            ref_l, ref_s = l, s
        else:
            assert np.array_equal(l, ref_l) and np.array_equal(s, ref_s)


def test_renderer_emits_hits_rule():
    strat = _two_filter_strategy("any")
    strat.name = "Logic Any"
    src = render_ea(strat)
    assert "int hits = 0;" in src
    assert "return(hits > 0);" in src
    assert ") hits++;" in src

    strat_all = _two_filter_strategy("all")
    strat_all.name = "Logic All"
    assert "return(hits == 2);" in render_ea(strat_all)

    three = _two_filter_strategy("majority")
    three.entry_filters.append(EntryFilter(
        type=EntryFilterType.WILLIAMS_R,
        params={"wpr_period": 14, "wpr_oversold": -80,
                "wpr_overbought": -20}))
    three.name = "Logic Maj"
    assert "return(hits >= 2);" in render_ea(three)


def test_generator_samples_and_mutates_logic():
    rng = random.Random(4)
    seen = set()
    for _ in range(300):
        s = random_strategy("EURUSD", "M15", rng=rng)
        seen.add(s.signal_logic)
        if len(s.entry_filters) < 2:
            assert s.signal_logic == "all"
    assert {"all", "any"} <= seen        # non-AND logics actually sampled

    base = _two_filter_strategy("all")
    flipped = False
    for _ in range(100):
        m = mutate(base, rng)
        if m.signal_logic != "all":
            flipped = True
            assert any("signal_logic" in x for x in m.lineage.mutations)
    assert flipped
    assert "ANY single filter" in describe_rules(_two_filter_strategy("any"))
