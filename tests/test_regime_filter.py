"""Adaptive regime-filter overlay: generator -> simulator -> MQL5 export."""
import random

import numpy as np
import pandas as pd

from factory.backtest import simulator as sim
from factory.backtest.simulator import SymbolSpec, run_simulation
from factory.generator import describe_trade_mgmt, random_trade_mgmt
from factory.models import (
    EntryFilter, EntryFilterType, ExecutionMechanic, ExecutionMechanicType,
    RiskBlock, StrategyDefinition, TradeManagement,
)
from factory.mql5.renderer import mql5_inputs_for, render_ea
from factory.regime import allowed_by_mask, classify_regimes_filter

SPEC = SymbolSpec(point=0.01, contract_size=100.0, leverage=100.0,
                  spread_points=0.0, slippage_points=0.0)


def _strategy(tm):
    return StrategyDefinition(
        symbol="TEST", timeframe="H1",
        entry_filters=[EntryFilter(
            type=EntryFilterType.RSI_REVERSION,
            params={"rsi_period": 14, "oversold": 30, "overbought": 70})],
        mechanic=ExecutionMechanic(
            type=ExecutionMechanicType.STANDARD_SLTP,
            params={"sl_points": 1000.0, "tp_points": 0.0}),
        risk=RiskBlock(fixed_lots=0.10, max_open_lots=50.0),
        trade_mgmt=tm)


def _trend_then_chop(n_trend=400, n_chop=400):
    """First half strong trend, second half tight oscillation."""
    trend = 100.0 + np.arange(n_trend) * 0.08
    chop = trend[-1] + 0.15 * np.sin(np.arange(n_chop) * 2 * np.pi / 10)
    closes = np.concatenate([trend, chop])
    n = len(closes)
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="1h", tz="utc"),
        "open": closes, "high": closes + 0.02, "low": closes - 0.02,
        "close": closes, "volume": np.ones(n)})


def test_generator_populates_regime_params():
    rng = random.Random(0)
    found = False
    for _ in range(60):
        tm = random_trade_mgmt(ExecutionMechanicType.STANDARD_SLTP, rng,
                               allowed=["regime_filter"])
        if tm.regime_filter:
            found = True
            for key in ("regime_allow_mask", "regime_adx_period",
                        "regime_adx_min", "regime_atr_period",
                        "regime_atr_mult"):
                assert key in tm.params and key in tm.ranges
            assert 1 <= tm.params["regime_allow_mask"] <= 14
            assert any("Regime filter" in ln for ln in describe_trade_mgmt(tm))
    assert found


def test_allowed_by_mask():
    codes = np.array([0, 1, 2, 3], dtype=np.int8)
    assert allowed_by_mask(codes, 0b0010).tolist() == [False, True, False, False]
    assert allowed_by_mask(codes, 0b1111).tolist() == [True] * 4
    assert allowed_by_mask(codes, 0b1010).tolist() == [False, True, False, True]


def _regime_tm(mask):
    return TradeManagement(
        regime_filter=True,
        params={"regime_allow_mask": float(mask), "regime_adx_period": 14,
                "regime_adx_min": 25.0, "regime_atr_period": 14,
                "regime_atr_mult": 1.25})


def test_simulator_gates_entries_by_regime(monkeypatch):
    df = _trend_then_chop()
    n = len(df)
    # force a long signal on every bar so only the regime gate decides
    monkeypatch.setattr(sim, "compute_signals",
                        lambda *a, **k: (np.ones(n, dtype=bool),
                                         np.zeros(n, dtype=bool), 1))

    codes = classify_regimes_filter(df, 14, 14, 25.0, 1.25)
    trend_mask = 0b1010          # trend regimes only (codes 1 and 3)
    range_mask = 0b0101          # range regimes only (codes 0 and 2)

    _, book_trend = run_simulation(df, _strategy(_regime_tm(trend_mask)),
                                   SPEC, 100_000.0)
    _, book_range = run_simulation(df, _strategy(_regime_tm(range_mask)),
                                   SPEC, 100_000.0)

    trend_ok = allowed_by_mask(codes, trend_mask)
    range_ok = allowed_by_mask(codes, range_mask)
    bar_of = {float(t): i for i, t in enumerate(
        pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        .to_numpy().astype("datetime64[s]").astype("int64"))}

    assert book_trend.closed and book_range.closed
    for tr in book_trend.closed:
        assert trend_ok[bar_of[tr.open_time]]
    for tr in book_range.closed:
        assert range_ok[bar_of[tr.open_time]]

    # unrestricted overlay trades everywhere both gated books traded
    _, book_all = run_simulation(df, _strategy(TradeManagement()),
                                 SPEC, 100_000.0)
    assert len(book_all.closed) >= max(len(book_trend.closed),
                                       len(book_range.closed))


def _sizing_tm(qr=1.0, qt=1.0, vr=1.0, vt=1.0):
    return TradeManagement(
        regime_sizing=True,
        params={"regime_adx_period": 14, "regime_adx_min": 25.0,
                "regime_atr_period": 14, "regime_atr_mult": 1.25,
                "regime_size_quiet_range": qr, "regime_size_quiet_trend": qt,
                "regime_size_vol_range": vr, "regime_size_vol_trend": vt})


def test_simulator_scales_lots_by_regime(monkeypatch):
    df = _trend_then_chop()
    n = len(df)
    monkeypatch.setattr(sim, "compute_signals",
                        lambda *a, **k: (np.ones(n, dtype=bool),
                                         np.zeros(n, dtype=bool), 1))
    codes = classify_regimes_filter(df, 14, 14, 25.0, 1.25)
    bar_of = {float(t): i for i, t in enumerate(
        pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        .to_numpy().astype("datetime64[s]").astype("int64"))}

    # quiet-trend entries at 2x lots, everything else at 0.5x; tight SL/TP
    # so positions cycle and entries land in both regimes
    strat = _strategy(_sizing_tm(qr=0.5, qt=2.0, vr=0.5, vt=0.5))
    strat.mechanic.params = {"sl_points": 10.0, "tp_points": 10.0}
    _, book = run_simulation(df, strat, SPEC, 100_000.0)
    assert book.closed
    seen_trend = seen_other = False
    for tr in book.closed:
        code = codes[bar_of[tr.open_time]]
        if code == 1:
            assert tr.lots == 0.20        # 0.10 base x 2.0
            seen_trend = True
        else:
            assert tr.lots == 0.05        # 0.10 base x 0.5
            seen_other = True
    assert seen_trend and seen_other


def test_generator_populates_sizing_params():
    rng = random.Random(1)
    found = False
    for _ in range(60):
        tm = random_trade_mgmt(ExecutionMechanicType.STANDARD_SLTP, rng,
                               allowed=["regime_sizing"])
        if tm.regime_sizing:
            found = True
            for key in ("regime_size_quiet_range", "regime_size_quiet_trend",
                        "regime_size_vol_range", "regime_size_vol_trend",
                        "regime_adx_period", "regime_atr_mult"):
                assert key in tm.params and key in tm.ranges
            assert not tm.regime_filter or "regime_allow_mask" in tm.params
            assert any("Regime sizing" in ln for ln in describe_trade_mgmt(tm))
    assert found


def test_rendered_ea_contains_sizing_block():
    strat = _strategy(_sizing_tm(qr=0.5, qt=2.0, vr=0.75, vt=1.5))
    strat.name = "Sizing Test"
    src = render_ea(strat)
    assert "Inp_X_regime_sizing            = 1;" in src
    assert "Inp_X_regime_size_quiet_trend  = 2;" in src
    assert "TM_RegimeLotMult" in src
    assert "NormalizeLots(lots * TM_RegimeLotMult())" in src
    inputs, ranges = mql5_inputs_for(strat)
    assert inputs["Inp_X_regime_size_vol_range"] == 0.75


def test_rendered_ea_contains_regime_block():
    tm = _regime_tm(0b1010)
    strat = _strategy(tm)
    strat.name = "Regime Test"
    src = render_ea(strat)
    assert "input int    Inp_X_regime_filter      = 1;" in src
    assert "Inp_X_regime_allow_mask  = 10;" in src
    assert "TM_RegimeAllowed" in src
    assert "iADX(_Symbol, _Period, Inp_X_regime_adx_period)" in src
    assert "IndicatorRelease(g_rg_adx_handle)" in src
    # inputs/ranges export includes the regime parameters
    inputs, _ranges = mql5_inputs_for(strat)
    assert inputs["Inp_X_regime_allow_mask"] == 10.0

    # disabled overlay renders the neutral no-op defaults
    off = render_ea(_strategy(TradeManagement()))
    assert "input int    Inp_X_regime_filter      = 0;" in off
