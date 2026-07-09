"""Trade-management (exit / risk overlay) behaviour tests.

Symbol economics: point = 0.01, contract size = 100 -> point_value = $1 per
point per 1.0 lot; zero spread/slippage so fills land exactly on the bar close
and every number below is hand-checkable.
"""
import re

import numpy as np
import pandas as pd
import pytest

from factory.backtest import simulator as sim
from factory.backtest.simulator import (
    SymbolSpec, _entry_sizing, _tm_entry_allowed, run_simulation,
)
from factory.mql5.renderer import mql5_inputs_for, render_ea
from factory.models import (
    EntryFilter, EntryFilterType, ExecutionMechanic, ExecutionMechanicType,
    LotMode, RiskBlock, StopLossMode, StrategyDefinition, TakeProfitMode,
    TradeManagement, TrailMode,
)

SPEC = SymbolSpec(point=0.01, contract_size=100.0, leverage=100.0,
                  spread_points=0.0, slippage_points=0.0)


def _bars(closes, highs=None, lows=None):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs, dtype=float) if highs is not None else closes.copy()
    lows = np.asarray(lows, dtype=float) if lows is not None else closes.copy()
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="1h", tz="utc"),
        "open": closes, "high": highs, "low": lows, "close": closes,
        "volume": np.ones(n),
    })


def _std_strategy(tm=None, sl_points=1000.0, tp_points=0.0, lots=0.10):
    return StrategyDefinition(
        symbol="TEST", timeframe="H1",
        entry_filters=[EntryFilter(
            type=EntryFilterType.RSI_REVERSION,
            params={"rsi_period": 14, "oversold": 30, "overbought": 70})],
        mechanic=ExecutionMechanic(
            type=ExecutionMechanicType.STANDARD_SLTP,
            params={"sl_points": sl_points, "tp_points": tp_points}),
        risk=RiskBlock(fixed_lots=lots, max_open_lots=50.0),
        trade_mgmt=tm or TradeManagement(),
    )


def _signals_at(monkeypatch, n, bars):
    long_sig = np.zeros(n, dtype=bool)
    for b in bars:
        long_sig[b] = True
    monkeypatch.setattr(sim, "compute_signals",
                        lambda *a, **k: (long_sig, np.zeros(n, dtype=bool), 1))


# ---------------------------------------------------------------------------
# Full-loop behaviour
# ---------------------------------------------------------------------------

def test_fixed_trailing_locks_profit(monkeypatch):
    # entry @100, trail arms at +100pts, trails 100pts behind, then price
    # retraces into the trailed stop and locks a gain.
    closes = [100, 100, 100, 100, 100, 100, 101.0, 101.5, 100.0]
    lows = list(closes)
    lows[7] = 100.7
    lows[8] = 100.0
    df = _bars(closes, lows=lows)
    _signals_at(monkeypatch, len(df), [5])

    tm = TradeManagement(
        sl_mode=StopLossMode.FIXED, tp_mode=TakeProfitMode.OFF,
        trail_mode=TrailMode.FIXED,
        params={"trail_start_points": 100.0, "trail_distance_points": 100.0,
                "trail_step_points": 10.0})
    metrics, book = run_simulation(df, _std_strategy(tm=tm), SPEC, 10_000.0)

    assert metrics.trade_count == 1
    trade = book.closed[0]
    assert trade.exit_price == pytest.approx(100.5)   # trailed stop at bar7
    assert trade.profit == pytest.approx(5.0)         # +50 pts * 0.10 lot * $1


def test_breakeven_moves_stop_to_entry_plus_offset(monkeypatch):
    closes = [100, 100, 100, 100, 100, 100, 101.0, 100.0]
    lows = list(closes)
    lows[7] = 100.0
    df = _bars(closes, lows=lows)
    _signals_at(monkeypatch, len(df), [5])

    tm = TradeManagement(
        sl_mode=StopLossMode.FIXED, tp_mode=TakeProfitMode.OFF, breakeven=True,
        params={"be_trigger_points": 100.0, "be_offset_points": 20.0})
    metrics, book = run_simulation(df, _std_strategy(tm=tm), SPEC, 10_000.0)

    assert metrics.trade_count == 1
    trade = book.closed[0]
    assert trade.exit_price == pytest.approx(100.2)   # entry + 20pts
    assert trade.profit == pytest.approx(2.0)


def test_time_filter_blocks_out_of_session_entries(monkeypatch):
    # bar hour == index % 24; session 9..17 lets bar 10 through, blocks bar 3.
    closes = [100.0] * 14
    closes[10] = 101.0
    df = _bars(closes)
    _signals_at(monkeypatch, len(df), [3, 10])

    tm = TradeManagement(time_filter=True,
                         params={"start_hour": 9.0, "end_hour": 17.0})
    metrics, book = run_simulation(df, _std_strategy(tm=tm), SPEC, 10_000.0)

    assert metrics.trade_count == 1
    assert book.closed[0].entry_price == pytest.approx(101.0)  # entered at bar 10


def test_cooldown_after_loss_blocks_next_entry(monkeypatch):
    closes = [100.0] * 14
    lows = list(closes)
    lows[6] = 98.0                       # forces the bar-5 stop-out
    df = _bars(closes, lows=lows)
    _signals_at(monkeypatch, len(df), [5, 7, 12])

    tm = TradeManagement(sl_mode=StopLossMode.FIXED, tp_mode=TakeProfitMode.OFF,
                         cooldown_enabled=True, params={"cooldown_bars": 3.0})
    metrics, book = run_simulation(
        df, _std_strategy(tm=tm, sl_points=100.0), SPEC, 10_000.0)

    # bar5 loses; bar7 is inside the 3-bar cooldown (blocked); bar12 re-enters.
    assert metrics.trade_count == 2
    chronological = sorted(book.closed, key=lambda t: t.open_time)
    assert chronological[0].profit < 0


def test_max_trades_per_day_caps_entries(monkeypatch):
    closes = [100.0] * 14
    lows = list(closes)
    lows[6] = 99.0                       # stop out first trade quickly
    df = _bars(closes, lows=lows)
    _signals_at(monkeypatch, len(df), [5, 8])

    tm = TradeManagement(sl_mode=StopLossMode.FIXED, tp_mode=TakeProfitMode.OFF,
                         limit_trades_per_day=True,
                         params={"max_trades_per_day": 1.0})
    metrics, _ = run_simulation(
        df, _std_strategy(tm=tm, sl_points=100.0), SPEC, 10_000.0)

    assert metrics.trade_count == 1     # second same-day signal is capped


# ---------------------------------------------------------------------------
# Pure sizing / gating helpers
# ---------------------------------------------------------------------------

def test_entry_sizing_atr_stop_and_rr_target():
    tm = TradeManagement(sl_mode=StopLossMode.ATR, tp_mode=TakeProfitMode.RR,
                         params={"atr_sl_mult": 2.0, "tp_rr": 3.0})
    sl_pts, tp_pts, lots = _entry_sizing(
        tm, tm.params, ExecutionMechanicType.STANDARD_SLTP,
        {"sl_points": 500.0, "tp_points": 500.0}, True,
        atr_price=0.10, spec=SPEC, base_lots=0.10, equity=10_000.0,
        max_open_lots=50.0)
    assert sl_pts == pytest.approx(20.0)   # 0.10 / 0.01 * 2
    assert tp_pts == pytest.approx(60.0)   # 20 * 3
    assert lots == pytest.approx(0.10)


def test_entry_sizing_risk_percent_lots():
    tm = TradeManagement(sl_mode=StopLossMode.FIXED, lot_mode=LotMode.RISK_PERCENT,
                         params={"risk_percent": 1.0})
    sl_pts, _tp, lots = _entry_sizing(
        tm, tm.params, ExecutionMechanicType.STANDARD_SLTP,
        {"sl_points": 20.0, "tp_points": 0.0}, True,
        atr_price=0.0, spec=SPEC, base_lots=0.10, equity=10_000.0,
        max_open_lots=50.0)
    # risk $100 over a 20pt stop @ $1/pt/lot -> 5.0 lots
    assert sl_pts == pytest.approx(20.0)
    assert lots == pytest.approx(5.0)


def test_entry_sizing_grid_is_unmanaged():
    tm = TradeManagement(sl_mode=StopLossMode.ATR, lot_mode=LotMode.RISK_PERCENT,
                         params={"atr_sl_mult": 2.0, "risk_percent": 1.0})
    sl_pts, tp_pts, lots = _entry_sizing(
        tm, tm.params, ExecutionMechanicType.DCA_GRID, {}, False,
        atr_price=0.10, spec=SPEC, base_lots=0.10, equity=10_000.0,
        max_open_lots=50.0)
    assert (sl_pts, tp_pts, lots) == (0.0, 0.0, 0.10)


@pytest.mark.parametrize("hour,ok", [(8, False), (9, True), (16, True), (17, False)])
def test_tm_entry_allowed_session(hour, ok):
    tm = TradeManagement(time_filter=True,
                         params={"start_hour": 9.0, "end_hour": 17.0})
    assert _tm_entry_allowed(tm, tm.params, hour, 0, 10_000.0, 10_000.0, 5, -1) is ok


def test_tm_entry_allowed_daily_loss_and_cooldown():
    loss = TradeManagement(daily_loss_enabled=True, params={"daily_loss_pct": 5.0})
    assert _tm_entry_allowed(loss, loss.params, 12, 0, 10_000.0, 9_400.0, 5, -1) is False
    assert _tm_entry_allowed(loss, loss.params, 12, 0, 10_000.0, 9_700.0, 5, -1) is True

    cd = TradeManagement(cooldown_enabled=True, params={"cooldown_bars": 3.0})
    assert _tm_entry_allowed(cd, cd.params, 12, 0, 10_000.0, 10_000.0, 4, 6) is False
    assert _tm_entry_allowed(cd, cd.params, 12, 0, 10_000.0, 10_000.0, 6, 6) is True


# ---------------------------------------------------------------------------
# MQL5 export
# ---------------------------------------------------------------------------

def test_render_full_management_no_placeholders():
    tm = TradeManagement(
        sl_mode=StopLossMode.ATR, tp_mode=TakeProfitMode.RR,
        trail_mode=TrailMode.CHANDELIER, lot_mode=LotMode.RISK_PERCENT,
        breakeven=True, time_filter=True, limit_trades_per_day=True,
        daily_loss_enabled=True, cooldown_enabled=True,
        params={"atr_period": 14, "atr_sl_mult": 2.5, "tp_rr": 2.0,
                "trail_start_points": 300, "trail_atr_mult": 3.0,
                "chandelier_lookback": 22, "trail_step_points": 10,
                "be_trigger_points": 300, "be_offset_points": 20,
                "risk_percent": 1.0, "start_hour": 8, "end_hour": 20,
                "max_trades_per_day": 3, "daily_loss_pct": 5, "cooldown_bars": 5})
    strat = _std_strategy(tm=tm, sl_points=400.0, tp_points=800.0)
    strat.name = "TM Test EA"
    ea = render_ea(strat)

    assert not re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", ea)   # no {token}
    assert not re.findall(r"__[A-Z_]+__", ea)                  # no __TOKEN__
    for needle in ("TM_ManageTrailing", "TM_InitialStopPoints", "iATR(_Symbol",
                   "g_tm_trades_today++", "Inp_X_trail_mode"):
        assert needle in ea

    inputs, _ranges = mql5_inputs_for(strat)
    assert "Inp_X_atr_period" in inputs
    assert "Inp_X_risk_percent" in inputs


def test_generated_management_params_are_optimizable():
    """A generated strategy with an active overlay exposes X_ ranges to the
    optimizer / .set exporter, and apply_flat_params round-trips them."""
    import random

    from factory.generator import random_strategy

    for seed in range(300):
        s = random_strategy(
            "EURUSD", "M15", random.Random(seed),
            allowed_mechanics=[ExecutionMechanicType.STANDARD_SLTP])
        x_ranges = {k: r for k, r in s.all_ranges().items() if k.startswith("X_")}
        if x_ranges:
            break
    assert x_ranges, "no generated strategy exposed a trade-management range"

    key = next(iter(x_ranges))
    r = x_ranges[key]
    tuned = s.apply_flat_params({key: r.max})
    assert tuned.trade_mgmt.params[key[2:]] == r.max
