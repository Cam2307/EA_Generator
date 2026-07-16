"""Hand-checked DCA/grid scenario: sequential fills + floating-DD accounting.

Symbol economics are chosen so the arithmetic is exact:
point = 0.01, contract size = 100  ->  point_value = $1 per point per 1.0 lot.
Spread and slippage are zero, so every fill lands exactly on the bar close.

Scenario (base lots 0.10, grid step 100 pts, lot multiplier 2.0,
max levels 3, basket TP 50 pts, basket SL 500 pts — far enough not to fire):

bar 10  close 100.0  -> long entry 0.10 @ 100.0
bar 11  close  99.0  -> 100 pts adverse -> grid add 0.20 @ 99.0
bar 12  close  98.0  -> 100 pts adverse -> grid add 0.40 @ 98.0 (max levels)
bar 13  close  97.5  -> no add (only 50 pts, and grid is full)
        floating = 0.1*(-250) + 0.2*(-150) + 0.4*(-50) = -75  -> max DD $75
bar 14  high   99.5  -> basket TP at avg 98.571429 + 0.50 = 99.071429
        realized = 0.7 lots * 50 pts = +$35
"""
import numpy as np
import pandas as pd
import pytest

from factory.backtest import simulator as sim
from factory.backtest.simulator import PositionBook, SymbolSpec, run_simulation
from factory.models import (
    EntryFilter, EntryFilterType, ExecutionMechanic, ExecutionMechanicType,
    ParamRange, RiskBlock, StrategyDefinition,
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


def _dca_strategy(**extra) -> StrategyDefinition:
    params = {"grid_step_points": 100.0, "lot_multiplier": 2.0,
              "max_levels": 3.0, "basket_tp_points": 50.0,
              "basket_sl_points": 500.0}
    params.update(extra)
    mech = ExecutionMechanic(
        type=ExecutionMechanicType.DCA_GRID,
        params=params,
        ranges={"grid_step_points": ParamRange(min=50, max=500, step=50)},
    )
    return StrategyDefinition(
        symbol="TEST", timeframe="H1",
        entry_filters=[EntryFilter(
            type=EntryFilterType.RSI_REVERSION,
            params={"rsi_period": 14, "oversold": 30, "overbought": 70})],
        mechanic=mech, risk=RiskBlock(fixed_lots=0.10, max_open_lots=5.0),
    )


@pytest.fixture
def dca_run(monkeypatch):
    closes = [100.0] * 11 + [99.0, 98.0, 97.5, 99.5, 99.5]
    highs = list(closes)
    highs[14] = 99.5
    df = _bars(closes, highs=highs)

    n = len(df)
    long_sig = np.zeros(n, dtype=bool)
    long_sig[10] = True
    short_sig = np.zeros(n, dtype=bool)
    monkeypatch.setattr(sim, "compute_signals",
                        lambda *a, **k: (long_sig, short_sig, 1))

    return run_simulation(df, _dca_strategy(), SPEC, deposit=10_000.0)


def test_dca_sequential_fills(dca_run):
    metrics, book = dca_run
    assert metrics.trade_count == 3

    trades = sorted(book.closed, key=lambda t: t.open_time)
    assert [t.lots for t in trades] == pytest.approx([0.10, 0.20, 0.40])
    assert [t.entry_price for t in trades] == pytest.approx([100.0, 99.0, 98.0])
    # fills are strictly sequential in time
    assert trades[0].open_time < trades[1].open_time < trades[2].open_time
    # all three legs closed together at the basket TP
    avg = (100.0 * 0.1 + 99.0 * 0.2 + 98.0 * 0.4) / 0.7
    target = avg + 50 * SPEC.point
    for t in trades:
        assert t.exit_price == pytest.approx(target)
        assert t.close_time == trades[0].close_time


def test_dca_profit_and_floating_drawdown(dca_run):
    metrics, book = dca_run
    # basket TP is 50 points from the volume-weighted average on 0.7 lots
    assert metrics.net_profit == pytest.approx(35.0)
    assert book.balance == pytest.approx(10_035.0)
    # worst floating loss at bar 13 (close 97.5):
    # 0.1*(-250) + 0.2*(-150) + 0.4*(-50) = -75
    assert metrics.max_dd_money == pytest.approx(75.0)
    assert metrics.max_dd_pct == pytest.approx(0.75)
    assert not book.positions   # everything flat at the end


def test_dca_respects_max_levels(dca_run):
    _, book = dca_run
    assert len(book.closed) == 3    # never a 4th level despite bar 13 drop


def test_dca_shared_basket_sl_closes_all(monkeypatch):
    """Shared basket SL from VWAP closes every open leg at the same stop."""
    # Entry @ 100, add @ 99 → avg = (100*0.1 + 99*0.2)/0.3 = 99.333...
    # basket_sl 50 pts → stop = avg - 0.50 = 98.833...
    # bar 12 low punches through the shared stop → both legs exit together.
    closes = [100.0] * 11 + [99.0, 98.5, 98.5]
    lows = list(closes)
    lows[12] = 98.5
    df = _bars(closes, lows=lows)
    n = len(df)
    long_sig = np.zeros(n, dtype=bool)
    long_sig[10] = True
    monkeypatch.setattr(sim, "compute_signals",
                        lambda *a, **k: (long_sig, np.zeros(n, dtype=bool), 1))

    strat = _dca_strategy(max_levels=3.0, basket_tp_points=200.0,
                          basket_sl_points=50.0)
    metrics, book = run_simulation(df, strat, SPEC, deposit=10_000.0)

    assert metrics.trade_count == 2
    trades = sorted(book.closed, key=lambda t: t.open_time)
    assert [t.lots for t in trades] == pytest.approx([0.10, 0.20])
    avg = (100.0 * 0.1 + 99.0 * 0.2) / 0.3
    stop = avg - 50 * SPEC.point
    for t in trades:
        assert t.exit_price == pytest.approx(stop)
        assert t.close_time == trades[0].close_time
    assert not book.positions


def test_position_book_margin_refusal():
    book = PositionBook(spec=SPEC, balance=10.0)   # tiny account
    # 1 lot at price 100 needs 100*100/100 = $100 margin > $10 equity
    pos = book.open(direction=1, lots=1.0, bid=100.0, time_s=0.0)
    assert pos is None
    assert not book.positions


def test_spread_and_slippage_costs():
    spec = SymbolSpec(point=0.01, contract_size=100.0, leverage=100.0,
                      spread_points=10.0, slippage_points=2.0)
    book = PositionBook(spec=spec, balance=10_000.0)
    pos = book.open(direction=1, lots=0.1, bid=100.0, time_s=0.0)
    # buy pays spread + slippage above bid
    assert pos.entry_price == pytest.approx(100.0 + 12 * 0.01)
    sell = book.open(direction=-1, lots=0.1, bid=100.0, time_s=0.0)
    # sell pays only slippage below bid
    assert sell.entry_price == pytest.approx(100.0 - 2 * 0.01)


def test_standard_sltp_stop_hit(monkeypatch):
    closes = [100.0] * 6 + [99.0, 98.0]
    lows = list(closes)
    lows[7] = 97.0
    df = _bars(closes, lows=lows)
    n = len(df)
    long_sig = np.zeros(n, dtype=bool)
    long_sig[5] = True
    monkeypatch.setattr(sim, "compute_signals",
                        lambda *a, **k: (long_sig, np.zeros(n, dtype=bool), 1))

    strat = _dca_strategy().model_copy(deep=True)
    strat.mechanic = ExecutionMechanic(
        type=ExecutionMechanicType.STANDARD_SLTP,
        params={"sl_points": 100.0, "tp_points": 300.0})
    metrics, book = run_simulation(df, strat, SPEC, deposit=10_000.0)

    assert metrics.trade_count == 1
    trade = book.closed[0]
    assert trade.exit_price == pytest.approx(100.0 - 100 * SPEC.point)
    assert trade.profit == pytest.approx(-0.1 * 100)   # 0.1 lots * -100 pts
