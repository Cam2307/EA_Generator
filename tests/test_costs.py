"""Session-aware dynamic cost model (factory.backtest.costs)."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from factory.backtest import costs
from factory.backtest.simulator import PositionBook, SymbolSpec


def _bars(start: str, n: int, freq: str = "1h", base: float = 1.10,
          rng_pts: float = 0.0010) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq, tz="utc")
    close = np.full(n, base)
    return pd.DataFrame({
        "time": idx,
        "open": close,
        "high": close + rng_pts / 2,
        "low": close - rng_pts / 2,
        "close": close,
        "volume": np.full(n, 100.0),
    })


def test_rollover_spread_wider_than_overlap():
    hours = np.arange(24)
    weekdays = np.full(24, 2)          # Wednesday
    mult = costs.spread_multipliers(hours, weekdays)
    assert mult[21] > 2.0 * mult[13]   # rollover vs London/NY overlap
    assert mult[13] < 1.0              # overlap is tighter than base
    assert mult[2] > 1.0               # Asian session wider than base


def test_friday_late_and_sunday_widening():
    hours = np.array([20, 20, 20])
    weekdays = np.array([2, 4, 6])     # Wed, Fri, Sun at the same hour
    wed, fri, sun = costs.spread_multipliers(hours, weekdays)
    assert fri > wed                   # Friday pre-weekend widening
    assert sun > fri                   # Sunday open is the widest


def test_slippage_scales_with_volatility():
    high = np.full(300, 1.101)
    low = np.full(300, 1.100)
    high[250] = 1.105                  # one 5x-range shock bar
    mult = costs.slippage_multipliers(high, low)
    assert mult[250] > 2.0
    assert mult[100] == pytest.approx(1.0, abs=0.01)
    assert mult.min() >= costs.SLIPPAGE_VOL_MIN
    assert mult.max() <= costs.SLIPPAGE_VOL_MAX


def test_build_cost_arrays_shapes_and_base_scaling():
    df = _bars("2024-01-03 00:00", 48)
    spread, slip = costs.build_cost_arrays(df, spread_points=15.0,
                                           slippage_points=2.0)
    assert len(spread) == len(df) and len(slip) == len(df)
    # hour 21 of the first day (index 21) is the rollover spike
    assert spread[21] == pytest.approx(15.0 * costs.HOUR_SPREAD_MULT[21])
    # constant-range bars -> slippage stays at the base once warmed up
    assert slip[30] == pytest.approx(2.0, abs=0.05)


def test_fill_price_per_fill_override():
    spec = SymbolSpec(point=0.0001, spread_points=10.0, slippage_points=2.0)
    book = PositionBook(spec=spec, balance=10_000.0)
    flat = book.fill_price(1, 1.1000)
    wide = book.fill_price(1, 1.1000, spread_points=30.0, slippage_points=6.0)
    assert wide > flat
    assert wide == pytest.approx(1.1000 + 36 * 0.0001)


def test_dynamic_costs_flag_on_inferred_spec(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "SIMULATOR_DYNAMIC_COSTS", True, raising=False)
    spec = SymbolSpec.infer(1.10)
    assert spec.dynamic_costs is True
    # explicitly constructed specs stay static (test determinism)
    assert SymbolSpec().dynamic_costs is False


def test_infer_jpy_not_treated_as_metal():
    """USDJPY (~150) must use 3-digit FX point scaling, not gold's 0.01.

    The old ``price >= 20 → metal`` rule made a 15-point spread cost $15,000
    per lot after a 100k contract override — Stage-1 screening never passed.
    """
    jpy = SymbolSpec.infer(150.5, {"contract_size": 100_000,
                                   "spread_points": 15.0}, symbol="USDJPY")
    assert jpy.point == pytest.approx(0.001)
    assert jpy.contract_size == 100_000.0
    assert jpy.pnl_divide_by_price is True
    assert jpy.point_value_at(150.0) == pytest.approx(100.0 / 150.0)
    # 15-point (=1.5 pip) spread on 1 lot ≈ $10 at 150 JPY
    assert abs(jpy.price_move_pnl(0.015, 1.0, 150.0)) == pytest.approx(10.0)

    # Mid-priced quote without a symbol hint still must not become a metal.
    mid = SymbolSpec.infer(150.5, {"contract_size": 100_000})
    assert mid.point == pytest.approx(0.001)
    assert mid.pnl_divide_by_price is True

    eurusd = SymbolSpec.infer(1.08, {"contract_size": 100_000,
                                     "spread_points": 15.0}, symbol="EURUSD")
    assert eurusd.point == pytest.approx(0.00001)
    assert eurusd.pnl_divide_by_price is False
    assert 15.0 * eurusd.point_value == pytest.approx(15.0)

    gold = SymbolSpec.infer(2650.0, symbol="XAUUSD")
    assert gold.point == pytest.approx(0.01)
    assert gold.contract_size == 100.0
    assert gold.spread_points == 25.0
    assert gold.pnl_divide_by_price is False


def test_defaults_for_symbol_covers_asset_classes():
    fx = SymbolSpec.defaults_for_symbol("EURUSD")
    assert fx.contract_size == 100_000.0
    assert fx.spread_points == 12.0
    assert fx.slippage_points == 2.0
    assert fx.point == pytest.approx(0.00001)

    jpy = SymbolSpec.defaults_for_symbol("USDJPY")
    assert jpy.point == pytest.approx(0.001)
    assert jpy.contract_size == 100_000.0
    assert jpy.slippage_points == 3.0
    assert jpy.pnl_divide_by_price is True

    idx = SymbolSpec.defaults_for_symbol("US30")
    assert idx.contract_size == 1.0
    assert idx.point == pytest.approx(0.1)
    assert idx.spread_points == 30.0
    assert idx.slippage_points == 4.0

    oil = SymbolSpec.defaults_for_symbol("USOIL")
    assert oil.contract_size == 100.0
    assert oil.point == pytest.approx(0.01)
    assert oil.slippage_points == 5.0

    btc = SymbolSpec.defaults_for_symbol("BTCUSD")
    assert btc.contract_size == 1.0
    assert btc.spread_points >= 100.0
    assert btc.slippage_points >= 50.0

    gold = SymbolSpec.defaults_for_symbol("XAUUSD")
    assert gold.slippage_points == 5.0

    # High-priced index must not inherit gold's 100-oz contract.
    assert SymbolSpec.infer(35000.0, symbol="US30").contract_size == 1.0


def test_simulation_runs_with_dynamic_costs():
    from factory.generator import random_strategy
    from factory.backtest.simulator import run_simulation
    import random

    df = _bars("2024-01-01 00:00", 2000, freq="15min")
    # give the walk some movement so signals fire
    rng = np.random.default_rng(7)
    steps = rng.normal(0, 0.0004, len(df))
    close = 1.10 + np.cumsum(steps)
    df["close"] = close
    df["open"] = np.roll(close, 1)
    df.loc[0, "open"] = 1.10
    df["high"] = np.maximum(df["open"], df["close"]) + 0.0002
    df["low"] = np.minimum(df["open"], df["close"]) - 0.0002

    strat = random_strategy("EURUSD", "M15", rng=random.Random(3))
    spec = SymbolSpec(point=0.0001, spread_points=10.0, slippage_points=2.0,
                      dynamic_costs=True)
    metrics, book = run_simulation(df, strat, spec, 10_000.0)
    assert metrics.initial_deposit == 10_000.0

    # a strategy with a spread ceiling below the rollover spread must never
    # fill during the rollover hours
    strat2 = strat.model_copy(deep=True)
    strat2.risk.max_spread_points = 12.0   # base 10 * 2.5 rollover = 25 > 12
    _, book2 = run_simulation(df, strat2, spec, 10_000.0)
    t = pd.to_datetime(df["time"], utc=True)
    hour_by_ts = dict(zip(
        t.dt.tz_localize(None).astype("datetime64[s]").astype("int64"),
        t.dt.hour))
    for tr in book2.closed:
        assert hour_by_ts.get(int(tr.open_time)) not in (21, 22)
