"""Regime classification + per-regime validation breakdown (factory.regime)."""
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from factory.models import RegimeStats
from factory.regime import (
    REGIME_NAMES, classify_regimes, regime_stats_from_trades,
    worst_regime_net,
)


def _df(close, spread=0.0002):
    n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="utc")
    close = np.asarray(close, dtype=float)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame({
        "time": idx, "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close, "volume": 100.0})


def test_classify_trend_vs_range():
    n = 600
    trend = 1.10 + np.arange(n) * 0.0008          # relentless one-way move
    # mean-reverting oscillation (period ~10 bars): directionless by design
    chop = 1.10 + 0.0015 * np.sin(np.arange(n) * 2 * np.pi / 10)

    trend_codes = classify_regimes(_df(trend))
    chop_codes = classify_regimes(_df(chop))
    # post-warmup, the one-way move should be overwhelmingly "trend" (odd codes)
    assert (trend_codes[100:] % 2 == 1).mean() > 0.9
    # tight chop should be overwhelmingly "range" (even codes)
    assert (chop_codes[100:] % 2 == 0).mean() > 0.9
    assert set(np.unique(trend_codes)) <= set(REGIME_NAMES)


def test_classify_flags_volatility_burst():
    n = 600
    rng = np.random.default_rng(9)
    steps = rng.normal(0, 0.0002, n)
    steps[400:460] = rng.normal(0, 0.0025, 60)     # 12x vol burst
    close = 1.10 + np.cumsum(steps)
    codes = classify_regimes(_df(close))
    assert (codes[420:460] >= 2).mean() > 0.5      # volatile codes during burst
    assert (codes[200:380] < 2).mean() > 0.9       # quiet before it


def test_regime_stats_aggregation():
    bar_times = np.arange(10, dtype=float) * 3600.0
    regimes = np.array([0, 0, 0, 1, 1, 1, 3, 3, 3, 3], dtype=np.int8)
    open_times = [0.0, 3 * 3600.0, 6 * 3600.0, 7 * 3600.0]
    profits = [50.0, -20.0, -30.0, -10.0]
    stats = regime_stats_from_trades(bar_times, regimes, open_times, profits)
    by_code = {s.code: s for s in stats}
    assert by_code[0].trades == 1 and by_code[0].net_profit == 50.0
    assert by_code[1].trades == 1 and by_code[1].net_profit == -20.0
    assert by_code[3].trades == 2 and by_code[3].net_profit == -40.0
    assert by_code[2].trades == 0
    assert abs(sum(s.bar_share for s in stats) - 1.0) < 1e-9
    assert worst_regime_net(stats) == -40.0
    assert worst_regime_net([RegimeStats(code=0, net_profit=5.0)]) == 0.0


def test_validation_report_carries_regime_stats():
    from factory.backtest.simulator import SimulatorEngine, SymbolSpec
    from factory.backtest.validation import validate_strategy
    from factory.generator import random_strategy

    n = 3000
    rng = np.random.default_rng(5)
    close = 1.10 + np.cumsum(rng.normal(0.00002, 0.0004, n))
    df = _df(close)
    df["time"] = pd.date_range("2023-01-01", periods=n, freq="15min", tz="utc")

    engine = SimulatorEngine(
        spec=SymbolSpec(point=0.0001, spread_points=5.0, slippage_points=1.0),
        ohlc=df)
    strat = random_strategy("EURUSD", "M15", rng=random.Random(9))
    report = validate_strategy(
        engine, strat,
        datetime(2023, 1, 1, tzinfo=timezone.utc),
        datetime(2023, 2, 1, tzinfo=timezone.utc),
        seed=1, run_montecarlo=False)
    # simulator engine -> breakdown must be present whenever OOS traded
    if report.oos_metrics.trade_count > 0:
        assert len(report.regime_stats) == 4
        assert sum(s.trades for s in report.regime_stats) > 0
