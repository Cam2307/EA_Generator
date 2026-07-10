"""Selection-bias statistics (factory.backtest.statistics)."""
import math

import numpy as np
import pytest

from factory.backtest.statistics import (
    _norm_cdf, _norm_ppf, deflated_sharpe_ratio, dsr_from_metrics,
    expected_max_sharpe, p_oos_loss,
)
from factory.metrics_display import dsr_badge, dsr_label
from factory.models import BacktestMetrics, WFOWindowResult


def test_norm_ppf_inverts_cdf():
    for p in (0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999):
        assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, abs=1e-6)
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)


def test_expected_max_sharpe_grows_with_trials():
    v = 0.01
    e10 = expected_max_sharpe(10, v)
    e100 = expected_max_sharpe(100, v)
    e10000 = expected_max_sharpe(10000, v)
    assert 0 < e10 < e100 < e10000
    assert expected_max_sharpe(1, v) == 0.0


def test_dsr_haircut_with_more_trials():
    # same observed Sharpe, more trials -> lower DSR
    sr, n_obs, skew, kurt = 0.08, 500, 0.0, 3.0
    d1 = deflated_sharpe_ratio(sr, n_obs, skew, kurt, n_trials=1)
    d1k = deflated_sharpe_ratio(sr, n_obs, skew, kurt, n_trials=1000)
    assert d1 > d1k
    assert 0.0 <= d1k <= 1.0
    # a strong per-period Sharpe over many observations survives even 1k trials
    strong = deflated_sharpe_ratio(0.3, 2000, 0.0, 3.0, n_trials=1000)
    assert strong > 0.99


def _metrics_from_curve(eq):
    return BacktestMetrics(equity=list(map(float, eq)),
                           equity_ts=[float(i * 3600) for i in range(len(eq))])


def test_dsr_from_metrics_separates_trend_from_noise():
    rng = np.random.default_rng(11)
    n = 800
    steady = 10_000 + np.cumsum(rng.normal(5.0, 8.0, n))    # strong drift
    noise = 10_000 + np.cumsum(rng.normal(0.0, 8.0, n))     # zero drift
    d_steady = dsr_from_metrics(_metrics_from_curve(steady), n_trials=500)
    d_noise = dsr_from_metrics(_metrics_from_curve(noise), n_trials=500)
    assert d_steady > 0.95
    assert d_noise < d_steady
    assert dsr_from_metrics(_metrics_from_curve([1, 2]), 10) == 0.0


def test_p_oos_loss():
    def _w(net):
        return WFOWindowResult(
            mode="rolling", index=0, is_start_ts=0, is_end_ts=1,
            oos_start_ts=1, oos_end_ts=2,
            is_metrics=BacktestMetrics(),
            oos_metrics=BacktestMetrics(net_profit=net))
    assert p_oos_loss([]) == 0.0
    assert p_oos_loss([_w(10), _w(-5), _w(3), _w(-1)]) == 0.5


def test_dsr_display_helpers():
    assert "very likely real" in dsr_label(0.97, 400)
    assert "selection luck" in dsr_label(0.30, 400)
    assert dsr_badge(0.97).startswith(":green")
    assert dsr_badge(0.30).startswith(":red")


def test_validation_report_carries_dsr(tmp_path):
    """End-to-end: validate_strategy populates dsr / n_trials / p_oos_loss."""
    import random
    from datetime import datetime, timezone

    import pandas as pd

    from factory.backtest.simulator import SimulatorEngine, SymbolSpec
    from factory.backtest.validation import validate_strategy
    from factory.generator import random_strategy

    n = 3000
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="utc")
    rng = np.random.default_rng(5)
    close = 1.10 + np.cumsum(rng.normal(0.00002, 0.0004, n))
    df = pd.DataFrame({
        "time": idx, "open": np.roll(close, 1), "high": close + 0.0002,
        "low": close - 0.0002, "close": close, "volume": 100.0})
    df.loc[0, "open"] = 1.10

    engine = SimulatorEngine(
        spec=SymbolSpec(point=0.0001, spread_points=5.0, slippage_points=1.0),
        ohlc=df)
    strat = random_strategy("EURUSD", "M15", rng=random.Random(9))
    report = validate_strategy(
        engine, strat,
        datetime(2023, 1, 1, tzinfo=timezone.utc),
        datetime(2023, 2, 1, tzinfo=timezone.utc),
        seed=1, run_montecarlo=False, n_trials=250)
    assert report.n_trials == 250
    assert 0.0 <= report.dsr <= 1.0
    assert 0.0 <= report.p_oos_loss <= 1.0
