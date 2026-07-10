"""Price-path block bootstrap (factory.backtest.montecarlo.block_bootstrap_ohlc)."""
import random

import numpy as np
import pandas as pd

from factory.backtest.montecarlo import block_bootstrap_ohlc


def _df(n=1000, seed=4):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="utc")
    close = 1.10 + np.cumsum(rng.normal(0, 0.0004, n))
    open_ = np.roll(close, 1)
    open_[0] = 1.10
    spread = np.abs(rng.normal(0, 0.0002, n))
    return pd.DataFrame({
        "time": idx, "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close, "volume": rng.integers(50, 500, n).astype(float)})


def test_bootstrap_preserves_shape_and_sanity():
    df = _df()
    out = block_bootstrap_ohlc(df, random.Random(1), block_bars=96)
    assert len(out) == len(df)
    assert (out["time"].to_numpy() == df["time"].to_numpy()).all()  # timestamps kept
    assert (out["high"] >= out["open"]).all()
    assert (out["high"] >= out["close"]).all()
    assert (out["low"] <= out["open"]).all()
    assert (out["low"] <= out["close"]).all()
    assert (out["close"] > 0).all()
    # same starting anchor, different realized path
    assert out["close"].iloc[0] != 0
    assert not np.allclose(out["close"].to_numpy(), df["close"].to_numpy())


def test_bootstrap_preserves_return_scale():
    df = _df()
    out = block_bootstrap_ohlc(df, random.Random(2), block_bars=96)
    r_orig = np.diff(np.log(df["close"].to_numpy()))
    r_boot = np.diff(np.log(out["close"].to_numpy()))
    # same volatility character (within a loose band)
    assert 0.5 < r_boot.std() / r_orig.std() < 2.0


def test_bootstrap_deterministic_and_seed_sensitive():
    df = _df()
    a = block_bootstrap_ohlc(df, random.Random(7), block_bars=64)
    b = block_bootstrap_ohlc(df, random.Random(7), block_bars=64)
    c = block_bootstrap_ohlc(df, random.Random(8), block_bars=64)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_bootstrap_short_series_passthrough():
    df = _df(n=100)
    out = block_bootstrap_ohlc(df, random.Random(1), block_bars=96)
    assert out is df                     # too short to resample -> unchanged


def test_montecarlo_result_carries_path_stats():
    from factory.backtest.montecarlo import MonteCarloConfig, run_montecarlo
    from factory.backtest.simulator import SymbolSpec
    from factory.generator import random_strategy

    df = _df(n=1500)
    spec = SymbolSpec(point=0.0001, spread_points=5.0, slippage_points=1.0)
    strat = random_strategy("EURUSD", "M15", rng=random.Random(3))
    cfg = MonteCarloConfig(n_runs=3, path_runs=4, path_block_bars=96,
                           n_resamples=20, seed=11)
    result = run_montecarlo(strat, df, spec, 10_000.0, cfg)
    assert result.path_runs == 4
    assert 0.0 <= result.path_pct_profitable <= 1.0
