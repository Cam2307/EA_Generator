"""Monte Carlo robustness math and gates."""
import random

import numpy as np
import pandas as pd
import pytest

from factory.backtest.montecarlo import (
    MonteCarloConfig, perturb_flat_params, resample_drawdowns, robustness_score,
    run_montecarlo,
)
from factory.backtest.simulator import SymbolSpec, run_simulation
from factory.generator import random_strategy
from factory.models import ParamRange


def test_perturb_respects_param_range():
    strat = random_strategy("EURUSD", "H1", random.Random(1))
    flat = strat.all_params()
    ranges = strat.all_ranges()
    rng = random.Random(42)
    out = perturb_flat_params(strat, rng, change_prob=1.0, max_steps=2)
    for key, val in out.items():
        r = ranges.get(key)
        if r:
            assert r.min <= val <= r.max


def test_resample_drawdowns_positive():
    profits = [100.0, -50.0, 80.0, -30.0, 60.0]
    dds = resample_drawdowns(profits, 10_000.0, 500, random.Random(7))
    assert len(dds) == 500
    assert all(d >= 0 for d in dds)


def test_robustness_score_bounds():
    score = robustness_score(0.9, 500.0, 1000.0, 10.0, 25.0)
    assert 0 <= score <= 100
    bad = robustness_score(0.2, -500.0, 100.0, 40.0, 25.0)
    assert bad < score


def test_run_montecarlo_produces_runs():
    rng_py = random.Random(99)
    strat = random_strategy("EURUSD", "H1", rng_py)
    n = 80
    np_rng = np.random.default_rng(99)
    closes = 1.10 + np.cumsum(np_rng.normal(0, 0.0003, n))
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="1h", tz="utc"),
        "open": closes, "high": closes + 0.0002,
        "low": closes - 0.0002, "close": closes,
        "volume": np.ones(n),
    })
    spec = SymbolSpec.infer(float(closes[0]))
    cfg = MonteCarloConfig(n_runs=5, n_resamples=20, seed=99,
                           min_profitable=0.0, max_dd_p95=100.0)
    result = run_montecarlo(strat, df, spec, 10_000.0, cfg)
    assert result.n_runs >= 1
    assert 0 <= result.robustness_score <= 100


def test_mc_gate_fails_low_profitable_fraction():
    cfg = MonteCarloConfig(min_profitable=0.99, max_dd_p95=0.1, n_runs=1,
                           n_resamples=0, seed=1)
    # empty runs path
    strat = random_strategy("EURUSD", "H1", random.Random(1))
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=5, freq="1h", tz="utc"),
        "open": [1.1] * 5, "high": [1.1] * 5,
        "low": [1.1] * 5, "close": [1.1] * 5,
        "volume": [1.0] * 5,
    })
    result = run_montecarlo(strat, df, SymbolSpec(), 10_000.0, cfg)
    assert not result.passed or result.n_runs == 0
