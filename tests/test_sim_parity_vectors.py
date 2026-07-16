"""Golden-vector sim parity: Python PositionBook vs Numba, and mechanic smoke.

Full sim↔MT5 Strategy Tester parity still requires a live terminal
(``scripts/reconcile_engines.py``). These tests lock in fill-path agreement
for the in-process engines that discovery actually screens with.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from config import settings
from factory.backtest.reconcile import compare_metrics, reconcile_strategies
from factory.backtest.base import BacktestEngine
from factory.backtest.simulator import SymbolSpec, run_simulation
from factory.generator import random_strategy
from factory.models import (
    BacktestMetrics, EntryFilter, EntryFilterType, ExecutionMechanic,
    ExecutionMechanicType, StrategyDefinition, TrailMode,
)


_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
_END = datetime(2023, 3, 1, tzinfo=timezone.utc)


def _synthetic_bars(n: int = 800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0, 0.0008, size=n)
    close = 1.10 + np.cumsum(ret)
    high = close + rng.uniform(0.0001, 0.0010, size=n)
    low = close - rng.uniform(0.0001, 0.0010, size=n)
    open_ = close - ret
    times = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "time": times,
        "open": open_,
        "high": np.maximum(high, np.maximum(open_, close)),
        "low": np.minimum(low, np.minimum(open_, close)),
        "close": close,
        "volume": rng.integers(100, 1000, size=n),
    })
    df.attrs["source"] = "synthetic"
    return df


def _mechanic_strategy(mech: ExecutionMechanicType) -> StrategyDefinition:
    params = {
        ExecutionMechanicType.STANDARD_SLTP: {
            "sl_points": 400.0, "tp_points": 600.0,
        },
        ExecutionMechanicType.PARTIAL_CLOSE: {
            "sl_points": 400.0, "tp_points": 800.0,
            "partial_tp_points": 200.0, "partial_fraction": 0.5,
        },
        ExecutionMechanicType.DCA_GRID: {
            "grid_step_points": 200.0, "max_levels": 3,
            "lot_multiplier": 1.0,
            "basket_tp_points": 150.0, "basket_sl_points": 500.0,
        },
        ExecutionMechanicType.HEDGE_LAYER: {
            "sl_points": 500.0, "tp_points": 500.0,
            "hedge_trigger_points": 300.0, "hedge_ratio": 1.0,
        },
    }[mech]
    strat = StrategyDefinition(
        name=f"parity_{mech.value}",
        symbol="EURUSD",
        timeframe="M15",
        entry_filters=[
            EntryFilter(
                type=EntryFilterType.RSI_REVERSION,
                params={"rsi_period": 14, "oversold": 30, "overbought": 70},
            ),
        ],
        mechanic=ExecutionMechanic(type=mech, params=params),
    )
    strat.trade_mgmt.trail_mode = TrailMode.OFF
    strat.trade_mgmt.breakeven = False
    strat.trade_mgmt.regime_filter = False
    strat.trade_mgmt.regime_sizing = False
    return strat


def test_numba_matches_python_on_standard_path(monkeypatch):
    pytest.importorskip("numba")
    from factory.backtest.sim_numba_core import (
        numba_available, strategy_numba_eligible,
    )

    if not numba_available():
        pytest.skip("numba not available")

    df = _synthetic_bars()
    strat = random_strategy(
        "EURUSD", "M15", __import__("random").Random(42),
        allowed_mechanics=[ExecutionMechanicType.STANDARD_SLTP],
    )
    strat.trade_mgmt.trail_mode = TrailMode.OFF
    strat.trade_mgmt.breakeven = False
    strat.trade_mgmt.regime_filter = False
    strat.trade_mgmt.regime_sizing = False
    if not strategy_numba_eligible(strat, intrabar_mode="path"):
        strat = _mechanic_strategy(ExecutionMechanicType.STANDARD_SLTP)
    assert strategy_numba_eligible(strat, intrabar_mode="path")

    spec = SymbolSpec(dynamic_costs=False)
    monkeypatch.setattr(settings, "SIMULATOR_NUMBA", False)
    m_py, _ = run_simulation(df, strat, spec, 10_000.0, intrabar_mode="path")
    monkeypatch.setattr(settings, "SIMULATOR_NUMBA", True)
    m_nb, _ = run_simulation(df, strat, spec, 10_000.0, intrabar_mode="path")

    deltas = compare_metrics(m_py, m_nb)
    by = {d.name: d for d in deltas}
    assert by["trade_count"].within_tolerance, by["trade_count"]
    assert by["profit_factor"].within_tolerance, by["profit_factor"]
    assert by["max_dd_pct"].within_tolerance, by["max_dd_pct"]
    assert abs(by["net_profit"].rel_delta) <= 0.50, by["net_profit"]


@pytest.mark.parametrize("mech", [
    ExecutionMechanicType.STANDARD_SLTP,
    ExecutionMechanicType.PARTIAL_CLOSE,
    ExecutionMechanicType.DCA_GRID,
    ExecutionMechanicType.HEDGE_LAYER,
])
def test_mechanic_smoke_runs_and_is_finite(mech, monkeypatch):
    monkeypatch.setattr(settings, "SIMULATOR_NUMBA", False)
    df = _synthetic_bars(seed=11)
    strat = _mechanic_strategy(mech)
    spec = SymbolSpec(dynamic_costs=False)
    m, _ = run_simulation(df, strat, spec, 10_000.0, intrabar_mode="path")
    assert np.isfinite(m.net_profit)
    assert np.isfinite(m.max_dd_pct)
    assert m.trade_count >= 0


def test_reconcile_stub_engines_agree_on_identical_outputs():
    """Documents the MT5 parity harness: identical engines → ok."""

    class _Twin(BacktestEngine):
        name = "twin"

        def run(self, strategy, start, end, params_override=None, deposit=10_000.0):
            return BacktestMetrics(
                net_profit=1000.0, profit_factor=1.5, max_dd_pct=10.0,
                trade_count=40)

    strategies = [_mechanic_strategy(ExecutionMechanicType.STANDARD_SLTP)]
    results = reconcile_strategies(
        _Twin(), _Twin(), strategies, _START, _END)
    assert results[0].ok
