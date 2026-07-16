"""Validation math: WFE and pass/fail gates."""
from datetime import datetime, timezone
from typing import Dict, Optional

import pytest

from config import settings
from factory.backtest.base import BacktestEngine
from factory.backtest.validation import compute_wfe, validate_strategy
from factory.models import (
    BacktestMetrics, EntryFilter, EntryFilterType, ExecutionMechanic,
    ExecutionMechanicType, ParamRange, StrategyDefinition,
)

YEAR = 365.25 * 86400


def _metrics(net, deposit, years, dd_pct=5.0, trades=50):
    return BacktestMetrics(
        net_profit=net, initial_deposit=deposit, start_ts=0.0,
        end_ts=years * YEAR, max_dd_pct=dd_pct, trade_count=trades,
    )


def test_wfe_definition():
    # IS: 1000 on 10k over 1 year -> 0.10/yr; OOS: 300 on 10k over 0.5y -> 0.06/yr
    is_m = _metrics(1000, 10_000, 1.0)
    oos_m = _metrics(300, 10_000, 0.5)
    assert compute_wfe(is_m, oos_m) == pytest.approx(0.06 / 0.10)


def test_wfe_zero_when_is_unprofitable():
    assert compute_wfe(_metrics(-500, 10_000, 1.0), _metrics(300, 10_000, 0.5)) == 0.0
    assert compute_wfe(_metrics(0, 10_000, 1.0), _metrics(300, 10_000, 0.5)) == 0.0


def test_validate_skips_higher_work_when_l1_fails(monkeypatch):
    """Failing L1 must not run WFO / higher-level scoring work."""
    from factory.backtest import validation as val_mod

    calls = {"wfo": 0}

    def _no_wfo(*args, **kwargs):
        calls["wfo"] += 1
        return []

    monkeypatch.setattr(val_mod, "walk_forward", _no_wfo)
    # DD 80% fails L1 (max 70%).
    report = validate_strategy(
        ConstantRateEngine(dd_pct=80.0), _strategy(), START, END,
        seed=1, run_montecarlo=False,
        floor_level=1, ceiling_level=16)
    assert report.highest_level_passed == 0
    assert report.passed is False
    assert report.levels_cleared.get("1") is False
    assert report.levels_cleared.get("2") is False
    assert calls["wfo"] == 0
    assert report.wfo_windows == []


def test_validate_nested_levels_never_skip_l1():
    """Clearing a higher band without L1 is impossible under nested scoring."""
    from factory import validation_levels
    oos = BacktestMetrics(
        net_profit=500.0, max_dd_pct=80.0, trade_count=100,
        profit_factor=2.0, sharpe=2.0, r_squared=0.9,
        initial_deposit=10_000.0)
    # Would look strong on many gates except L1 DD.
    assert validation_levels.highest_level_cleared(oos, wfe=0.9) == 0


def test_annualized_profit_rate():
    m = _metrics(2000, 10_000, 2.0)
    assert m.annualized_profit_rate() == pytest.approx(0.10)


class ConstantRateEngine(BacktestEngine):
    """Stub engine: profit strictly proportional to the tested duration, so
    IS and OOS annualized rates are identical (WFE == 1)."""
    name = "stub"

    def __init__(self, rate_per_year=0.2, dd_pct=5.0, trades=50):
        self.rate = rate_per_year
        self.dd_pct = dd_pct
        self.trades = trades

    def run(self, strategy: StrategyDefinition, start: datetime, end: datetime,
            params_override: Optional[Dict[str, float]] = None,
            deposit: float = 10_000.0) -> BacktestMetrics:
        years = (end - start).total_seconds() / YEAR
        return BacktestMetrics(
            net_profit=deposit * self.rate * years, initial_deposit=deposit,
            start_ts=start.timestamp(), end_ts=end.timestamp(),
            max_dd_pct=self.dd_pct, trade_count=self.trades,
        )


def _strategy():
    return StrategyDefinition(
        symbol="TEST", timeframe="H1",
        entry_filters=[EntryFilter(
            type=EntryFilterType.RSI_REVERSION,
            params={"rsi_period": 14, "oversold": 30, "overbought": 70},
            ranges={"rsi_period": ParamRange(min=7, max=21, step=7)})],
        mechanic=ExecutionMechanic(
            type=ExecutionMechanicType.DCA_GRID,
            params={"grid_step_points": 200.0, "lot_multiplier": 1.5,
                    "max_levels": 3.0, "basket_tp_points": 100.0,
                    "basket_sl_points": 400.0},
            ranges={"grid_step_points": ParamRange(min=100, max=500, step=50),
                    "lot_multiplier": ParamRange(min=1.0, max=2.0, step=0.25),
                    "basket_sl_points": ParamRange(min=150, max=800, step=25)}),
    )


START = datetime(2023, 1, 1, tzinfo=timezone.utc)
END = datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_validate_passes_consistent_strategy():
    report = validate_strategy(ConstantRateEngine(), _strategy(), START, END, seed=1)
    assert report.wfe == pytest.approx(1.0)
    assert report.passed
    assert report.reasons == []
    assert report.degradation_pct == pytest.approx(0.0)
    assert report.stability_ratio == pytest.approx(1.0)
    # chronological 70/30 split
    total = END.timestamp() - START.timestamp()
    assert report.is_range[1] - report.is_range[0] == pytest.approx(0.7 * total)
    assert report.oos_range[1] - report.oos_range[0] == pytest.approx(0.3 * total)
    # both configured WFO modes ran
    modes = {w.mode for w in report.wfo_windows}
    assert modes == set(settings.WFO_MODES)


def test_validate_fails_on_oos_drawdown():
    report = validate_strategy(ConstantRateEngine(dd_pct=20.0), _strategy(),
                               START, END, seed=1)
    assert not report.passed
    assert any("drawdown" in r for r in report.reasons)


def test_validate_fails_on_too_few_trades():
    report = validate_strategy(ConstantRateEngine(trades=2), _strategy(),
                               START, END, seed=1)
    assert not report.passed
    assert any("trade count" in r for r in report.reasons)


def test_optimizer_searches_mechanic_ranges():
    """The IS optimizer's search space must include execution-mechanic
    parameters (grid step, lot multiplier), not just filter parameters."""
    seen: list = []

    class RecordingEngine(ConstantRateEngine):
        def run(self, strategy, start, end, params_override=None,
                deposit=10_000.0):
            if params_override:
                seen.append(dict(params_override))
            return super().run(strategy, start, end, params_override, deposit)

    validate_strategy(RecordingEngine(), _strategy(), START, END, seed=42)
    sampled_keys = set().union(*seen) if seen else set()
    assert "M_DCA_GRID_grid_step_points" in sampled_keys
    assert "M_DCA_GRID_lot_multiplier" in sampled_keys
    assert "M_DCA_GRID_basket_sl_points" in sampled_keys
    assert "F0_RSI_REVERSION_rsi_period" in sampled_keys
    # and the sampled grid steps actually vary across the range
    grid_values = {p["M_DCA_GRID_grid_step_points"] for p in seen
                   if "M_DCA_GRID_grid_step_points" in p}
    assert len(grid_values) > 1
