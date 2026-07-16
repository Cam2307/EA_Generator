"""Untouched-holdout layer (factory.holdout)."""
from datetime import datetime, timedelta, timezone

from factory.backtest.base import BacktestEngine
from factory.holdout import (
    clamp_discovery_end, evaluate_holdout, factory_hit_rate, holdout_boundary,
)
from factory.models import (
    BacktestMetrics, EntryFilter, EntryFilterType, ExecutionMechanic,
    ExecutionMechanicType, StrategyDefinition, ValidationReport,
)
from factory.storage import Storage

NOW = datetime(2026, 7, 10, tzinfo=timezone.utc)


def test_boundary_and_clamp(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "HOLDOUT_MONTHS", 12, raising=False)
    boundary = holdout_boundary(NOW)
    assert (NOW - boundary).days == int(12 * settings.DAYS_PER_MONTH)

    # end beyond the boundary is clamped
    clamped, was = clamp_discovery_end(NOW, now=NOW)
    assert was and clamped == boundary
    # historical ranges pass through untouched
    old = NOW - timedelta(days=900)
    same, was = clamp_discovery_end(old, now=NOW)
    assert not was and same == old
    # disabled -> no-op
    monkeypatch.setattr(settings, "HOLDOUT_ENABLED", False, raising=False)
    same, was = clamp_discovery_end(NOW, now=NOW)
    assert not was and same == NOW


def test_history_window_survives_equal_months_holdout(monkeypatch):
    """Overnight preset (12m history, 12m holdout) must not collapse to hours."""
    from config import settings
    from factory.discovery_config import history_start_end

    monkeypatch.setattr(settings, "HOLDOUT_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "HOLDOUT_MONTHS", 12, raising=False)
    start, end = history_start_end(12, today=NOW.date())
    clamped, was = clamp_discovery_end(end, now=NOW)
    assert was or clamped == end or clamped <= end
    # Rebase the way the worker does after clamp.
    if was:
        start = clamped - timedelta(days=12 * settings.DAYS_PER_MONTH)
        end = clamped
    assert (end - start).days >= 360
    assert end <= holdout_boundary(NOW) + timedelta(days=1)


class _Engine(BacktestEngine):
    name = "stub"

    def __init__(self, profit=1000.0):
        self.profit = profit
        self.calls = 0
        self.last_range = None

    def run(self, strategy, start, end, params_override=None, deposit=10_000.0):
        self.calls += 1
        self.last_range = (start, end)
        return BacktestMetrics(net_profit=self.profit, trade_count=30,
                               max_dd_pct=8.0, profit_factor=1.4)


def _seed(tmp_path):
    storage = Storage(tmp_path / "h.db")
    strat = StrategyDefinition(
        id="h1", name="Holdout One", symbol="EURUSD", timeframe="M15",
        entry_filters=[EntryFilter(type=EntryFilterType.RSI_REVERSION,
                                   params={"rsi_period": 14, "oversold": 30,
                                           "overbought": 70})],
        mechanic=ExecutionMechanic(type=ExecutionMechanicType.STANDARD_SLTP,
                                   params={"sl_points": 300, "tp_points": 300}))
    storage.save_strategy(strat)
    storage.save_validation(ValidationReport(
        strategy_id="h1", is_metrics=BacktestMetrics(),
        oos_metrics=BacktestMetrics(), passed=True,
        best_params={"M_STANDARD_SLTP_sl_points": 400.0}))
    return storage


def test_one_shot_discipline(tmp_path):
    storage = _seed(tmp_path)
    engine = _Engine(profit=1000.0)
    res1 = evaluate_holdout(storage, "h1", engine=engine, now=NOW)
    assert res1.passed and res1.error is None
    assert engine.calls == 1
    # window is exactly the reserved trailing months
    start, end = engine.last_range
    assert start == holdout_boundary(NOW) and end == NOW

    # second call returns the STORED result without re-running the engine
    res2 = evaluate_holdout(storage, "h1", engine=_Engine(profit=-5000.0),
                            now=NOW)
    assert res2.passed and res2.net_profit == 1000.0
    # force=True is the only way to re-evaluate
    res3 = evaluate_holdout(storage, "h1", engine=_Engine(profit=-5000.0),
                            now=NOW, force=True)
    assert not res3.passed


def test_holdout_fail_conditions(tmp_path, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "HOLDOUT_MAX_DD_PCT", 25.0, raising=False)
    storage = _seed(tmp_path)

    class DDEngine(_Engine):
        def run(self, *a, **k):
            return BacktestMetrics(net_profit=500.0, trade_count=30,
                                   max_dd_pct=40.0, profit_factor=1.2)

    res = evaluate_holdout(storage, "h1", engine=DDEngine(), now=NOW)
    assert not res.passed                # profitable but DD too deep

    missing = evaluate_holdout(storage, "ghost", engine=_Engine(), now=NOW)
    assert missing.error == "strategy not found"


def test_factory_hit_rate(tmp_path):
    storage = _seed(tmp_path)
    evaluate_holdout(storage, "h1", engine=_Engine(profit=1000.0), now=NOW)
    evaluate_holdout(storage, "ghost", engine=_Engine(), now=NOW)  # error row
    stats = factory_hit_rate(storage)
    assert stats["evaluated"] == 1.0
    assert stats["passed"] == 1.0
    assert stats["hit_rate"] == 1.0
    assert stats["errors"] == 1.0
