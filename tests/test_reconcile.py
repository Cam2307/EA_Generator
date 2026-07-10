"""Simulator<->MT5 reconciliation compare logic (factory.backtest.reconcile)."""
from datetime import datetime, timezone

from factory.backtest.base import BacktestEngine
from factory.backtest.reconcile import (
    bias_summary, compare_metrics, format_report, reconcile_strategies,
)
from factory.models import (
    BacktestMetrics, EntryFilter, EntryFilterType, ExecutionMechanic,
    ExecutionMechanicType, StrategyDefinition,
)


def _m(**kw):
    base = dict(net_profit=1000.0, profit_factor=1.5, max_dd_pct=10.0,
                trade_count=50)
    base.update(kw)
    return BacktestMetrics(**base)


def test_compare_metrics_within_and_out_of_tolerance():
    deltas = compare_metrics(_m(net_profit=1100.0), _m(net_profit=1000.0))
    by = {d.name: d for d in deltas}
    assert by["net_profit"].rel_delta == 0.1
    assert by["net_profit"].within_tolerance          # 10% < 35%
    assert by["trade_count"].rel_delta == 0.0

    deltas = compare_metrics(_m(profit_factor=2.5), _m(profit_factor=1.5))
    by = {d.name: d for d in deltas}
    assert not by["profit_factor"].within_tolerance   # +67% > 25%


def test_compare_metrics_floor_prevents_blowup():
    # near-zero MT5 profit must not create an infinite relative delta
    deltas = compare_metrics(_m(net_profit=50.0), _m(net_profit=1.0))
    by = {d.name: d for d in deltas}
    assert abs(by["net_profit"].rel_delta) <= 1.0     # floored at 100


class _StubEngine(BacktestEngine):
    name = "stub"

    def __init__(self, scale=1.0, fail=False):
        self.scale = scale
        self.fail = fail

    def run(self, strategy, start, end, params_override=None, deposit=10_000.0):
        if self.fail:
            raise RuntimeError("engine exploded")
        return _m(net_profit=1000.0 * self.scale,
                  profit_factor=1.5 * self.scale)


def _strategies(n):
    return [StrategyDefinition(
        name=f"S{i}", symbol="EURUSD", timeframe="M15",
        entry_filters=[EntryFilter(type=EntryFilterType.RSI_REVERSION,
                                   params={"rsi_period": 14, "oversold": 30,
                                           "overbought": 70})],
        mechanic=ExecutionMechanic(type=ExecutionMechanicType.STANDARD_SLTP,
                                   params={"sl_points": 300, "tp_points": 300}),
    ) for i in range(n)]


_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
_END = datetime(2024, 6, 1, tzinfo=timezone.utc)


def test_reconcile_and_bias_summary():
    # simulator 20% optimistic across the board
    results = reconcile_strategies(_StubEngine(scale=1.2), _StubEngine(),
                                   _strategies(4), _START, _END)
    assert len(results) == 4
    summary = bias_summary(results)
    assert summary["net_profit"]["mean_rel"] == 0.2
    assert summary["profit_factor"]["median_rel"] == 0.2
    assert summary["net_profit"]["agree_frac"] == 1.0    # 20% < 35% tolerance
    assert summary["profit_factor"]["agree_frac"] == 1.0  # 20% < 25%

    report = format_report(results)
    assert "Aggregate simulator bias" in report
    assert "4/4 strategies reconcile" in report


def test_reconcile_captures_engine_failures():
    results = reconcile_strategies(_StubEngine(), _StubEngine(fail=True),
                                   _strategies(2), _START, _END)
    assert all(r.error for r in results)
    assert all(not r.ok for r in results)
    report = format_report(results)
    assert "FAIL" in report and "2 failed to run" in report
