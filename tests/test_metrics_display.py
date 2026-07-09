"""Canonical metric display helpers."""
import numpy as np

from factory.metrics_display import (
    gate_drawdown_pct, report_zone_drawdown, sortino_ratio, zone_drawdown_label,
)
from factory.models import BacktestMetrics, ValidationReport


def test_gate_drawdown_uses_simulator_metric():
    m = BacktestMetrics(max_dd_pct=14.2, equity=[10_000, 9_900, 10_100])
    assert gate_drawdown_pct(m) == 14.2


def test_zone_drawdown_label():
    assert zone_drawdown_label("OOS") == "OOS Max DD %"


def test_report_zone_drawdown():
    report = ValidationReport(
        strategy_id="x",
        is_metrics=BacktestMetrics(max_dd_pct=8.0),
        oos_metrics=BacktestMetrics(max_dd_pct=12.5),
    )
    assert report_zone_drawdown(report, "OOS") == 12.5
    assert report_zone_drawdown(report, "IS") == 8.0


def test_sortino_ratio_derives_from_equity_curve():
    eq = [10_000.0, 10_050.0, 10_020.0, 10_080.0, 10_060.0]
    ts = [0.0, 86_400.0, 172_800.0, 259_200.0, 345_600.0]
    metrics = BacktestMetrics(equity=eq, equity_ts=ts, sortino=0.0)
    rets = np.diff(np.asarray(eq, dtype=float)) / np.asarray(eq[:-1], dtype=float)
    downside = rets[rets < 0]
    downside_std = float(downside.std())
    bars_per_year = 365.25
    expected = float(rets.mean() / downside_std * np.sqrt(bars_per_year))
    assert sortino_ratio(metrics) == expected


def test_sortino_ratio_falls_back_to_stored_field():
    metrics = BacktestMetrics(sortino=1.42)
    assert sortino_ratio(metrics) == 1.42
