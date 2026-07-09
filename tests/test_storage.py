"""Storage layer tests — durable persistence of complete strategies."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from factory.models import (
    BacktestMetrics, EntryFilter, EntryFilterType, ExecutionMechanic,
    ExecutionMechanicType, StrategyDefinition, ValidationReport,
)
from factory.storage import Storage


def _minimal_strategy() -> StrategyDefinition:
    return StrategyDefinition(
        symbol="EURUSD", timeframe="H1",
        entry_filters=[EntryFilter(
            type=EntryFilterType.RSI_REVERSION,
            params={"rsi_period": 14, "oversold": 30, "overbought": 70})],
        mechanic=ExecutionMechanic(
            type=ExecutionMechanicType.STANDARD_SLTP,
            params={"sl_points": 100.0, "tp_points": 200.0}),
    )


def _report_for(strategy: StrategyDefinition, *, passed: bool) -> ValidationReport:
    metrics = BacktestMetrics(net_profit=100.0, trade_count=10, max_dd_pct=5.0)
    return ValidationReport(
        strategy_id=strategy.id,
        is_metrics=metrics,
        oos_metrics=metrics,
        wfe=0.8 if passed else 0.3,
        passed=passed,
        reasons=[] if passed else ["WFE below threshold"],
        engine="simulator",
    )


@pytest.fixture
def temp_db() -> Path:
    return Path(tempfile.mkdtemp(prefix="eaf_storage_")) / "test.db"


def test_save_complete_is_atomic(temp_db: Path) -> None:
    """Strategy + validation land together; a fresh Storage sees both."""
    storage = Storage(temp_db)
    strategy = _minimal_strategy()
    report = _report_for(strategy, passed=False)

    storage.save_complete(strategy, report)

    storage2 = Storage(temp_db)
    assert storage2.get_strategy(strategy.id) is not None
    loaded = storage2.get_validation(strategy.id)
    assert loaded is not None
    assert loaded.passed is False
    assert loaded.wfe == pytest.approx(0.3)


def test_list_validated_includes_failed(temp_db: Path) -> None:
    storage = Storage(temp_db)
    passed = _minimal_strategy()
    failed = _minimal_strategy()
    storage.save_complete(passed, _report_for(passed, passed=True))
    storage.save_complete(failed, _report_for(failed, passed=False))

    assert len(storage.list_validated(passed_only=True)) == 1
    assert len(storage.list_validated(passed_only=False)) == 2
