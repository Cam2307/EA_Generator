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
    storage.save_complete(passed, _report_for(passed, passed=True), job_id="job_a")
    storage.save_complete(failed, _report_for(failed, passed=False), job_id="job_a")

    assert len(storage.list_validated(passed_only=True)) == 1
    assert len(storage.list_validated(passed_only=False)) == 2
    assert storage.count_validated(passed_only=True) == 1
    assert storage.count_validated(passed_only=None) == 2
    assert storage.count_strategies() == 2
    assert storage.count_validated("job_a") == (1, 2)
    assert storage.count_validated_by_jobs(["job_a", "missing"]) == {
        "job_a": (1, 2),
        "missing": (0, 0),
    }

    summaries = storage.list_validation_summaries(passed_only=None)
    assert len(summaries) == 2
    assert all(s.oos_metrics.equity == [] for s in summaries)
    assert storage.list_validation_summaries(passed_only=True)[0].passed is True

    light = storage.list_validation_summaries(
        passed_only=None, job_id="job_a", include_body_metrics=False)
    assert len(light) == 2
    assert all(s.oos_metrics.net_profit == 0.0 for s in light)

def test_cancel_active_discovery_jobs(temp_db: Path) -> None:
    from factory.models import Job, JobStatus

    storage = Storage(temp_db)
    storage.upsert_job(Job(id="a", kind="discovery", status=JobStatus.RUNNING))
    storage.upsert_job(Job(id="b", kind="discovery", status=JobStatus.PENDING))
    storage.upsert_job(Job(id="c", kind="discovery", status=JobStatus.DONE))

    n = storage.cancel_active_discovery_jobs(message="stopped")
    assert n == 2
    assert storage.get_job("a").status == JobStatus.CANCELLED
    assert storage.get_job("b").status == JobStatus.CANCELLED
    assert storage.get_job("a").cancel_requested is True
    assert storage.get_job("c").status == JobStatus.DONE


def test_agent_state_aliases_and_upsert(temp_db: Path) -> None:
    storage = Storage(temp_db)

    # Start from a clean state row and verify defaults exist.
    initial = storage.get_agent_state()
    assert initial["id"] == 1
    assert "status" in initial

    # Modern API.
    storage.update_agent_state(enabled=1, status="running", queue_depth=3)
    state = storage.get_agent_state()
    assert int(state["enabled"]) == 1
    assert state["status"] == "running"
    assert int(state["queue_depth"]) == 3

    # Backward-compatible aliases.
    storage.upsert_agent_state(status="stopping")
    storage.set_agent_state(status="stopped", enabled=0)
    state = storage.get_agent_state()
    assert state["status"] == "stopped"
    assert int(state["enabled"]) == 0


def test_app_settings_persist_recipient_email(temp_db: Path) -> None:
    storage = Storage(temp_db)
    storage.upsert_app_settings({"recipient_email": "alerts@example.com"})

    restored = Storage(temp_db).get_app_settings()
    assert restored["recipient_email"] == "alerts@example.com"


def test_agent_state_migrates_legacy_schema(temp_db: Path) -> None:
    """Pre-migration DBs missing agent_state columns must upgrade on connect."""
    import sqlite3

    with sqlite3.connect(temp_db) as con:
        con.execute(
            """
            CREATE TABLE agent_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                enabled INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'stopped',
                pid INTEGER,
                heartbeat_at REAL,
                queue_depth INTEGER NOT NULL DEFAULT 0,
                jobs_submitted INTEGER NOT NULL DEFAULT 0,
                cursor INTEGER NOT NULL DEFAULT 0,
                updated_at REAL
            )
            """
        )
        con.execute(
            "INSERT INTO agent_state (id, enabled, status, updated_at) VALUES (1, 0, 'stopped', 0)"
        )

    storage = Storage(temp_db)
    storage.update_agent_state(message="migrated", sweep_total=12)
    state = storage.get_agent_state()
    assert state["message"] == "migrated"
    assert int(state["sweep_total"]) == 12
