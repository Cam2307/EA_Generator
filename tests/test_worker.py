"""Integration tests for the discovery worker orchestration."""
from __future__ import annotations

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pytest

from config import settings
from factory.backtest.base import BacktestEngine
from factory.backtest.validation import validate_strategy
from factory.models import (
    BacktestMetrics, JobStatus, StrategyDefinition,
)
from factory.storage import Storage
from jobs.worker import JobQueue

YEAR = 365.25 * 86400


class StubEngine(BacktestEngine):
    """Fast stub: profitable runs with configurable failure modes."""

    name = "simulator"  # skip real MC in validate_strategy by default... 
    # Actually MC runs when settings.MC_ENABLED and name==simulator
    # Use run_montecarlo=False via validation_level 1

    def __init__(self, rate_per_year: float = 0.25, dd_pct: float = 5.0,
                 trades: int = 50, fail_ids: Optional[set] = None):
        self.rate = rate_per_year
        self.dd_pct = dd_pct
        self.trades = trades
        self.fail_ids = fail_ids or set()
        self.run_count = 0

    def run(self, strategy: StrategyDefinition, start: datetime, end: datetime,
            params_override: Optional[Dict[str, float]] = None,
            deposit: float = 10_000.0) -> BacktestMetrics:
        self.run_count += 1
        if strategy.id in self.fail_ids:
            raise RuntimeError(f"simulated failure for {strategy.id}")
        years = max((end - start).total_seconds() / YEAR, 1e-9)
        return BacktestMetrics(
            net_profit=deposit * self.rate * years,
            initial_deposit=deposit,
            start_ts=start.timestamp(),
            end_ts=end.timestamp(),
            max_dd_pct=self.dd_pct,
            trade_count=self.trades,
            profit_factor=1.5,
        )


START = datetime(2023, 1, 1, tzinfo=timezone.utc)
END = datetime(2024, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def temp_storage():
    db = Path(tempfile.mkdtemp(prefix="eaf_worker_")) / "test.db"
    return Storage(db)


@pytest.fixture
def queue(temp_storage):
    return JobQueue(temp_storage)


def _wait_job(storage: Storage, job_id: str, timeout: float = 120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = storage.get_job(job_id)
        if job.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
            return job
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish")


def _base_payload(**overrides):
    payload = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "engine": "simulator",
        "batch_size": 4,
        "target_survivors": 1,
        "max_candidates": 8,
        "genetic": False,
        "seed": 42,
        "start": START.isoformat(),
        "end": END.isoformat(),
        "validation_level": 1,
        "data_source": "synthetic",
        "wfo_train_months": 2,
        "wfo_test_months": 1,
        "wfo_windows": 2,
    }
    payload.update(overrides)
    return payload


def test_discovery_two_stage_screen_and_validate(queue, temp_storage, monkeypatch):
    """Promising candidates pass quick_screen then full validate_strategy."""
    stub = StubEngine()
    monkeypatch.setattr(queue, "_make_engine", lambda _name: stub)

    job_id = "worker_screen_validate"
    assert queue.submit_discovery(job_id, _base_payload())
    job = _wait_job(temp_storage, job_id)
    assert job.status == JobStatus.DONE, job.error

    reports = temp_storage.list_validated(passed_only=False)
    assert reports, "expected validation reports"
    assert any(r.passed for r in reports)
    assert all(r.data_source == "synthetic" for r in reports)
    assert stub.run_count > len(reports), "quick_screen should run before validate"


def test_mt5_discovery_screens_with_simulator_first(queue, temp_storage, monkeypatch):
    """engine=mt5 must simulator-screen first; MT5 only sees promising ones."""
    # First few screens look like junk so they never reach MT5.
    reject_ids: set = set()

    class SelectiveScreen(StubEngine):
        def run(self, strategy, start, end, params_override=None, deposit=10_000.0):
            self.run_count += 1
            if len(reject_ids) < 3:
                reject_ids.add(strategy.id)
            if strategy.id in reject_ids:
                return BacktestMetrics(
                    net_profit=-100.0, initial_deposit=deposit,
                    start_ts=start.timestamp(), end_ts=end.timestamp(),
                    max_dd_pct=50.0, trade_count=1, profit_factor=0.5)
            return BacktestMetrics(
                net_profit=deposit * 0.25, initial_deposit=deposit,
                start_ts=start.timestamp(), end_ts=end.timestamp(),
                max_dd_pct=5.0, trade_count=50, profit_factor=1.5)

    screen = SelectiveScreen()
    mt5 = StubEngine()
    mt5.name = "mt5"
    engines = {"simulator": screen, "mt5": mt5}
    monkeypatch.setattr(queue, "_make_engine", lambda name: engines[name])
    monkeypatch.setattr(
        "jobs.worker.interactive_terminal_running", lambda: False)

    job_id = "worker_mt5_prefilter"
    assert queue.submit_discovery(job_id, _base_payload(
        engine="mt5", max_candidates=6, batch_size=6, target_survivors=1))
    job = _wait_job(temp_storage, job_id)
    assert job.status == JobStatus.DONE, job.error

    reports = temp_storage.list_validated(passed_only=False)
    assert reports, "at least one promising candidate should reach MT5"
    assert all(r.engine == "mt5" for r in reports)
    # First 3 screens are rejected; the 4th is promising and becomes the
    # survivor, so discovery stops before exhausting max_candidates.
    assert screen.run_count >= 4, "rejected + at least one promising screen"
    assert len(reject_ids) == 3
    assert mt5.run_count > 0, "promising candidates must reach MT5 validation"
    assert len(reports) == screen.run_count - len(reject_ids)


def test_discovery_validation_level_wiring(queue, temp_storage, monkeypatch):
    """Floor L1 accepts modest DD; floor L13 (Strict·A) rejects the same stub."""
    monkeypatch.setattr(queue, "_make_engine",
                        lambda _name: StubEngine(dd_pct=12.0))

    job_id = "worker_level_pass"
    assert queue.submit_discovery(job_id, _base_payload(
        validation_level=1, max_candidates=4, target_survivors=1))
    job = _wait_job(temp_storage, job_id)
    assert job.status == JobStatus.DONE
    passed = temp_storage.list_validated(passed_only=True)
    assert passed, "level 1 should accept 12% DD"
    assert all(r.highest_level_passed >= 1 for r in passed)

    db2 = Path(tempfile.mkdtemp(prefix="eaf_worker2_")) / "test.db"
    storage2 = Storage(db2)
    queue2 = JobQueue(storage2)
    monkeypatch.setattr(queue2, "_make_engine",
                        lambda _name: StubEngine(dd_pct=12.0))
    job_id2 = "worker_level_fail"
    assert queue2.submit_discovery(job_id2, _base_payload(
        validation_level=13,
        validation_level_floor=13,
        validation_level_ceiling=13,
        max_candidates=4, target_survivors=1))
    job2 = _wait_job(storage2, job_id2)
    assert job2.status == JobStatus.DONE
    reports2 = storage2.list_validated(passed_only=False)
    assert reports2
    assert not any(r.passed for r in reports2)


def test_discovery_population_tiers_with_high_ceiling(queue, temp_storage, monkeypatch):
    """High ceiling still persists mid-tier clears; floor L1 marks them passed."""
    monkeypatch.setattr(queue, "_make_engine",
                        lambda _name: StubEngine(dd_pct=12.0, trades=20))

    job_id = "worker_tier_pop"
    assert queue.submit_discovery(job_id, _base_payload(
        validation_level=1,
        validation_level_floor=1,
        validation_level_ceiling=13,
        max_candidates=4, target_survivors=1, batch_size=4))
    job = _wait_job(temp_storage, job_id)
    assert job.status == JobStatus.DONE, job.error

    reports = temp_storage.list_validated(passed_only=False)
    assert reports
    assert any(r.passed and r.highest_level_passed >= 1 for r in reports)
    # Population filter by min_level must return the mid-tier clears.
    ge1 = temp_storage.list_validated(passed_only=False, min_level=1)
    assert ge1
    assert all(r.highest_level_passed >= 1 for r in ge1)
    # Stub lacks Sharpe/R²/MC for Standard·A+ — high tiers stay empty.
    ge7 = temp_storage.list_validated(passed_only=False, min_level=7)
    assert ge7 == []
    hist = temp_storage.level_counts(job_id)
    assert sum(hist.get(k, 0) for k in hist if k >= 1) >= 1



def test_failed_complete_validation_persisted(queue, temp_storage, monkeypatch):
    """Candidates that finish validation but fail gates are still in SQLite."""
    monkeypatch.setattr(queue, "_make_engine",
                        lambda _name: StubEngine(dd_pct=12.0))

    job_id = "worker_failed_persist"
    assert queue.submit_discovery(job_id, _base_payload(
        validation_level=13, max_candidates=6, target_survivors=1, batch_size=6))
    job = _wait_job(temp_storage, job_id)
    assert job.status == JobStatus.DONE

    all_reports = temp_storage.list_validated(passed_only=False)
    failed = [r for r in all_reports if not r.passed]
    assert failed, "expected at least one complete-but-failed validation"

    for report in failed:
        strategy = temp_storage.get_strategy(report.strategy_id)
        assert strategy is not None


def test_complete_strategies_survive_storage_reopen(queue, temp_storage, monkeypatch):
    """Simulate a server restart: new Storage on the same DB sees all results."""
    monkeypatch.setattr(queue, "_make_engine", lambda _name: StubEngine())

    job_id = "worker_restart_persist"
    db_path = temp_storage.db_path
    assert queue.submit_discovery(job_id, _base_payload(max_candidates=6))
    job = _wait_job(temp_storage, job_id)
    assert job.status == JobStatus.DONE

    n_before = len(temp_storage.list_validated(passed_only=False))
    assert n_before >= 1

    storage2 = Storage(db_path)
    n_after = len(storage2.list_validated(passed_only=False))
    assert n_after == n_before
    for report in storage2.list_validated(passed_only=False):
        assert storage2.get_strategy(report.strategy_id) is not None


def test_validation_abort_persisted_with_report(queue, temp_storage, monkeypatch):
    """When full validation throws, strategy + aborted report are still stored."""
    from factory.backtest import validation as val_mod

    real_validate = val_mod.validate_strategy
    calls = {"n": 0}

    def flaky_validate(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated validation crash")
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(queue, "_make_engine", lambda _name: StubEngine())
    monkeypatch.setattr("jobs.worker.validate_strategy", flaky_validate)

    job_id = "worker_abort_persist"
    assert queue.submit_discovery(job_id, _base_payload(
        max_candidates=4, batch_size=4, target_survivors=10))
    job = _wait_job(temp_storage, job_id)
    assert job.status == JobStatus.DONE

    reports = temp_storage.list_validated(passed_only=False)
    aborted = [r for r in reports if r.reasons and "did not complete" in r.reasons[0]]
    assert aborted, "aborted validation should leave a failed report"
    assert temp_storage.get_strategy(aborted[0].strategy_id) is not None
    # Non-infra RuntimeError must not be tagged INFRA
    assert not aborted[0].infra_failure
    assert not aborted[0].reasons[0].startswith("INFRA:")


def test_aborted_report_tags_mt5_infra():
    from factory.backtest.mt5_runner import MT5RunnerError
    from factory.generator import random_strategy
    from jobs.worker import _aborted_validation_report
    import random

    strat = random_strategy("EURUSD", "H1", rng=random.Random(1))
    report = _aborted_validation_report(
        strat, "mt5",
        MT5RunnerError("MetaTrader 5 terminal is already running interactively."))
    assert report.infra_failure is True
    assert report.reasons[0].startswith("INFRA:")
    assert report.oos_metrics.trade_count == 0


def test_mt5_preflight_falls_back_to_simulator_when_terminal_running(
        queue, temp_storage, monkeypatch):
    """MT5 discovery falls back to simulator instead of failing the sweep."""
    engines: list[str] = []

    def make_engine(name):
        engines.append(name)
        return StubEngine()

    monkeypatch.setattr(
        "jobs.worker.interactive_terminal_running", lambda: True)
    monkeypatch.setattr(queue, "_make_engine", make_engine)

    job_id = "worker_mt5_fallback"
    assert queue.submit_discovery(
        job_id, _base_payload(engine="mt5", max_candidates=4, batch_size=4))
    job = _wait_job(temp_storage, job_id)
    assert job.status == JobStatus.DONE
    assert "simulator" in engines
    assert not (job.error or "").startswith("INFRA:")


def test_mt5_preflight_fails_fast_when_fallback_disabled(
        queue, temp_storage, monkeypatch):
    """With fallback off, MT5 discovery still fails before burning candidates."""
    monkeypatch.setattr(
        "jobs.worker.interactive_terminal_running", lambda: True)
    monkeypatch.setattr(
        "config.settings.DISCOVERY_MT5_FALLBACK_TO_SIMULATOR", False)
    monkeypatch.setattr(queue, "_make_engine", lambda _name: StubEngine())

    job_id = "worker_mt5_preflight"
    assert queue.submit_discovery(
        job_id, _base_payload(engine="mt5", max_candidates=8))
    job = _wait_job(temp_storage, job_id)
    assert job.status == JobStatus.FAILED
    assert "INFRA:" in (job.error or "")
    assert "already running" in (job.error or "").lower()
    assert temp_storage.list_validated(passed_only=False) == []


def test_discovery_progress_accounting(queue, temp_storage, monkeypatch):
    monkeypatch.setattr(queue, "_make_engine", lambda _name: StubEngine())

    job_id = "worker_progress"
    queue.submit_discovery(job_id, _base_payload(max_candidates=6))
    job = _wait_job(temp_storage, job_id)
    assert job.progress == pytest.approx(1.0)
    assert "done" in (job.message or "").lower()
    assert "tested" in (job.message or "").lower()


def test_discovery_per_strategy_error_handling(queue, temp_storage, monkeypatch):
    """A failing strategy must not abort the whole batch."""
    from factory.backtest import validation as val_mod

    calls = {"n": 0}
    real_quick_screen = val_mod.quick_screen

    def flaky_quick_screen(engine, strategy, start, end, deposit, criteria):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("boom")
        return real_quick_screen(engine, strategy, start, end, deposit, criteria)

    monkeypatch.setattr(queue, "_make_engine", lambda _name: StubEngine())
    monkeypatch.setattr("jobs.worker.quick_screen", flaky_quick_screen)

    job_id = "worker_errors"
    queue.submit_discovery(job_id, _base_payload(
        max_candidates=5, batch_size=5, target_survivors=10))
    job = _wait_job(temp_storage, job_id)
    assert job.status == JobStatus.DONE
    assert job.error and "boom" in job.error
    assert len(temp_storage.list_strategies()) >= 1


def test_month_based_wfo_windows():
    """Rolling WFO with month boundaries produces rolling-mode windows."""
    report = validate_strategy(
        StubEngine(), _minimal_strategy(), START, END,
        seed=1, run_montecarlo=False,
        wfo_train_months=2, wfo_test_months=1, wfo_windows=2,
        data_source="synthetic",
    )
    rolling = [w for w in report.wfo_windows if w.mode == "rolling"]
    assert rolling
    assert report.wfo_train_months == 2
    assert report.wfo_test_months == 1
    span = rolling[0].oos_end_ts - rolling[0].oos_start_ts
    assert span == pytest.approx(1 * settings.DAYS_PER_MONTH * 86400, rel=0.02)


def _minimal_strategy() -> StrategyDefinition:
    from factory.models import (
        EntryFilter, EntryFilterType, ExecutionMechanic, ExecutionMechanicType,
    )
    return StrategyDefinition(
        symbol="TEST", timeframe="H1",
        entry_filters=[EntryFilter(
            type=EntryFilterType.RSI_REVERSION,
            params={"rsi_period": 14, "oversold": 30, "overbought": 70})],
        mechanic=ExecutionMechanic(
            type=ExecutionMechanicType.STANDARD_SLTP,
            params={"sl_points": 100.0, "tp_points": 200.0}),
    )
