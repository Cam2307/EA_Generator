from __future__ import annotations

import time
from pathlib import Path

import pytest

from factory.agent_alerts import (
    is_exceptional_ea,
    maybe_send_progress_digest,
    maybe_send_quality_alerts,
)
from factory.models import BacktestMetrics, EntryFilter, EntryFilterType, ExecutionMechanic, ExecutionMechanicType, StrategyDefinition, ValidationReport
from factory.promotion import evaluate_promotion
from factory.storage import Storage


def _report(**overrides) -> ValidationReport:
    # Metrics sized for edge_positive (~76): clears min_score=75 but stays
    # below the 80 promote_live threshold so the min_score gate is meaningful.
    base = ValidationReport(
        strategy_id="s-alert",
        is_metrics=BacktestMetrics(),
        oos_metrics=BacktestMetrics(
            net_profit=1500.0,
            profit_factor=1.8,
            sharpe=1.4,
            max_dd_pct=8.0,
            trade_count=60,
        ),
        wfe=0.85,
        passed=True,
        stability_ratio=0.85,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "alerts.db"


def test_is_exceptional_ea_requires_strong_promotion(temp_db: Path) -> None:
    strong = evaluate_promotion(
        _report(), holdout_passed=True, mt5_confirmed=False)
    weak = evaluate_promotion(
        _report(oos_metrics=BacktestMetrics(net_profit=10.0, profit_factor=1.0, sharpe=0.1)),
        holdout_passed=True,
    )
    assert is_exceptional_ea(strong, min_score=75.0)
    assert not is_exceptional_ea(strong, min_score=85.0)
    assert not is_exceptional_ea(weak, min_score=75.0)


def test_quality_alert_only_once_per_strategy(temp_db: Path, monkeypatch) -> None:
    storage = Storage(temp_db)

    class _Holdout:
        passed = True
        error = None

    monkeypatch.setattr(
        "factory.holdout.evaluate_holdout",
        lambda *a, **k: _Holdout(),
    )
    strategy = StrategyDefinition(
        id="s-alert",
        symbol="EURUSD",
        timeframe="H1",
        entry_filters=[EntryFilter(
            type=EntryFilterType.RSI_REVERSION,
            params={"rsi_period": 14, "oversold": 30, "overbought": 70},
        )],
        mechanic=ExecutionMechanic(
            type=ExecutionMechanicType.STANDARD_SLTP,
            params={"sl_points": 100.0, "tp_points": 200.0},
        ),
    )
    report = _report()
    report.holdout_passed = True
    report.mt5_confirmed = False  # edge_positive, not watchlist
    storage.save_complete(strategy, report, job_id="job1")

    sent: list[tuple[str, str, str]] = []

    def _fake_send(recipient: str, subject: str, body: str) -> None:
        sent.append((recipient, subject, body))

    monkeypatch.setattr("factory.agent_alerts.send_email", _fake_send)

    count1 = maybe_send_quality_alerts(storage, recipient="user@example.com", min_score=70.0)
    count2 = maybe_send_quality_alerts(storage, recipient="user@example.com", min_score=70.0)

    assert count1 == 1
    assert count2 == 0
    assert len(sent) == 1


def test_progress_digest_respects_interval(temp_db: Path, monkeypatch) -> None:
    storage = Storage(temp_db)
    storage.update_agent_state(enabled=1, status="running", last_progress_email_at=time.time())

    calls: list[str] = []
    monkeypatch.setattr(
        "factory.agent_alerts.send_email",
        lambda recipient, subject, body: calls.append(subject),
    )

    assert maybe_send_progress_digest(
        storage, recipient="user@example.com", progress_email_hours=1.0
    ) is False
    assert calls == []


def test_progress_digest_sends_when_due(temp_db: Path, monkeypatch) -> None:
    storage = Storage(temp_db)
    storage.update_agent_state(
        enabled=1,
        status="running",
        last_progress_email_at=time.time() - 7200,
        message="Running sweep",
    )

    calls: list[str] = []
    monkeypatch.setattr(
        "factory.agent_alerts.send_email",
        lambda recipient, subject, body: calls.append(subject),
    )

    assert maybe_send_progress_digest(
        storage, recipient="user@example.com", progress_email_hours=1.0
    ) is True
    assert calls == ["EA Generator — hourly discovery progress"]
