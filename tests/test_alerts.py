from __future__ import annotations

from factory import alerts


def test_smtp_diagnostics_reports_missing_host(monkeypatch) -> None:
    monkeypatch.delenv("EA_SMTP_HOST", raising=False)
    monkeypatch.delenv("EA_SMTP_FROM", raising=False)
    monkeypatch.setenv("EA_SMTP_USER", "bot@example.com")
    diag = alerts.smtp_diagnostics()
    assert diag.configured is False
    assert "EA_SMTP_HOST" in diag.missing_keys


def test_smtp_diagnostics_accepts_user_when_from_missing(monkeypatch) -> None:
    monkeypatch.setenv("EA_SMTP_HOST", "smtp.example.com")
    monkeypatch.delenv("EA_SMTP_FROM", raising=False)
    monkeypatch.setenv("EA_SMTP_USER", "bot@example.com")
    monkeypatch.setenv("EA_SMTP_PORT", "not-an-int")
    diag = alerts.smtp_diagnostics()
    assert diag.configured is True
    assert diag.port == 587


def test_send_email_raises_actionable_missing_error(monkeypatch) -> None:
    monkeypatch.delenv("EA_SMTP_HOST", raising=False)
    monkeypatch.delenv("EA_SMTP_FROM", raising=False)
    monkeypatch.delenv("EA_SMTP_USER", raising=False)
    try:
        alerts.send_email("user@example.com", "x", "y")
    except RuntimeError as exc:
        msg = str(exc)
        assert "Missing" in msg
        assert "EA_SMTP_HOST" in msg
    else:
        raise AssertionError("send_email should fail when SMTP is missing")
