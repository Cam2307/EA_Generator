"""SMTP email helpers for discovery alerts."""
from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage


@dataclass(frozen=True)
class SmtpDiagnostics:
    host: str
    port: int
    username: str
    from_email: str
    use_tls: bool
    missing_keys: list[str]
    has_password: bool

    @property
    def configured(self) -> bool:
        return len(self.missing_keys) == 0


def _parse_port(raw: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 587


def smtp_config_from_env() -> dict:
    return {
        "host": os.getenv("EA_SMTP_HOST", ""),
        "port": _parse_port(os.getenv("EA_SMTP_PORT", "587")),
        "username": os.getenv("EA_SMTP_USER", ""),
        "password": os.getenv("EA_SMTP_PASS", ""),
        "from_email": os.getenv("EA_SMTP_FROM", os.getenv("EA_SMTP_USER", "")),
        "use_tls": os.getenv("EA_SMTP_TLS", "1") not in ("0", "false", "False"),
    }


def smtp_diagnostics() -> SmtpDiagnostics:
    cfg = smtp_config_from_env()
    missing: list[str] = []
    if not cfg["host"]:
        missing.append("EA_SMTP_HOST")
    if not cfg["from_email"] and not cfg["username"]:
        missing.append("EA_SMTP_FROM or EA_SMTP_USER")
    return SmtpDiagnostics(
        host=str(cfg["host"]),
        port=int(cfg["port"]),
        username=str(cfg["username"]),
        from_email=str(cfg["from_email"]),
        use_tls=bool(cfg["use_tls"]),
        missing_keys=missing,
        has_password=bool(cfg["password"]),
    )


def smtp_missing_message(diag: SmtpDiagnostics) -> str:
    if diag.configured:
        return ""
    return (
        "SMTP is not fully configured. Missing: "
        + ", ".join(diag.missing_keys)
        + ". Set these in your process environment and restart if changed outside the app."
    )


def send_email(recipient: str, subject: str, body: str) -> None:
    cfg = smtp_config_from_env()
    diag = smtp_diagnostics()
    if not diag.configured:
        raise RuntimeError(smtp_missing_message(diag))
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["from_email"]
    msg["To"] = recipient
    msg.set_content(body)

    with smtplib.SMTP(cfg["host"], cfg["port"], timeout=20) as smtp:
        if cfg["use_tls"]:
            smtp.starttls()
        if cfg["username"]:
            smtp.login(cfg["username"], cfg["password"])
        smtp.send_message(msg)
