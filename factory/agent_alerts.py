"""Discovery-agent email rules: hourly progress digests and exceptional-EA alerts."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from factory.alerts import send_email
from factory.promotion import evaluate_promotion

if TYPE_CHECKING:
    from factory.models import ValidationReport
    from factory.storage import Storage


def is_exceptional_ea(decision, *, min_score: float) -> bool:
    """Return True only for genuinely strong strategies worth emailing about."""
    if not decision.hard_gates_passed:
        return False
    if decision.promotion_state == "promoted_live_watchlist":
        return True
    return (
        decision.promotion_state == "edge_positive"
        and decision.quality_score >= float(min_score)
    )


def _duplicate_signatures(storage: Storage, reports: list) -> tuple[dict[str, int], dict[str, str]]:
    signatures: dict[str, int] = {}
    report_sig: dict[str, str] = {}
    for rep in reports:
        strat = storage.get_strategy(rep.strategy_id)
        sig = ""
        if strat is not None and strat.profile.portfolio_signature:
            sig = strat.profile.portfolio_signature
        report_sig[rep.strategy_id] = sig
        if sig:
            signatures[sig] = signatures.get(sig, 0) + 1
    return signatures, report_sig


def sync_promotion_scores(storage: Storage, *, limit: int = 100) -> int:
    """Refresh promotion metadata for unscored / recently updated rows only.

    Returns how many rows were processed. Caps work per call so the
    orchestrator loop never stalls on a full-library deserialize.
    """
    state = storage.get_agent_state()
    since = float(state.get("last_promotion_sync_at") or 0.0)
    reports = storage.list_validations_needing_promotion(since=since, limit=limit)
    if not reports:
        return 0
    signatures, report_sig = _duplicate_signatures(storage, reports)
    for report in reports:
        sig = report_sig.get(report.strategy_id, "")
        duplicate_penalty = 0.0
        if sig and signatures.get(sig, 0) > 1:
            duplicate_penalty = min(10.0, (signatures[sig] - 1) * 3.0)
        decision = evaluate_promotion(report, duplicate_penalty=duplicate_penalty)
        storage.update_validation_promotion(
            report.strategy_id,
            promotion_state=decision.promotion_state,
            quality_score=decision.quality_score,
            hard_gates_passed=decision.hard_gates_passed,
            quality_breakdown=decision.breakdown,
        )
    return len(reports)


def maybe_send_quality_alerts(
    storage: Storage,
    *,
    recipient: str,
    min_score: float,
) -> int:
    """Email at most once per strategy when it first qualifies as exceptional."""
    if not recipient.strip():
        return 0
    # Score a batch of unscored/recent rows first so hard_gates / quality_score
    # columns are available for the cheap unalerted query below.
    sync_promotion_scores(storage, limit=100)
    reports = storage.list_unalerted_quality_candidates(
        min_score=float(min_score), limit=50,
    )
    if not reports:
        return 0
    signatures, report_sig = _duplicate_signatures(storage, reports)
    sent = 0
    for report in reports:
        sig = report_sig.get(report.strategy_id, "")
        duplicate_penalty = 0.0
        if sig and signatures.get(sig, 0) > 1:
            duplicate_penalty = min(10.0, (signatures[sig] - 1) * 3.0)
        decision = evaluate_promotion(report, duplicate_penalty=duplicate_penalty)
        if not is_exceptional_ea(decision, min_score=min_score):
            continue
        alert_state = storage.get_alert_state(report.strategy_id)
        if alert_state.get("last_alert_at"):
            continue
        subject = f"EA discovery alert: {decision.promotion_state}"
        body = _format_quality_alert(report, decision)
        try:
            send_email(recipient, subject, body)
            storage.mark_alert_sent(
                report.strategy_id,
                fingerprint=f"{decision.promotion_state}|{round(decision.quality_score, 2)}",
            )
            sent += 1
        except Exception:
            continue
    return sent


def maybe_send_progress_digest(
    storage: Storage,
    *,
    recipient: str,
    progress_email_hours: float,
) -> bool:
    """Send a single hourly-style progress summary (not per-candidate)."""
    if not recipient.strip():
        return False
    state = storage.get_agent_state()
    if not int(state.get("enabled", 0) or 0):
        return False
    interval = max(0.25, float(progress_email_hours)) * 3600.0
    last_sent = float(state.get("last_progress_email_at") or 0.0)
    now = time.time()
    if now - last_sent < interval:
        return False

    agent_status = str(state.get("status", "stopped"))
    jobs_submitted = int(state.get("jobs_submitted", 0) or 0)
    sweep_total = int(state.get("sweep_total", 0) or 0)
    cursor = int(state.get("cursor", 0) or 0)
    passing = storage.count_validated(passed_only=True)
    tested_total = storage.count_strategies()
    message = str(state.get("message") or "").strip()

    current_job_id = state.get("current_job_id")
    job_line = "none"
    if current_job_id:
        job = storage.get_job(str(current_job_id))
        if job is not None:
            tested = int(getattr(job, "tested", 0) or 0)
            survivors = int(getattr(job, "survivors", 0) or 0)
            pct = min(max(job.progress, 0.0), 1.0) * 100.0
            job_line = (
                f"{job.id} — {job.status.value} — {pct:.0f}% — "
                f"tested {tested}, passed {survivors}"
            )

    sweep_line = "n/a"
    if sweep_total > 0:
        sweep_idx = max(cursor - 1, 0) % sweep_total
        sweep_line = f"{sweep_idx + 1} / {sweep_total}"

    body = (
        "EA Generator — automated discovery progress\n"
        f"Agent status: {agent_status}\n"
        f"Sweep cycle position: {sweep_line}\n"
        f"Sweeps submitted (lifetime): {jobs_submitted}\n"
        f"Current sweep: {job_line}\n"
        f"Activity: {message or 'idle'}\n"
        f"Winning strategies (library): {passing}\n"
        f"Total candidates evaluated (library): {tested_total}\n"
    )
    try:
        send_email(recipient, "EA Generator — hourly discovery progress", body)
        storage.update_agent_state(last_progress_email_at=now)
        return True
    except Exception:
        return False


def _format_quality_alert(report: ValidationReport, decision) -> str:
    return (
        f"Strategy: {report.strategy_id}\n"
        f"Promotion: {decision.promotion_state}\n"
        f"Quality score: {decision.quality_score:.2f}\n"
        f"WFE: {report.wfe:.2f}\n"
        f"OOS PF: {report.oos_metrics.profit_factor:.2f}\n"
        f"OOS Sharpe: {report.oos_metrics.sharpe:.2f}\n"
    )
