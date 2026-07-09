"""Long-running discovery orchestrator independent from Streamlit lifecycle."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

from config import settings
from factory.alerts import send_email
from factory.promotion import evaluate_promotion
from factory.storage import Storage
from jobs.sweep import plan_sweeps
from jobs.worker import get_job_queue

LOCK_PATH = settings.DATA_DIR / "discovery_orchestrator.lock"
SERVICE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "discovery_agent_service.py"


def start_orchestrator_process() -> bool:
    """Spawn detached orchestrator process; returns False if already running."""
    if LOCK_PATH.exists():
        return False
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
    subprocess.Popen(  # noqa: S603
        [sys.executable, str(SERVICE_SCRIPT)],
        cwd=str(Path(__file__).resolve().parents[1]),
        creationflags=flags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


def stop_orchestrator_process() -> None:
    storage = Storage()
    state = storage.get_agent_state()
    pid = state.get("pid")
    storage.update_agent_state(enabled=0, status="stopping")
    if pid:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)  # noqa: S603,S607
            else:
                os.kill(int(pid), 15)
        except Exception:
            pass


class OrchestratorSingleton:
    def __enter__(self):
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            LOCK_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def run_orchestrator_forever(sleep_seconds: int = 5) -> None:
    storage = Storage()
    queue = get_job_queue()
    with OrchestratorSingleton():
        storage.update_agent_state(enabled=1, status="running", pid=os.getpid())
        while True:
            cfg = _load_agent_config(storage)
            if not cfg["enabled"]:
                storage.update_agent_state(status="stopped", heartbeat_at=time.time(), pid=None)
                break
            _tick(storage, queue, cfg)
            time.sleep(sleep_seconds)


def _load_agent_config(storage: Storage) -> dict:
    app = storage.get_app_settings()
    return {
        "enabled": bool(storage.get_agent_state().get("enabled", 0)),
        "symbols": app.get("agent_symbols", settings.SYMBOLS[:5]),
        "timeframes": app.get("agent_timeframes", ["M15", "H1"]),
        "strictness_profiles": app.get("agent_strictness_profiles", ["easy", "normal", "hard", "custom"]),
        "months": int(app.get("agent_history_months", 12)),
        "batch_size": int(app.get("agent_batch_size", 100)),
        "max_candidates": int(app.get("agent_max_candidates", 1000)),
        "target_survivors": int(app.get("agent_target_survivors", 2)),
        "cooldown_minutes": int(app.get("alert_cooldown_minutes", 60)),
        "alert_min_score": float(app.get("alert_min_score", 70.0)),
        "recipient_email": str(app.get("recipient_email", "camdwg@gmail.com")),
        "custom_criteria": app.get("agent_custom_criteria", {}),
        "base_seed": int(app.get("agent_base_seed", 1337)),
        "advanced_mode": bool(app.get("agent_advanced_mode", True)),
        "complexity_cap": int(app.get("agent_complexity_cap", 6)),
        "enable_regime_switching": bool(app.get("agent_enable_regime_switching", True)),
        "enable_mtf_context": bool(app.get("agent_enable_mtf_context", True)),
        "feature_toggles": app.get(
            "agent_feature_toggles",
            ["momentum", "mean_reversion", "volatility", "market_structure"],
        ),
    }


def _tick(storage: Storage, queue, cfg: dict) -> None:
    jobs = storage.list_jobs("discovery")
    active = [j for j in jobs if j.status.value in ("PENDING", "RUNNING")]
    state = storage.get_agent_state()
    cursor = int(state.get("cursor", 0) or 0)
    plans = plan_sweeps(
        symbols=list(cfg["symbols"]),
        timeframes=list(cfg["timeframes"]),
        strictness_profiles=list(cfg["strictness_profiles"]),
        months=cfg["months"],
        base_seed=cfg["base_seed"],
        custom_criteria=cfg["custom_criteria"],
    )
    if plans and len(active) == 0:
        plan = plans[cursor % len(plans)]
        payload = {
            "engine": settings.DEFAULT_ENGINE,
            "batch_size": cfg["batch_size"],
            "target_survivors": cfg["target_survivors"],
            "max_candidates": cfg["max_candidates"],
            "genetic": True,
            "seed": int(plan.seed),
            "advanced_mode": cfg["advanced_mode"],
            "complexity_cap": cfg["complexity_cap"],
            "enable_regime_switching": cfg["enable_regime_switching"],
            "enable_mtf_context": cfg["enable_mtf_context"],
            "feature_toggles": list(cfg["feature_toggles"]),
            **plan.payload_patch,
        }
        job_id = f"auto_{int(time.time())}_{cursor % len(plans):03d}"
        queue.submit_discovery(job_id, payload)
        storage.update_agent_state(cursor=cursor + 1, jobs_submitted=int(state.get("jobs_submitted", 0)) + 1)
    storage.update_agent_state(status="running", heartbeat_at=time.time(), queue_depth=len(active), pid=os.getpid())
    _run_alert_pass(storage, cfg)


def _run_alert_pass(storage: Storage, cfg: dict) -> None:
    recipient = cfg["recipient_email"].strip()
    if not recipient:
        return
    cooldown = max(1, int(cfg["cooldown_minutes"])) * 60
    min_score = float(cfg["alert_min_score"])
    reports = storage.list_validated(passed_only=False)
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
        if not decision.hard_gates_passed or decision.quality_score < min_score:
            continue
        alert_state = storage.get_alert_state(report.strategy_id)
        now = time.time()
        last_sent = float(alert_state.get("last_alert_at") or 0.0)
        fingerprint = hashlib.sha1(
            f"{decision.promotion_state}|{round(decision.quality_score, 2)}".encode("utf-8")
        ).hexdigest()
        if alert_state.get("alert_fingerprint") == fingerprint and now - last_sent < cooldown:
            continue
        subject = f"EA discovery alert: {decision.promotion_state}"
        body = (
            f"Strategy: {report.strategy_id}\n"
            f"Promotion: {decision.promotion_state}\n"
            f"Quality score: {decision.quality_score:.2f}\n"
            f"WFE: {report.wfe:.2f}\n"
            f"OOS PF: {report.oos_metrics.profit_factor:.2f}\n"
            f"OOS Sharpe: {report.oos_metrics.sharpe:.2f}\n"
        )
        try:
            send_email(recipient, subject, body)
            storage.mark_alert_sent(report.strategy_id, fingerprint=fingerprint)
        except Exception:
            # Keep orchestrator resilient if SMTP is unavailable.
            continue
