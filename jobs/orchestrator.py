"""Long-running discovery orchestrator independent from Streamlit lifecycle."""
from __future__ import annotations

import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

from config import settings
from factory.logutil import get_logger
from factory.agent_alerts import (
    maybe_send_ops_alerts,
    maybe_send_progress_digest,
    maybe_send_quality_alerts,
    sync_promotion_scores,
)
from factory.discovery_config import (
    DiscoverySettings,
    build_discovery_payload,
    effective_validation_level,
    settings_from_app,
)
from factory.storage import Storage
from jobs.bandit import (
    corr_penalties_for_plans,
    dump_bandit_stats,
    load_bandit_stats,
    outcome_weight,
    record_outcome,
    record_pull,
    select_plan,
)
from jobs.sweep import plan_sweeps
from jobs.worker import _pid_alive, get_job_queue

log = get_logger(__name__)

LOCK_PATH = settings.DATA_DIR / "discovery_orchestrator.lock"
ERROR_LOG_PATH = settings.DATA_DIR / "discovery_agent_error.log"
SERVICE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "discovery_agent_service.py"
_STARTING_STALE_SECONDS = 5
_RECOVER_RETRY_SECONDS = 15
_MAX_RECOVER_SPAWNS = 5
_ALERT_LOCK = threading.Lock()
_ALERT_THREAD: threading.Thread | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def _python_executable(*, windowless: bool = False) -> str:
    """Launch with an interpreter that has project dependencies.

    On Windows, Streamlit is often started as
    ``base_python.exe .venv\\Scripts\\streamlit.exe``, so ``sys.executable`` is
    the system interpreter (no numpy). Always prefer this repo's ``.venv`` when
    it exists, then ``VIRTUAL_ENV``, then a ``streamlit.exe`` sibling, then
    ``sys.executable``.
    """
    candidates: list[Path] = []
    project_venv = _venv_python(_project_root())
    if project_venv.exists():
        candidates.append(project_venv)
    ve = os.environ.get("VIRTUAL_ENV")
    if ve:
        ve_py = (
            Path(ve) / "Scripts" / "python.exe"
            if os.name == "nt"
            else Path(ve) / "bin" / "python"
        )
        candidates.append(ve_py)
    exe = Path(sys.executable)
    if exe.stem.lower() == "streamlit":
        sibling = exe.with_name("python.exe" if os.name == "nt" else "python")
        candidates.append(sibling)
        root_py = exe.parent.parent / ("python.exe" if os.name == "nt" else "python")
        candidates.append(root_py)
    candidates.append(exe)

    chosen = next((c for c in candidates if c.exists()), exe)
    if windowless and os.name == "nt":
        pyw = chosen.with_name("pythonw.exe")
        if pyw.exists():
            return str(pyw)
    return str(chosen)


def _windows_subprocess_flags(*, detached: bool = False) -> int:
    """Hide console windows when spawning subprocesses on Windows."""
    if os.name != "nt":
        return 0
    flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    if detached:
        flags |= (
            subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
        )
    return flags


def _read_lock_pid() -> int:
    if not LOCK_PATH.exists():
        return 0
    try:
        raw = LOCK_PATH.read_text(encoding="utf-8").strip()
        return int(raw) if raw else 0
    except (OSError, ValueError):
        return 0


def clear_stale_orchestrator_lock() -> bool:
    """Remove a lock left behind by a dead orchestrator process."""
    if not LOCK_PATH.exists():
        return False
    pid = _read_lock_pid()
    if pid > 0 and _pid_alive(pid):
        return False
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def _startup_error_detail() -> str:
    if not ERROR_LOG_PATH.exists():
        return ""
    try:
        text = ERROR_LOG_PATH.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def sync_agent_with_orchestrator_lock(storage: Storage | None = None) -> bool:
    """Promote agent_state to running when the lock is held by a live process."""
    store = storage or Storage()
    pid = _read_lock_pid()
    if pid <= 0 or not _pid_alive(pid):
        return False
    state = store.get_agent_state()
    if not bool(state.get("enabled", 0)):
        return False
    if str(state.get("status") or "") == "running" and int(state.get("pid") or 0) == pid:
        return True
    store.update_agent_state(
        enabled=1,
        status="running",
        pid=pid,
        spawn_attempts=0,
        message=str(state.get("message") or "Discovery agent running"),
    )
    return True


def start_orchestrator_process() -> bool:
    """Spawn detached orchestrator process; returns False if already running."""
    if sync_agent_with_orchestrator_lock():
        return True
    clear_stale_orchestrator_lock()
    if LOCK_PATH.exists():
        return sync_agent_with_orchestrator_lock()
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    err_log = open(ERROR_LOG_PATH, "ab")  # noqa: SIM115 - inherited by child briefly
    subprocess.Popen(  # noqa: S603
        [_python_executable(windowless=True), str(SERVICE_SCRIPT)],
        cwd=str(Path(__file__).resolve().parents[1]),
        creationflags=_windows_subprocess_flags(detached=True),
        stdout=subprocess.DEVNULL,
        stderr=err_log,
        close_fds=False,
    )
    err_log.close()
    return True


def recover_stuck_starting_agent(storage: Storage | None = None) -> bool:
    """Retry spawn when the UI enabled the agent but no live process took over."""
    store = storage or Storage()
    if sync_agent_with_orchestrator_lock(store):
        return True
    state = store.get_agent_state()
    if not bool(state.get("enabled", 0)):
        return False
    if str(state.get("status") or "") != "starting":
        return False
    pid = int(state.get("pid") or 0)
    if pid > 0 and _pid_alive(pid):
        return False
    now = time.time()
    updated = float(state.get("updated_at") or 0)
    # A fresh click with no pid yet gets a short grace period for spawn.
    if pid <= 0 and updated and (now - updated) < _STARTING_STALE_SECONDS:
        return False
    detail = _startup_error_detail()
    if detail and not LOCK_PATH.exists():
        store.update_agent_state(
            enabled=0,
            status="stopped",
            pid=None,
            message=f"Discovery agent failed to start: {detail}",
        )
        clear_stale_orchestrator_lock()
        return False
    attempts = int(state.get("spawn_attempts") or 0)
    if attempts >= _MAX_RECOVER_SPAWNS:
        store.update_agent_state(
            enabled=0,
            status="stopped",
            pid=None,
            message=(
                "Discovery agent failed to start after several attempts. "
                "Stop any stale process, then try again."
                + (f" Last error: {detail}" if detail else "")
            ),
        )
        clear_stale_orchestrator_lock()
        return False
    # Rate-limit recovery so the live-status poll cannot spawn every 2s.
    if attempts > 0 and updated and (now - updated) < _RECOVER_RETRY_SECONDS:
        return False
    clear_stale_orchestrator_lock()
    spawned = start_orchestrator_process()
    store.update_agent_state(
        updated_at=now,
        spawn_attempts=attempts + 1,
    )
    sync_agent_with_orchestrator_lock(store)
    return spawned


def stop_orchestrator_process() -> None:
    """Disable agent, cancel discovery jobs immediately, force-kill in background.

    DB updates happen on the caller thread so the UI can clear Active runs on
    the next fragment poll. ``taskkill`` / SIGTERM runs in a daemon thread so
    Streamlit is never blocked waiting on process teardown.
    """
    import threading

    storage = Storage()
    state = storage.get_agent_state()
    pid = int(state.get("pid") or 0) or _read_lock_pid()
    storage.update_agent_state(
        enabled=0,
        status="stopping",
        message="Stop requested — cancelling jobs and stopping agent",
    )
    storage.cancel_active_discovery_jobs(
        message="Cancelled — agent stop requested",
    )

    def _force_stop(target_pid: int) -> None:
        try:
            if target_pid:
                if os.name == "nt":
                    subprocess.run(  # noqa: S603,S607
                        ["taskkill", "/PID", str(target_pid), "/T", "/F"],
                        check=False,
                        creationflags=_windows_subprocess_flags(),
                    )
                else:
                    try:
                        os.kill(int(target_pid), 15)
                    except ProcessLookupError:
                        pass
        except Exception:
            pass
        clear_stale_orchestrator_lock()
        try:
            store = Storage()
            store.update_agent_state(
                enabled=0,
                status="stopped",
                pid=None,
                current_job_id=None,
                message="Agent stopped",
            )
        except Exception:
            pass

    threading.Thread(
        target=_force_stop,
        args=(pid,),
        daemon=True,
        name="orchestrator-stop",
    ).start()


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


def run_orchestrator_forever(sleep_seconds: int = 1) -> None:
    storage = Storage()
    queue = get_job_queue()
    try:
        with OrchestratorSingleton():
            storage.update_agent_state(enabled=1, status="running", pid=os.getpid())
            while True:
                cfg = _load_discovery_config(storage)
                if not cfg["enabled"]:
                    storage.update_agent_state(
                        status="stopped",
                        heartbeat_at=time.time(),
                        pid=None,
                        message="Stopped",
                        current_job_id=None,
                    )
                    break
                _tick(storage, queue, cfg)
                time.sleep(sleep_seconds)
    except FileExistsError:
        if sync_agent_with_orchestrator_lock(storage):
            return
        raise


def _load_discovery_config(storage: Storage) -> dict:
    app = storage.get_app_settings()
    discovery = settings_from_app(app)
    state = storage.get_agent_state()
    return {
        "enabled": bool(state.get("enabled", 0)),
        "mode": str(state.get("mode") or "continuous"),
        "discovery": discovery,
        "alert_min_score": discovery.alert_min_score,
        "progress_email_hours": discovery.progress_email_hours,
        "recipient_email": discovery.recipient_email,
    }


def _tick(storage: Storage, queue, cfg: dict) -> None:
    import random
    from datetime import datetime, timezone

    discovery: DiscoverySettings = cfg["discovery"]
    jobs = storage.list_jobs("discovery")
    active = [j for j in jobs if j.status.value in ("PENDING", "RUNNING")]
    state = storage.get_agent_state()
    cursor = int(state.get("cursor", 0) or 0)
    mode = str(cfg.get("mode") or "continuous")
    plans = plan_sweeps(
        symbols=list(discovery.symbols),
        timeframes=list(discovery.timeframes),
        months=discovery.months,
        base_seed=discovery.base_seed,
    )
    sweep_total = len(plans)
    bandit = load_bandit_stats(state.get("bandit_stats"))
    # Heartbeat first so a slow later step cannot leave the UI looking dead.
    agent_updates: dict = {
        "status": "running",
        "heartbeat_at": time.time(),
        "queue_depth": len(active),
        "pid": os.getpid(),
        "sweep_total": sweep_total,
        "mode": mode,
    }
    storage.update_agent_state(**agent_updates)

    # Learn from recently finished jobs tagged with sweep symbol/TF.
    _update_bandit_from_finished_jobs(storage, state, bandit)

    # Daily wall-clock budget: pause submissions until next UTC day.
    budget_hours = float(getattr(discovery, "daily_budget_hours", 0.0) or 0.0)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    budget_day = str(state.get("budget_day_utc") or "")
    used = float(state.get("budget_seconds_used_today") or 0.0)
    if budget_day != today:
        used = 0.0
        agent_updates["budget_day_utc"] = today
        agent_updates["budget_seconds_used_today"] = 0.0
        agent_updates["budget_paused_until"] = None
    if budget_hours > 0 and used >= budget_hours * 3600.0 and not active:
        # Pause until next UTC midnight.
        from datetime import timedelta
        tomorrow = (datetime.now(timezone.utc).date() + timedelta(days=1))
        resume_at = datetime(
            tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=timezone.utc
        ).timestamp()
        agent_updates["status"] = "paused_budget"
        agent_updates["budget_paused_until"] = resume_at
        agent_updates["message"] = (
            f"Daily budget exhausted ({budget_hours:.1f}h) — "
            f"pausing until next UTC day"
        )
        agent_updates["bandit_stats"] = dump_bandit_stats(bandit)
        storage.update_agent_state(**agent_updates)
        _schedule_alert_pass(storage, cfg)
        return
    # Accrue wall-clock while a job is active.
    if active and budget_hours > 0:
        agent_updates["budget_seconds_used_today"] = used + 1.0  # tick interval
        agent_updates["budget_day_utc"] = today

    # One cheap bandit penalty map per tick (must not touch the validations blob).
    penalties = corr_penalties_for_plans(plans, storage=storage) if plans else {}

    if active:
        job = active[0]
        sweep_idx = max(cursor - 1, 0) % sweep_total if sweep_total else 0
        plan = plans[sweep_idx] if plans else None
        # Prefer the symbol/TF stored on the in-flight job payload.
        if job.payload.get("symbol") and job.payload.get("timeframe"):
            from jobs.sweep import SweepPlan
            plan = SweepPlan(
                symbol=str(job.payload["symbol"]),
                timeframe=str(job.payload["timeframe"]),
                seed=int(job.payload.get("seed") or 0),
                payload_patch={},
            )
        agent_updates["current_job_id"] = job.id
        eff_level = job.payload.get("validation_level")
        if eff_level is not None:
            agent_updates["effective_validation_level"] = int(eff_level)
        if plan is not None:
            sweep_label = f"{plan.symbol} · {plan.timeframe}"
            job_detail = job.message or job.status.value.lower()
            agent_updates["message"] = f"Running sweep — {sweep_label} — {job_detail}"
        else:
            agent_updates["message"] = job.message or "Running discovery sweep"
    elif plans:
        agent_updates["current_job_id"] = None
        next_level = effective_validation_level(
            discovery,
            sweep_index=cursor,
            sweep_total=sweep_total,
        )
        agent_updates["effective_validation_level"] = next_level
        # Preview the bandit-chosen next arm without recording a pull yet.
        try:
            _idx, preview = select_plan(
                plans, bandit, rng=random.Random(cursor + discovery.base_seed),
                corr_penalty=penalties)
            agent_updates["message"] = (
                f"Idle — next sweep: {preview.symbol} · {preview.timeframe}"
            )
        except Exception:
            plan = plans[cursor % len(plans)]
            agent_updates["message"] = (
                f"Idle — next sweep: {plan.symbol} · {plan.timeframe}"
            )
    else:
        agent_updates["current_job_id"] = None
        agent_updates["message"] = "No sweeps configured — check symbols/timeframes"

    if plans and len(active) == 0:
        if mode == "batch" and cursor >= len(plans):
            agent_updates["enabled"] = 0
            agent_updates["status"] = "stopped"
            agent_updates["message"] = (
                f"Batch complete — {len(plans)} sweeps finished"
            )
        else:
            _idx, plan = select_plan(
                plans, bandit,
                # Deterministic: same cursor + seed → same arm order.
                rng=random.Random(
                    (int(cursor) * 1_000_003
                     + int(discovery.base_seed) * 97
                     + len(plans)) & 0x7FFFFFFF),
                corr_penalty=penalties)
            record_pull(bandit, plan.symbol, plan.timeframe)
            level = effective_validation_level(
                discovery,
                sweep_index=cursor,
                sweep_total=sweep_total,
            )
            payload = build_discovery_payload(
                discovery,
                symbol=plan.symbol,
                timeframe=plan.timeframe,
                seed=plan.seed,
                validation_level=level,
            )
            if mode == "continuous":
                # Keep searching until the user stops the agent. Survivor
                # targets only apply to one-shot / batch jobs.
                payload["continuous"] = True
                payload["target_survivors"] = 10**9
                # Continuous mass search always uses the simulator for Stage-2
                # so interactive MT5 / exclusive locks cannot empty the funnel.
                # Single-run UI can still pick MT5 explicitly.
                payload["engine"] = "simulator"
            job_id = f"auto_{int(time.time())}_{cursor % len(plans):03d}"
            queue.submit_discovery(job_id, payload)
            storage.update_agent_state(
                cursor=cursor + 1,
                jobs_submitted=int(state.get("jobs_submitted", 0)) + 1,
                bandit_stats=dump_bandit_stats(bandit),
                last_bandit_job_id=job_id,
            )
            agent_updates["current_job_id"] = job_id
            agent_updates["effective_validation_level"] = level
            lvl_name = ""
            if not discovery.use_custom:
                from factory import validation_levels

                lvl_name = validation_levels.get_level(level).name
            level_note = (
                f"level {level} ({lvl_name})"
                if lvl_name
                else f"level {level}"
            )
            if discovery.progressive_strictness and not discovery.use_custom:
                level_note += f" · progressive (max {discovery.validation_level})"
            if mode == "continuous":
                agent_updates["message"] = (
                    f"Submitted sweep — {plan.symbol} · {plan.timeframe} · "
                    f"{level_note} · continuous (runs until Stop)"
                )
            else:
                agent_updates["message"] = (
                    f"Submitted sweep — {plan.symbol} · {plan.timeframe} · {level_note}"
                )
    agent_updates["bandit_stats"] = dump_bandit_stats(bandit)
    agent_updates["heartbeat_at"] = time.time()
    storage.update_agent_state(**agent_updates)
    # Never block the 1s tick loop on SMTP / promotion scoring.
    _schedule_alert_pass(storage, cfg)


def _schedule_alert_pass(storage: Storage, cfg: dict) -> None:
    """Fire-and-forget alert/promotion work so the tick loop stays snappy."""
    global _ALERT_THREAD
    if not _ALERT_LOCK.acquire(blocking=False):
        return  # previous alert pass still running

    def _run() -> None:
        try:
            _run_alert_pass(storage, cfg)
        except Exception as exc:
            log.warning("alert/promotion pass failed: %s", exc)
        finally:
            _ALERT_LOCK.release()

    _ALERT_THREAD = threading.Thread(
        target=_run, name="ea-alert-pass", daemon=True)
    _ALERT_THREAD.start()


def _update_bandit_from_finished_jobs(storage: Storage, state: dict,
                                      bandit: dict) -> None:
    """Credit Thompson arms when a sweep job finishes with survivors.

    Soft-weights successes: +1 for any survivor, +2 when the job cleared
    L4+ (or the job's validation floor) so productive niches get more pulls.
    """
    last_seen = str(state.get("last_bandit_learned_job") or "")
    jobs = storage.list_jobs("discovery")
    for job in jobs:
        if job.id == last_seen:
            break
        if job.status.value not in ("DONE", "FAILED", "CANCELLED"):
            continue
        symbol = str(job.payload.get("symbol") or "").strip().upper()
        timeframe = str(job.payload.get("timeframe") or "")
        if not symbol or not timeframe:
            continue
        max_level = 0
        try:
            max_level = int(storage.job_max_level_passed(job.id) or 0)
        except Exception:
            max_level = 0
        floor = int(
            job.payload.get("validation_level_floor")
            or job.payload.get("validation_level")
            or 4
        )
        success, weight = outcome_weight(
            survivors=int(job.survivors or 0) if job.status.value == "DONE" else 0,
            max_level=max_level if job.status.value == "DONE" else 0,
            floor_level=floor,
        )
        record_outcome(bandit, symbol, timeframe, success=success, weight=weight)
        storage.update_agent_state(last_bandit_learned_job=job.id)
        break


def _run_alert_pass(storage: Storage, cfg: dict) -> None:
    """Refresh promotion scores, email exceptional EAs once, and hourly progress."""
    state = storage.get_agent_state()
    now = time.time()
    last_sync = float(state.get("last_promotion_sync_at") or 0)
    if now - last_sync >= 60:
        sync_promotion_scores(storage)
        storage.update_agent_state(last_promotion_sync_at=now)
    recipient = str(cfg["recipient_email"]).strip()
    if not recipient:
        return
    maybe_send_quality_alerts(
        storage, recipient=recipient, min_score=cfg["alert_min_score"]
    )
    maybe_send_progress_digest(
        storage,
        recipient=recipient,
        progress_email_hours=cfg["progress_email_hours"],
    )
    maybe_send_ops_alerts(storage, recipient=recipient)
