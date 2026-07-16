"""Watchdog for the detached discovery agent.

Polls ``agent_state.heartbeat_at``. When the continuous agent is enabled but
the heartbeat goes stale (process crashed / hung), restarts the orchestrator
via ``start_orchestrator_process`` and records a restart alert fingerprint so
``factory.agent_alerts`` can email the operator.

Intended to run as a Windows Scheduled Task or a small always-on loop::

    py -3.11 scripts/discovery_agent_watchdog.py
    py -3.11 scripts/discovery_agent_watchdog.py --once
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from factory.storage import Storage  # noqa: E402
from jobs.orchestrator import (  # noqa: E402
    clear_stale_orchestrator_lock,
    start_orchestrator_process,
    sync_agent_with_orchestrator_lock,
)
from jobs.worker import _pid_alive  # noqa: E402

# Heartbeat older than this (seconds) while enabled => restart.
STALE_SECONDS = 90.0
# Minimum gap between automatic restarts.
RESTART_COOLDOWN_SECONDS = 120.0


def check_and_restart(*, stale_seconds: float = STALE_SECONDS) -> str:
    """Return an action label: ok | restarted | skipped | stopped."""
    storage = Storage()
    state = storage.get_agent_state()
    if not bool(state.get("enabled", 0)):
        return "stopped"

    status = str(state.get("status") or "")
    if status == "paused_budget":
        # Budget pause is intentional — do not restart.
        storage.update_agent_state(heartbeat_at=time.time())  # keep UI fresh
        return "ok"

    if sync_agent_with_orchestrator_lock(storage):
        hb = float(state.get("heartbeat_at") or 0.0)
        # Re-read after sync.
        state = storage.get_agent_state()
        hb = float(state.get("heartbeat_at") or hb)
        if hb and (time.time() - hb) < stale_seconds:
            return "ok"
        # Lock held but heartbeat stale — process may be wedged.
        pid = int(state.get("pid") or 0)
        if pid > 0 and _pid_alive(pid) and hb and (time.time() - hb) < stale_seconds * 2:
            return "ok"

    last_restart = float(state.get("last_watchdog_restart_at") or 0.0)
    if last_restart and (time.time() - last_restart) < RESTART_COOLDOWN_SECONDS:
        return "skipped"

    clear_stale_orchestrator_lock()
    start_orchestrator_process()
    storage.update_agent_state(
        last_watchdog_restart_at=time.time(),
        watchdog_restart_count=int(state.get("watchdog_restart_count") or 0) + 1,
        message="Watchdog restarted discovery agent (stale heartbeat)",
        status="starting",
    )
    # Fingerprint for email alert dedup.
    try:
        from factory.agent_alerts import note_watchdog_restart
        note_watchdog_restart(storage)
    except Exception:
        pass
    return "restarted"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discovery agent heartbeat watchdog")
    parser.add_argument("--once", action="store_true",
                        help="Single check then exit (for Scheduled Tasks)")
    parser.add_argument("--interval", type=float, default=30.0,
                        help="Seconds between checks in loop mode")
    parser.add_argument("--stale", type=float, default=STALE_SECONDS,
                        help="Heartbeat age (seconds) that triggers a restart")
    args = parser.parse_args(argv)

    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.once:
        action = check_and_restart(stale_seconds=args.stale)
        print(action)
        return 0
    while True:
        action = check_and_restart(stale_seconds=args.stale)
        if action != "ok":
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} watchdog: {action}")
        time.sleep(max(5.0, float(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
