"""Entry point for the detached discovery orchestrator service."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _fail(message: str, exc: BaseException | None = None) -> None:
    from factory.storage import Storage
    from jobs.orchestrator import ERROR_LOG_PATH, clear_stale_orchestrator_lock

    if exc is not None:
        ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ERROR_LOG_PATH.write_text(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            encoding="utf-8",
        )
    Storage().update_agent_state(
        enabled=0,
        status="stopped",
        pid=None,
        message=message,
    )
    clear_stale_orchestrator_lock()


if __name__ == "__main__":
    try:
        from jobs.orchestrator import (  # noqa: E402
            run_orchestrator_forever,
            sync_agent_with_orchestrator_lock,
        )

        run_orchestrator_forever()
    except FileExistsError:
        from jobs.orchestrator import sync_agent_with_orchestrator_lock  # noqa: E402

        if sync_agent_with_orchestrator_lock():
            sys.exit(0)
        _fail("Discovery agent is already running.")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - surface startup failures in the UI
        _fail(f"Discovery agent crashed on startup: {type(exc).__name__}: {exc}", exc)
        sys.exit(1)
