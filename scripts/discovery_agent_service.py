"""Entry point for the detached discovery orchestrator service."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jobs.orchestrator import run_orchestrator_forever  # noqa: E402


if __name__ == "__main__":
    run_orchestrator_forever()
