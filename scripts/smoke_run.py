"""End-to-end smoke run with the simulated engine.

Generates a batch of strategies, backtests + validates them through the job
worker (SQLite-persisted progress), then exports one Marketplace Package.
Also attempts MT5 terminal auto-detection and reports the outcome without
failing when MT5 is absent.

Usage:  python scripts/smoke_run.py
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from factory.models import JobStatus
from factory.storage import Storage
from jobs.worker import JobQueue


def main() -> int:
    settings.ensure_dirs()
    db_path = Path(tempfile.mkdtemp(prefix="eaf_smoke_")) / "smoke.db"
    storage = Storage(db_path)
    queue = JobQueue(storage)

    print(f"[smoke] SQLite at {db_path} (WAL mode)")

    # 1) MT5 detection (informational — never fatal)
    try:
        from factory.backtest.mt5_runner import detect_mt5
        paths = detect_mt5()
        print(f"[smoke] MT5 detected: terminal={paths.terminal_exe}")
        mt5_status = f"detected at {paths.terminal_exe}"
    except Exception as exc:
        print(f"[smoke] MT5 not available (expected on machines without MT5): {exc}")
        mt5_status = f"not available: {exc}"

    # 2) discovery batch through the worker
    job_id = "smoke_job_001"
    submitted = queue.submit_discovery(job_id, {
        "symbol": "EURUSD", "timeframe": "H1",
        "batch_size": 6, "genetic_rounds": 1,
        "engine": "simulator", "seed": 20260708,
        "start": "2023-01-01T00:00:00+00:00",
        "end": "2024-07-01T00:00:00+00:00",
    })
    assert submitted, "job submission failed"
    # idempotency check: duplicate submit must be dropped
    assert not queue.submit_discovery(job_id, {}), "duplicate submit was accepted!"
    print("[smoke] duplicate submission correctly ignored")

    deadline = time.time() + 600
    while time.time() < deadline:
        job = storage.get_job(job_id)
        if job.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
            break
        print(f"[smoke] {job.status.value} {job.progress:.0%} {job.message}")
        time.sleep(2)

    job = storage.get_job(job_id)
    print(f"[smoke] final job status: {job.status.value} — {job.message}")
    if job.status != JobStatus.DONE:
        print(f"[smoke] FAILED: {job.error}")
        return 1

    # 3) inspect results
    strategies = storage.list_strategies()
    all_reports = storage.list_validated(passed_only=False)
    passed = storage.list_validated(passed_only=True)
    print(f"[smoke] strategies: {len(strategies)}, validated: {len(all_reports)}, "
          f"passed gates: {len(passed)}")
    assert all_reports, "no validation reports were produced"

    # 4) export one Marketplace Package (best by WFE, pass or not)
    from factory.assets.exporter import export_marketplace_package
    best = max(all_reports, key=lambda r: r.wfe)
    strategy = storage.get_strategy(best.strategy_id)
    out_dir = export_marketplace_package(strategy, best)
    files = sorted(p.name for p in out_dir.iterdir())
    print(f"[smoke] exported package -> {out_dir}")
    for f in files:
        print(f"[smoke]   {f}")
    assert any(f.endswith(".mq5") for f in files)
    assert any(f.endswith(".set") for f in files)
    assert any(f.endswith(".md") for f in files)

    # the .set must expose optimizable mechanic parameters
    set_file = next(out_dir.glob("*.set"))
    set_text = set_file.read_text(encoding="utf-16")
    assert "||Y" in set_text, ".set carries no optimization ranges"

    print(f"[smoke] MT5 status: {mt5_status}")
    print("[smoke] SMOKE RUN PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
