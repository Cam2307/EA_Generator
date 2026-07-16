"""Backfill ``results/{job_id}/`` archives from existing SQLite discovery jobs.

Rebuilds the standard results layout (config, levels, manifest, candidates,
summary, run.json) for finished jobs. Stage-1 screens are not available
historically and are noted in run.json.

Usage:
    python scripts/backfill_results_archive.py [--force] [--job-id ID]
        [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from factory import results_archive, validation_levels  # noqa: E402
from factory.models import JobStatus, StrategyDefinition, ValidationReport  # noqa: E402
from factory.storage import Storage  # noqa: E402

_FINISHED = {"DONE", "FAILED", "CANCELLED"}


def _status_str(job) -> str:
    st = getattr(job, "status", None)
    if st is None:
        return "UNKNOWN"
    return st.value if hasattr(st, "value") else str(st)


def _iter_job_validations(storage: Storage, job_id: str):
    """Yield (strategy, report, metadata) one row at a time (memory-friendly)."""
    with storage.connection() as con:
        rows = con.execute(
            "SELECT strategy_id FROM validations WHERE job_id=? "
            "ORDER BY highest_level_passed DESC, wfe DESC",
            (job_id,),
        ).fetchall()
    for row in rows:
        sid = row["strategy_id"]
        report = storage.get_validation(sid)
        strat = storage.get_strategy(sid)
        if report is None or strat is None:
            continue
        meta = storage.get_strategy_metadata(sid) or {}
        yield strat, report, meta


def backfill_job(
    storage: Storage,
    job,
    *,
    force: bool = False,
    root: Optional[Path] = None,
) -> bool:
    job_id = job.id
    d = results_archive.job_dir(job_id, root=root)
    run_path = d / "run.json"
    if run_path.is_file() and not force:
        existing = results_archive._read_json(run_path) or {}
        if existing.get("status") and existing.get("source") != "partial":
            return False

    payload = dict(getattr(job, "payload", None) or {})
    started = float(getattr(job, "created_at", None) or time.time())
    finished = float(getattr(job, "updated_at", None) or started)
    status = _status_str(job)

    print(f"  init {job_id} …", flush=True)
    manifest = storage.get_run_manifest(job_id)
    results_archive.init_run(
        job_id, payload, manifest=manifest,
        started_at=started, source="backfill", root=root,
    )

    n = 0
    for strat, report, meta in _iter_job_validations(storage, job_id):
        if not getattr(report, "levels_cleared", None):
            hi = int(getattr(report, "highest_level_passed", 0) or 0)
            report.levels_cleared = {
                str(lvl): (lvl <= hi)
                for lvl in range(
                    validation_levels.MIN_LEVEL,
                    validation_levels.MAX_LEVEL + 1,
                )
            }
        results_archive.write_candidate(
            job_id, strat, report, metadata=dict(meta), root=root)
        n += 1
        if n % 50 == 0:
            print(f"  … {n} candidates", flush=True)

    notes = [
        "Backfilled from SQLite; Stage-1 screens.jsonl unavailable historically.",
        "Per-candidate duration_ms may be missing for pre-timing runs.",
    ]
    results_archive.update_run(
        job_id,
        status=status,
        tested=int(getattr(job, "tested", 0) or 0),
        promising=int(getattr(job, "promising", 0) or 0),
        survivors=int(getattr(job, "survivors", 0) or 0),
        generation=int(getattr(job, "generation", 0) or 0),
        finished_at=finished,
        notes=notes,
        extra={
            "source": "backfill",
            "duration_s": round(max(0.0, finished - started), 3),
            "candidates_archived": n,
        },
        root=root,
    )
    results_archive.rebuild_summary(job_id, root=root)
    print(f"  wrote {n} candidates for {job_id}", flush=True)
    return True


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill results/ archives from factory.db discovery jobs")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing results/{job_id}/ archives")
    parser.add_argument("--job-id", default=None,
                        help="Backfill only this job id")
    parser.add_argument("--db", default=None,
                        help="Path to factory.db (default: config settings)")
    parser.add_argument("--all-statuses", action="store_true",
                        help="Include non-finished jobs")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max jobs to write (0 = no limit)")
    args = parser.parse_args(argv)

    settings.ensure_dirs()
    db_path = Path(args.db) if args.db else settings.DB_PATH
    print(f"db={db_path}", flush=True)
    storage = Storage(db_path)

    if args.job_id:
        job = storage.get_job(args.job_id)
        jobs = [job] if job is not None else []
    else:
        jobs = storage.list_jobs(kind="discovery")
        if not jobs:
            jobs = list(storage.list_jobs())

    written = 0
    skipped = 0
    for job in jobs:
        if job is None:
            continue
        st = _status_str(job)
        if not args.all_statuses and st not in _FINISHED:
            skipped += 1
            continue
        print(f"job {job.id} ({st})", flush=True)
        if backfill_job(storage, job, force=args.force):
            written += 1
            if args.limit and written >= args.limit:
                break
        else:
            skipped += 1
            print(f"  skip (already present; use --force)", flush=True)

    print(f"done — wrote {written}, skipped {skipped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
