"""Filesystem archive of discovery run results.

Every discovery job writes a self-contained folder under ``results/{job_id}/``
with the exact config, validation-level schema, Stage-1 screens, Stage-2
candidates, timings, and a flat summary CSV for review / optimization.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config import settings
from factory import validation_levels
from factory.manifest import _RESULT_AFFECTING_SETTINGS

_write_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _job_lock(job_id: str) -> threading.Lock:
    with _locks_guard:
        lock = _write_locks.get(job_id)
        if lock is None:
            lock = threading.Lock()
            _write_locks[job_id] = lock
        return lock


def job_dir(job_id: str, *, root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else Path(settings.RESULTS_DIR)
    return base / str(job_id)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, default=str, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def settings_snapshot() -> dict:
    return {name: getattr(settings, name, None)
            for name in _RESULT_AFFECTING_SETTINGS}


def init_run(
    job_id: str,
    payload: dict,
    *,
    manifest: Optional[dict] = None,
    started_at: Optional[float] = None,
    source: str = "live",
    root: Optional[Path] = None,
) -> Path:
    """Create ``results/{job_id}/`` and write config / levels / initial run.json."""
    d = job_dir(job_id, root=root)
    d.mkdir(parents=True, exist_ok=True)
    (d / "candidates").mkdir(exist_ok=True)
    started = float(started_at if started_at is not None else time.time())

    _atomic_write_json(d / "config.json", {
        "job_id": job_id,
        "payload": dict(payload or {}),
        "settings": settings_snapshot(),
        "created_at": started,
    })
    _atomic_write_json(d / "levels.json", validation_levels.levels_snapshot())
    if manifest is not None:
        _atomic_write_json(d / "manifest.json", manifest)

    run = {
        "job_id": job_id,
        "status": "RUNNING",
        "source": source,
        "started_at": started,
        "finished_at": None,
        "duration_s": None,
        "tested": 0,
        "promising": 0,
        "survivors": 0,
        "generation": 0,
        "screen_ms_total": 0.0,
        "validate_ms_total": 0.0,
        "level_schema_version": validation_levels.LEVEL_SCHEMA_VERSION,
        "notes": [],
    }
    _atomic_write_json(d / "run.json", run)
    return d


def append_screen(
    job_id: str,
    record: dict,
    *,
    root: Optional[Path] = None,
) -> None:
    """Append one Stage-1 screen outcome to ``screens.jsonl``."""
    d = job_dir(job_id, root=root)
    d.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str, ensure_ascii=False)
    with _job_lock(job_id):
        with (d / "screens.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def write_candidate(
    job_id: str,
    strategy,
    report,
    *,
    metadata: Optional[dict] = None,
    root: Optional[Path] = None,
) -> Path:
    """Write one Stage-2 candidate JSON (pass or fail)."""
    d = job_dir(job_id, root=root)
    cand_dir = d / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    sid = getattr(strategy, "id", None) or getattr(report, "strategy_id", "unknown")
    strat_body = (
        strategy.model_dump() if hasattr(strategy, "model_dump")
        else dict(strategy) if isinstance(strategy, dict) else {"id": sid}
    )
    report_body = (
        report.model_dump() if hasattr(report, "model_dump")
        else dict(report) if isinstance(report, dict) else {}
    )
    # Ensure levels_cleared is populated when missing (backfill / legacy).
    if not report_body.get("levels_cleared") and report_body.get(
            "highest_level_passed") is not None:
        hi = int(report_body.get("highest_level_passed") or 0)
        report_body["levels_cleared"] = {
            str(lvl): (lvl <= hi)
            for lvl in range(
                validation_levels.MIN_LEVEL, validation_levels.MAX_LEVEL + 1)
        }
    payload = {
        "strategy_id": sid,
        "job_id": job_id,
        "strategy": strat_body,
        "report": report_body,
        "metadata": dict(metadata or {}),
        "archived_at": time.time(),
    }
    path = cand_dir / f"{sid}.json"
    _atomic_write_json(path, payload)
    return path


def _candidate_summary_row(path: Path) -> Optional[dict]:
    data = _read_json(path)
    if not data:
        return None
    report = data.get("report") or {}
    oos = report.get("oos_metrics") or {}
    meta = data.get("metadata") or {}
    reasons = list(report.get("reasons") or [])
    levels = report.get("levels_cleared") or {}
    cleared = sorted(
        int(k) for k, v in levels.items() if v and str(k).isdigit())
    return {
        "strategy_id": data.get("strategy_id"),
        "passed": report.get("passed"),
        "highest_level_passed": report.get("highest_level_passed"),
        "wfe": report.get("wfe"),
        "oos_net_profit": oos.get("net_profit"),
        "oos_max_dd_pct": oos.get("max_dd_pct"),
        "oos_trade_count": oos.get("trade_count"),
        "oos_profit_factor": oos.get("profit_factor"),
        "oos_sharpe": oos.get("sharpe"),
        "oos_r_squared": oos.get("r_squared"),
        "quality_score": report.get("quality_score"),
        "duration_ms": report.get("duration_ms"),
        "engine": report.get("engine"),
        "generation": meta.get("generation"),
        "operation": meta.get("operation"),
        "primary_reason": (reasons[0] if reasons else ""),
        "levels_cleared": ",".join(str(x) for x in cleared),
    }


def rebuild_summary(job_id: str, *, root: Optional[Path] = None) -> Path:
    """Rebuild ``summary.csv`` from ``candidates/*.json``."""
    d = job_dir(job_id, root=root)
    cand_dir = d / "candidates"
    rows: List[dict] = []
    if cand_dir.is_dir():
        for path in sorted(cand_dir.glob("*.json")):
            row = _candidate_summary_row(path)
            if row:
                rows.append(row)
    path = d / "summary.csv"
    fieldnames = [
        "strategy_id", "passed", "highest_level_passed", "wfe",
        "oos_net_profit", "oos_max_dd_pct", "oos_trade_count",
        "oos_profit_factor", "oos_sharpe", "oos_r_squared", "quality_score",
        "duration_ms", "engine", "generation", "operation",
        "primary_reason", "levels_cleared",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def update_run(
    job_id: str,
    *,
    status: Optional[str] = None,
    tested: Optional[int] = None,
    promising: Optional[int] = None,
    survivors: Optional[int] = None,
    generation: Optional[int] = None,
    screen_ms_total: Optional[float] = None,
    validate_ms_total: Optional[float] = None,
    finished_at: Optional[float] = None,
    notes: Optional[Iterable[str]] = None,
    extra: Optional[dict] = None,
    root: Optional[Path] = None,
) -> dict:
    """Merge fields into ``run.json`` (creates a stub if missing)."""
    d = job_dir(job_id, root=root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "run.json"
    with _job_lock(job_id):
        run = _read_json(path) or {
            "job_id": job_id,
            "status": "RUNNING",
            "source": "live",
            "started_at": time.time(),
            "finished_at": None,
            "duration_s": None,
            "tested": 0,
            "promising": 0,
            "survivors": 0,
            "generation": 0,
            "screen_ms_total": 0.0,
            "validate_ms_total": 0.0,
            "level_schema_version": validation_levels.LEVEL_SCHEMA_VERSION,
            "notes": [],
        }
        if status is not None:
            run["status"] = status
        if tested is not None:
            run["tested"] = int(tested)
        if promising is not None:
            run["promising"] = int(promising)
        if survivors is not None:
            run["survivors"] = int(survivors)
        if generation is not None:
            run["generation"] = int(generation)
        if screen_ms_total is not None:
            run["screen_ms_total"] = float(screen_ms_total)
        if validate_ms_total is not None:
            run["validate_ms_total"] = float(validate_ms_total)
        if finished_at is not None:
            run["finished_at"] = float(finished_at)
            started = float(run.get("started_at") or finished_at)
            run["duration_s"] = round(float(finished_at) - started, 3)
        if notes is not None:
            existing = list(run.get("notes") or [])
            for note in notes:
                if note and note not in existing:
                    existing.append(note)
            run["notes"] = existing
        if extra:
            run.update(extra)
        _atomic_write_json(path, run)
        return run


def finalize_run(
    job_id: str,
    *,
    status: str,
    tested: int = 0,
    promising: int = 0,
    survivors: int = 0,
    generation: int = 0,
    screen_ms_total: float = 0.0,
    validate_ms_total: float = 0.0,
    notes: Optional[List[str]] = None,
    root: Optional[Path] = None,
) -> dict:
    """Mark the run finished and rebuild summary.csv."""
    d = job_dir(job_id, root=root)
    # Prefer live totals; fall back to summing archived records.
    if screen_ms_total <= 0 and (d / "screens.jsonl").is_file():
        try:
            total = 0.0
            with (d / "screens.jsonl").open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    total += float(json.loads(line).get("duration_ms") or 0.0)
            screen_ms_total = total
        except Exception:
            pass
    if validate_ms_total <= 0 and (d / "candidates").is_dir():
        try:
            total = 0.0
            for path in (d / "candidates").glob("*.json"):
                data = _read_json(path) or {}
                total += float((data.get("report") or {}).get("duration_ms") or 0.0)
            validate_ms_total = total
        except Exception:
            pass
    finished = time.time()
    run = update_run(
        job_id,
        status=status,
        tested=tested,
        promising=promising,
        survivors=survivors,
        generation=generation,
        screen_ms_total=screen_ms_total,
        validate_ms_total=validate_ms_total,
        finished_at=finished,
        notes=notes,
        root=root,
    )
    try:
        rebuild_summary(job_id, root=root)
    except Exception:
        pass
    return run


def metrics_brief(m) -> dict:
    """Compact metrics dict suitable for screens.jsonl."""
    if m is None:
        return {}
    if hasattr(m, "model_dump"):
        raw = m.model_dump()
    elif isinstance(m, dict):
        raw = m
    else:
        raw = {}
    keys = (
        "net_profit", "max_dd_pct", "trade_count", "profit_factor",
        "sharpe", "r_squared", "win_rate", "initial_deposit",
    )
    return {k: raw.get(k) for k in keys if k in raw}


def screen_record(
    strategy,
    *,
    promising: bool,
    metrics=None,
    duration_ms: float = 0.0,
    fitness: float = 0.0,
    error: Optional[str] = None,
    generation: Optional[int] = None,
) -> dict:
    sid = getattr(strategy, "id", None) or "unknown"
    return {
        "strategy_id": sid,
        "promising": bool(promising),
        "duration_ms": float(duration_ms),
        "fitness": float(fitness),
        "metrics": metrics_brief(metrics),
        "error": error,
        "generation": generation,
        "ts": time.time(),
    }
