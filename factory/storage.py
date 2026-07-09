"""SQLite persistence layer.

Rules (see plan):
- WAL mode + busy_timeout on every connection.
- No shared module-level connection: every access opens a short-lived,
  context-managed connection, so worker threads and the Streamlit process
  never share sqlite objects across threads.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

from config import settings
from factory.models import Job, JobStatus, StrategyDefinition, ValidationReport

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id TEXT PRIMARY KEY,
    name TEXT,
    symbol TEXT,
    timeframe TEXT,
    created_at REAL,
    body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS validations (
    strategy_id TEXT PRIMARY KEY,
    passed INTEGER NOT NULL DEFAULT 0,
    wfe REAL,
    body TEXT NOT NULL,
    updated_at REAL,
    job_id TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT,
    status TEXT,
    progress REAL,
    message TEXT,
    error TEXT,
    payload TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at REAL,
    updated_at REAL,
    tested INTEGER NOT NULL DEFAULT 0,
    promising INTEGER NOT NULL DEFAULT 0,
    survivors INTEGER NOT NULL DEFAULT 0,
    generation INTEGER NOT NULL DEFAULT 0
);
"""

# Columns added after the original jobs table shipped; carried live during a
# run so the UI can show determinate progress. ALTER TABLE keeps pre-existing
# databases readable/writable without a manual migration.
_JOB_COUNTER_COLUMNS = ("tested", "promising", "survivors", "generation")

# Columns added to `validations` after it first shipped. `job_id` links every
# result back to the discovery run that produced it, so the UI can show the
# results of a single run.
_VALIDATION_COLUMNS = (("job_id", "TEXT"),)


class Storage:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else settings.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as con:
            con.executescript(_SCHEMA)
            self._migrate_job_counters(con)
            self._migrate_validation_columns(con)

    @staticmethod
    def _migrate_job_counters(con: sqlite3.Connection) -> None:
        """Add live-counter columns to a jobs table created before they existed."""
        existing = {r["name"] for r in con.execute("PRAGMA table_info(jobs)")}
        for col in _JOB_COUNTER_COLUMNS:
            if col not in existing:
                con.execute(
                    f"ALTER TABLE jobs ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")

    @staticmethod
    def _migrate_validation_columns(con: sqlite3.Connection) -> None:
        """Add columns to a validations table created before they existed."""
        existing = {r["name"] for r in con.execute("PRAGMA table_info(validations)")}
        for col, decl in _VALIDATION_COLUMNS:
            if col not in existing:
                con.execute(f"ALTER TABLE validations ADD COLUMN {col} {decl}")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(str(self.db_path), timeout=10)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA busy_timeout=5000;")
            con.row_factory = sqlite3.Row
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------
    def save_strategy(self, s: StrategyDefinition) -> None:
        with self.connection() as con:
            con.execute(
                "INSERT OR REPLACE INTO strategies (id, name, symbol, timeframe, created_at, body)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (s.id, s.name, s.symbol, s.timeframe, s.created_at, s.model_dump_json()),
            )

    def save_complete(self, strategy: StrategyDefinition,
                      report: ValidationReport,
                      job_id: Optional[str] = None) -> None:
        """Atomically persist a strategy and its validation report.

        Every candidate that finishes (or aborts) the validation pipeline
        should go through this method so a crash between the two writes
        cannot leave orphan rows. ``job_id`` links the result to the run that
        produced it so the UI can show a single run's results.
        """
        body_s = strategy.model_dump_json()
        body_v = report.model_dump_json()
        now = time.time()
        with self.connection() as con:
            con.execute(
                "INSERT OR REPLACE INTO strategies (id, name, symbol, timeframe, created_at, body)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (strategy.id, strategy.name, strategy.symbol, strategy.timeframe,
                 strategy.created_at, body_s),
            )
            con.execute(
                "INSERT OR REPLACE INTO validations (strategy_id, passed, wfe, body, updated_at, job_id)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (report.strategy_id, int(report.passed), report.wfe, body_v, now,
                 job_id),
            )

    def get_strategy(self, strategy_id: str) -> Optional[StrategyDefinition]:
        with self.connection() as con:
            row = con.execute("SELECT body FROM strategies WHERE id=?", (strategy_id,)).fetchone()
        return StrategyDefinition.model_validate_json(row["body"]) if row else None

    def list_strategies(self) -> List[StrategyDefinition]:
        with self.connection() as con:
            rows = con.execute("SELECT body FROM strategies ORDER BY created_at DESC").fetchall()
        return [StrategyDefinition.model_validate_json(r["body"]) for r in rows]

    # ------------------------------------------------------------------
    # Validation reports
    # ------------------------------------------------------------------
    def save_validation(self, report: ValidationReport) -> None:
        with self.connection() as con:
            con.execute(
                "INSERT OR REPLACE INTO validations (strategy_id, passed, wfe, body, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (report.strategy_id, int(report.passed), report.wfe,
                 report.model_dump_json(), time.time()),
            )

    def get_validation(self, strategy_id: str) -> Optional[ValidationReport]:
        with self.connection() as con:
            row = con.execute(
                "SELECT body, job_id FROM validations WHERE strategy_id=?",
                (strategy_id,)
            ).fetchone()
        return self._row_to_report(row) if row else None

    def list_validated(self, passed_only: bool = True,
                       job_id: Optional[str] = None) -> List[ValidationReport]:
        q = "SELECT body, job_id FROM validations"
        clauses, args = [], []
        if passed_only:
            clauses.append("passed=1")
        if job_id is not None:
            clauses.append("job_id=?")
            args.append(job_id)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY wfe DESC"
        with self.connection() as con:
            rows = con.execute(q, tuple(args)).fetchall()
        return [self._row_to_report(r) for r in rows]

    @staticmethod
    def _row_to_report(row: sqlite3.Row) -> ValidationReport:
        """Deserialize a validations row, stamping the originating run id.

        ``run_id`` lives in the table's ``job_id`` column (not the JSON body),
        so it is applied here on load and stays authoritative even for results
        saved before the field existed on the model.
        """
        report = ValidationReport.model_validate_json(row["body"])
        keys = row.keys()
        report.run_id = row["job_id"] if "job_id" in keys else None
        return report

    def count_validated(self, job_id: str) -> tuple[int, int]:
        """Return ``(passed, total)`` result counts for a single run."""
        with self.connection() as con:
            row = con.execute(
                "SELECT COALESCE(SUM(passed), 0) AS passed, COUNT(*) AS total"
                " FROM validations WHERE job_id=?", (job_id,)).fetchone()
        return (int(row["passed"]), int(row["total"])) if row else (0, 0)

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------
    def upsert_job(self, job: Job) -> None:
        job.updated_at = time.time()
        with self.connection() as con:
            con.execute(
                "INSERT OR REPLACE INTO jobs"
                " (id, kind, status, progress, message, error, payload,"
                "  cancel_requested, created_at, updated_at,"
                "  tested, promising, survivors, generation)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (job.id, job.kind, job.status.value, job.progress, job.message,
                 job.error, json.dumps(job.payload), int(job.cancel_requested),
                 job.created_at, job.updated_at,
                 int(job.tested), int(job.promising), int(job.survivors),
                 int(job.generation)),
            )

    def get_job(self, job_id: str) -> Optional[Job]:
        with self.connection() as con:
            row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self, kind: Optional[str] = None) -> List[Job]:
        q, args = "SELECT * FROM jobs", ()
        if kind:
            q, args = q + " WHERE kind=?", (kind,)
        q += " ORDER BY created_at DESC"
        with self.connection() as con:
            rows = con.execute(q, args).fetchall()
        return [self._row_to_job(r) for r in rows]

    def update_job_progress(self, job_id: str, progress: float, message: str = "",
                            tested: Optional[int] = None,
                            promising: Optional[int] = None,
                            survivors: Optional[int] = None,
                            generation: Optional[int] = None) -> None:
        """Persist the progress bar fraction/message plus optional live counters.

        Counters left as ``None`` are not written, so callers that only have a
        fraction (legacy behaviour) keep the previously stored counts.
        """
        sets = ["progress=?", "message=?", "updated_at=?"]
        args: list = [progress, message, time.time()]
        for name, value in (("tested", tested), ("promising", promising),
                            ("survivors", survivors), ("generation", generation)):
            if value is not None:
                sets.append(f"{name}=?")
                args.append(int(value))
        args.append(job_id)
        with self.connection() as con:
            con.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", tuple(args))

    def set_job_status(self, job_id: str, status: JobStatus,
                       error: Optional[str] = None, message: str = "") -> None:
        with self.connection() as con:
            con.execute(
                "UPDATE jobs SET status=?, error=COALESCE(?, error),"
                " message=CASE WHEN ?='' THEN message ELSE ? END, updated_at=?"
                " WHERE id=?",
                (status.value, error, message, message, time.time(), job_id),
            )

    def request_cancel(self, job_id: str) -> None:
        with self.connection() as con:
            con.execute(
                "UPDATE jobs SET cancel_requested=1, updated_at=? WHERE id=?",
                (time.time(), job_id),
            )

    def is_cancel_requested(self, job_id: str) -> bool:
        with self.connection() as con:
            row = con.execute(
                "SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        keys = row.keys()
        return Job(
            id=row["id"], kind=row["kind"], status=JobStatus(row["status"]),
            progress=row["progress"], message=row["message"] or "",
            error=row["error"], payload=json.loads(row["payload"] or "{}"),
            cancel_requested=bool(row["cancel_requested"]),
            created_at=row["created_at"], updated_at=row["updated_at"],
            tested=row["tested"] if "tested" in keys else 0,
            promising=row["promising"] if "promising" in keys else 0,
            survivors=row["survivors"] if "survivors" in keys else 0,
            generation=row["generation"] if "generation" in keys else 0,
        )
