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
    job_id TEXT,
    promotion_state TEXT,
    quality_score REAL,
    hard_gates_passed INTEGER NOT NULL DEFAULT 0,
    quality_breakdown TEXT,
    last_alert_at REAL,
    alert_fingerprint TEXT
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
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS agent_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    enabled INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'stopped',
    pid INTEGER,
    heartbeat_at REAL,
    queue_depth INTEGER NOT NULL DEFAULT 0,
    jobs_submitted INTEGER NOT NULL DEFAULT 0,
    cursor INTEGER NOT NULL DEFAULT 0,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS strategy_metadata (
    strategy_id TEXT PRIMARY KEY,
    sweep_symbol TEXT,
    sweep_timeframe TEXT,
    strictness_profile TEXT,
    seed INTEGER,
    parameter_snapshot TEXT,
    parent_id TEXT,
    generation INTEGER,
    updated_at REAL
);
"""

# Columns added after the original jobs table shipped; carried live during a
# run so the UI can show determinate progress. ALTER TABLE keeps pre-existing
# databases readable/writable without a manual migration.
_JOB_COUNTER_COLUMNS = ("tested", "promising", "survivors", "generation")

# Columns added to `validations` after it first shipped. `job_id` links every
# result back to the discovery run that produced it, so the UI can show the
# results of a single run.
_VALIDATION_COLUMNS = (
    ("job_id", "TEXT"),
    ("promotion_state", "TEXT"),
    ("quality_score", "REAL"),
    ("hard_gates_passed", "INTEGER NOT NULL DEFAULT 0"),
    ("quality_breakdown", "TEXT"),
    ("last_alert_at", "REAL"),
    ("alert_fingerprint", "TEXT"),
)


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
                      job_id: Optional[str] = None,
                      metadata: Optional[dict] = None) -> None:
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
            if metadata:
                con.execute(
                    "INSERT OR REPLACE INTO strategy_metadata "
                    "(strategy_id, sweep_symbol, sweep_timeframe, strictness_profile, "
                    " seed, parameter_snapshot, parent_id, generation, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        strategy.id,
                        metadata.get("sweep_symbol"),
                        metadata.get("sweep_timeframe"),
                        metadata.get("strictness_profile"),
                        metadata.get("seed"),
                        json.dumps(metadata.get("parameter_snapshot", {})),
                        metadata.get("parent_id"),
                        metadata.get("generation"),
                        now,
                    ),
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
        q = (
            "SELECT body, job_id, promotion_state, quality_score, hard_gates_passed, "
            "quality_breakdown, last_alert_at, alert_fingerprint FROM validations"
        )
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
        if "promotion_state" in keys and row["promotion_state"]:
            report.promotion_state = row["promotion_state"]
        if "quality_score" in keys and row["quality_score"] is not None:
            report.quality_score = float(row["quality_score"])
        if "hard_gates_passed" in keys:
            report.hard_gates_passed = bool(row["hard_gates_passed"])
        if "quality_breakdown" in keys and row["quality_breakdown"]:
            try:
                report.quality_breakdown = json.loads(row["quality_breakdown"])
            except json.JSONDecodeError:
                report.quality_breakdown = {}
        return report

    def get_strategy_metadata(self, strategy_id: str) -> Optional[dict]:
        with self.connection() as con:
            row = con.execute(
                "SELECT * FROM strategy_metadata WHERE strategy_id=?",
                (strategy_id,),
            ).fetchone()
        if not row:
            return None
        out = dict(row)
        if out.get("parameter_snapshot"):
            try:
                out["parameter_snapshot"] = json.loads(out["parameter_snapshot"])
            except json.JSONDecodeError:
                out["parameter_snapshot"] = {}
        return out

    def update_validation_promotion(
        self,
        strategy_id: str,
        *,
        promotion_state: str,
        quality_score: float,
        hard_gates_passed: bool,
        quality_breakdown: Optional[dict] = None,
    ) -> None:
        with self.connection() as con:
            con.execute(
                "UPDATE validations SET promotion_state=?, quality_score=?, "
                "hard_gates_passed=?, quality_breakdown=?, updated_at=? "
                "WHERE strategy_id=?",
                (
                    promotion_state,
                    float(quality_score),
                    int(bool(hard_gates_passed)),
                    json.dumps(quality_breakdown or {}),
                    time.time(),
                    strategy_id,
                ),
            )

    def mark_alert_sent(self, strategy_id: str, fingerprint: str) -> None:
        with self.connection() as con:
            con.execute(
                "UPDATE validations SET last_alert_at=?, alert_fingerprint=? WHERE strategy_id=?",
                (time.time(), fingerprint, strategy_id),
            )

    def get_alert_state(self, strategy_id: str) -> dict:
        with self.connection() as con:
            row = con.execute(
                "SELECT last_alert_at, alert_fingerprint FROM validations WHERE strategy_id=?",
                (strategy_id,),
            ).fetchone()
        if not row:
            return {"last_alert_at": None, "alert_fingerprint": None}
        return {
            "last_alert_at": row["last_alert_at"],
            "alert_fingerprint": row["alert_fingerprint"],
        }

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

    # ------------------------------------------------------------------
    # App settings + discovery agent state
    # ------------------------------------------------------------------
    def get_app_settings(self) -> dict:
        with self.connection() as con:
            rows = con.execute("SELECT key, value FROM app_settings").fetchall()
        out: dict = {}
        for row in rows:
            value = row["value"]
            try:
                out[row["key"]] = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                out[row["key"]] = value
        return out

    def upsert_app_settings(self, values: dict) -> None:
        now = time.time()
        with self.connection() as con:
            for key, value in values.items():
                con.execute(
                    "INSERT OR REPLACE INTO app_settings (key, value, updated_at) VALUES (?, ?, ?)",
                    (key, json.dumps(value), now),
                )

    def get_agent_state(self) -> dict:
        with self.connection() as con:
            self._ensure_agent_state_row(con)
            row = con.execute("SELECT * FROM agent_state WHERE id=1").fetchone()
        return dict(row) if row else {}

    def update_agent_state(self, **fields) -> None:
        if not fields:
            return
        fields["updated_at"] = time.time()
        sets = ", ".join(f"{k}=?" for k in fields)
        args = list(fields.values()) + [1]
        with self.connection() as con:
            self._ensure_agent_state_row(con)
            con.execute(f"UPDATE agent_state SET {sets} WHERE id=?", args)

    def upsert_agent_state(self, **fields) -> None:
        """Compatibility alias for callers that expect an upsert method."""
        self.update_agent_state(**fields)

    def set_agent_state(self, **fields) -> None:
        """Compatibility alias for legacy callers."""
        self.update_agent_state(**fields)

    @staticmethod
    def _ensure_agent_state_row(con: sqlite3.Connection) -> None:
        row = con.execute("SELECT id FROM agent_state WHERE id=1").fetchone()
        if row:
            return
        con.execute(
            "INSERT INTO agent_state (id, updated_at) VALUES (1, ?)",
            (time.time(),),
        )
