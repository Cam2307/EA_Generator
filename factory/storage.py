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
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

from config import settings
from factory.models import Job, JobStatus, StrategyDefinition, ValidationReport


@dataclass
class MetricsSummary:
    """Scalar backtest metrics without equity curves (list-view safe)."""

    net_profit: float = 0.0
    profit_factor: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_dd_pct: float = 0.0
    r_squared: float = 0.0
    trade_count: int = 0
    equity: List[float] = field(default_factory=list)
    equity_ts: List[float] = field(default_factory=list)


@dataclass
class MonteCarloSummary:
    robustness_score: float = 0.0


@dataclass
class ValidationSummary:
    """Lightweight validation row for gallery/export chrome (no equity JSON)."""

    strategy_id: str
    run_id: Optional[str] = None
    passed: bool = False
    wfe: float = 0.0
    engine: str = "simulator"
    data_source: str = "unknown"
    degradation_pct: float = 0.0
    stability_ratio: float = 1.0
    promotion_state: str = "candidate"
    quality_score: float = 0.0
    hard_gates_passed: bool = False
    reasons: List[str] = field(default_factory=list)
    oos_metrics: MetricsSummary = field(default_factory=MetricsSummary)
    is_metrics: MetricsSummary = field(default_factory=MetricsSummary)
    montecarlo: Optional[MonteCarloSummary] = None


@dataclass
class StrategySummary:
    """Strategy index row without deserializing the full definition body."""

    id: str
    name: str = ""
    symbol: str = ""
    timeframe: str = ""
    created_at: float = 0.0

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
    message TEXT,
    current_job_id TEXT,
    sweep_total INTEGER NOT NULL DEFAULT 0,
    last_progress_email_at REAL,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS run_manifests (
    job_id TEXT PRIMARY KEY,
    seed INTEGER,
    data_sha256 TEXT,
    body TEXT NOT NULL,
    created_at REAL
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
_JOB_EXTRA_COLUMNS = (("runner_pid", "INTEGER"),)

_AGENT_STATE_COLUMNS = (
    ("message", "TEXT"),
    ("current_job_id", "TEXT"),
    ("sweep_total", "INTEGER NOT NULL DEFAULT 0"),
    ("last_progress_email_at", "REAL"),
    ("mode", "TEXT NOT NULL DEFAULT 'continuous'"),
    ("spawn_attempts", "INTEGER NOT NULL DEFAULT 0"),
    ("last_promotion_sync_at", "REAL"),
)

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

# Indexes for gallery / per-run lookups. job_id is filtered on every
# Results-per-run open; without it each COUNT/list scans the full body table.
_VALIDATION_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_validations_job_id ON validations(job_id)",
)


class Storage:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else settings.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection():
            pass  # bootstrap tables + incremental migrations

    def _ensure_schema(self, con: sqlite3.Connection) -> None:
        """Apply idempotent schema upgrades on every connection.

        Migrations must not live only in ``__init__``: Streamlit's
        ``cache_resource`` can keep a Storage instance across hot reloads, and
        other long-lived processes may open the DB before new columns ship.
        """
        con.executescript(_SCHEMA)
        self._migrate_job_counters(con)
        self._migrate_job_extra_columns(con)
        self._migrate_validation_columns(con)
        self._migrate_validation_indexes(con)
        self._migrate_agent_state_columns(con)

    @staticmethod
    def _migrate_job_counters(con: sqlite3.Connection) -> None:
        """Add live-counter columns to a jobs table created before they existed."""
        if not Storage._table_exists(con, "jobs"):
            return
        existing = {r["name"] for r in con.execute("PRAGMA table_info(jobs)")}
        for col in _JOB_COUNTER_COLUMNS:
            if col not in existing:
                con.execute(
                    f"ALTER TABLE jobs ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")

    @staticmethod
    def _migrate_job_extra_columns(con: sqlite3.Connection) -> None:
        if not Storage._table_exists(con, "jobs"):
            return
        existing = {r["name"] for r in con.execute("PRAGMA table_info(jobs)")}
        for col, decl in _JOB_EXTRA_COLUMNS:
            if col not in existing:
                con.execute(f"ALTER TABLE jobs ADD COLUMN {col} {decl}")

    @staticmethod
    def _migrate_validation_columns(con: sqlite3.Connection) -> None:
        """Add columns to a validations table created before they existed."""
        if not Storage._table_exists(con, "validations"):
            return
        existing = {r["name"] for r in con.execute("PRAGMA table_info(validations)")}
        for col, decl in _VALIDATION_COLUMNS:
            if col not in existing:
                con.execute(f"ALTER TABLE validations ADD COLUMN {col} {decl}")

    @staticmethod
    def _migrate_validation_indexes(con: sqlite3.Connection) -> None:
        """Ensure lookup indexes exist (idempotent)."""
        if not Storage._table_exists(con, "validations"):
            return
        for ddl in _VALIDATION_INDEXES:
            con.execute(ddl)

    @staticmethod
    def _migrate_agent_state_columns(con: sqlite3.Connection) -> None:
        """Add live-status columns to agent_state created before they existed."""
        if not Storage._table_exists(con, "agent_state"):
            return
        existing = {r["name"] for r in con.execute("PRAGMA table_info(agent_state)")}
        for col, decl in _AGENT_STATE_COLUMNS:
            if col not in existing:
                con.execute(f"ALTER TABLE agent_state ADD COLUMN {col} {decl}")

    @staticmethod
    def _table_exists(con: sqlite3.Connection, name: str) -> bool:
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row is not None

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(str(self.db_path), timeout=10)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA busy_timeout=5000;")
            con.row_factory = sqlite3.Row
            self._ensure_schema(con)
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

    def list_strategy_summaries(self) -> List[StrategySummary]:
        """Index rows only — avoids deserializing full strategy JSON bodies."""
        with self.connection() as con:
            rows = con.execute(
                "SELECT id, name, symbol, timeframe, created_at "
                "FROM strategies ORDER BY created_at DESC"
            ).fetchall()
        return [
            StrategySummary(
                id=r["id"],
                name=r["name"] or "",
                symbol=r["symbol"] or "",
                timeframe=r["timeframe"] or "",
                created_at=float(r["created_at"] or 0),
            )
            for r in rows
        ]

    def count_strategies(self) -> int:
        with self.connection() as con:
            row = con.execute("SELECT COUNT(*) AS n FROM strategies").fetchone()
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------
    # Run manifests (reproducibility)
    # ------------------------------------------------------------------
    def save_run_manifest(self, manifest: dict) -> None:
        """Persist a discovery-run reproducibility manifest (see factory.manifest)."""
        with self.connection() as con:
            con.execute(
                "INSERT OR REPLACE INTO run_manifests"
                " (job_id, seed, data_sha256, body, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (manifest["job_id"], manifest.get("seed"),
                 (manifest.get("data") or {}).get("sha256"),
                 json.dumps(manifest), manifest.get("created_at", time.time())),
            )

    def get_run_manifest(self, job_id: str) -> Optional[dict]:
        with self.connection() as con:
            row = con.execute(
                "SELECT body FROM run_manifests WHERE job_id=?",
                (job_id,)).fetchone()
        return json.loads(row["body"]) if row else None

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
                "SELECT body, job_id, promotion_state, quality_score, hard_gates_passed, "
                "quality_breakdown FROM validations WHERE strategy_id=?",
                (strategy_id,)
            ).fetchone()
        return self._row_to_report(row) if row else None

    def get_validations(self, strategy_ids: Sequence[str]) -> Dict[str, ValidationReport]:
        """Load full validation bodies for a small id set (visible page / export)."""
        ids = [sid for sid in strategy_ids if sid]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        q = (
            "SELECT strategy_id, body, job_id, promotion_state, quality_score, "
            "hard_gates_passed, quality_breakdown "
            f"FROM validations WHERE strategy_id IN ({placeholders})"
        )
        with self.connection() as con:
            rows = con.execute(q, tuple(ids)).fetchall()
        return {r["strategy_id"]: self._row_to_report(r) for r in rows}

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

    def list_validation_summaries(
        self,
        passed_only: bool | None = True,
        job_id: Optional[str] = None,
        *,
        limit: Optional[int] = None,
        offset: int = 0,
        include_body_metrics: bool = True,
    ) -> List[ValidationSummary]:
        """List validations without loading equity / WFO JSON into Python.

        Uses table columns plus (optionally) ``json_extract`` for scalar body
        fields so gallery/export chrome never pays full pydantic + equity
        deserialize cost. Set ``include_body_metrics=False`` for per-run index
        lists that only need ``passed`` / ``wfe`` / ids — avoids scanning huge
        body blobs just to populate the run picker grid.
        """
        if include_body_metrics:
            select = (
                "SELECT strategy_id, passed, wfe, job_id, promotion_state, quality_score, "
                "hard_gates_passed, "
                "json_extract(body, '$.engine') AS engine, "
                "json_extract(body, '$.data_source') AS data_source, "
                "json_extract(body, '$.degradation_pct') AS degradation_pct, "
                "json_extract(body, '$.stability_ratio') AS stability_ratio, "
                "json_extract(body, '$.oos_metrics.net_profit') AS oos_net_profit, "
                "json_extract(body, '$.oos_metrics.profit_factor') AS oos_profit_factor, "
                "json_extract(body, '$.oos_metrics.sharpe') AS oos_sharpe, "
                "json_extract(body, '$.oos_metrics.sortino') AS oos_sortino, "
                "json_extract(body, '$.oos_metrics.max_dd_pct') AS oos_max_dd_pct, "
                "json_extract(body, '$.oos_metrics.r_squared') AS oos_r_squared, "
                "json_extract(body, '$.oos_metrics.trade_count') AS oos_trade_count, "
                "json_extract(body, '$.is_metrics.max_dd_pct') AS is_max_dd_pct, "
                "json_extract(body, '$.is_metrics.net_profit') AS is_net_profit, "
                "json_extract(body, '$.montecarlo.robustness_score') AS mc_robustness, "
                "json_extract(body, '$.reasons') AS reasons_json "
                "FROM validations"
            )
        else:
            select = (
                "SELECT strategy_id, passed, wfe, job_id, promotion_state, quality_score, "
                "hard_gates_passed FROM validations"
            )
        q = select
        clauses, args = [], []
        if passed_only is True:
            clauses.append("passed=1")
        elif passed_only is False:
            clauses.append("passed=0")
        if job_id is not None:
            clauses.append("job_id=?")
            args.append(job_id)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY wfe DESC"
        if limit is not None:
            q += " LIMIT ? OFFSET ?"
            args.extend([int(limit), int(offset)])
        with self.connection() as con:
            rows = con.execute(q, tuple(args)).fetchall()
        return [self._row_to_summary(r) for r in rows]

    def list_validations_needing_promotion(
        self,
        *,
        since: float = 0.0,
        limit: int = 100,
    ) -> List[ValidationReport]:
        """Full bodies for rows never scored, or body-updated since ``since``.

        When ``since <= 0`` (first catch-up), only unscored rows are returned so
        we do not re-deserialize the entire library every cycle.
        """
        base = (
            "SELECT strategy_id, body, job_id, promotion_state, quality_score, "
            "hard_gates_passed, quality_breakdown FROM validations "
        )
        unscored = (
            "(promotion_state IS NULL OR promotion_state='' OR quality_score IS NULL)"
        )
        if since <= 0:
            q = base + f"WHERE {unscored} ORDER BY updated_at DESC LIMIT ?"
            args: tuple = (int(limit),)
        else:
            q = (
                base
                + f"WHERE {unscored} OR updated_at > ? "
                + "ORDER BY updated_at DESC LIMIT ?"
            )
            args = (float(since), int(limit))
        with self.connection() as con:
            rows = con.execute(q, args).fetchall()
        return [self._row_to_report(r) for r in rows]

    @staticmethod
    def _row_to_summary(row: sqlite3.Row) -> ValidationSummary:
        keys = row.keys()
        reasons: List[str] = []
        raw_reasons = row["reasons_json"] if "reasons_json" in keys else None
        if raw_reasons:
            try:
                parsed = json.loads(raw_reasons)
                if isinstance(parsed, list):
                    reasons = [str(x) for x in parsed]
            except (TypeError, json.JSONDecodeError):
                reasons = []
        mc_score = row["mc_robustness"] if "mc_robustness" in keys else None
        has_body = "oos_net_profit" in keys
        return ValidationSummary(
            strategy_id=row["strategy_id"],
            run_id=row["job_id"] if "job_id" in keys else None,
            passed=bool(row["passed"]),
            wfe=float(row["wfe"] or 0.0),
            engine=str(row["engine"] or "simulator") if "engine" in keys else "simulator",
            data_source=(
                str(row["data_source"] or "unknown") if "data_source" in keys
                else "unknown"
            ),
            degradation_pct=(
                float(row["degradation_pct"] or 0.0) if "degradation_pct" in keys
                else 0.0
            ),
            stability_ratio=(
                float(row["stability_ratio"] or 1.0) if "stability_ratio" in keys
                else 1.0
            ),
            promotion_state=str(row["promotion_state"] or "candidate"),
            quality_score=float(row["quality_score"] or 0.0),
            hard_gates_passed=bool(row["hard_gates_passed"])
            if "hard_gates_passed" in keys else False,
            reasons=reasons,
            oos_metrics=MetricsSummary(
                net_profit=float(row["oos_net_profit"] or 0.0) if has_body else 0.0,
                profit_factor=(
                    float(row["oos_profit_factor"] or 0.0) if has_body else 0.0
                ),
                sharpe=float(row["oos_sharpe"] or 0.0) if has_body else 0.0,
                sortino=float(row["oos_sortino"] or 0.0) if has_body else 0.0,
                max_dd_pct=float(row["oos_max_dd_pct"] or 0.0) if has_body else 0.0,
                r_squared=float(row["oos_r_squared"] or 0.0) if has_body else 0.0,
                trade_count=int(row["oos_trade_count"] or 0) if has_body else 0,
            ),
            is_metrics=MetricsSummary(
                net_profit=(
                    float(row["is_net_profit"] or 0.0) if "is_net_profit" in keys
                    else 0.0
                ),
                max_dd_pct=(
                    float(row["is_max_dd_pct"] or 0.0) if "is_max_dd_pct" in keys
                    else 0.0
                ),
            ),
            montecarlo=(
                MonteCarloSummary(robustness_score=float(mc_score))
                if mc_score is not None else None
            ),
        )

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
        # Do not bump updated_at — that column tracks when the validation body
        # was written so incremental promotion sync can key off it.
        with self.connection() as con:
            con.execute(
                "UPDATE validations SET promotion_state=?, quality_score=?, "
                "hard_gates_passed=?, quality_breakdown=? "
                "WHERE strategy_id=?",
                (
                    promotion_state,
                    float(quality_score),
                    int(bool(hard_gates_passed)),
                    json.dumps(quality_breakdown or {}),
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

    def count_validated(
        self,
        job_id: Optional[str] = None,
        *,
        passed_only: bool | None = None,
    ) -> tuple[int, int] | int:
        """Count validations.

        - ``count_validated(job_id)`` → ``(passed, total)`` for one run (legacy).
        - ``count_validated(passed_only=True|False|None)`` → single COUNT(*) for
          the library (``None`` = all rows).
        """
        if job_id is not None and passed_only is None:
            with self.connection() as con:
                row = con.execute(
                    "SELECT COALESCE(SUM(passed), 0) AS passed, COUNT(*) AS total"
                    " FROM validations WHERE job_id=?", (job_id,)).fetchone()
            return (int(row["passed"]), int(row["total"])) if row else (0, 0)

        clauses, args = [], []
        if job_id is not None:
            clauses.append("job_id=?")
            args.append(job_id)
        if passed_only is True:
            clauses.append("passed=1")
        elif passed_only is False:
            clauses.append("passed=0")
        q = "SELECT COUNT(*) AS n FROM validations"
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        with self.connection() as con:
            row = con.execute(q, tuple(args)).fetchone()
        return int(row["n"]) if row else 0

    def promotion_state_counts(self) -> Dict[str, int]:
        """Row counts per promotion_state (single GROUP BY, KPI strip)."""
        with self.connection() as con:
            rows = con.execute(
                "SELECT COALESCE(promotion_state, 'candidate') AS state,"
                " COUNT(*) AS n FROM validations GROUP BY state").fetchall()
        return {str(r["state"]): int(r["n"]) for r in rows}

    def count_validated_by_jobs(
        self, job_ids: Sequence[str],
    ) -> Dict[str, tuple[int, int]]:
        """Batch ``(passed, total)`` counts keyed by job id (one GROUP BY)."""
        ids = [jid for jid in job_ids if jid]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        q = (
            "SELECT job_id, COALESCE(SUM(passed), 0) AS passed, COUNT(*) AS total "
            f"FROM validations WHERE job_id IN ({placeholders}) GROUP BY job_id"
        )
        with self.connection() as con:
            rows = con.execute(q, tuple(ids)).fetchall()
        out = {jid: (0, 0) for jid in ids}
        for row in rows:
            out[str(row["job_id"])] = (int(row["passed"]), int(row["total"]))
        return out

    def list_unalerted_quality_candidates(
        self,
        *,
        min_score: float,
        limit: int = 50,
    ) -> List[ValidationReport]:
        """Full bodies for scored, unalerted rows at/above ``min_score``."""
        q = (
            "SELECT strategy_id, body, job_id, promotion_state, quality_score, "
            "hard_gates_passed, quality_breakdown FROM validations "
            "WHERE hard_gates_passed=1 AND last_alert_at IS NULL "
            "AND COALESCE(quality_score, 0) >= ? "
            "ORDER BY quality_score DESC LIMIT ?"
        )
        with self.connection() as con:
            rows = con.execute(q, (float(min_score), int(limit))).fetchall()
        return [self._row_to_report(r) for r in rows]

    def cancel_active_discovery_jobs(self, *, message: str = "Cancelled by stop") -> int:
        """Flag cancel and immediately clear PENDING/RUNNING discovery jobs in UI."""
        now = time.time()
        with self.connection() as con:
            con.execute(
                "UPDATE jobs SET cancel_requested=1, updated_at=? "
                "WHERE kind='discovery' AND status IN ('PENDING', 'RUNNING')",
                (now,),
            )
            cur = con.execute(
                "UPDATE jobs SET status=?, message=?, updated_at=? "
                "WHERE kind='discovery' AND status IN ('PENDING', 'RUNNING') "
                "AND cancel_requested=1",
                (JobStatus.CANCELLED.value, message, now),
            )
            return int(cur.rowcount or 0)

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
                "  tested, promising, survivors, generation, runner_pid)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (job.id, job.kind, job.status.value, job.progress, job.message,
                 job.error, json.dumps(job.payload), int(job.cancel_requested),
                 job.created_at, job.updated_at,
                 int(job.tested), int(job.promising), int(job.survivors),
                 int(job.generation), job.runner_pid),
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

    def set_job_runner(self, job_id: str, pid: Optional[int]) -> None:
        with self.connection() as con:
            con.execute(
                "UPDATE jobs SET runner_pid=?, updated_at=? WHERE id=?",
                (pid, time.time(), job_id),
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
            runner_pid=row["runner_pid"] if "runner_pid" in keys else None,
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
