"""Background job queue.

- Process-wide singleton (`get_job_queue()` is wrapped in st.cache_resource by
  the dashboard, with a plain module-level fallback for scripts/tests), so
  Streamlit reruns can never spawn duplicate workers.
- ThreadPoolExecutor for CPU work (simulation, generation); a dedicated
  single-slot lane (lock) for MT5 terminal runs.
- All progress/status/errors are persisted to SQLite through the worker's own
  short-lived connections; the UI only reads.
- Cooperative cancellation via a DB flag; idempotent submission by job id.
"""
from __future__ import annotations

import os
import pickle
import random
import threading
import time
import traceback
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

from config import settings
from factory import data as data_mod
from factory.backtest.montecarlo import MonteCarloConfig
from factory.backtest.mt5_runner import MT5Runner
from factory.backtest.simulator import SimulatorEngine, SymbolSpec
from factory.backtest.validation import (
    default_criteria, quick_screen, screen_fitness, validate_strategy,
)
from factory.generator import (
    GenerationSettings, evolve, evolve_pareto, random_strategy,
)
from factory.pareto import objectives_from_metrics
from factory.models import (
    AcceptanceCriteria, BacktestMetrics, ExecutionMechanicType, Job,
    JobCancelled, JobStatus, StrategyDefinition, ValidationReport,
)
from factory import validation_levels
from factory.storage import Storage

# How long a cancel probe caches its DB answer, so the deep validation loops
# can poll it thousands of times without hammering SQLite.
_CANCEL_POLL_INTERVAL = 0.25


def _mechanics_from_payload(payload: Dict):
    """Parse the user's allowed execution mechanics from a job payload.

    Returns a list of :class:`ExecutionMechanicType` (or ``None`` to allow
    every mechanic). Unknown/invalid entries are ignored; an empty result
    falls back to all mechanics so a run is never left unable to generate.
    """
    raw = payload.get("mechanics")
    if not raw:
        return None
    out = []
    for name in raw:
        try:
            out.append(ExecutionMechanicType(name))
        except ValueError:
            continue
    return out or None


def _tm_features_from_payload(payload: Dict):
    """Parse allowed trade-management overlay features from a job payload.

    Returns a list of feature keys (subset of ``generator.TM_FEATURES``) or
    ``None`` to allow every overlay. An explicit empty list means "no overlays"
    (plain fixed SL/TP), which we preserve as an empty list.
    """
    raw = payload.get("tm_features")
    if raw is None:
        return None
    from factory.generator import TM_FEATURES
    valid = set(TM_FEATURES)
    return [f for f in raw if f in valid]


def _spec_overrides_from_payload(payload: Dict) -> Dict[str, float]:
    """Collect the user-chosen account/execution economics from a payload.

    Only keys the user actually supplied are returned, so unset values keep
    the engine's inferred defaults (backward compatible). ``point`` is never
    user-controlled — it is always inferred from the price scale.
    """
    out: Dict[str, float] = {}
    for name in SymbolSpec.OVERRIDABLE:      # contract_size, leverage, spread, slippage
        value = payload.get(name)
        if value is not None:
            out[name] = float(value)
    return out


def _generation_settings_from_payload(payload: Dict) -> GenerationSettings:
    """Parse advanced-generation controls from the payload."""
    return GenerationSettings(
        advanced_mode=bool(payload.get("advanced_mode", False)),
        complexity_cap=int(payload.get("complexity_cap", 4)),
        enable_regime_switching=bool(payload.get("enable_regime_switching", False)),
        enable_mtf_context=bool(payload.get("enable_mtf_context", False)),
        feature_toggles=list(payload.get("feature_toggles") or []),
    )


def _aborted_validation_report(strategy: StrategyDefinition, engine: str,
                               exc: Exception) -> ValidationReport:
    """Build a failed report when the validation pipeline aborts mid-run."""
    return ValidationReport(
        strategy_id=strategy.id,
        is_metrics=BacktestMetrics(),
        oos_metrics=BacktestMetrics(),
        passed=False,
        reasons=[f"Validation did not complete: {type(exc).__name__}: {exc}"],
        engine=engine,
    )


def _persist_complete(storage: Storage, strategy: StrategyDefinition,
                      report: ValidationReport,
                      job_id: Optional[str] = None,
                      metadata: Optional[Dict] = None) -> None:
    """Durably store a finished (pass or fail) validation result."""
    storage.save_complete(strategy, report, job_id=job_id, metadata=metadata)


def _apply_spec_overrides(engine, spec_overrides: Dict[str, float]) -> None:
    """Push the user economics into whichever engine will run the backtests.

    The simulator infers ``point`` from price and applies these overrides on
    top; MT5 only consumes leverage (spread/slippage/contract size are broker-
    side in the real terminal). A stub/test engine is left untouched.
    """
    if not spec_overrides:
        return
    if isinstance(engine, SimulatorEngine):
        engine._spec_overrides = dict(spec_overrides)
    elif isinstance(engine, MT5Runner) and "leverage" in spec_overrides:
        engine.leverage = int(spec_overrides["leverage"])


# ---------------------------------------------------------------------------
# Parallel candidate evaluation (process pool)
#
# Simulator backtests are CPU-bound pure-Python loops. Running them in the
# Streamlit process (even on a worker thread) starves the UI via the GIL and
# limits throughput to a single core — which is why discovery felt slow and the
# Cancel button appeared frozen. Evaluating a whole generation across a process
# pool uses every core AND keeps the UI thread free, so Cancel responds at once.
# ---------------------------------------------------------------------------

def _discovery_pool_size(gen_size: int) -> int:
    """Worker count: use most cores but leave one for the UI, capped sanely."""
    cores = os.cpu_count() or 2
    return max(1, min(cores - 1, 16, gen_size))


def _pool_worker_init(ohlc_blob: bytes) -> None:
    """Share the preloaded discovery OHLC range with a pool worker process."""
    symbol, timeframe, df = pickle.loads(ohlc_blob)
    data_mod.register_range_cache(symbol, timeframe, df)


def _pool_cancel_check(db_path, job_id: str) -> Callable[[], bool]:
    """A throttled DB cancel probe for a pool worker process.

    Each worker opens its own short-lived SQLite connections; caching the
    answer for a fraction of a second keeps the deep validation loops cheap
    while still aborting the in-flight backtest within ~0.25s of a click.
    """
    storage = Storage(Path(db_path))
    state = {"ts": 0.0, "cancelled": False}

    def _check() -> bool:
        if state["cancelled"]:
            return True
        now = time.monotonic()
        if now - state["ts"] >= _CANCEL_POLL_INTERVAL:
            state["ts"] = now
            try:
                state["cancelled"] = storage.is_cancel_requested(job_id)
            except Exception:
                pass
        return state["cancelled"]

    return _check


def _evaluate_candidate(task: Dict) -> Dict:
    """Screen one candidate and, if promising, fully validate it.

    Runs inside a pool worker process; returns a picklable result dict. All
    exceptions are captured (never raised across the process boundary) so the
    dispatcher can account for every candidate deterministically.
    """
    strat = task["strategy"]
    start, end, deposit = task["start"], task["end"], task["deposit"]
    criteria = task["criteria"]
    spec_overrides = task["spec_overrides"] or None
    storage = Storage(Path(task["db_path"]))
    cancel_check = _pool_cancel_check(task["db_path"], task["job_id"])

    engine = SimulatorEngine(spec_overrides=spec_overrides)
    engine._cancel_check = cancel_check
    result = {"strategy": strat, "fitness": 0.0, "promising": False,
              "passed": False, "error": None, "cancelled": False,
              "objectives": None}
    try:
        promising, m = quick_screen(engine, strat, start, end, deposit, criteria)
        result["fitness"] = screen_fitness(m)
        result["objectives"] = objectives_from_metrics(m)
        if not promising:
            return result
        result["promising"] = True
        report = validate_strategy(
            engine, strat, start, end, deposit=deposit, seed=task["seed"],
            criteria=criteria, run_montecarlo=task["run_mc"],
            mc_config=task["mc_config"], wfo_train_months=task["wfo_train"],
            wfo_test_months=task["wfo_test"], wfo_windows=task["wfo_n"],
            data_source=task["data_source"], cancel_check=cancel_check,
            spec_overrides=spec_overrides,
            n_trials=int(task.get("n_trials", 1)))
        result["passed"] = report.passed
        storage.save_complete(
            strat,
            report,
            job_id=task["job_id"],
            metadata={
                "sweep_symbol": task.get("sweep_symbol"),
                "sweep_timeframe": task.get("sweep_timeframe"),
                "strictness_profile": task.get("strictness_profile"),
                "seed": task.get("seed"),
                "parameter_snapshot": report.best_params,
                "parent_id": (strat.lineage.parents[0] if strat.lineage.parents else None),
                "generation": strat.lineage.generation,
            },
        )
        return result
    except JobCancelled:
        result["cancelled"] = True
        return result
    except Exception as exc:                       # noqa: BLE001 - abort, don't crash
        result["error"] = f"{type(exc).__name__}: {exc}"
        if result["promising"]:
            report = _aborted_validation_report(strat, "simulator", exc)
            storage.save_complete(
                strat,
                report,
                job_id=task["job_id"],
                metadata={
                    "sweep_symbol": task.get("sweep_symbol"),
                    "sweep_timeframe": task.get("sweep_timeframe"),
                    "strictness_profile": task.get("strictness_profile"),
                    "seed": task.get("seed"),
                    "parameter_snapshot": {},
                    "parent_id": (strat.lineage.parents[0] if strat.lineage.parents else None),
                    "generation": strat.lineage.generation,
                },
            )
        return result


class JobQueue:
    def __init__(self, storage: Optional[Storage] = None):
        self.storage = storage or Storage()
        self._executor = ThreadPoolExecutor(max_workers=2,
                                            thread_name_prefix="eafactory")
        self._mt5_lane = threading.Lock()   # max one concurrent MT5 terminal
        # Optional multi-instance MT5 pool (portable installs). None keeps
        # the legacy single-lane behavior above.
        try:
            from jobs.mt5_pool import pool_from_settings
            self._mt5_pool = pool_from_settings()
        except Exception:
            self._mt5_pool = None
        self._submitted: Dict[str, bool] = {}
        self._submit_lock = threading.Lock()
        self._reconcile_orphaned_jobs()
        self._resume_pending_jobs()

    def _reconcile_orphaned_jobs(self) -> None:
        """Close out jobs left ``RUNNING`` by a dead process.

        Workers are threads inside this process, so a shutdown/restart kills
        them mid-run while their SQLite row stays ``RUNNING``. Only cancel a
        row when its ``runner_pid`` is missing or no longer alive — never
        touch jobs actively executed by another live process (e.g. the detached
        discovery orchestrator).
        """
        for job in self.storage.list_jobs():
            if job.status != JobStatus.RUNNING:
                continue
            pid = job.runner_pid
            if pid and _pid_alive(int(pid)):
                continue
            self.storage.set_job_status(
                job.id, JobStatus.CANCELLED,
                message="stopped — interrupted by an app restart")
            self.storage.set_job_runner(job.id, None)

    def _resume_pending_jobs(self) -> None:
        """Pick up discovery jobs queued while this process was down."""
        if self._has_live_running_job():
            return
        for job in reversed(self.storage.list_jobs("discovery")):
            if job.status != JobStatus.PENDING or job.cancel_requested:
                continue
            self.submit_discovery(job.id, job.payload)
            break

    def _has_live_running_job(self) -> bool:
        for job in self.storage.list_jobs("discovery"):
            if job.status != JobStatus.RUNNING:
                continue
            if job.runner_pid and _pid_alive(int(job.runner_pid)):
                return True
        return False

    # ------------------------------------------------------------------
    # Submission (idempotent by job id)
    # ------------------------------------------------------------------
    def submit_discovery(self, job_id: str, payload: Dict) -> bool:
        """Enqueue a discovery batch. Returns False when the id is already
        pending/running (rerun-triggered double clicks are dropped)."""
        with self._submit_lock:
            existing = self.storage.get_job(job_id)
            if existing and existing.status in (JobStatus.PENDING, JobStatus.RUNNING):
                return False
            if self._submitted.get(job_id):
                return False
            self._submitted[job_id] = True

        job = Job(id=job_id, kind="discovery", payload=payload)
        self.storage.upsert_job(job)
        self._executor.submit(self._run_discovery_safe, job_id)
        return True

    def cancel(self, job_id: str) -> None:
        self.storage.request_cancel(job_id)

    def _make_cancel_check(self, job_id: str) -> Callable[[], bool]:
        """A throttled probe of the DB cancel flag for the validation pipeline.

        The expensive per-candidate work (IS optimization, walk-forward, Monte
        Carlo) calls this at many safe points; caching the answer for a short
        interval keeps that cheap while still aborting within ~0.25s of a click.
        """
        state = {"ts": 0.0, "cancelled": False}

        def _check() -> bool:
            if state["cancelled"]:
                return True
            now = time.monotonic()
            if now - state["ts"] >= _CANCEL_POLL_INTERVAL:
                state["ts"] = now
                state["cancelled"] = self.storage.is_cancel_requested(job_id)
            return state["cancelled"]

        return _check

    # ------------------------------------------------------------------
    # Discovery pipeline
    # ------------------------------------------------------------------
    @contextmanager
    def _mt5_engine_lease(self, default_engine, spec_overrides: Dict[str, float],
                          cancel_check):
        """Yield an engine for ONE MT5 validation run.

        With a pool configured, lease a portable instance exclusively and
        bind a fresh non-exclusive runner to it (several leases can then run
        testers concurrently). Without a pool, fall back to the shared
        engine under the legacy single lane.
        """
        if self._mt5_pool is not None:
            lev = spec_overrides.get("leverage") if spec_overrides else None
            with self._mt5_pool.lease() as instance:
                runner = self._mt5_pool.runner_for(
                    instance, leverage=int(lev) if lev is not None else None)
                runner._cancel_check = cancel_check
                yield runner
        else:
            with self._mt5_lane:
                yield default_engine

    def _run_discovery_safe(self, job_id: str) -> None:
        try:
            self._run_discovery(job_id)
        except Exception:
            self.storage.set_job_status(job_id, JobStatus.FAILED,
                                        error=traceback.format_exc())
        finally:
            self.storage.set_job_runner(job_id, None)
            with self._submit_lock:
                self._submitted.pop(job_id, None)

    def _make_engine(self, engine_name: str):
        if engine_name == "mt5":
            return MT5Runner()
        return SimulatorEngine()

    def _run_discovery(self, job_id: str) -> None:
        """Continuous, two-stage discovery.

        Stage 1 (fast screen): every generated candidate gets a single cheap
        full-range simulation. Obvious losers are discarded immediately, so
        thousands of parameter combinations can be triaged quickly.

        Stage 2 (full validation): only promising candidates run the expensive
        IS/OOS + walk-forward + Monte Carlo pipeline. Every candidate that
        completes (or aborts) full validation is persisted to SQLite — pass
        or fail — so nothing is lost on restart or when the survivor quota
        is met.

        The loop keeps generating new generations (random + genetic evolution
        of the best screened candidates) until it has found the requested
        number of survivors, exhausted the candidate budget, or is cancelled.
        """
        storage = self.storage
        job = storage.get_job(job_id)
        payload = job.payload
        storage.set_job_status(job_id, JobStatus.RUNNING, message="starting")
        storage.set_job_runner(job_id, os.getpid())

        symbol = payload.get("symbol", settings.DEFAULT_SYMBOL)
        timeframe = payload.get("timeframe", settings.DEFAULT_TIMEFRAME)
        engine_name = payload.get("engine", settings.DEFAULT_ENGINE)
        deposit = float(payload.get("deposit", settings.DEFAULT_DEPOSIT))
        spec_overrides = _spec_overrides_from_payload(payload)
        # Reproducibility: a run without an explicit seed gets a concrete one
        # drawn here and recorded in the manifest, so *every* run — not just
        # deliberately seeded ones — can be exactly re-derived later.
        seed = payload.get("seed")
        if seed is None:
            seed = random.SystemRandom().randint(0, 2**31)
            payload["seed"] = seed

        # generation size + loop budget (thousands of candidates supported)
        gen_size = int(payload.get("batch_size", 100))
        gen_size = max(1, gen_size)
        target_survivors = int(payload.get("target_survivors", 5))
        max_candidates = int(payload.get("max_candidates", 1000))
        use_genetic = bool(payload.get("genetic", True))
        allowed_mechanics = _mechanics_from_payload(payload)
        allowed_tm_features = _tm_features_from_payload(payload)
        generation_settings = _generation_settings_from_payload(payload)

        start = datetime.fromisoformat(payload["start"]) if payload.get("start") \
            else datetime.now(timezone.utc) - timedelta(days=365)
        end = datetime.fromisoformat(payload["end"]) if payload.get("end") \
            else datetime.now(timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        # Validation gates come from a named level (easy mode) unless the user
        # supplied explicit custom criteria (expert override).
        if payload.get("criteria"):
            criteria = AcceptanceCriteria(**payload["criteria"])
            run_mc = payload.get("montecarlo")       # None -> settings default
            mc_config = None
            if payload.get("mc_runs"):
                mc_config = MonteCarloConfig(n_runs=int(payload["mc_runs"]))
        elif payload.get("validation_level") is not None:
            level = validation_levels.get_level(int(payload["validation_level"]))
            criteria = level.criteria
            run_mc = level.montecarlo
            mc_config = validation_levels.mc_config_for(level)
        else:
            criteria = default_criteria()
            run_mc = payload.get("montecarlo")
            mc_config = None

        wfo_train = int(payload.get("wfo_train_months", settings.WFO_TRAIN_MONTHS))
        wfo_test = int(payload.get("wfo_test_months", settings.WFO_TEST_MONTHS))
        wfo_n = int(payload.get("wfo_windows", settings.WFO_WINDOWS))
        full_ohlc = data_mod.load_ohlc(symbol, timeframe, start, end)
        data_mod.register_range_cache(symbol, timeframe, full_ohlc)
        data_source = str(payload.get("data_source") or full_ohlc.attrs.get("source", "unknown"))
        ohlc_blob = pickle.dumps((symbol, timeframe, full_ohlc))

        # Persist the reproducibility manifest before any candidate runs, so
        # even a crashed/cancelled run documents exactly what it saw.
        try:
            from factory.manifest import build_manifest
            storage.save_run_manifest(build_manifest(job_id, payload, seed,
                                                     full_ohlc))
        except Exception:
            pass  # manifests are diagnostics — never block a run on them

        rng = random.Random(seed)
        engine = self._make_engine(engine_name)
        _apply_spec_overrides(engine, spec_overrides)
        cancel_check = self._make_cancel_check(job_id)
        # Let the engine abort a long backtest mid-loop, not just between
        # candidates — the simulator polls this probe inside its bar loop.
        if isinstance(engine, SimulatorEngine):
            engine._cancel_check = cancel_check

        tested = 0            # candidates that ran the fast screen
        screened_in = 0       # candidates promising enough for full validation
        survivors = 0         # strategies passing all gates
        errors = 0
        scored: list = []     # (strategy, screen_fitness, objectives) triples
        generation = 0
        persist_state = {"ts": 0.0, "tested": 0}

        def _progress_msg() -> str:
            return (f"tested {tested}/{max_candidates} · promising {screened_in}"
                    f" · survivors {survivors}/{target_survivors}"
                    f" · gen {generation}")

        def _persist_progress(*, force: bool = False) -> None:
            """Write live counters for the UI without hammering SQLite."""
            now = time.monotonic()
            if (not force and tested == persist_state["tested"]
                    and now - persist_state["ts"] < 2.0):
                return
            if (not force and tested - persist_state["tested"] < 5
                    and now - persist_state["ts"] < 2.0):
                return
            persist_state["ts"] = now
            persist_state["tested"] = tested
            frac = max(tested / max_candidates,
                       survivors / max(target_survivors, 1))
            storage.update_job_progress(
                job_id, min(frac, 0.999), _progress_msg(),
                tested=tested, promising=screened_in,
                survivors=survivors, generation=generation)

        def _cancelled_now() -> bool:
            return storage.is_cancel_requested(job_id)

        def _finish_cancelled() -> None:
            storage.set_job_status(
                job_id, JobStatus.CANCELLED,
                message=f"cancelled — {_progress_msg()}")

        def _build_population(this_gen: int):
            if generation == 0 or not (use_genetic and scored):
                return [random_strategy(symbol, timeframe, rng,
                                        allowed_mechanics=allowed_mechanics,
                                        allowed_tm_features=allowed_tm_features,
                                        generation_settings=generation_settings)
                        for _ in range(this_gen)]
            if getattr(settings, "PARETO_EVOLUTION", True):
                # NSGA-II over the most recent candidates with objective
                # vectors; keeps the population spread along the whole
                # profit/risk/stability front instead of one scalar peak.
                recent = [(s, obj) for s, _f, obj in scored[-500:]
                          if obj is not None]
                if len(recent) >= 2:
                    pop = evolve_pareto(recent, this_gen, rng)
                else:
                    top = sorted(scored, key=lambda t: t[1], reverse=True)[:50]
                    pop = evolve([(s, f) for s, f, _o in top], this_gen, rng)
            else:
                top = sorted(scored, key=lambda t: t[1], reverse=True)[:50]
                pop = evolve([(s, f) for s, f, _o in top], this_gen, rng)
            n_fresh = max(1, this_gen // 3)      # fresh blood vs. convergence
            pop[:n_fresh] = [random_strategy(symbol, timeframe, rng,
                                             allowed_mechanics=allowed_mechanics,
                                             allowed_tm_features=allowed_tm_features,
                                             generation_settings=generation_settings)
                             for _ in range(n_fresh)]
            return pop

        # Simulator work fans out across a process pool (parallel, off the UI
        # thread). MT5 and non-simulator engines stay sequential.
        #
        # Test suites monkeypatch ``_make_engine`` with in-process stubs that
        # track call counts / injected failures. Those stubs are not picklable
        # across process boundaries, so parallel pool mode must only be used
        # for the real SimulatorEngine.
        pool = None
        if engine_name != "mt5" and isinstance(engine, SimulatorEngine):
            pool = ProcessPoolExecutor(
                max_workers=_discovery_pool_size(gen_size),
                initializer=_pool_worker_init,
                initargs=(ohlc_blob,),
            )

        def _make_task(strat) -> Dict:
            return {
                "strategy": strat, "start": start, "end": end,
                "deposit": deposit, "criteria": criteria, "run_mc": run_mc,
                "mc_config": mc_config, "wfo_train": wfo_train,
                "wfo_test": wfo_test, "wfo_n": wfo_n, "data_source": data_source,
                "spec_overrides": spec_overrides, "seed": rng.randint(0, 2**31),
                "n_trials": tested + 1,
                "db_path": str(self.storage.db_path), "job_id": job_id,
                "sweep_symbol": payload.get("symbol"),
                "sweep_timeframe": payload.get("timeframe"),
                "strictness_profile": payload.get("strictness_profile", "normal"),
            }

        try:
            while tested < max_candidates and survivors < target_survivors:
                if _cancelled_now():
                    _finish_cancelled()
                    return

                remaining = max_candidates - tested
                population = _build_population(min(gen_size, remaining))

                if (pool is None and engine_name == "mt5"
                        and self._mt5_pool is not None
                        and self._mt5_pool.size > 1):
                    # -- MT5 pool: validate the generation in parallel -------
                    # Each task leases one portable instance for its whole
                    # validation (IS opt + OOS + WFO), so at most pool.size
                    # testers run concurrently and no instance is shared.
                    def _candidate_meta(strat, report) -> Dict:
                        return {
                            "sweep_symbol": payload.get("symbol"),
                            "sweep_timeframe": payload.get("timeframe"),
                            "strictness_profile": payload.get(
                                "strictness_profile", "normal"),
                            "seed": payload.get("seed"),
                            "parameter_snapshot": report.best_params,
                            "parent_id": (strat.lineage.parents[0]
                                          if strat.lineage.parents else None),
                            "generation": strat.lineage.generation,
                        }

                    def _mt5_one(strat, seed_val: int, trial_no: int):
                        try:
                            with self._mt5_engine_lease(
                                    engine, spec_overrides,
                                    cancel_check) as m5:
                                rep = validate_strategy(
                                    m5, strat, start, end, deposit=deposit,
                                    seed=seed_val, criteria=criteria,
                                    run_montecarlo=run_mc, mc_config=mc_config,
                                    wfo_train_months=wfo_train,
                                    wfo_test_months=wfo_test,
                                    wfo_windows=wfo_n,
                                    data_source=data_source,
                                    cancel_check=cancel_check,
                                    spec_overrides=spec_overrides,
                                    n_trials=trial_no)
                            return "ok", rep, None
                        except JobCancelled:
                            return "cancelled", None, None
                        except Exception as exc:  # noqa: BLE001
                            return ("error",
                                    _aborted_validation_report(strat, "mt5", exc),
                                    f"{type(exc).__name__}: {exc}")

                    was_cancelled = False
                    tex = ThreadPoolExecutor(
                        max_workers=self._mt5_pool.size,
                        thread_name_prefix="mt5pool")
                    futures = {
                        tex.submit(_mt5_one, s, rng.randint(0, 2**31),
                                   tested + k + 1): s
                        for k, s in enumerate(population)}
                    try:
                        for fut in as_completed(futures):
                            strat = futures[fut]
                            status, report, err = fut.result()
                            if status == "cancelled":
                                was_cancelled = True
                                continue
                            tested += 1
                            screened_in += 1
                            if err:
                                errors += 1
                                storage.set_job_status(
                                    job_id, JobStatus.RUNNING, error=err)
                            _persist_complete(storage, strat, report,
                                              job_id=job_id,
                                              metadata=_candidate_meta(strat, report))
                            scored.append((strat,
                                           screen_fitness(report.oos_metrics),
                                           objectives_from_metrics(report.oos_metrics)))
                            if report.passed:
                                survivors += 1
                            _persist_progress()
                            if survivors >= target_survivors or _cancelled_now():
                                for f in futures:
                                    f.cancel()
                    finally:
                        tex.shutdown(wait=True, cancel_futures=True)

                    if was_cancelled or _cancelled_now():
                        _finish_cancelled()
                        return
                    generation += 1
                    continue

                if pool is None:
                    # -- Sequential lane: MT5 (always) and non-simulator stubs --
                    for strat in population:
                        if _cancelled_now():
                            _finish_cancelled()
                            return
                        if survivors >= target_survivors:
                            break
                        tested += 1
                        _persist_progress()
                        engine_label = getattr(engine, "name", engine_name)
                        if engine_name == "mt5":
                            try:
                                with self._mt5_engine_lease(
                                        engine, spec_overrides,
                                        cancel_check) as mt5_engine:
                                    report = validate_strategy(
                                        mt5_engine, strat, start, end, deposit=deposit,
                                        seed=rng.randint(0, 2**31), criteria=criteria,
                                        run_montecarlo=run_mc, mc_config=mc_config,
                                        wfo_train_months=wfo_train,
                                        wfo_test_months=wfo_test, wfo_windows=wfo_n,
                                        data_source=data_source,
                                        cancel_check=cancel_check,
                                        spec_overrides=spec_overrides,
                                        n_trials=tested)
                            except JobCancelled:
                                _finish_cancelled()
                                return
                            except Exception as exc:
                                errors += 1
                                report = _aborted_validation_report(
                                    strat, engine_label, exc)
                                _persist_complete(storage, strat, report,
                                                  job_id=job_id,
                                                  metadata={
                                                      "sweep_symbol": payload.get("symbol"),
                                                      "sweep_timeframe": payload.get("timeframe"),
                                                      "strictness_profile": payload.get("strictness_profile", "normal"),
                                                      "seed": payload.get("seed"),
                                                      "parameter_snapshot": report.best_params,
                                                      "parent_id": (strat.lineage.parents[0] if strat.lineage.parents else None),
                                                      "generation": strat.lineage.generation,
                                                  })
                                storage.set_job_status(
                                    job_id, JobStatus.RUNNING,
                                    error=f"{type(exc).__name__}: {exc}")
                            else:
                                _persist_complete(storage, strat, report,
                                                  job_id=job_id,
                                                  metadata={
                                                      "sweep_symbol": payload.get("symbol"),
                                                      "sweep_timeframe": payload.get("timeframe"),
                                                      "strictness_profile": payload.get("strictness_profile", "normal"),
                                                      "seed": payload.get("seed"),
                                                      "parameter_snapshot": report.best_params,
                                                      "parent_id": (strat.lineage.parents[0] if strat.lineage.parents else None),
                                                      "generation": strat.lineage.generation,
                                                  })
                            scored.append((strat, screen_fitness(report.oos_metrics),
                                           objectives_from_metrics(report.oos_metrics)))
                            if report.passed:
                                survivors += 1
                        else:
                            # Non-MT5 sequential fallback mirrors pool semantics:
                            # quick screen -> full validation only for promising.
                            try:
                                promising, m = quick_screen(
                                    engine, strat, start, end, deposit, criteria)
                            except JobCancelled:
                                _finish_cancelled()
                                return
                            except Exception as exc:
                                errors += 1
                                storage.set_job_status(
                                    job_id, JobStatus.RUNNING,
                                    error=f"{type(exc).__name__}: {exc}")
                                continue

                            scored.append((strat, screen_fitness(m),
                                           objectives_from_metrics(m)))
                            if not promising:
                                continue
                            screened_in += 1
                            try:
                                report = validate_strategy(
                                    engine, strat, start, end, deposit=deposit,
                                    seed=rng.randint(0, 2**31), criteria=criteria,
                                    run_montecarlo=run_mc, mc_config=mc_config,
                                    wfo_train_months=wfo_train,
                                    wfo_test_months=wfo_test, wfo_windows=wfo_n,
                                    data_source=data_source,
                                    cancel_check=cancel_check,
                                    spec_overrides=spec_overrides,
                                    n_trials=tested)
                            except JobCancelled:
                                _finish_cancelled()
                                return
                            except Exception as exc:
                                errors += 1
                                report = _aborted_validation_report(
                                    strat, engine_label, exc)
                                _persist_complete(storage, strat, report,
                                                  job_id=job_id,
                                                  metadata={
                                                      "sweep_symbol": payload.get("symbol"),
                                                      "sweep_timeframe": payload.get("timeframe"),
                                                      "strictness_profile": payload.get("strictness_profile", "normal"),
                                                      "seed": payload.get("seed"),
                                                      "parameter_snapshot": report.best_params,
                                                      "parent_id": (strat.lineage.parents[0] if strat.lineage.parents else None),
                                                      "generation": strat.lineage.generation,
                                                  })
                                storage.set_job_status(
                                    job_id, JobStatus.RUNNING,
                                    error=f"{type(exc).__name__}: {exc}")
                            else:
                                _persist_complete(storage, strat, report,
                                                  job_id=job_id,
                                                  metadata={
                                                      "sweep_symbol": payload.get("symbol"),
                                                      "sweep_timeframe": payload.get("timeframe"),
                                                      "strictness_profile": payload.get("strictness_profile", "normal"),
                                                      "seed": payload.get("seed"),
                                                      "parameter_snapshot": report.best_params,
                                                      "parent_id": (strat.lineage.parents[0] if strat.lineage.parents else None),
                                                      "generation": strat.lineage.generation,
                                                  })
                            if report.passed:
                                survivors += 1
                        _persist_progress()
                else:
                    # -- Simulator: evaluate the whole generation in parallel --
                    futures = [pool.submit(_evaluate_candidate, _make_task(s))
                               for s in population]
                    try:
                        for fut in as_completed(futures):
                            try:
                                res = fut.result()
                            except Exception as exc:       # pool/worker failure
                                tested += 1
                                errors += 1
                                storage.set_job_status(
                                    job_id, JobStatus.RUNNING,
                                    error=f"pool: {type(exc).__name__}: {exc}")
                                continue
                            tested += 1
                            scored.append((res["strategy"], res["fitness"],
                                           res.get("objectives")))
                            if res["promising"]:
                                screened_in += 1
                                if res["passed"]:
                                    survivors += 1
                                if res["error"]:
                                    errors += 1
                                    storage.set_job_status(
                                        job_id, JobStatus.RUNNING,
                                        error=res["error"])
                            _persist_progress()
                            if survivors >= target_survivors or _cancelled_now():
                                break
                    finally:
                        for fut in futures:
                            fut.cancel()

                    if _cancelled_now():
                        _finish_cancelled()
                        return

                generation += 1
        finally:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            data_mod.clear_range_cache()

        total_passing = self.storage.count_validated(passed_only=True)
        storage.update_job_progress(
            job_id, 1.0,
            f"done — {_progress_msg()} ({total_passing} passing in library)",
            tested=tested, promising=screened_in,
            survivors=survivors, generation=generation)
        storage.set_job_status(job_id, JobStatus.DONE)


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_queue_lock = threading.Lock()
_queue: Optional[JobQueue] = None


def _pid_alive(pid: int) -> bool:
    """Return True when ``pid`` still refers to a live OS process."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_job_queue() -> JobQueue:
    """Module-level singleton. The dashboard additionally wraps this in
    st.cache_resource so Streamlit reruns always reuse one instance."""
    global _queue
    with _queue_lock:
        if _queue is None:
            _queue = JobQueue()
        return _queue
