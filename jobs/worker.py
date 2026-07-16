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
import sys
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
from factory.backtest.mt5_runner import MT5Runner, MT5RunnerError, interactive_terminal_running
from factory.backtest.simulator import SimulatorEngine, SymbolSpec
from factory.backtest.validation import (
    default_criteria, quick_screen, screen_fitness, validate_strategy,
)
from factory.edge import (
    apply_edge_best_params, expand_execution_variants, is_edge_probe,
    is_execution_variant,
)
from factory.generator import (
    GenerationSettings, blend_family_weights, blend_filter_weights,
    blend_mechanic_weights, evolve, evolve_pareto, mutate, random_strategy,
)
from factory.symbol_class import classify_symbol
from factory.pareto import objectives_from_metrics
from factory.models import (
    AcceptanceCriteria, BacktestMetrics, ExecutionMechanicType, Job,
    JobCancelled, JobStatus, StrategyDefinition, ValidationReport,
)
from factory import validation_levels
from factory.logutil import get_logger
from factory.storage import Storage
from factory import results_archive

log = get_logger(__name__)

# How long a cancel probe caches its DB answer, so the deep validation loops
# can poll it thousands of times without hammering SQLite.
_CANCEL_POLL_INTERVAL = 0.25
# Consecutive cancel-probe DB failures before fail-closed (treat as cancelled).
_CANCEL_PROBE_FAIL_LIMIT = 5


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
        hypothesis_families=bool(payload.get("hypothesis_families", True)),
    )


def _is_mt5_already_running_error(exc: BaseException) -> bool:
    """True when MT5 refused the run because an interactive terminal is open."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return "already running" in text


def _is_infra_exception(exc: BaseException) -> bool:
    """Classify abort exceptions as infrastructure (not strategy quality)."""
    if isinstance(exc, MT5RunnerError) or _is_mt5_already_running_error(exc):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "metatrader" in text
        or "mt5" in text
        or "terminal64" in text
        or "failed to initialize" in text
        or "failed to connect" in text
    )


def _aborted_validation_report(strategy: StrategyDefinition, engine: str,
                               exc: Exception) -> ValidationReport:
    """Build a failed report when the validation pipeline aborts mid-run."""
    infra = _is_infra_exception(exc)
    prefix = "INFRA: " if infra else ""
    return ValidationReport(
        strategy_id=strategy.id,
        is_metrics=BacktestMetrics(),
        oos_metrics=BacktestMetrics(),
        passed=False,
        reasons=[f"{prefix}Validation did not complete: {type(exc).__name__}: {exc}"],
        engine=engine,
        infra_failure=infra,
    )


_MT5_ALREADY_RUNNING_RETRY_SLEEP_S = 2.0


def _run_mt5_validate_with_retry(validate_fn):
    """Run MT5 validation; retry once on 'terminal already running'."""
    try:
        return validate_fn()
    except JobCancelled:
        raise
    except Exception as exc:
        if not _is_mt5_already_running_error(exc):
            raise
        time.sleep(_MT5_ALREADY_RUNNING_RETRY_SLEEP_S)
        return validate_fn()


def _lineage_metadata(strat: StrategyDefinition) -> Dict:
    """Persistable lineage fields for strategy_metadata."""
    parents = list(strat.lineage.parents or [])
    mutations = list(strat.lineage.mutations or [])
    if len(parents) > 1:
        operation = "crossover"
    elif parents:
        operation = "mutate"
    else:
        operation = "random"
    return {
        "parent_id": parents[0] if parents else None,
        "generation": strat.lineage.generation,
        "parents_json": parents,
        "mutations_json": mutations,
        "operation": operation,
        "pareto_rank": None,
        "crowding_distance": None,
    }


def _candidate_metadata(strat: StrategyDefinition, payload: Dict,
                        *, parameter_snapshot: Optional[Dict] = None,
                        sweep_symbol: Optional[str] = None,
                        sweep_timeframe: Optional[str] = None,
                        strictness_profile: Optional[str] = None,
                        seed=None) -> Dict:
    """Build the metadata dict passed to ``storage.save_complete``."""
    meta = {
        "sweep_symbol": (sweep_symbol if sweep_symbol is not None
                         else payload.get("symbol") or payload.get("sweep_symbol")),
        "sweep_timeframe": (sweep_timeframe if sweep_timeframe is not None
                            else payload.get("timeframe")
                            or payload.get("sweep_timeframe")),
        "strictness_profile": (
            strictness_profile if strictness_profile is not None
            else payload.get("strictness_profile", "normal")),
        "seed": seed if seed is not None else payload.get("seed"),
        "parameter_snapshot": parameter_snapshot or {},
    }
    meta.update(_lineage_metadata(strat))
    return meta


def _persist_complete(storage: Storage, strategy: StrategyDefinition,
                      report: ValidationReport,
                      job_id: Optional[str] = None,
                      metadata: Optional[Dict] = None) -> None:
    """Durably store a finished (pass or fail) validation result + archive it."""
    storage.save_complete(strategy, report, job_id=job_id, metadata=metadata)
    if job_id:
        try:
            results_archive.write_candidate(
                job_id, strategy, report, metadata=metadata)
        except Exception as exc:
            log.warning(
                "results archive write failed for %s/%s: %s",
                job_id, strategy.id, exc)


def _maybe_mt5_confirm_survivor(
    strategy: StrategyDefinition,
    report: ValidationReport,
    *,
    start: datetime,
    end: datetime,
    deposit: float,
    enabled: bool = True,
) -> ValidationReport:
    """Cheap MT5 agreement check for high-level simulator survivors.

    Sets ``report.mt5_confirmed`` to True/False when a tester run completes,
    or leaves it ``None`` when MT5 is unavailable / infra-blocked.
    """
    if not enabled or not getattr(settings, "DISCOVERY_MT5_CONFIRM_SURVIVORS", True):
        return report
    if not report.passed:
        return report
    min_lvl = int(getattr(settings, "MT5_CONFIRM_MIN_LEVEL", 7))
    if int(getattr(report, "highest_level_passed", 0) or 0) < min_lvl:
        return report
    if str(getattr(report, "engine", "") or "").lower() == "mt5":
        report.mt5_confirmed = True
        return report
    try:
        from factory.backtest.mt5_runner import MT5Runner, detect_mt5
        detect_mt5()
        runner = MT5Runner()
        strat = strategy
        if report.best_params:
            strat = strategy.apply_flat_params(report.best_params)
        metrics = runner.run(strat, start, end, deposit=deposit)
        report.mt5_confirmed = bool(
            metrics.net_profit > 0.0 and metrics.trade_count >= 3
            and metrics.max_dd_pct < 80.0
        )
        if not report.mt5_confirmed:
            report.reasons = list(report.reasons or []) + [
                "MT5 confirm: unprofitable or empty on tester re-run"
            ]
    except Exception as exc:  # noqa: BLE001
        log.info("MT5 survivor confirm skipped for %s: %s", strategy.id, exc)
        report.mt5_confirmed = None
    return report


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


def _hide_pool_console_on_windows() -> None:
    """Optionally use pythonw.exe for ProcessPoolExecutor workers on Windows.

    Pool workers are spawned as separate ``python.exe`` processes by default,
    each of which opens a blank console window. ``pythonw.exe`` is the same
    interpreter without a console subsystem — but on some Windows/venv setups
    switching the multiprocessing executable to pythonw causes children to die
    immediately (``BrokenProcessPool``). Opt in with ``EA_POOL_PYTHONW=1``.
    """
    if os.name != "nt":
        return
    if os.environ.get("EA_POOL_PYTHONW", "").strip().lower() not in (
            "1", "true", "yes"):
        return
    import multiprocessing

    pyw = Path(sys.executable).with_name("pythonw.exe")
    if pyw.is_file():
        multiprocessing.set_executable(str(pyw))


def _pack_ohlc_blob(frames: list) -> bytes:
    """Pickle one or more ``(symbol, timeframe, df)`` frames for pool workers."""
    return pickle.dumps(list(frames))


def _unpack_ohlc_blob(ohlc_blob: bytes) -> list:
    """Accept both legacy single-triple blobs and multi-frame lists."""
    payload = pickle.loads(ohlc_blob)
    if (isinstance(payload, tuple) and len(payload) == 3
            and not isinstance(payload[0], (list, tuple))):
        return [payload]
    return list(payload)


def _warm_numba_jit() -> None:
    """Compile the Numba bar loop once so the first real screen is not cold."""
    try:
        from factory.backtest.sim_numba_core import numba_available, run_simulation_numba
        from factory.backtest.simulator import SymbolSpec
        from factory.generator import random_strategy
        if not numba_available():
            return
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)
        df = data_mod.synthetic_ohlc("EURUSD", "M15", start, end)
        if len(df) < 20:
            return
        strat = random_strategy("EURUSD", "M15", random.Random(0))
        spec = SymbolSpec.infer(float(df["close"].iloc[0]), None, symbol="EURUSD")
        run_simulation_numba(df, strat, spec, 10_000.0, intrabar_mode="path")
    except Exception as exc:
        log.warning("Numba JIT warm-up failed; Python sim path will be used: %s", exc)


def _pool_worker_init(ohlc_blob: bytes) -> None:
    """Share preloaded discovery OHLC (+ optional M1) with a pool worker.

    Also marks this process as an outer discovery-pool worker so nested
    Monte Carlo (and similar) stays serial and does not oversubscribe cores,
    and warms the Numba JIT so Stage-1 screens start hot.
    """
    os.environ["EA_DISCOVERY_POOL_WORKER"] = "1"
    for symbol, timeframe, df in _unpack_ohlc_blob(ohlc_blob):
        data_mod.register_range_cache(symbol, timeframe, df)
    _warm_numba_jit()


def _screen_engine(spec_overrides, *, confirm: bool = False) -> SimulatorEngine:
    """Simulator for Stage-1 triage (path) or Stage-1.5 confirm (Stage-2 mode)."""
    engine = SimulatorEngine(spec_overrides=spec_overrides or None)
    if confirm:
        # Match Stage-2 realism (no override → SIMULATOR_INTRABAR_MODE).
        engine._intrabar_mode_override = None
    else:
        engine._intrabar_mode_override = str(
            getattr(settings, "SIMULATOR_SCREEN_INTRABAR_MODE", "path"))
    return engine


def _screen_candidate(task: Dict) -> Dict:
    """Stage-1 quick_screen for one candidate (picklable process-pool entry).

    Used by the MT5 lane and the simulator screen-then-validate path to fan
    out cheap triage across cores. Returns a picklable result dict.
    Set ``task["confirm"]=True`` to re-screen with Stage-2 intrabar realism.
    """
    strat = task["strategy"]
    confirm = bool(task.get("confirm"))
    result = {
        "strategy": strat, "fitness": 0.0, "promising": False,
        "error": None, "cancelled": False, "objectives": None,
        "fingerprint": None, "duration_ms": 0.0, "metrics": None,
        "confirm": confirm,
    }
    t0 = time.perf_counter()
    try:
        cancel_check = _pool_cancel_check(task["db_path"], task["job_id"])
        engine = _screen_engine(task.get("spec_overrides"), confirm=confirm)
        engine._cancel_check = cancel_check
        if cancel_check():
            result["cancelled"] = True
            return result
        ohlc = None
        spec = None
        try:
            from factory.backtest.simulator import SymbolSpec
            sym = (task.get("sweep_symbol") or getattr(strat, "symbol", None)
                   or "EURUSD")
            tf = (task.get("sweep_timeframe")
                  or getattr(strat, "timeframe", None) or "M15")
            ohlc = data_mod.get_range_cache(
                sym, tf, task["start"], task["end"])
            if ohlc is not None and len(ohlc) > 0:
                spec = SymbolSpec.infer(
                    float(ohlc["close"].iloc[-1]),
                    task.get("spec_overrides"),
                    symbol=str(sym),
                )
        except Exception:
            ohlc, spec = None, None
        promising, m = quick_screen(
            engine, strat, task["start"], task["end"], task["deposit"],
            task["criteria"], ohlc=ohlc, spec=spec)
        result["fitness"] = screen_fitness(m)
        result["objectives"] = objectives_from_metrics(m)
        from factory.correlation import daily_returns
        result["fingerprint"] = daily_returns(m)
        result["promising"] = bool(promising)
        result["metrics"] = m
    except JobCancelled:
        result["cancelled"] = True
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return result


def _screen_confirm_enabled() -> bool:
    if not bool(getattr(settings, "SIMULATOR_SCREEN_CONFIRM", True)):
        return False
    screen_mode = str(getattr(settings, "SIMULATOR_SCREEN_INTRABAR_MODE", "path"))
    validate_mode = str(getattr(settings, "SIMULATOR_INTRABAR_MODE", "m1"))
    return screen_mode != validate_mode


def _confirm_promising_batch(
    pool: ProcessPoolExecutor,
    promising_batch: list,
    *,
    start, end, deposit, screen_criteria, spec_overrides, db_path, job_id,
    cancelled_now: Callable[[], bool],
    archive_screen: Optional[Callable] = None,
    as_strategies: bool = False,
) -> list:
    """Re-screen path-triaged candidates with Stage-2 intrabar realism.

    ``promising_batch`` may be Stage-1 result dicts (simulator lane) or bare
    ``StrategyDefinition`` objects (MT5 lane when ``as_strategies=True``).
    """
    if not promising_batch or not _screen_confirm_enabled():
        return promising_batch

    def _strat_of(item):
        return item if as_strategies else item["strategy"]

    tasks = [
        {
            "strategy": _strat_of(item), "start": start, "end": end,
            "deposit": deposit, "criteria": screen_criteria,
            "spec_overrides": spec_overrides,
            "db_path": db_path, "job_id": job_id, "confirm": True,
        }
        for item in promising_batch
    ]
    futures = [pool.submit(_screen_candidate, t) for t in tasks]
    by_id = {_strat_of(item).id: item for item in promising_batch}
    confirmed: list = []
    try:
        for fut in as_completed(futures):
            if cancelled_now():
                for f in futures:
                    f.cancel()
                break
            try:
                res = fut.result()
            except Exception as exc:
                log.warning("Stage-1.5 confirm screen failed: %s", exc)
                continue
            if res.get("cancelled"):
                for f in futures:
                    f.cancel()
                break
            if res.get("error"):
                log.warning("Stage-1.5 confirm error: %s", res["error"])
                continue
            if archive_screen is not None:
                archive_screen(
                    res["strategy"],
                    promising=bool(res.get("promising")),
                    metrics=res.get("metrics"),
                    duration_ms=float(res.get("duration_ms") or 0.0),
                    fitness=float(res.get("fitness") or 0.0),
                    error=res.get("error"),
                )
            if not res.get("promising"):
                continue
            prior = by_id.get(res["strategy"].id)
            if prior is None:
                continue
            if as_strategies:
                confirmed.append(res["strategy"])
            else:
                # Keep Stage-1 fitness/objectives; only gate on confirm pass.
                confirmed.append(prior)
    finally:
        for fut in futures:
            fut.cancel()
    return confirmed


def _pool_cancel_check(db_path, job_id: str) -> Callable[[], bool]:
    """A throttled DB cancel probe for a pool worker process.

    Each worker opens its own short-lived SQLite connections; caching the
    answer for a fraction of a second keeps the deep validation loops cheap
    while still aborting the in-flight backtest within ~0.25s of a click.
    After ``_CANCEL_PROBE_FAIL_LIMIT`` consecutive DB errors, fail closed
    (treat as cancelled) so a wedged DB cannot strand a worker forever.
    """
    storage = Storage(Path(db_path))
    state = {"ts": 0.0, "cancelled": False, "fails": 0}

    def _check() -> bool:
        if state["cancelled"]:
            return True
        now = time.monotonic()
        if now - state["ts"] >= _CANCEL_POLL_INTERVAL:
            state["ts"] = now
            try:
                state["cancelled"] = storage.is_cancel_requested(job_id)
                state["fails"] = 0
            except Exception as exc:
                state["fails"] += 1
                log.warning(
                    "cancel probe failed (%d/%d) for %s: %s",
                    state["fails"], _CANCEL_PROBE_FAIL_LIMIT, job_id, exc)
                if state["fails"] >= _CANCEL_PROBE_FAIL_LIMIT:
                    state["cancelled"] = True
        return state["cancelled"]

    return _check


def _validate_candidate(task: Dict) -> Dict:
    """Stage-2 full validation for a promising candidate (pool entry)."""
    strat = task["strategy"]
    start, end, deposit = task["start"], task["end"], task["deposit"]
    criteria = task.get("gate_criteria") or task["criteria"]
    spec_overrides = task.get("spec_overrides") or None
    storage = Storage(Path(task["db_path"]))
    cancel_check = _pool_cancel_check(task["db_path"], task["job_id"])

    engine = SimulatorEngine(spec_overrides=spec_overrides)
    engine._cancel_check = cancel_check
    result = {"strategy": strat, "fitness": float(task.get("fitness", 0.0)),
              "promising": True, "passed": False, "error": None,
              "cancelled": False, "objectives": task.get("objectives"),
              "fingerprint": task.get("fingerprint"),
              "highest_level_passed": 0}
    try:
        report = validate_strategy(
            engine, strat, start, end, deposit=deposit, seed=task["seed"],
            criteria=criteria,
            run_montecarlo=task["run_mc"],
            mc_config=task["mc_config"], wfo_train_months=task["wfo_train"],
            wfo_test_months=task["wfo_test"], wfo_windows=task["wfo_n"],
            data_source=task["data_source"], cancel_check=cancel_check,
            spec_overrides=spec_overrides,
            n_trials=int(task.get("n_trials", 1)),
            floor_level=task.get("floor_level"),
            ceiling_level=task.get("ceiling_level"))
        result["passed"] = report.passed
        result["best_params"] = dict(report.best_params or {})
        result["highest_level_passed"] = int(
            getattr(report, "highest_level_passed", 0) or 0)
        result["infra_failure"] = bool(getattr(report, "infra_failure", False))
        # Pool workers skip MT5 confirm (exclusive tester lock); main process
        # re-confirms survivors after the batch joins.
        if os.environ.get("EA_DISCOVERY_POOL_WORKER") != "1":
            report = _maybe_mt5_confirm_survivor(
                strat, report, start=start, end=end, deposit=deposit,
                enabled=bool(task.get("mt5_confirm_survivors", True)),
            )
        result["mt5_confirmed"] = getattr(report, "mt5_confirmed", None)
        storage.save_complete(
            strat,
            report,
            job_id=task["job_id"],
            metadata=_candidate_metadata(
                strat, task,
                parameter_snapshot=report.best_params,
                sweep_symbol=task.get("sweep_symbol"),
                sweep_timeframe=task.get("sweep_timeframe"),
                strictness_profile=task.get("strictness_profile"),
                seed=task.get("seed"),
            ),
        )
        try:
            results_archive.write_candidate(
                task["job_id"], strat, report,
                metadata=_candidate_metadata(
                    strat, task,
                    parameter_snapshot=report.best_params,
                    sweep_symbol=task.get("sweep_symbol"),
                    sweep_timeframe=task.get("sweep_timeframe"),
                    strictness_profile=task.get("strictness_profile"),
                    seed=task.get("seed"),
                ),
            )
        except Exception as exc:
            log.warning(
                "results archive write failed for %s/%s: %s",
                task["job_id"], strat.id, exc)
        return result
    except JobCancelled:
        result["cancelled"] = True
        return result
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        report = _aborted_validation_report(strat, "simulator", exc)
        result["infra_failure"] = bool(getattr(report, "infra_failure", False))
        meta = _candidate_metadata(
            strat, task,
            parameter_snapshot={},
            sweep_symbol=task.get("sweep_symbol"),
            sweep_timeframe=task.get("sweep_timeframe"),
            strictness_profile=task.get("strictness_profile"),
            seed=task.get("seed"),
        )
        storage.save_complete(
            strat,
            report,
            job_id=task["job_id"],
            metadata=meta,
        )
        try:
            results_archive.write_candidate(
                task["job_id"], strat, report, metadata=meta)
        except Exception as archive_exc:
            log.warning(
                "results archive write failed for %s/%s: %s",
                task["job_id"], strat.id, archive_exc)
        return result


def _evaluate_candidate(task: Dict) -> Dict:
    """Screen one candidate and, if promising, fully validate it.

    Kept for callers/tests that still use the combined entry point. The live
    simulator discovery path now screens then validates in two pool phases so
    ``tested`` advances after cheap triage instead of after full Optuna/WFO.
    """
    screen = _screen_candidate(task)
    if screen.get("cancelled") or screen.get("error") or not screen.get("promising"):
        return {
            "strategy": screen["strategy"], "fitness": screen["fitness"],
            "promising": bool(screen.get("promising")), "passed": False,
            "error": screen.get("error"), "cancelled": bool(screen.get("cancelled")),
            "objectives": screen.get("objectives"),
            "fingerprint": screen.get("fingerprint"),
            "highest_level_passed": 0,
        }
    task = dict(task)
    task["fitness"] = screen["fitness"]
    task["objectives"] = screen.get("objectives")
    task["fingerprint"] = screen.get("fingerprint")
    return _validate_candidate(task)


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
            try:
                results_archive.finalize_run(job_id, status="FAILED")
            except Exception:
                pass
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
        full-range *simulator* run. Obvious losers are discarded immediately,
        so thousands of parameter combinations can be triaged quickly.

        Stage 2 (full validation): only promising candidates run the expensive
        IS/OOS + walk-forward + Monte Carlo pipeline — on the selected engine.
        When ``engine=mt5``, Stage 1 still uses the simulator; Stage 2 is the
        only place the real Strategy Tester runs. Every candidate that
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
        continuous = bool(payload.get("continuous", False))
        if continuous:
            # Continuous agent: never stop a sweep early because enough
            # survivors were found — rotate only when the candidate budget
            # is exhausted (then the orchestrator starts the next sweep).
            target_survivors = max(target_survivors, 10**9)

        use_genetic = bool(payload.get("genetic", True))
        edge_first = bool(payload.get("edge_first", True))
        allowed_mechanics = _mechanics_from_payload(payload)
        allowed_tm_features = _tm_features_from_payload(payload)
        generation_settings = _generation_settings_from_payload(payload)
        max_edge_variants = int(payload.get(
            "max_edge_variants",
            getattr(settings, "DISCOVERY_MAX_EDGE_VARIANTS", 8)))

        start = datetime.fromisoformat(payload["start"]) if payload.get("start") \
            else datetime.now(timezone.utc) - timedelta(days=365)
        end = datetime.fromisoformat(payload["end"]) if payload.get("end") \
            else datetime.now(timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        # Untouched holdout: discovery may never see the reserved trailing
        # window (factory/holdout.py). Clamp the end, then re-anchor start so
        # the requested duration still fits *before* the boundary. Without
        # that rebase, months≈HOLDOUT_MONTHS collapses to a few hours and
        # every Stage-2 OOS window is empty (0 survivors forever).
        from factory.holdout import clamp_discovery_end
        original_span = end - start
        end, holdout_clamped = clamp_discovery_end(end)
        if holdout_clamped:
            months = payload.get("test_duration_months")
            if months is not None:
                start = end - timedelta(
                    days=max(1, int(months)) * settings.DAYS_PER_MONTH)
            else:
                start = end - original_span
            payload["holdout_clamped_end"] = end.isoformat()
            payload["holdout_adjusted_start"] = start.isoformat()
        min_span = timedelta(days=14)
        if end <= start or (end - start) < min_span:
            storage.set_job_status(
                job_id, JobStatus.FAILED,
                error="Discovery range is too short after holdout clamp "
                      f"({start.date()} → {end.date()}). Increase test "
                      "duration, lower HOLDOUT_MONTHS, or disable holdout.")
            return

        # Validation gates: named levels use population tiering (floor/ceiling).
        # Custom criteria keep legacy boolean gating.
        floor_level = None
        ceiling_level = None
        if payload.get("criteria"):
            criteria = AcceptanceCriteria(**payload["criteria"])
            run_mc = payload.get("montecarlo")       # None -> settings default
            mc_config = None
            if payload.get("mc_runs"):
                mc_config = MonteCarloConfig(n_runs=int(payload["mc_runs"]))
            # Loose screen still uses the custom criteria as provided.
            screen_criteria = criteria
        elif payload.get("validation_level") is not None:
            floor_level = int(payload.get(
                "validation_level_floor", payload["validation_level"]))
            # Always score L1–L16; UI dial is survivor floor / progressive target.
            ceiling_level = validation_levels.MAX_LEVEL
            floor_level = max(
                validation_levels.MIN_LEVEL,
                min(validation_levels.MAX_LEVEL, floor_level))
            floor_def = validation_levels.get_level(floor_level)
            ceil_def = validation_levels.get_level(ceiling_level)
            criteria = floor_def.criteria
            # Screen with L1 so high floors don't starve the promising funnel.
            screen_criteria = validation_levels.get_level(1).criteria
            # MC once at Elite·B depth; re-scored against each MC-gated level.
            run_mc = True
            mc_config = validation_levels.mc_config_for(ceil_def)
            if mc_config is None:
                run_mc = False
        else:
            criteria = default_criteria()
            screen_criteria = criteria
            run_mc = payload.get("montecarlo")
            mc_config = None

        wfo_train = int(payload.get("wfo_train_months", settings.WFO_TRAIN_MONTHS))
        wfo_test = int(payload.get("wfo_test_months", settings.WFO_TEST_MONTHS))
        wfo_n = int(payload.get("wfo_windows", settings.WFO_WINDOWS))

        def _phase(message: str, *, frac: float = 0.0) -> None:
            """Immediate UI-visible startup status (bypasses progress throttle)."""
            storage.update_job_progress(
                job_id, min(max(frac, 0.0), 0.05), message,
                tested=0, promising=0, survivors=0, generation=0)

        _phase(f"Loading {symbol} {timeframe} bars…")
        full_ohlc = data_mod.load_ohlc(symbol, timeframe, start, end)
        data_mod.register_range_cache(symbol, timeframe, full_ohlc)
        data_source = str(payload.get("data_source") or full_ohlc.attrs.get("source", "unknown"))
        n_bars = len(full_ohlc)
        src = full_ohlc.attrs.get("source", data_source)
        _phase(f"Loaded {n_bars:,} {symbol} {timeframe} bars ({src})")

        ohlc_frames = [(symbol, timeframe, full_ohlc)]
        # Preload M1 once for the whole job when full validation uses m1 mode.
        # Without this, every Optuna/WFO trial re-queries MT5 for the same range
        # (and on miss, re-pays the initialize/shutdown cost every time).
        intrabar_mode = str(getattr(settings, "SIMULATOR_INTRABAR_MODE", "path"))
        if (intrabar_mode == "m1" and timeframe != "M1"
                and src not in (None, "synthetic")
                and engine_name != "mt5"):
            _phase(f"Preloading {symbol} M1 intrabar data…", frac=0.01)
            try:
                m1_df = data_mod.load_ohlc(
                    symbol, "M1", start, end, allow_synthetic=False)
                data_mod.register_range_cache(symbol, "M1", m1_df)
                ohlc_frames.append((symbol, "M1", m1_df))
                _phase(
                    f"M1 ready — {len(m1_df):,} bars "
                    f"({m1_df.attrs.get('source', 'cache')})",
                    frac=0.02)
            except Exception:
                data_mod.mark_unavailable(symbol, "M1", start, end)
                _phase(
                    f"M1 unavailable — using path heuristic for intrabar",
                    frac=0.02)

        ohlc_blob = _pack_ohlc_blob(ohlc_frames)

        # Persist the reproducibility manifest before any candidate runs, so
        # even a crashed/cancelled run documents exactly what it saw.
        manifest_body = None
        try:
            from factory.manifest import build_manifest
            manifest_body = build_manifest(job_id, payload, seed, full_ohlc)
            storage.save_run_manifest(manifest_body)
        except Exception as exc:
            log.warning("manifest save failed for %s: %s", job_id, exc)

        # Filesystem results archive (config + levels + timing + every test).
        job_started_at = time.time()
        archive_timing = {"screen_ms": 0.0, "validate_ms": 0.0}
        try:
            results_archive.init_run(
                job_id, payload, manifest=manifest_body,
                started_at=job_started_at)
        except Exception as exc:
            log.warning("results archive init failed for %s: %s", job_id, exc)

        def _archive_screen(strat, *, promising: bool, metrics=None,
                            duration_ms: float = 0.0, fitness: float = 0.0,
                            error: Optional[str] = None) -> None:
            archive_timing["screen_ms"] += float(duration_ms or 0.0)
            try:
                results_archive.append_screen(
                    job_id,
                    results_archive.screen_record(
                        strat, promising=promising, metrics=metrics,
                        duration_ms=duration_ms, fitness=fitness,
                        error=error, generation=generation),
                )
            except Exception as exc:
                log.warning("screen archive append failed for %s: %s", job_id, exc)

        def _note_validate_ms(report) -> None:
            try:
                archive_timing["validate_ms"] += float(
                    getattr(report, "duration_ms", 0.0) or 0.0)
            except Exception as exc:
                log.warning("validate timing note failed: %s", exc)

        rng = random.Random(seed)
        engine = self._make_engine(engine_name)
        _apply_spec_overrides(engine, spec_overrides)
        cancel_check = self._make_cancel_check(job_id)
        # Let the engine abort a long backtest mid-loop, not just between
        # candidates — the simulator polls this probe inside its bar loop.
        if isinstance(engine, SimulatorEngine):
            engine._cancel_check = cancel_check
        # MT5 is expensive: always pre-filter with the simulator so only
        # promising candidates ever touch the Strategy Tester.
        screen_engine = None
        mt5_fallback_note = ""
        if engine_name == "mt5":
            # Interactive terminal owns the shared data directory — Stage-2
            # would abort empty. Prefer falling back to the simulator so the
            # sweep still scores real quality instead of INFRA spam.
            terminal_busy = False
            try:
                terminal_busy = bool(interactive_terminal_running())
            except Exception:
                terminal_busy = False
            fallback = bool(getattr(
                settings, "DISCOVERY_MT5_FALLBACK_TO_SIMULATOR", True))
            if terminal_busy and fallback:
                log.warning(
                    "MT5 interactive terminal open — falling back to simulator "
                    "for job %s", job_id)
                engine_name = "simulator"
                engine = self._make_engine("simulator")
                _apply_spec_overrides(engine, spec_overrides)
                if isinstance(engine, SimulatorEngine):
                    engine._cancel_check = cancel_check
                mt5_fallback_note = (
                    "MT5 terminal open — using simulator for Stage-2"
                )
            elif terminal_busy:
                storage.set_job_status(
                    job_id, JobStatus.FAILED,
                    error="INFRA: MetaTrader 5 terminal is already running "
                          "interactively. Close MT5 and retry — headless "
                          "tester runs need exclusive use of the terminal "
                          "data directory.")
                return
            else:
                screen_engine = self._make_engine("simulator")
                _apply_spec_overrides(screen_engine, spec_overrides)
                if isinstance(screen_engine, SimulatorEngine):
                    screen_engine._intrabar_mode_override = str(
                        getattr(settings, "SIMULATOR_SCREEN_INTRABAR_MODE", "path"))
                    screen_engine._cancel_check = cancel_check
                elif hasattr(screen_engine, "_cancel_check"):
                    screen_engine._cancel_check = cancel_check

        tested = 0            # candidates that ran the fast screen
        screened_in = 0       # candidates promising enough for full validation
        survivors = 0         # strategies passing all gates
        edges_found = 0       # validated signal edges (edge-first mode)
        errors = 0
        infra_aborts = 0      # Stage-2 incomplete (not a strategy quality fail)
        scored: list = []     # (strategy, screen_fitness, objectives) triples
        # Strategy ids whose Stage-2 outcome was an infra abort (empty trades).
        # Excluded from genetic parents so incomplete MT5 runs don't breed.
        infra_abort_ids: set = set()
        # Execution variants waiting to be screened after an edge qualifies.
        expansion_queue: list = []

        # Bias fresh randoms toward mechanics / families / filters that
        # historically clear L4+. In edge-first mode random generation stays
        # on STANDARD_SLTP probes; mechanic weights still shape expansion.
        try:
            _mech_weights = blend_mechanic_weights(
                self.storage.mechanic_clear_counts(min_level=4))
        except Exception:
            _mech_weights = None
        try:
            _family_weights = blend_family_weights(
                self.storage.family_clear_counts(min_level=4))
        except Exception:
            _family_weights = None
        try:
            _filter_weights = blend_filter_weights(
                self.storage.filter_clear_counts(min_level=4))
        except Exception:
            _filter_weights = None
        # Inject empirical priors into generation settings (frozen dataclass).
        if _family_weights or _filter_weights:
            generation_settings = GenerationSettings(
                advanced_mode=generation_settings.advanced_mode,
                complexity_cap=generation_settings.complexity_cap,
                enable_regime_switching=generation_settings.enable_regime_switching,
                enable_mtf_context=generation_settings.enable_mtf_context,
                feature_toggles=generation_settings.feature_toggles,
                hypothesis_families=generation_settings.hypothesis_families,
                family_weights=_family_weights,
                filter_weights=_filter_weights,
            )
        _elite_seed_n = int(getattr(settings, "DISCOVERY_ELITE_SEED_COUNT", 8))
        _elite_fetch = max(1, _elite_seed_n * 3)
        try:
            _elite_seeds = self.storage.list_cleared_strategies(
                symbol=symbol, timeframe=timeframe, min_level=4,
                limit=_elite_fetch,
            )
            # Same symbol-class transfer when exact symbol+TF elites are thin.
            if len(_elite_seeds) < _elite_seed_n:
                seen_ids = {s.id for s in _elite_seeds}
                try:
                    class_peers = self.storage.list_cleared_strategies(
                        timeframe=timeframe,
                        symbol_class=classify_symbol(symbol).value,
                        min_level=4,
                        limit=_elite_fetch,
                    )
                except Exception:
                    class_peers = []
                for s in class_peers:
                    if s.id in seen_ids:
                        continue
                    _elite_seeds.append(s)
                    seen_ids.add(s.id)
                    if len(_elite_seeds) >= _elite_fetch:
                        break
            if edge_first:
                # Prefer prior edge probes as elite seeds (not DCA/grid EAs).
                _elite_seeds = [
                    s for s in _elite_seeds
                    if is_edge_probe(s) and not is_execution_variant(s)
                ] or _elite_seeds
        except Exception:
            _elite_seeds = []

        def _queue_edge_expansions(strat, report) -> None:
            """After a validated edge, enqueue mechanic/TM EA variants."""
            nonlocal edges_found
            if not edge_first or report is None or not report.passed:
                return
            if is_execution_variant(strat) or not is_edge_probe(strat):
                return
            edges_found += 1
            try:
                variants = expand_execution_variants(
                    strat,
                    mechanics=allowed_mechanics,
                    tm_features=allowed_tm_features,
                    rng=rng,
                    max_variants=max(1, max_edge_variants),
                )
            except Exception as exc:
                log.warning("edge expansion failed for %s: %s", strat.id, exc)
                return
            for v in variants:
                try:
                    v = apply_edge_best_params(v, strat, report.best_params)
                except Exception:
                    pass
                expansion_queue.append(v)

        # Behavioral-novelty reservoir: recent candidates' daily-return
        # fingerprints. Novelty (1 - max|corr| vs the reservoir) is appended
        # as an extra NSGA-II objective so the search explores new behaviors
        # instead of rediscovering the same edge under different indicators.
        from factory.correlation import daily_returns as _fp_of, novelty_score
        fingerprints: list = []
        _novelty_cap = int(getattr(settings, "NOVELTY_RESERVOIR", 200))
        _novelty_on = bool(getattr(settings, "NOVELTY_ENABLED", True))

        def _with_novelty(objectives, fingerprint):
            """Append the novelty objective and grow the reservoir."""
            if objectives is None:
                return None
            nov = (novelty_score(fingerprint or {}, fingerprints)
                   if _novelty_on else 1.0)
            if fingerprint:
                fingerprints.append(fingerprint)
                del fingerprints[:-_novelty_cap]
            return tuple(objectives) + (nov,)
        generation = 0
        persist_state = {"ts": 0.0, "tested": 0, "promising": 0, "survivors": 0}
        phase_note = {"text": ""}

        def _progress_msg() -> str:
            tier = ""
            if floor_level is not None:
                tier = f" · floor L{floor_level}/ceil L{ceiling_level}"
            if continuous:
                surv = f"survivors {survivors}"
                mode_note = " · continuous"
            else:
                surv = f"survivors {survivors}/{target_survivors}"
                mode_note = ""
            infra_note = f" · infra {infra_aborts}" if infra_aborts else ""
            fallback = f" · {mt5_fallback_note}" if mt5_fallback_note else ""
            edge_note = ""
            if edge_first:
                pending = len(expansion_queue)
                edge_note = (
                    f" · edges {edges_found}"
                    + (f" · EA queue {pending}" if pending else "")
                )
            base = (f"tested {tested}/{max_candidates} · promising {screened_in}"
                    f" · {surv}{edge_note}{infra_note} · gen {generation}"
                    f"{tier}{mode_note}{fallback}")
            if phase_note["text"]:
                return f"{phase_note['text']} — {base}"
            return base

        def _note_infra(report: ValidationReport, strat_id: str) -> None:
            nonlocal infra_aborts
            if getattr(report, "infra_failure", False):
                infra_aborts += 1
                infra_abort_ids.add(strat_id)

        def _persist_progress(*, force: bool = False,
                              note: Optional[str] = None) -> None:
            """Write live counters for the UI without hammering SQLite."""
            if note is not None:
                phase_note["text"] = note
            now = time.monotonic()
            counters_changed = (
                tested != persist_state["tested"]
                or screened_in != persist_state["promising"]
                or survivors != persist_state["survivors"]
            )
            # Always surface the first few counter ticks; then rate-limit ~2.5Hz.
            if not force:
                if not counters_changed and now - persist_state["ts"] < 1.0:
                    return
                min_gap = 0.25 if tested < 20 else 0.4
                if counters_changed and now - persist_state["ts"] < min_gap:
                    return
            persist_state["ts"] = now
            persist_state["tested"] = tested
            persist_state["promising"] = screened_in
            persist_state["survivors"] = survivors
            if continuous:
                frac = tested / max(max_candidates, 1)
            else:
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
            try:
                results_archive.finalize_run(
                    job_id, status="CANCELLED",
                    tested=tested, promising=screened_in,
                    survivors=survivors, generation=generation,
                    screen_ms_total=archive_timing["screen_ms"],
                    validate_ms_total=archive_timing["validate_ms"])
            except Exception:
                pass

        def _build_population(this_gen: int):
            # Prefer draining gated execution variants (validated edges only).
            pop: list = []
            while expansion_queue and len(pop) < this_gen:
                pop.append(expansion_queue.pop(0))
            remaining = this_gen - len(pop)
            if remaining <= 0:
                return pop

            search_phase = "edge" if edge_first else None
            # Edge search always probes STANDARD_SLTP; user mechanics are for
            # post-edge expansion only.
            gen_mechanics = (
                [ExecutionMechanicType.STANDARD_SLTP]
                if edge_first else allowed_mechanics
            )

            # Drop infra-aborted Stage-2 candidates (empty trades) from breeding.
            # Edge-first: breed only signal probes so genetics optimize entries.
            breedable = [
                (s, f, obj) for s, f, obj in scored
                if s.id not in infra_abort_ids
                and (not edge_first or not is_execution_variant(s))
            ]
            if generation == 0 or not (use_genetic and breedable):
                pop.extend([
                    random_strategy(
                        symbol, timeframe, rng,
                        allowed_mechanics=gen_mechanics,
                        allowed_tm_features=allowed_tm_features,
                        generation_settings=generation_settings,
                        mechanic_weights=_mech_weights,
                        search_phase=search_phase,
                    )
                    for _ in range(remaining)
                ])
            else:
                if getattr(settings, "PARETO_EVOLUTION", True):
                    # NSGA-II over the most recent candidates with objective
                    # vectors; keeps the population spread along the whole
                    # profit/risk/stability front instead of one scalar peak.
                    recent = [(s, obj) for s, _f, obj in breedable[-500:]
                              if obj is not None]
                    if len(recent) >= 2:
                        evolved = evolve_pareto(recent, remaining, rng)
                    else:
                        top = sorted(breedable, key=lambda t: t[1], reverse=True)[:50]
                        evolved = evolve([(s, f) for s, f, _o in top], remaining, rng)
                else:
                    top = sorted(breedable, key=lambda t: t[1], reverse=True)[:50]
                    evolved = evolve([(s, f) for s, f, _o in top], remaining, rng)
                # Adaptive fresh blood: explore more when parents are thin or
                # the job has not found a survivor yet; else exploit harder.
                if len(breedable) < 20 or survivors == 0:
                    n_fresh = max(1, remaining // 2)   # ~50%
                else:
                    n_fresh = max(1, remaining // 4)   # ~25%
                evolved[:n_fresh] = [
                    random_strategy(
                        symbol, timeframe, rng,
                        allowed_mechanics=gen_mechanics,
                        allowed_tm_features=allowed_tm_features,
                        generation_settings=generation_settings,
                        mechanic_weights=_mech_weights,
                        search_phase=search_phase,
                    )
                    for _ in range(n_fresh)
                ]
                pop.extend(evolved)
            # Seed a few slots with mutants of prior L4+ survivors so search
            # spends time near known-working structure instead of pure noise.
            if _elite_seeds and _elite_seed_n > 0 and remaining > 0:
                n_seed = min(_elite_seed_n, remaining, len(_elite_seeds))
                for i in range(n_seed):
                    parent = _elite_seeds[i % len(_elite_seeds)]
                    child = mutate(parent, rng, rate=0.35)
                    child.symbol = symbol
                    child.timeframe = timeframe
                    child.lineage.generation = generation
                    if edge_first:
                        child.lineage.role = child.lineage.role or "edge"
                        child.profile.search_phase = (
                            child.profile.search_phase or "edge")
                    pop[-(i + 1)] = child
            return pop

        # Simulator work fans out across a process pool (parallel, off the UI
        # thread). MT5 and non-simulator engines stay sequential.
        #
        # Test suites monkeypatch ``_make_engine`` with in-process stubs that
        # track call counts / injected failures. Those stubs are not picklable
        # across process boundaries, so parallel pool mode must only be used
        # for the real SimulatorEngine.
        pool = None
        pool_workers = 0
        if engine_name != "mt5" and isinstance(engine, SimulatorEngine):
            pool_workers = _discovery_pool_size(gen_size)
            _phase(
                f"Starting worker pool ({pool_workers} processes)…",
                frac=0.03)
            _hide_pool_console_on_windows()
            pool = ProcessPoolExecutor(
                max_workers=pool_workers,
                initializer=_pool_worker_init,
                initargs=(ohlc_blob,),
            )

        def _make_task(strat) -> Dict:
            return {
                "strategy": strat, "start": start, "end": end,
                "deposit": deposit, "criteria": screen_criteria,
                "gate_criteria": criteria,
                "run_mc": run_mc,
                "mc_config": mc_config, "wfo_train": wfo_train,
                "wfo_test": wfo_test, "wfo_n": wfo_n, "data_source": data_source,
                "spec_overrides": spec_overrides, "seed": rng.randint(0, 2**31),
                "n_trials": tested + 1,
                "db_path": str(self.storage.db_path), "job_id": job_id,
                "sweep_symbol": payload.get("symbol"),
                "sweep_timeframe": payload.get("timeframe"),
                "strictness_profile": payload.get("strictness_profile", "normal"),
                "floor_level": floor_level,
                "ceiling_level": ceiling_level,
                "mt5_confirm_survivors": bool(
                    payload.get("mt5_confirm_survivors",
                                getattr(settings, "DISCOVERY_MT5_CONFIRM_SURVIVORS", True))),
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
                    # -- MT5 pool: sim-screen first, then validate in parallel --
                    # Each MT5 task leases one portable instance for its whole
                    # validation (IS opt + OOS + WFO), so at most pool.size
                    # testers run concurrently and no instance is shared.
                    def _candidate_meta(strat, report) -> Dict:
                        return _candidate_metadata(
                            strat, payload,
                            parameter_snapshot=report.best_params,
                        )

                    def _mt5_one(strat, seed_val: int, trial_no: int):
                        def _validate():
                            with self._mt5_engine_lease(
                                    engine, spec_overrides,
                                    cancel_check) as m5:
                                return validate_strategy(
                                    m5, strat, start, end, deposit=deposit,
                                    seed=seed_val, criteria=criteria,
                                    run_montecarlo=run_mc, mc_config=mc_config,
                                    wfo_train_months=wfo_train,
                                    wfo_test_months=wfo_test,
                                    wfo_windows=wfo_n,
                                    data_source=data_source,
                                    cancel_check=cancel_check,
                                    spec_overrides=spec_overrides,
                                    n_trials=trial_no,
                                    floor_level=floor_level,
                                    ceiling_level=ceiling_level)

                        try:
                            rep = _run_mt5_validate_with_retry(_validate)
                            return "ok", rep, None
                        except JobCancelled:
                            return "cancelled", None, None
                        except Exception as exc:  # noqa: BLE001
                            return ("error",
                                    _aborted_validation_report(strat, "mt5", exc),
                                    f"{type(exc).__name__}: {exc}")

                    # Stage 1: cheap simulator screen over the whole generation
                    # (process pool — MT5 lane is not already inside a pool).
                    promising_batch: list = []
                    screen_pool = ProcessPoolExecutor(
                        max_workers=_discovery_pool_size(len(population)),
                        initializer=_pool_worker_init,
                        initargs=(ohlc_blob,),
                    )
                    try:
                        screen_tasks = [
                            {
                                "strategy": s, "start": start, "end": end,
                                "deposit": deposit, "criteria": screen_criteria,
                                "spec_overrides": spec_overrides,
                                "db_path": str(self.storage.db_path),
                                "job_id": job_id,
                            }
                            for s in population
                        ]
                        futures = [
                            screen_pool.submit(_screen_candidate, t)
                            for t in screen_tasks
                        ]
                        for fut in as_completed(futures):
                            if _cancelled_now() or survivors >= target_survivors:
                                for f in futures:
                                    f.cancel()
                                break
                            tested += 1
                            try:
                                res = fut.result()
                            except Exception as exc:
                                errors += 1
                                storage.set_job_status(
                                    job_id, JobStatus.RUNNING,
                                    error=f"{type(exc).__name__}: {exc}")
                                _persist_progress()
                                continue
                            if res.get("cancelled"):
                                _finish_cancelled()
                                return
                            if res.get("error"):
                                errors += 1
                                storage.set_job_status(
                                    job_id, JobStatus.RUNNING,
                                    error=res["error"])
                                _persist_progress()
                                continue
                            strat = res["strategy"]
                            scored.append((
                                strat, res["fitness"],
                                _with_novelty(res["objectives"],
                                              res.get("fingerprint"))))
                            _archive_screen(
                                strat,
                                promising=bool(res.get("promising")),
                                metrics=res.get("metrics"),
                                duration_ms=float(res.get("duration_ms") or 0.0),
                                fitness=float(res.get("fitness") or 0.0),
                                error=res.get("error"),
                            )
                            if res["promising"]:
                                screened_in += 1
                                promising_batch.append(strat)
                            _persist_progress()

                        if (promising_batch
                                and not _cancelled_now()
                                and survivors < target_survivors):
                            # Stage-1.5: re-screen with Stage-2 realism.
                            promising_batch = _confirm_promising_batch(
                                screen_pool, promising_batch,
                                start=start, end=end, deposit=deposit,
                                screen_criteria=screen_criteria,
                                spec_overrides=spec_overrides,
                                db_path=str(self.storage.db_path),
                                job_id=job_id,
                                cancelled_now=_cancelled_now,
                                archive_screen=_archive_screen,
                                as_strategies=True,
                            )
                    finally:
                        screen_pool.shutdown(wait=True, cancel_futures=True)

                    if _cancelled_now():
                        _finish_cancelled()
                        return
                    if survivors >= target_survivors or not promising_batch:
                        generation += 1
                        continue

                    # Stage 2: MT5 full validation only for promising candidates.
                    was_cancelled = False
                    tex = ThreadPoolExecutor(
                        max_workers=self._mt5_pool.size,
                        thread_name_prefix="mt5pool")
                    futures = {
                        tex.submit(_mt5_one, s, rng.randint(0, 2**31),
                                   tested - len(promising_batch) + k + 1): s
                        for k, s in enumerate(promising_batch)}
                    try:
                        for fut in as_completed(futures):
                            strat = futures[fut]
                            status, report, err = fut.result()
                            if status == "cancelled":
                                was_cancelled = True
                                continue
                            if err:
                                errors += 1
                                storage.set_job_status(
                                    job_id, JobStatus.RUNNING, error=err)
                            if (report is not None
                                    and getattr(report, "infra_failure", False)):
                                _note_infra(report, strat.id)
                            _persist_complete(storage, strat, report,
                                              job_id=job_id,
                                              metadata=_candidate_meta(strat, report))
                            if report is not None and not report.passed:
                                try:
                                    from factory.agent_alerts import note_mt5_disagreement
                                    reason = (report.reasons[0]
                                              if report.reasons else "MT5 failed")
                                    note_mt5_disagreement(
                                        storage, strat.id, reason)
                                except Exception:
                                    pass
                            if report.passed:
                                survivors += 1
                                _queue_edge_expansions(strat, report)
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
                            # Simulator pre-filter, then MT5 only if promising.
                            t_screen = time.perf_counter()
                            try:
                                promising, m = quick_screen(
                                    screen_engine, strat, start, end, deposit,
                                    screen_criteria)
                            except JobCancelled:
                                _finish_cancelled()
                                return
                            except Exception as exc:
                                errors += 1
                                storage.set_job_status(
                                    job_id, JobStatus.RUNNING,
                                    error=f"{type(exc).__name__}: {exc}")
                                continue
                            screen_ms = round(
                                (time.perf_counter() - t_screen) * 1000.0, 3)
                            fit = screen_fitness(m)
                            scored.append((strat, fit,
                                           _with_novelty(objectives_from_metrics(m),
                                                         _fp_of(m))))
                            _archive_screen(
                                strat, promising=bool(promising), metrics=m,
                                duration_ms=screen_ms, fitness=fit)
                            if not promising:
                                continue
                            if _screen_confirm_enabled():
                                t_confirm = time.perf_counter()
                                try:
                                    confirm_engine = _screen_engine(
                                        spec_overrides, confirm=True)
                                    promising, m = quick_screen(
                                        confirm_engine, strat, start, end,
                                        deposit, screen_criteria)
                                except JobCancelled:
                                    _finish_cancelled()
                                    return
                                confirm_ms = round(
                                    (time.perf_counter() - t_confirm) * 1000.0, 3)
                                _archive_screen(
                                    strat, promising=bool(promising), metrics=m,
                                    duration_ms=confirm_ms,
                                    fitness=screen_fitness(m))
                                if not promising:
                                    continue
                            screened_in += 1
                            try:
                                def _validate_seq():
                                    with self._mt5_engine_lease(
                                            engine, spec_overrides,
                                            cancel_check) as mt5_engine:
                                        return validate_strategy(
                                            mt5_engine, strat, start, end,
                                            deposit=deposit,
                                            seed=rng.randint(0, 2**31),
                                            criteria=criteria,
                                            run_montecarlo=run_mc,
                                            mc_config=mc_config,
                                            wfo_train_months=wfo_train,
                                            wfo_test_months=wfo_test,
                                            wfo_windows=wfo_n,
                                            data_source=data_source,
                                            cancel_check=cancel_check,
                                            spec_overrides=spec_overrides,
                                            n_trials=tested,
                                            floor_level=floor_level,
                                            ceiling_level=ceiling_level)

                                report = _run_mt5_validate_with_retry(_validate_seq)
                            except JobCancelled:
                                _finish_cancelled()
                                return
                            except Exception as exc:
                                errors += 1
                                report = _aborted_validation_report(
                                    strat, engine_label, exc)
                                _note_infra(report, strat.id)
                                _persist_complete(storage, strat, report,
                                                  job_id=job_id,
                                                  metadata=_candidate_metadata(
                                                      strat, payload,
                                                      parameter_snapshot=report.best_params,
                                                  ))
                                storage.set_job_status(
                                    job_id, JobStatus.RUNNING,
                                    error=f"{type(exc).__name__}: {exc}")
                            else:
                                if report.passed:
                                    report = _maybe_mt5_confirm_survivor(
                                        strat, report, start=start, end=end,
                                        deposit=deposit,
                                        enabled=bool(payload.get(
                                            "mt5_confirm_survivors", True)),
                                    )
                                _persist_complete(storage, strat, report,
                                                  job_id=job_id,
                                                  metadata=_candidate_metadata(
                                                      strat, payload,
                                                      parameter_snapshot=report.best_params,
                                                  ))
                            if report.passed:
                                survivors += 1
                                _queue_edge_expansions(strat, report)
                        else:
                            # Non-MT5 sequential fallback mirrors pool semantics:
                            # quick screen -> full validation only for promising.
                            # (Real SimulatorEngine uses the process-pool path above;
                            # this branch is mainly for in-process test stubs.)
                            t_screen = time.perf_counter()
                            try:
                                promising, m = quick_screen(
                                    engine, strat, start, end, deposit,
                                    screen_criteria)
                            except JobCancelled:
                                _finish_cancelled()
                                return
                            except Exception as exc:
                                errors += 1
                                storage.set_job_status(
                                    job_id, JobStatus.RUNNING,
                                    error=f"{type(exc).__name__}: {exc}")
                                continue

                            screen_ms = round(
                                (time.perf_counter() - t_screen) * 1000.0, 3)
                            fit = screen_fitness(m)
                            scored.append((strat, fit,
                                           _with_novelty(objectives_from_metrics(m),
                                                         _fp_of(m))))
                            _archive_screen(
                                strat, promising=bool(promising), metrics=m,
                                duration_ms=screen_ms, fitness=fit)
                            if not promising:
                                _persist_progress(note="screening")
                                continue
                            screened_in += 1
                            _persist_progress(note="validating", force=True)
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
                                    n_trials=tested,
                                    floor_level=floor_level,
                                    ceiling_level=ceiling_level)
                            except JobCancelled:
                                _finish_cancelled()
                                return
                            except Exception as exc:
                                errors += 1
                                report = _aborted_validation_report(
                                    strat, engine_label, exc)
                                _note_infra(report, strat.id)
                                _persist_complete(storage, strat, report,
                                                  job_id=job_id,
                                                  metadata=_candidate_metadata(
                                                      strat, payload,
                                                      parameter_snapshot=report.best_params,
                                                  ))
                                storage.set_job_status(
                                    job_id, JobStatus.RUNNING,
                                    error=f"{type(exc).__name__}: {exc}")
                            else:
                                if report.passed:
                                    report = _maybe_mt5_confirm_survivor(
                                        strat, report, start=start, end=end,
                                        deposit=deposit,
                                        enabled=bool(payload.get(
                                            "mt5_confirm_survivors", True)),
                                    )
                                _persist_complete(storage, strat, report,
                                                  job_id=job_id,
                                                  metadata=_candidate_metadata(
                                                      strat, payload,
                                                      parameter_snapshot=report.best_params,
                                                  ))
                            if report.passed:
                                survivors += 1
                                _queue_edge_expansions(strat, report)
                        _persist_progress()
                else:
                    # -- Simulator: Stage 1 screen (fast) then Stage 2 validate --
                    # Progress advances after each cheap screen so the UI is not
                    # stuck at tested=0 while the first promising candidate runs
                    # Optuna + WFO (+ MC).
                    _persist_progress(
                        note=f"screening gen {generation}", force=True)
                    screen_tasks = [
                        {
                            "strategy": s, "start": start, "end": end,
                            "deposit": deposit, "criteria": screen_criteria,
                            "spec_overrides": spec_overrides,
                            "db_path": str(self.storage.db_path),
                            "job_id": job_id,
                            "sweep_symbol": payload.get("symbol"),
                            "sweep_timeframe": payload.get("timeframe"),
                        }
                        for s in population
                    ]
                    futures = [pool.submit(_screen_candidate, t)
                               for t in screen_tasks]
                    promising_batch: list = []
                    try:
                        for fut in as_completed(futures):
                            try:
                                res = fut.result()
                            except Exception as exc:
                                tested += 1
                                errors += 1
                                storage.set_job_status(
                                    job_id, JobStatus.RUNNING,
                                    error=f"pool: {type(exc).__name__}: {exc}")
                                _persist_progress(note=f"screening gen {generation}")
                                continue
                            tested += 1
                            if res.get("cancelled"):
                                for f in futures:
                                    f.cancel()
                                _finish_cancelled()
                                return
                            if res.get("error"):
                                errors += 1
                                storage.set_job_status(
                                    job_id, JobStatus.RUNNING,
                                    error=res["error"])
                                _persist_progress(note=f"screening gen {generation}")
                                continue
                            scored.append((res["strategy"], res["fitness"],
                                           _with_novelty(res.get("objectives"),
                                                         res.get("fingerprint"))))
                            _archive_screen(
                                res["strategy"],
                                promising=bool(res.get("promising")),
                                metrics=res.get("metrics"),
                                duration_ms=float(res.get("duration_ms") or 0.0),
                                fitness=float(res.get("fitness") or 0.0),
                                error=res.get("error"),
                            )
                            if res["promising"]:
                                screened_in += 1
                                promising_batch.append(res)
                            _persist_progress(note=f"screening gen {generation}")
                            if survivors >= target_survivors or _cancelled_now():
                                break

                        if (promising_batch
                                and not _cancelled_now()
                                and survivors < target_survivors):
                            _persist_progress(
                                note=f"confirming {len(promising_batch)}",
                                force=True)
                            promising_batch = _confirm_promising_batch(
                                pool, promising_batch,
                                start=start, end=end, deposit=deposit,
                                screen_criteria=screen_criteria,
                                spec_overrides=spec_overrides,
                                db_path=str(self.storage.db_path),
                                job_id=job_id,
                                cancelled_now=_cancelled_now,
                                archive_screen=_archive_screen,
                                as_strategies=False,
                            )
                    finally:
                        for fut in futures:
                            fut.cancel()

                    if _cancelled_now():
                        _finish_cancelled()
                        return
                    if survivors >= target_survivors or not promising_batch:
                        generation += 1
                        continue

                    _persist_progress(
                        note=f"validating {len(promising_batch)} promising",
                        force=True)
                    val_tasks = []
                    for i, res in enumerate(promising_batch):
                        task = _make_task(res["strategy"])
                        task["fitness"] = res["fitness"]
                        task["objectives"] = res.get("objectives")
                        task["fingerprint"] = res.get("fingerprint")
                        task["n_trials"] = tested - len(promising_batch) + i + 1
                        val_tasks.append(task)
                    val_futures = [
                        pool.submit(_validate_candidate, t) for t in val_tasks]
                    try:
                        done_val = 0
                        for fut in as_completed(val_futures):
                            done_val += 1
                            try:
                                res = fut.result()
                            except Exception as exc:
                                errors += 1
                                storage.set_job_status(
                                    job_id, JobStatus.RUNNING,
                                    error=f"pool: {type(exc).__name__}: {exc}")
                                _persist_progress(
                                    note=f"validating {done_val}/{len(val_tasks)}")
                                continue
                            if res.get("cancelled"):
                                for f in val_futures:
                                    f.cancel()
                                _finish_cancelled()
                                return
                            if res.get("error"):
                                errors += 1
                                storage.set_job_status(
                                    job_id, JobStatus.RUNNING,
                                    error=res["error"])
                            if res.get("infra_failure"):
                                infra_aborts += 1
                                strat_obj = res.get("strategy")
                                if strat_obj is not None:
                                    infra_abort_ids.add(getattr(
                                        strat_obj, "id", strat_obj))
                            if res.get("passed"):
                                survivors += 1
                                # Main-process MT5 confirm (pool workers skip it).
                                try:
                                    strat_obj = res.get("strategy")
                                    if strat_obj is not None:
                                        vrep = storage.get_validation(strat_obj.id)
                                        if vrep is not None:
                                            vrep = _maybe_mt5_confirm_survivor(
                                                strat_obj, vrep,
                                                start=start, end=end,
                                                deposit=deposit,
                                                enabled=bool(payload.get(
                                                    "mt5_confirm_survivors", True)),
                                            )
                                            storage.save_complete(
                                                strat_obj, vrep, job_id=job_id,
                                                metadata=_candidate_metadata(
                                                    strat_obj, payload,
                                                    parameter_snapshot=vrep.best_params,
                                                ),
                                            )
                                            _queue_edge_expansions(strat_obj, vrep)
                                        else:
                                            stub = ValidationReport(
                                                strategy_id=strat_obj.id,
                                                is_metrics=BacktestMetrics(),
                                                oos_metrics=BacktestMetrics(),
                                                passed=True,
                                                best_params=dict(
                                                    res.get("best_params") or {}),
                                            )
                                            _queue_edge_expansions(strat_obj, stub)
                                except Exception as exc:
                                    log.info("post-pool MT5 confirm skipped: %s", exc)
                                    # Still expand EAs from the validated edge.
                                    try:
                                        strat_obj = res.get("strategy")
                                        if strat_obj is not None:
                                            stub = ValidationReport(
                                                strategy_id=strat_obj.id,
                                                is_metrics=BacktestMetrics(),
                                                oos_metrics=BacktestMetrics(),
                                                passed=True,
                                                best_params=dict(
                                                    res.get("best_params") or {}),
                                            )
                                            _queue_edge_expansions(strat_obj, stub)
                                    except Exception:
                                        pass
                            _persist_progress(
                                note=f"validating {done_val}/{len(val_tasks)}")
                            if survivors >= target_survivors or _cancelled_now():
                                break
                    finally:
                        for fut in val_futures:
                            fut.cancel()

                    if _cancelled_now():
                        _finish_cancelled()
                        return

                    generation += 1
                    continue

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
        try:
            results_archive.finalize_run(
                job_id, status="DONE",
                tested=tested, promising=screened_in,
                survivors=survivors, generation=generation,
                screen_ms_total=archive_timing["screen_ms"],
                validate_ms_total=archive_timing["validate_ms"])
        except Exception as exc:
            log.warning("results archive finalize failed for %s: %s", job_id, exc)


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
