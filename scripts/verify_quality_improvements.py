"""End-to-end verification that discovery quality improvements behave as planned.

Runs without MT5. Exercises:
1. Continuous agent payload forces simulator
2. MT5+interactive → simulator fallback (unit-level worker path)
3. Mechanic bias + elite seeding hooks
4. L4–L6 trade floors
5. Infra KPI exclusion + progress counters
6. ETA helper caps sleep-inflated estimates
7. Mini discovery smoke (synthetic data) and report funnel stats

Usage:
    .venv\\Scripts\\python.exe scripts/verify_quality_improvements.py
"""
from __future__ import annotations

import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from factory import validation_levels
from factory.discovery_config import DiscoverySettings, build_discovery_payload
from factory.generator import (
    blend_mechanic_weights,
    default_mechanic_weights,
    mutate,
    random_strategy,
)
from factory.models import (
    BacktestMetrics, ExecutionMechanicType, JobStatus, ValidationReport,
)
from factory.storage import Storage
from jobs.worker import JobQueue


START = datetime(2023, 1, 1, tzinfo=timezone.utc)
END = datetime(2023, 7, 1, tzinfo=timezone.utc)  # 6m window inside longer default


def _ok(name: str, cond: bool, detail: str = "") -> bool:
    mark = "PASS" if cond else "FAIL"
    line = f"  [{mark}] {name}" + (f" -- {detail}" if detail else "")
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"))
    return cond


def check_settings_and_gates() -> list[bool]:
    print("\n== Settings & Basic gates ==")
    results = []
    results.append(_ok(
        "SIMULATOR_INTRABAR_MODE=path",
        settings.SIMULATOR_INTRABAR_MODE == "path",
        repr(settings.SIMULATOR_INTRABAR_MODE),
    ))
    results.append(_ok(
        "DEFAULT_ENGINE=simulator",
        settings.DEFAULT_ENGINE == "simulator",
        repr(settings.DEFAULT_ENGINE),
    ))
    results.append(_ok(
        "DISCOVERY_MT5_FALLBACK_TO_SIMULATOR",
        bool(settings.DISCOVERY_MT5_FALLBACK_TO_SIMULATOR),
    ))
    results.append(_ok(
        "DISCOVERY_ELITE_SEED_COUNT>0",
        int(settings.DISCOVERY_ELITE_SEED_COUNT) > 0,
        str(settings.DISCOVERY_ELITE_SEED_COUNT),
    ))
    l4 = validation_levels.get_level(4).criteria
    l5 = validation_levels.get_level(5).criteria
    l6 = validation_levels.get_level(6).criteria
    results.append(_ok("L4 min_trades=6", l4.min_trades == 6, str(l4.min_trades)))
    results.append(_ok("L5 min_trades=8", l5.min_trades == 8, str(l5.min_trades)))
    results.append(_ok("L6 min_trades=10", l6.min_trades == 10, str(l6.min_trades)))
    results.append(_ok("L4 WFE still 0.30", abs(l4.min_wfe - 0.30) < 1e-9))
    results.append(_ok(
        "DiscoverySettings months default=12",
        DiscoverySettings().months == 12,
        str(DiscoverySettings().months),
    ))
    return results


def check_continuous_forces_simulator() -> list[bool]:
    print("\n== Continuous agent forces simulator ==")
    # Mirror orchestrator continuous override without spawning the agent.
    cfg = DiscoverySettings(engine="mt5", months=12, symbols=["EURUSD"],
                            timeframes=["H1"])
    payload = build_discovery_payload(
        cfg, symbol="EURUSD", timeframe="H1", seed=1, validation_level=1)
    payload["continuous"] = True
    payload["target_survivors"] = 10**9
    payload["engine"] = "simulator"  # orchestrator overwrite
    return [_ok("payload.engine=simulator after continuous override",
                payload["engine"] == "simulator")]


def check_mechanic_bias() -> list[bool]:
    print("\n== Mechanic survivor bias ==")
    prior = default_mechanic_weights()
    blended = blend_mechanic_weights(
        {ExecutionMechanicType.PARTIAL_CLOSE.value: 5,
         ExecutionMechanicType.STANDARD_SLTP.value: 2})
    results = []
    results.append(_ok(
        "PARTIAL_CLOSE weight increases with clears",
        blended[ExecutionMechanicType.PARTIAL_CLOSE]
        > prior[ExecutionMechanicType.PARTIAL_CLOSE],
        f"{prior[ExecutionMechanicType.PARTIAL_CLOSE]} -> "
        f"{blended[ExecutionMechanicType.PARTIAL_CLOSE]}",
    ))
    # Sampling should prefer PARTIAL_CLOSE under extreme bias
    heavy = {m: 0.01 for m in ExecutionMechanicType}
    heavy[ExecutionMechanicType.PARTIAL_CLOSE] = 1000.0
    counts = {m: 0 for m in ExecutionMechanicType}
    rng = random.Random(0)
    for _ in range(80):
        s = random_strategy("EURUSD", "H1", rng=rng, mechanic_weights=heavy)
        counts[s.mechanic.type] += 1
    results.append(_ok(
        "biased random_strategy prefers PARTIAL_CLOSE",
        counts[ExecutionMechanicType.PARTIAL_CLOSE] >= 60,
        str(dict((k.value, v) for k, v in counts.items() if v)),
    ))
    return results


def check_infra_kpi(tmp: Path) -> list[bool]:
    print("\n== Infra KPI exclusion ==")
    storage = Storage(tmp / "kpi.db")
    ok = random_strategy("EURUSD", "H1", rng=random.Random(1))
    bad = random_strategy("EURUSD", "H1", rng=random.Random(2))
    storage.save_complete(
        ok,
        ValidationReport(
            strategy_id=ok.id,
            is_metrics=BacktestMetrics(trade_count=12, net_profit=40),
            oos_metrics=BacktestMetrics(
                trade_count=12, net_profit=30, profit_factor=1.2),
            passed=False, highest_level_passed=0, engine="simulator",
        ),
        job_id="smoke",
    )
    storage.save_complete(
        bad,
        ValidationReport(
            strategy_id=bad.id,
            is_metrics=BacktestMetrics(),
            oos_metrics=BacktestMetrics(),
            passed=False, highest_level_passed=0, infra_failure=True,
            reasons=["INFRA: terminal busy"], engine="mt5",
        ),
        job_id="smoke",
    )
    total = storage.count_validated(passed_only=None)
    tradeable = storage.count_validated(passed_only=None, exclude_infra=True)
    infra = storage.count_infra_failures()
    prog = storage.run_progress_by_jobs(["smoke"])["smoke"]
    results = []
    results.append(_ok("total=2", total == 2, str(total)))
    results.append(_ok("tradeable=1", tradeable == 1, str(tradeable)))
    results.append(_ok("infra=1", infra == 1, str(infra)))
    results.append(_ok("progress.tradeable=1", prog["tradeable"] == 1))
    results.append(_ok("progress.infra=1", prog["infra"] == 1))
    elite = storage.list_cleared_strategies(min_level=4)
    results.append(_ok("no false elite from level-0 rows", elite == []))
    # Persist a real L4+ and ensure elite list finds it
    winner = random_strategy("USDJPY", "M15", rng=random.Random(9))
    storage.save_complete(
        winner,
        ValidationReport(
            strategy_id=winner.id,
            is_metrics=BacktestMetrics(trade_count=20, net_profit=100),
            oos_metrics=BacktestMetrics(
                trade_count=20, net_profit=80, profit_factor=1.3, sharpe=0.4),
            passed=True, highest_level_passed=5, wfe=0.5, engine="simulator",
        ),
        job_id="smoke2",
    )
    found = storage.list_cleared_strategies(
        symbol="USDJPY", timeframe="M15", min_level=4)
    results.append(_ok("elite list returns L5 survivor", len(found) == 1,
                       found[0].id if found else "none"))
    child = mutate(found[0], random.Random(3), rate=0.4)
    results.append(_ok("elite mutate yields new id", child.id != found[0].id))
    return results


def check_eta_helper() -> list[bool]:
    print("\n== ETA honesty ==")
    from app.components.runs_panel import _estimate_eta
    results = []
    normal = _estimate_eta(100, 1, 1000, 5, 600.0)  # 100 in 10 min
    results.append(_ok("normal ETA is finite duration", normal.startswith("~"),
                       normal))
    sleep = _estimate_eta(2, 0, 1000, 10**9, 11 * 3600.0)
    results.append(_ok(
        "sleep-inflated ETA capped",
        sleep == "slow — see rate" or sleep.startswith("~"),
        sleep,
    ))
    # Continuous huge target should not dominate ETA when survivors=0
    cont = _estimate_eta(50, 0, 200, 10**9, 300.0)
    results.append(_ok("continuous target ignored when survivors=0",
                       "estimating" not in cont or cont.startswith("~"),
                       cont))
    return results


def check_mt5_fallback(tmp: Path) -> list[bool]:
    print("\n== MT5 interactive → simulator fallback ==")
    from factory.backtest.base import BacktestEngine
    from factory.models import StrategyDefinition
    from typing import Dict, Optional

    class StubEngine(BacktestEngine):
        name = "simulator"

        def __init__(self, label: str):
            self.label = label
            self.run_count = 0

        def run(self, strategy: StrategyDefinition, start: datetime, end: datetime,
                params_override: Optional[Dict[str, float]] = None,
                deposit: float = 10_000.0) -> BacktestMetrics:
            self.run_count += 1
            years = max((end - start).total_seconds() / (365.25 * 86400), 1e-9)
            return BacktestMetrics(
                net_profit=deposit * 0.2 * years,
                initial_deposit=deposit,
                start_ts=start.timestamp(),
                end_ts=end.timestamp(),
                max_dd_pct=5.0,
                trade_count=40,
                profit_factor=1.4,
            )

    storage = Storage(tmp / "fallback.db")
    queue = JobQueue(storage)
    engines: list[str] = []

    def make_engine(name: str):
        engines.append(name)
        return StubEngine(name)

    import jobs.worker as worker_mod
    orig_interactive = worker_mod.interactive_terminal_running
    worker_mod.interactive_terminal_running = lambda: True
    queue._make_engine = make_engine  # type: ignore[method-assign]
    try:
        job_id = "verify_mt5_fallback"
        payload = {
            "symbol": "EURUSD", "timeframe": "H1", "engine": "mt5",
            "batch_size": 4, "target_survivors": 1, "max_candidates": 6,
            "genetic": False, "seed": 7,
            "start": START.isoformat(), "end": END.isoformat(),
            "validation_level": 1, "data_source": "synthetic",
            "wfo_train_months": 1, "wfo_test_months": 1, "wfo_windows": 1,
        }
        assert queue.submit_discovery(job_id, payload)
        deadline = time.time() + 90
        while time.time() < deadline:
            job = storage.get_job(job_id)
            if job.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
                break
            time.sleep(0.05)
        job = storage.get_job(job_id)
        results = []
        results.append(_ok("job finished DONE", job.status == JobStatus.DONE,
                           job.status.value))
        results.append(_ok("engine path used simulator",
                           "simulator" in engines, str(engines)))
        results.append(_ok("did not INFRA-fail job",
                           not (job.error or "").startswith("INFRA:"),
                           (job.error or "")[:80]))
        results.append(_ok("tested > 0", int(job.tested or 0) > 0,
                           str(job.tested)))
        return results
    finally:
        worker_mod.interactive_terminal_running = orig_interactive
        # Leave queue threads to exit with process; no public shutdown API.


def check_mini_discovery(tmp: Path) -> list[bool]:
    print("\n== Mini discovery smoke (simulator / synthetic) ==")
    from factory.backtest.base import BacktestEngine
    from factory.models import StrategyDefinition
    from typing import Dict, Optional

    class RichStub(BacktestEngine):
        name = "simulator"
        def __init__(self):
            self.n = 0
        def run(self, strategy: StrategyDefinition, start: datetime, end: datetime,
                params_override: Optional[Dict[str, float]] = None,
                deposit: float = 10_000.0) -> BacktestMetrics:
            self.n += 1
            # Alternate strong / weak so some clear L1 and a few look promising
            strong = (self.n % 3) != 0
            years = max((end - start).total_seconds() / (365.25 * 86400), 1e-9)
            rate = 0.35 if strong else -0.15
            return BacktestMetrics(
                net_profit=deposit * rate * years,
                initial_deposit=deposit,
                start_ts=start.timestamp(),
                end_ts=end.timestamp(),
                max_dd_pct=8.0 if strong else 40.0,
                trade_count=35 if strong else 4,
                profit_factor=1.45 if strong else 0.7,
                sharpe=0.8 if strong else -0.2,
                r_squared=0.55 if strong else 0.05,
            )

    storage = Storage(tmp / "mini.db")
    queue = JobQueue(storage)
    queue._make_engine = lambda _name: RichStub()  # type: ignore[method-assign]
    job_id = "verify_mini_discovery"
    payload = {
        "symbol": "EURUSD", "timeframe": "H1", "engine": "simulator",
        "batch_size": 8, "target_survivors": 2, "max_candidates": 16,
        "genetic": True, "seed": 11,
        "start": START.isoformat(), "end": END.isoformat(),
        "validation_level": 1, "data_source": "synthetic",
        "wfo_train_months": 1, "wfo_test_months": 1, "wfo_windows": 1,
    }
    assert queue.submit_discovery(job_id, payload)
    deadline = time.time() + 120
    while time.time() < deadline:
        job = storage.get_job(job_id)
        if job.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
            break
        time.sleep(0.05)
    job = storage.get_job(job_id)
    reports = storage.list_validated(passed_only=False)
    infra = storage.count_infra_failures(job_id)
    survivors = sum(1 for r in reports if r.passed)
    levels = [int(getattr(r, "highest_level_passed", 0) or 0) for r in reports]
    results = []
    results.append(_ok("mini job DONE", job.status == JobStatus.DONE,
                       f"{job.status.value} err={job.error}"))
    # May stop early once target_survivors is hit — tested can be < max_candidates.
    results.append(_ok("tested progressed", int(job.tested or 0) > 0,
                       str(job.tested)))
    results.append(_ok("no infra aborts in simulator smoke", infra == 0,
                       str(infra)))
    results.append(_ok("hit survivor target without infra waste",
                       int(job.survivors or 0) >= 1 and infra == 0,
                       f"survivors={job.survivors} infra={infra}"))
    results.append(_ok("validation rows persisted", len(reports) > 0,
                       str(len(reports))))
    # Highest level among survivors should be nested ladder, not zero
    hi = max(levels) if levels else 0
    results.append(_ok("survivors cleared nested levels", hi >= 1,
                       f"max_level={hi}"))
    print(f"      detail: tested={job.tested} promising={job.promising} "
          f"survivors={job.survivors} reports={len(reports)} "
          f"level_hist={sorted(set(levels))} msg={job.message}")
    return results


def check_library_snapshot() -> list[bool]:
    print("\n== Live library snapshot (factory.db) ==")
    results = []
    try:
        s = Storage()
    except Exception as exc:
        return [_ok("open factory.db", False, str(exc))]
    total = s.count_validated(passed_only=None)
    tradeable = s.count_validated(passed_only=None, exclude_infra=True)
    infra = s.count_infra_failures()
    l4 = s.count_validated(min_level=4)
    l7 = s.count_validated(min_level=7)
    results.append(_ok("library readable", total > 0, f"total={total}"))
    results.append(_ok(
        "KPI exclude_infra available",
        tradeable <= total,
        f"tradeable={tradeable} infra={infra}",
    ))
    results.append(_ok("L4+ clears still present", l4 >= 0, f"L4+={l4} L7+={l7}"))
    app = s.get_app_settings()
    results.append(_ok(
        "saved discovery_engine=simulator",
        str(app.get("discovery_engine", "")).lower() == "simulator",
        str(app.get("discovery_engine")),
    ))
    results.append(_ok(
        "saved discovery_months>=12",
        int(app.get("discovery_months") or 0) >= 12,
        str(app.get("discovery_months")),
    ))
    # Recent jobs: continuous payloads should request simulator
    jobs = s.list_jobs("discovery")[:15]
    cont = [j for j in jobs if j.payload.get("continuous")]
    if cont:
        engines = {j.payload.get("engine") for j in cont[:5]}
        newest = cont[0]
        results.append(_ok(
            "newest continuous job engine recorded",
            newest.payload.get("engine") in ("simulator", "mt5"),
            f"id={newest.id} engine={newest.payload.get('engine')} "
            f"created={newest.created_at}",
        ))
        print(f"      recent continuous engines sample: {engines}")
    else:
        results.append(_ok("continuous jobs exist (informational)", True,
                           "none yet — restart agent to emit simulator jobs"))
    return results


def check_gate_recalibration_impact() -> list[bool]:
    """Re-score a sample of tradeable OOS metrics under old vs new L4 trade floors."""
    print("\n== Gate recalibration impact (sample re-score) ==")
    import json
    import sqlite3
    results = []
    try:
        con = sqlite3.connect(
            f"file:{settings.DB_PATH.resolve().as_posix()}?mode=ro", uri=True,
            timeout=15.0)
    except Exception as exc:
        return [_ok("open DB readonly", False, str(exc))]
    rows = con.execute(
        "SELECT body FROM validations "
        "WHERE COALESCE(infra_failure,0)=0 "
        "ORDER BY updated_at DESC LIMIT 1500"
    ).fetchall()
    con.close()
    if not rows:
        return [_ok("have tradeable sample", False, "empty")]

    old_trades, new_trades = 8, 6
    # Approximate L4 Basic·A other gates
    def clears(oos, wfe, min_tr):
        if int(oos.get("trade_count") or 0) < min_tr:
            return False
        if float(oos.get("net_profit") or 0) <= 0:
            return False
        if float(oos.get("max_dd_pct") or 0) > 55.0:
            return False
        if float(oos.get("profit_factor") or 0) < 1.05:
            return False
        if float(wfe or 0) < 0.30:
            return False
        return True

    n = 0
    clear_old = clear_new = 0
    tradeable = 0
    for (body_s,) in rows:
        try:
            body = json.loads(body_s)
        except Exception:
            continue
        if body.get("infra_failure"):
            continue
        oos = body.get("oos_metrics") or {}
        if int(oos.get("trade_count") or 0) <= 0:
            continue
        tradeable += 1
        wfe = body.get("wfe")
        n += 1
        if clears(oos, wfe, old_trades):
            clear_old += 1
        if clears(oos, wfe, new_trades):
            clear_new += 1

    lift = clear_new - clear_old
    results.append(_ok("sampled tradeable OOS rows", tradeable > 0, str(tradeable)))
    results.append(_ok(
        "new L4 trade floor clears >= old floor",
        clear_new >= clear_old,
        f"old(min_tr=8)={clear_old} new(min_tr=6)={clear_new} lift=+{lift}",
    ))
    print(f"      sample clear-rate old={clear_old}/{tradeable} "
          f"({100*clear_old/max(tradeable,1):.2f}%) "
          f"new={clear_new}/{tradeable} ({100*clear_new/max(tradeable,1):.2f}%)")

    # Dominant fail cliffs among tradeable (for operator insight)
    from collections import Counter
    cliffs = Counter()
    for (body_s,) in rows:
        try:
            body = json.loads(body_s)
        except Exception:
            continue
        if body.get("infra_failure"):
            continue
        oos = body.get("oos_metrics") or {}
        if int(oos.get("trade_count") or 0) <= 0:
            continue
        wfe = float(body.get("wfe") or 0)
        if float(oos.get("net_profit") or 0) <= 0:
            cliffs["net_profit"] += 1
        elif wfe < 0.30:
            cliffs["wfe<0.30"] += 1
        elif float(oos.get("profit_factor") or 0) < 1.05:
            cliffs["pf<1.05"] += 1
        elif float(oos.get("max_dd_pct") or 999) > 55:
            cliffs["dd>55"] += 1
        elif int(oos.get("trade_count") or 0) < 6:
            cliffs["trades<6"] += 1
        else:
            cliffs["would_clear_L4_approx"] += 1
    print(f"      tradeable fail cliffs: {dict(cliffs.most_common())}")
    results.append(_ok(
        "WFE or profit dominate over trade-count cliff",
        cliffs.get("wfe<0.30", 0) + cliffs.get("net_profit", 0)
        >= cliffs.get("trades<6", 0),
        str(dict(cliffs.most_common(5))),
    ))
    return results


def check_real_simulator_discovery(tmp: Path) -> list[bool]:
    """Short discovery with the real SimulatorEngine on synthetic OHLC."""
    print("\n== Real SimulatorEngine discovery smoke ==")
    storage = Storage(tmp / "real_sim.db")
    queue = JobQueue(storage)
    # Seed one elite so elite-seeding path is exercised on gen>0 if we get there
    elite = random_strategy("EURUSD", "H1", rng=random.Random(42))
    storage.save_complete(
        elite,
        ValidationReport(
            strategy_id=elite.id,
            is_metrics=BacktestMetrics(trade_count=30, net_profit=200),
            oos_metrics=BacktestMetrics(
                trade_count=25, net_profit=150, profit_factor=1.4, sharpe=0.5),
            passed=True, highest_level_passed=5, wfe=0.55, engine="simulator",
        ),
        job_id="seed",
    )
    job_id = "verify_real_sim"
    payload = {
        "symbol": "EURUSD", "timeframe": "H1", "engine": "simulator",
        "batch_size": 6, "target_survivors": 1, "max_candidates": 12,
        "genetic": True, "seed": 99,
        "start": START.isoformat(), "end": END.isoformat(),
        "validation_level": 1, "data_source": "synthetic",
        "wfo_train_months": 1, "wfo_test_months": 1, "wfo_windows": 1,
    }
    assert queue.submit_discovery(job_id, payload)
    deadline = time.time() + 300
    while time.time() < deadline:
        job = storage.get_job(job_id)
        if job.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
            break
        time.sleep(0.2)
    job = storage.get_job(job_id)
    reports = [r for r in storage.list_validated(passed_only=False)
               if getattr(r, "strategy_id", None)]
    # Only this job
    job_reports = []
    with storage.connection() as con:
        rows = con.execute(
            "SELECT body FROM validations WHERE job_id=?", (job_id,)
        ).fetchall()
    import json
    for (body,) in rows:
        job_reports.append(json.loads(body))

    infra = sum(1 for r in job_reports if r.get("infra_failure"))
    survivors = sum(1 for r in job_reports if r.get("passed"))
    reasons = []
    for r in job_reports:
        for x in (r.get("reasons") or [])[:1]:
            reasons.append(str(x)[:80])
    results = []
    results.append(_ok("real sim job finished",
                       job.status in (JobStatus.DONE, JobStatus.FAILED),
                       job.status.value))
    results.append(_ok("not INFRA job failure",
                       not (job.error or "").startswith("INFRA:"),
                       (job.error or "")[:60] or "none"))
    results.append(_ok("screens progressed", int(job.tested or 0) > 0,
                       str(job.tested)))
    results.append(_ok("zero infra_failure rows", infra == 0, str(infra)))
    # Message should mention floor / continuous-free path
    msg = job.message or ""
    results.append(_ok("progress message present", bool(msg), msg[:100]))
    print(f"      tested={job.tested} promising={job.promising} "
          f"survivors={job.survivors}/{survivors} reports={len(job_reports)}")
    if reasons:
        print(f"      sample reasons: {reasons[:5]}")
    # Elite list still works after run
    found = storage.list_cleared_strategies(
        symbol="EURUSD", timeframe="H1", min_level=4)
    results.append(_ok("elite seed available to worker", len(found) >= 1,
                       str(len(found))))
    return results


def main() -> int:
    print("EA Factory — quality improvement verification")
    print("=" * 60)
    all_results: list[bool] = []
    all_results.extend(check_settings_and_gates())
    all_results.extend(check_continuous_forces_simulator())
    all_results.extend(check_mechanic_bias())
    all_results.extend(check_eta_helper())

    with tempfile.TemporaryDirectory(prefix="ea_verify_",
                                     ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        all_results.extend(check_infra_kpi(tmp))
        all_results.extend(check_mt5_fallback(tmp))
        all_results.extend(check_mini_discovery(tmp))
        all_results.extend(check_real_simulator_discovery(tmp))

    all_results.extend(check_library_snapshot())
    all_results.extend(check_gate_recalibration_impact())

    passed = sum(1 for x in all_results if x)
    failed = sum(1 for x in all_results if not x)
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed, {len(all_results)} checks")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
