"""Online re-optimization of promoted strategies.

Markets drift; a parameter set fit a year ago slowly walks off its plateau.
This service re-runs the in-sample optimizer for promoted/edge-positive
strategies on a *trailing* window, compares the fresh optimum against the
incumbent ``best_params``, and — when the plateau has genuinely shifted —
writes an updated ``.set`` file and flags the strategy for review. The
factory becomes a maintenance system, not just a generator.

Drift criteria (both must hold):
- the re-optimized parameters differ from the incumbent, AND
- the incumbent's fitness on the trailing window trails the fresh optimum
  by more than ``improvement_threshold`` (relative).

Run it via ``scripts/reoptimize_promoted.py`` (schedule it weekly/monthly),
or call :func:`reoptimize_promoted` from the orchestrator.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config import settings
from factory.backtest.validation import optimize_is, screen_fitness
from factory.storage import Storage

# Relative fitness shortfall of the incumbent that counts as real drift.
DEFAULT_IMPROVEMENT_THRESHOLD = 0.10


@dataclass
class ReoptimizationResult:
    strategy_id: str
    strategy_name: str = ""
    window_days: int = 0
    old_params: Dict[str, float] = field(default_factory=dict)
    new_params: Dict[str, float] = field(default_factory=dict)
    old_fitness: float = 0.0
    new_fitness: float = 0.0
    improvement: float = 0.0        # relative fitness gain of the fresh optimum
    params_changed: bool = False
    drifted: bool = False           # plateau moved enough to act on
    set_path: Optional[Path] = None
    error: Optional[str] = None
    checked_at: float = field(default_factory=time.time)

    @property
    def changed_params(self) -> Dict[str, tuple]:
        """Only the parameters whose values moved: name -> (old, new)."""
        out = {}
        for k, new in self.new_params.items():
            old = self.old_params.get(k)
            if old is None or abs(float(old) - float(new)) > 1e-12:
                out[k] = (old, new)
        return out


def reoptimize_strategy(storage: Storage, strategy_id: str, *,
                        window_days: int = 180,
                        n_samples: Optional[int] = None,
                        deposit: float = settings.DEFAULT_DEPOSIT,
                        engine=None,
                        seed: Optional[int] = None,
                        improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD,
                        out_dir: Optional[Path] = None,
                        now: Optional[datetime] = None
                        ) -> ReoptimizationResult:
    """Re-optimize one strategy on its trailing window (see module docs)."""
    res = ReoptimizationResult(strategy_id=strategy_id,
                               window_days=window_days)
    strategy = storage.get_strategy(strategy_id)
    if strategy is None:
        res.error = "strategy not found"
        return res
    res.strategy_name = strategy.name
    report = storage.get_validation(strategy_id)
    incumbent = dict(report.best_params) if report is not None else {}
    res.old_params = incumbent

    if engine is None:
        from factory.backtest.simulator import SimulatorEngine
        engine = SimulatorEngine()

    end = now or datetime.now(timezone.utc)
    start = end - timedelta(days=window_days)
    rng = random.Random(seed)
    current = strategy.apply_flat_params(incumbent) if incumbent else strategy

    try:
        # incumbent's performance on the trailing window
        old_m = engine.run(current, start, end, deposit=deposit)
        res.old_fitness = screen_fitness(old_m)
        # fresh optimization (candidate 0 is the incumbent, so the result
        # can never be worse than what we already run)
        new_params, new_m, *_rest = optimize_is(
            engine, current, start, end, deposit,
            n_samples or settings.OPT_SAMPLES, rng,
            stability=settings.NEIGHBORHOOD_STABILITY)
        res.new_params = dict(new_params)
        res.new_fitness = screen_fitness(new_m)
    except Exception as exc:                       # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"
        return res

    denom = max(abs(res.old_fitness), 1e-9)
    res.improvement = round((res.new_fitness - res.old_fitness) / denom, 4)
    res.params_changed = bool(res.changed_params)
    res.drifted = res.params_changed and res.improvement > improvement_threshold

    if res.drifted and out_dir is not None:
        from factory.assets.set_writer import write_set_file
        try:
            res.set_path = write_set_file(strategy, Path(out_dir),
                                          params_override=res.new_params)
        except Exception as exc:                   # noqa: BLE001
            res.error = f"set write failed: {type(exc).__name__}: {exc}"
    return res


def reoptimize_promoted(storage: Storage, *,
                        states: tuple = ("edge_positive",
                                         "promoted_live_watchlist"),
                        limit: int = 20,
                        **kwargs) -> List[ReoptimizationResult]:
    """Re-optimize every promoted / edge-positive strategy (capped)."""
    reports = [r for r in storage.list_validated(passed_only=True)
               if r.promotion_state in states][:limit]
    return [reoptimize_strategy(storage, r.strategy_id, **kwargs)
            for r in reports]


def format_reopt_report(results: List[ReoptimizationResult]) -> str:
    """Human-readable summary for the CLI / alert email."""
    lines: List[str] = []
    for r in results:
        if r.error:
            lines.append(f"[FAIL]  {r.strategy_name or r.strategy_id}: {r.error}")
            continue
        if r.drifted:
            changed = ", ".join(
                f"{k}: {old} -> {new}"
                for k, (old, new) in sorted(r.changed_params.items()))
            extra = f" — updated .set: {r.set_path}" if r.set_path else ""
            lines.append(
                f"[DRIFT] {r.strategy_name}: fresh optimum beats incumbent by"
                f" {r.improvement:+.0%} on the last {r.window_days}d"
                f" ({changed}){extra}")
        else:
            lines.append(
                f"[ok]    {r.strategy_name}: incumbent still on the plateau"
                f" (fresh optimum {r.improvement:+.0%})")
    n_drift = sum(1 for r in results if r.drifted)
    lines.append("")
    lines.append(f"{n_drift}/{len(results)} promoted strategies drifted")
    return "\n".join(lines)
