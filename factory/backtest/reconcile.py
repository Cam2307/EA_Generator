"""Simulator <-> MT5 reconciliation: quantify the pre-filter's bias.

The simulator is a pre-filter and the MT5 Strategy Tester is the source of
truth (README). What the pipeline never measured until now is *how far apart
they are* — if the simulator overstates profit factor by 12% for grid
strategies, every screening gate is silently 12% too loose for them.

This module runs the same strategies through both engines on the same window
and reports per-metric deltas plus an aggregate bias summary. Run it via
``scripts/reconcile_engines.py`` on a machine with MT5 installed; the compare
logic is engine-agnostic so tests exercise it with stubs.

Delta convention: ``rel = (sim - mt5) / max(|mt5|, floor)`` — positive means
the simulator is *more optimistic* than the truth for profit-like metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from factory.backtest.base import BacktestEngine
from factory.models import BacktestMetrics, StrategyDefinition

# Metrics compared and their default relative tolerances. Trade count is the
# most diagnostic: a large mismatch there means the two engines are not even
# seeing the same signals, and per-trade economics comparisons are moot.
DEFAULT_TOLERANCES: Dict[str, float] = {
    "trade_count": 0.25,
    "net_profit": 0.35,
    "profit_factor": 0.25,
    "max_dd_pct": 0.50,
}

# Absolute floors so tiny denominators do not explode relative deltas.
_REL_FLOORS: Dict[str, float] = {
    "trade_count": 5.0,
    "net_profit": 100.0,
    "profit_factor": 0.5,
    "max_dd_pct": 2.0,
}


@dataclass
class MetricDelta:
    name: str
    sim_value: float
    mt5_value: float
    rel_delta: float          # signed; positive = simulator higher
    within_tolerance: bool


@dataclass
class ReconcileResult:
    strategy_id: str
    strategy_name: str
    deltas: List[MetricDelta] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and all(d.within_tolerance for d in self.deltas)


def compare_metrics(sim_m: BacktestMetrics, mt5_m: BacktestMetrics,
                    tolerances: Optional[Dict[str, float]] = None
                    ) -> List[MetricDelta]:
    tol = dict(DEFAULT_TOLERANCES)
    if tolerances:
        tol.update(tolerances)
    out: List[MetricDelta] = []
    for name, limit in tol.items():
        sv = float(getattr(sim_m, name))
        mv = float(getattr(mt5_m, name))
        denom = max(abs(mv), _REL_FLOORS.get(name, 1.0))
        rel = (sv - mv) / denom
        out.append(MetricDelta(name=name, sim_value=sv, mt5_value=mv,
                               rel_delta=round(rel, 4),
                               within_tolerance=abs(rel) <= limit))
    return out


def reconcile_strategies(sim_engine: BacktestEngine, mt5_engine: BacktestEngine,
                         strategies: Sequence[StrategyDefinition],
                         start: datetime, end: datetime,
                         deposit: float = 10_000.0,
                         tolerances: Optional[Dict[str, float]] = None
                         ) -> List[ReconcileResult]:
    """Run every strategy through both engines and compare."""
    results: List[ReconcileResult] = []
    for strat in strategies:
        res = ReconcileResult(strategy_id=strat.id, strategy_name=strat.name)
        try:
            sim_m = sim_engine.run(strat, start, end, deposit=deposit)
            mt5_m = mt5_engine.run(strat, start, end, deposit=deposit)
            res.deltas = compare_metrics(sim_m, mt5_m, tolerances)
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
        results.append(res)
    return results


def bias_summary(results: Sequence[ReconcileResult]) -> Dict[str, Dict[str, float]]:
    """Per-metric aggregate: mean/median signed bias + agreement fraction.

    A consistently positive ``mean_rel`` on profit-like metrics is the
    simulator's systematic optimism for this strategy population — the
    number to feed back into screening-gate calibration.
    """
    import statistics

    per_metric: Dict[str, List[float]] = {}
    ok_flags: Dict[str, List[bool]] = {}
    for res in results:
        if res.error:
            continue
        for d in res.deltas:
            per_metric.setdefault(d.name, []).append(d.rel_delta)
            ok_flags.setdefault(d.name, []).append(d.within_tolerance)
    out: Dict[str, Dict[str, float]] = {}
    for name, vals in per_metric.items():
        out[name] = {
            "n": float(len(vals)),
            "mean_rel": round(statistics.fmean(vals), 4),
            "median_rel": round(statistics.median(vals), 4),
            "agree_frac": round(
                sum(ok_flags[name]) / len(ok_flags[name]), 4),
        }
    return out


def format_report(results: Sequence[ReconcileResult]) -> str:
    """Human-readable reconciliation report for the CLI harness."""
    lines: List[str] = []
    for res in results:
        if res.error:
            lines.append(f"[FAIL] {res.strategy_name}: {res.error}")
            continue
        status = "ok  " if res.ok else "DIVERGES"
        parts = ", ".join(
            f"{d.name} sim={d.sim_value:g} mt5={d.mt5_value:g}"
            f" ({d.rel_delta:+.0%})" for d in res.deltas)
        lines.append(f"[{status}] {res.strategy_name}: {parts}")
    summary = bias_summary(results)
    if summary:
        lines.append("")
        lines.append("Aggregate simulator bias vs MT5 (positive = simulator optimistic):")
        for name, s in summary.items():
            lines.append(
                f"  {name:<14} mean {s['mean_rel']:+.1%}  median"
                f" {s['median_rel']:+.1%}  within-tolerance {s['agree_frac']:.0%}"
                f"  (n={s['n']:.0f})")
    n_err = sum(1 for r in results if r.error)
    n_ok = sum(1 for r in results if r.ok)
    lines.append("")
    lines.append(f"{n_ok}/{len(results)} strategies reconcile within tolerance"
                 + (f"; {n_err} failed to run" if n_err else ""))
    return "\n".join(lines)
