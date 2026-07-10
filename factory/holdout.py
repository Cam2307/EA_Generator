"""Untouched-holdout layer: data discovery is never allowed to see.

The most recent ``HOLDOUT_MONTHS`` of history are reserved: the discovery
worker clamps every run's end date to the holdout boundary, so no candidate
is ever generated, optimized, or gate-tuned on that window. A strategy is
scored on the holdout exactly ONCE (one-shot discipline — repeated peeks
would turn the holdout into just another training set), and the aggregate
hit rate of those one-shot evaluations is the factory's master KPI: it
measures whether the pipeline finds real edge, not whether it fits gates.

Publication (factory/publication.py) requires a profitable holdout result.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

from config import settings
from factory.storage import Storage


def holdout_boundary(now: Optional[datetime] = None) -> datetime:
    """First instant of the reserved window (end of usable discovery data)."""
    now = now or datetime.now(timezone.utc)
    months = getattr(settings, "HOLDOUT_MONTHS", 12)
    return now - timedelta(days=months * settings.DAYS_PER_MONTH)


def clamp_discovery_end(end: datetime,
                        now: Optional[datetime] = None
                        ) -> Tuple[datetime, bool]:
    """Clamp a discovery range's end to the holdout boundary.

    Returns ``(clamped_end, was_clamped)``. No-op when the holdout is
    disabled or the requested range already stops before the boundary.
    """
    if not getattr(settings, "HOLDOUT_ENABLED", True):
        return end, False
    boundary = holdout_boundary(now)
    if end <= boundary:
        return end, False
    return boundary, True


@dataclass
class HoldoutResult:
    strategy_id: str
    strategy_name: str = ""
    start_ts: float = 0.0
    end_ts: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    max_dd_pct: float = 0.0
    trade_count: int = 0
    passed: bool = False               # profitable with sane drawdown
    error: Optional[str] = None
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "start_ts": self.start_ts, "end_ts": self.end_ts,
            "net_profit": self.net_profit,
            "profit_factor": self.profit_factor,
            "max_dd_pct": self.max_dd_pct,
            "trade_count": self.trade_count,
            "passed": self.passed, "error": self.error,
            "evaluated_at": self.evaluated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "HoldoutResult":
        return cls(**d)


def evaluate_holdout(storage: Storage, strategy_id: str, *,
                     engine=None,
                     deposit: float = settings.DEFAULT_DEPOSIT,
                     now: Optional[datetime] = None,
                     force: bool = False) -> HoldoutResult:
    """One-shot holdout evaluation of a strategy at its ``best_params``.

    A stored result is returned as-is unless ``force=True`` — re-running the
    holdout until it passes is exactly the selection bias this layer exists
    to prevent, so ``force`` is for genuine corrections only (e.g. a data
    outage produced an error result).
    """
    existing = storage.get_holdout_result(strategy_id)
    if existing is not None and not force:
        return HoldoutResult.from_dict(existing)

    res = HoldoutResult(strategy_id=strategy_id)
    strategy = storage.get_strategy(strategy_id)
    if strategy is None:
        res.error = "strategy not found"
        storage.save_holdout_result(res.to_dict())
        return res
    res.strategy_name = strategy.name
    report = storage.get_validation(strategy_id)
    params = dict(report.best_params) if report is not None else {}

    if engine is None:
        from factory.backtest.simulator import SimulatorEngine
        engine = SimulatorEngine()

    now = now or datetime.now(timezone.utc)
    start = holdout_boundary(now)
    res.start_ts, res.end_ts = start.timestamp(), now.timestamp()
    try:
        m = engine.run(strategy, start, now,
                       params_override=params or None, deposit=deposit)
        res.net_profit = m.net_profit
        res.profit_factor = m.profit_factor
        res.max_dd_pct = m.max_dd_pct
        res.trade_count = m.trade_count
        dd_limit = getattr(settings, "HOLDOUT_MAX_DD_PCT", 25.0)
        res.passed = (m.net_profit > 0.0 and m.trade_count >= 3
                      and m.max_dd_pct < dd_limit)
    except Exception as exc:                       # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"
    storage.save_holdout_result(res.to_dict())
    return res


def factory_hit_rate(storage: Storage) -> Dict[str, float]:
    """Aggregate one-shot holdout outcomes — the factory's master KPI."""
    rows = storage.list_holdout_results()
    usable = [r for r in rows if not r.get("error")]
    passed = sum(1 for r in usable if r.get("passed"))
    return {
        "evaluated": float(len(usable)),
        "passed": float(passed),
        "hit_rate": round(passed / len(usable), 4) if usable else 0.0,
        "errors": float(len(rows) - len(usable)),
    }
