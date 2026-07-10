"""Return-stream correlation between strategies (novelty / curation).

A gallery that rediscovers the same RSI-reversion edge forty times is not
forty strategies — it is one strategy with forty names, and promoting several
of them concentrates risk instead of diversifying it. This module reduces
each validated strategy to a daily-return fingerprint (derived from the OOS
equity curve already stored in its validation report) and measures Pearson
correlation between fingerprints on their overlapping days.

Used by the promotion flow to flag (and score-penalize) candidates that are
highly correlated with an already-promoted strategy.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from factory.models import BacktestMetrics, ValidationReport

# Correlation above this against a promoted strategy marks a near-duplicate.
DUPLICATE_CORR_THRESHOLD = 0.6
# Minimum overlapping days for a correlation to mean anything.
MIN_OVERLAP_DAYS = 10


def daily_returns(metrics: BacktestMetrics) -> Dict[int, float]:
    """Map of trading-day index -> daily return, from the stored equity curve.

    Uses the last equity sample of each UTC day; days without samples are
    simply absent (correlation aligns on common days only).
    """
    ts = np.asarray(metrics.equity_ts, dtype=float)
    eq = np.asarray(metrics.equity, dtype=float)
    if len(ts) < 2 or len(ts) != len(eq):
        return {}
    days = (ts // 86400).astype(np.int64)
    # last sample per day
    last_eq: Dict[int, float] = {}
    for d, e in zip(days, eq):
        last_eq[int(d)] = float(e)
    ordered = sorted(last_eq.items())
    out: Dict[int, float] = {}
    for (d0, e0), (d1, e1) in zip(ordered[:-1], ordered[1:]):
        if e0 > 0:
            out[d1] = e1 / e0 - 1.0
    return out


def return_correlation(a: Dict[int, float],
                       b: Dict[int, float]) -> Optional[float]:
    """Pearson correlation on overlapping days; None when not measurable."""
    common = sorted(set(a) & set(b))
    if len(common) < MIN_OVERLAP_DAYS:
        return None
    xa = np.asarray([a[d] for d in common])
    xb = np.asarray([b[d] for d in common])
    if float(xa.std()) <= 1e-15 or float(xb.std()) <= 1e-15:
        return None
    return round(float(np.corrcoef(xa, xb)[0, 1]), 4)


def max_correlation(report: ValidationReport,
                    others: Sequence[ValidationReport]
                    ) -> Tuple[Optional[float], Optional[str]]:
    """Highest |correlation| of ``report`` against ``others`` (same symbol).

    Returns ``(correlation, other_strategy_id)`` for the strongest overlap,
    or ``(None, None)`` when nothing is measurable. Only strategies whose
    OOS windows actually overlap contribute.
    """
    mine = daily_returns(report.oos_metrics)
    if not mine:
        return None, None
    best: Optional[float] = None
    best_id: Optional[str] = None
    for other in others:
        if other.strategy_id == report.strategy_id:
            continue
        corr = return_correlation(mine, daily_returns(other.oos_metrics))
        if corr is None:
            continue
        if best is None or abs(corr) > abs(best):
            best, best_id = corr, other.strategy_id
    return best, best_id


def duplicate_penalty_from_corr(corr: Optional[float],
                                threshold: float = DUPLICATE_CORR_THRESHOLD,
                                max_penalty: float = 15.0) -> float:
    """Quality-score penalty for near-duplicate return streams.

    0 below the threshold, then linear up to ``max_penalty`` at |corr| = 1.
    """
    if corr is None or abs(corr) < threshold:
        return 0.0
    span = max(1e-9, 1.0 - threshold)
    return round(max_penalty * (abs(corr) - threshold) / span, 2)
