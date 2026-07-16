"""Promotion scoring and hard-gate evaluation for discovery alerts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import settings
from factory.models import ValidationReport


@dataclass(frozen=True)
class PromotionDecision:
    promotion_state: str
    quality_score: float
    hard_gates_passed: bool
    breakdown: dict[str, float]


def evaluate_promotion(
    report: ValidationReport,
    *,
    min_profit: float = 0.0,
    min_pf: float = 1.15,
    min_sharpe: float = 0.6,
    min_wfe: float = 0.55,
    edge_threshold: float = 65.0,
    promote_threshold: float = 80.0,
    duplicate_penalty: float = 0.0,
    holdout_passed: Optional[bool] = None,
    mt5_confirmed: Optional[bool] = None,
) -> PromotionDecision:
    """Compute quality and the promotion state for one validation report.

    ``edge_positive`` / ``promoted_live_watchlist`` require a holdout pass when
    ``PROMOTION_REQUIRE_HOLDOUT`` is enabled. Watchlist additionally requires
    ``mt5_confirmed=True`` when ``PROMOTION_REQUIRE_MT5_FOR_WATCHLIST`` is on.
    """
    oos = report.oos_metrics
    hard_pass = (
        report.passed
        and oos.net_profit >= min_profit
        and oos.profit_factor >= min_pf
        and oos.sharpe >= min_sharpe
        and report.wfe >= min_wfe
    )
    # Positive components are budgeted to sum to 100 at a perfect score.
    breakdown = {
        "profit_factor": min(max(oos.profit_factor / 2.0, 0.0), 1.0) * 25.0,
        "sharpe": min(max(oos.sharpe / 2.5, 0.0), 1.0) * 18.0,
        "wfe": min(max(report.wfe / 1.0, 0.0), 1.0) * 18.0,
        "drawdown": max(0.0, 1.0 - min(oos.max_dd_pct, 40.0) / 40.0) * 14.0,
        "stability": min(max(report.stability_ratio, 0.0), 1.0) * 10.0,
        "degradation": max(0.0, 1.0 - max(report.degradation_pct, 0.0) / 100.0) * 5.0,
        "sample_size": min(max(oos.trade_count / 60.0, 0.0), 1.0) * 5.0,
        "mc": (
            min(max((report.montecarlo.robustness_score or 0.0) / 100.0, 0.0), 1.0) * 5.0
            if report.montecarlo
            else 0.0
        ),
    }
    complexity = len(report.best_params)
    complexity_penalty = max(0.0, (complexity - 14) * 0.9) * (
        1.0 - min(max(report.stability_ratio, 0.0), 1.0)
    )
    breakdown["complexity_penalty"] = -round(complexity_penalty, 2)
    if duplicate_penalty > 0:
        breakdown["duplicate_penalty"] = -round(duplicate_penalty, 2)

    holdout = (
        holdout_passed if holdout_passed is not None
        else getattr(report, "holdout_passed", None)
    )
    mt5_ok = (
        mt5_confirmed if mt5_confirmed is not None
        else getattr(report, "mt5_confirmed", None)
    )
    if holdout is True:
        breakdown["holdout"] = 0.0  # gate only — recorded for transparency
    elif holdout is False:
        breakdown["holdout"] = 0.0
    if mt5_ok is True:
        breakdown["mt5_confirm"] = 0.0
    elif mt5_ok is False:
        breakdown["mt5_confirm"] = 0.0

    score = round(min(100.0, max(0.0, sum(
        v for k, v in breakdown.items()
        if k not in ("holdout", "mt5_confirm")
    ))), 2)

    require_holdout = bool(getattr(settings, "PROMOTION_REQUIRE_HOLDOUT", True))
    require_mt5 = bool(
        getattr(settings, "PROMOTION_REQUIRE_MT5_FOR_WATCHLIST", True))
    holdout_ok = (not require_holdout) or (holdout is True)
    mt5_gate_ok = (not require_mt5) or (mt5_ok is True)

    if not report.passed:
        state = "candidate"
    elif (score >= promote_threshold and hard_pass and holdout_ok
          and mt5_gate_ok):
        state = "promoted_live_watchlist"
    elif score >= edge_threshold and hard_pass and holdout_ok:
        state = "edge_positive"
    elif report.passed:
        state = "validated"
    else:
        state = "candidate"

    return PromotionDecision(
        promotion_state=state,
        quality_score=score,
        hard_gates_passed=hard_pass,
        breakdown=breakdown,
    )
