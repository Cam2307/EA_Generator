"""Promotion scoring and hard-gate evaluation for discovery alerts."""
from __future__ import annotations

from dataclasses import dataclass

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
) -> PromotionDecision:
    """Compute quality and the promotion state for one validation report."""
    oos = report.oos_metrics
    hard_pass = (
        report.passed
        and oos.net_profit >= min_profit
        and oos.profit_factor >= min_pf
        and oos.sharpe >= min_sharpe
        and report.wfe >= min_wfe
    )
    breakdown = {
        "profit_factor": min(max(oos.profit_factor / 2.0, 0.0), 1.0) * 30.0,
        "sharpe": min(max(oos.sharpe / 2.5, 0.0), 1.0) * 20.0,
        "wfe": min(max(report.wfe / 1.0, 0.0), 1.0) * 20.0,
        "drawdown": max(0.0, 1.0 - min(oos.max_dd_pct, 40.0) / 40.0) * 15.0,
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
    score = round(sum(breakdown.values()), 2)

    if not report.passed:
        state = "candidate"
    elif score >= promote_threshold and hard_pass:
        state = "promoted_live_watchlist"
    elif score >= edge_threshold and hard_pass:
        state = "edge_positive"
    else:
        state = "validated"
    return PromotionDecision(
        promotion_state=state,
        quality_score=score,
        hard_gates_passed=hard_pass,
        breakdown=breakdown,
    )
