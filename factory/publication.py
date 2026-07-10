"""Publication pipeline: a distinctly higher bar than discovery gates.

Discovery gates exist to feed the genetic search; the *publication* tier
decides what carries your marketplace reputation. A strategy is publishable
only when every check below holds:

- validation passed, on real (non-synthetic) data
- statistically meaningful OOS sample (trade count)
- Deflated Sharpe Ratio clears the multiple-testing haircut
- walk-forward efficiency and Monte Carlo robustness at publication level
- edge exists in at least two market regimes
- return stream is not a near-duplicate of anything already published
- the one-shot untouched-holdout evaluation exists and passed

Aggressive-recovery styles (martingale grids, hedge recovery) are *flagged*,
not blocked — the risk label follows the strategy onto its result card and
into the publication record so the decision is always explicit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import settings
from factory.models import ExecutionMechanicType, StrategyDefinition
from factory.storage import Storage


@dataclass
class PublicationDecision:
    strategy_id: str
    ready: bool = False
    checks: Dict[str, bool] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)   # failed-check details
    warnings: List[str] = field(default_factory=list)  # non-blocking flags


def risk_style(strategy: StrategyDefinition) -> Optional[Tuple[str, str]]:
    """Aggressive-recovery label for a strategy, or None for plain styles.

    Returns ``(label, tone)`` where tone matches the UI chip palette:
    - DCA grid with lot multiplier > 1.0  -> "Martingale grid" (red)
    - DCA grid, flat lots                 -> "DCA grid" (amber)
    - Hedge-recovery layer                -> "Hedge recovery" (violet)
    """
    mech = strategy.mechanic
    if mech.type == ExecutionMechanicType.DCA_GRID:
        if float(mech.params.get("lot_multiplier", 1.0)) > 1.0:
            return "Martingale grid", "red"
        return "DCA grid", "amber"
    if mech.type == ExecutionMechanicType.HEDGE_LAYER:
        return "Hedge recovery", "violet"
    return None


def evaluate_publication(storage: Storage, strategy_id: str
                         ) -> PublicationDecision:
    """Run every publication-tier check; nothing is mutated."""
    d = PublicationDecision(strategy_id=strategy_id)
    strategy = storage.get_strategy(strategy_id)
    report = storage.get_validation(strategy_id)
    if strategy is None or report is None:
        d.reasons.append("strategy or validation record missing")
        return d

    min_trades = int(getattr(settings, "PUB_MIN_OOS_TRADES", 200))
    min_dsr = float(getattr(settings, "PUB_MIN_DSR", 0.95))
    min_wfe = float(getattr(settings, "PUB_MIN_WFE", 0.70))
    min_mc = float(getattr(settings, "PUB_MIN_MC_SCORE", 85.0))
    max_corr = float(getattr(settings, "PUB_MAX_CORR", 0.5))
    min_regimes = int(getattr(settings, "PUB_MIN_POSITIVE_REGIMES", 2))
    allowed_sources = tuple(getattr(settings, "PUB_ALLOWED_DATA_SOURCES",
                                    ("mt5", "cache")))

    def check(name: str, ok: bool, why: str) -> None:
        d.checks[name] = bool(ok)
        if not ok:
            d.reasons.append(why)

    oos = report.oos_metrics
    check("validation_passed", report.passed,
          "validation gates not passed")
    check("real_data", report.data_source in allowed_sources,
          f"data source '{report.data_source}' is not publication-grade")
    check("oos_trades", oos.trade_count >= min_trades,
          f"OOS trades {oos.trade_count} < {min_trades}")
    check("dsr", report.dsr >= min_dsr,
          f"Deflated Sharpe {report.dsr:.2f} < {min_dsr} "
          f"(over {report.n_trials} trials)")
    check("wfe", report.wfe >= min_wfe,
          f"walk-forward efficiency {report.wfe:.2f} < {min_wfe}")
    mc_score = (report.montecarlo.robustness_score
                if report.montecarlo else 0.0)
    check("montecarlo", mc_score >= min_mc,
          f"Monte Carlo robustness {mc_score:.0f} < {min_mc:.0f}")

    positive_regimes = sum(1 for s in report.regime_stats
                           if s.trades > 0 and s.net_profit > 0)
    check("regimes", positive_regimes >= min_regimes,
          f"edge positive in only {positive_regimes} regime(s); "
          f"need {min_regimes}")

    # near-duplicate check against everything already published
    from factory.correlation import max_correlation
    published_reports = []
    for rec in storage.list_publications():
        rep = storage.get_validation(rec["strategy_id"])
        if rep is not None and rec["strategy_id"] != strategy_id:
            published_reports.append(rep)
    corr, corr_id = (max_correlation(report, published_reports)
                     if published_reports else (None, None))
    check("uncorrelated", corr is None or abs(corr) < max_corr,
          f"return stream {corr if corr is not None else 0:.2f} correlated "
          f"with already-published {corr_id}")

    # one-shot untouched holdout
    if getattr(settings, "PUB_REQUIRE_HOLDOUT", True):
        hres = storage.get_holdout_result(strategy_id)
        if hres is None:
            check("holdout", False, "holdout not evaluated yet")
        elif hres.get("error"):
            check("holdout", False, f"holdout errored: {hres['error']}")
        else:
            check("holdout", bool(hres.get("passed")),
                  f"holdout failed (net {hres.get('net_profit', 0):,.0f})")
    else:
        d.checks["holdout"] = True

    style = risk_style(strategy)
    if style is not None:
        d.warnings.append(
            f"{style[0]} execution style — disclose prominently in the "
            "listing; buyers must understand the drawdown profile")
    if storage.get_publication(strategy_id) is not None:
        d.warnings.append("already published — publishing again bumps the "
                          "version")

    d.ready = all(d.checks.values())
    return d


def publish(storage: Storage, strategy_id: str, *,
            version: str = "1.0.0", force: bool = False) -> Dict:
    """Export the marketplace package and record the publication.

    Refuses when the readiness checks fail unless ``force=True`` (which is
    recorded in the publication record — forced publications are visible).
    """
    decision = evaluate_publication(storage, strategy_id)
    if not decision.ready and not force:
        raise ValueError("not publication-ready: "
                         + "; ".join(decision.reasons))
    strategy = storage.get_strategy(strategy_id)
    report = storage.get_validation(strategy_id)

    from factory.assets.exporter import export_marketplace_package
    out_dir = export_marketplace_package(strategy, report)

    style = risk_style(strategy)
    record = {
        "strategy_id": strategy_id,
        "strategy_name": strategy.name,
        "version": version,
        "published_at": time.time(),
        "package_dir": str(out_dir),
        "forced": bool(force and not decision.ready),
        "checks": dict(decision.checks),
        "warnings": list(decision.warnings),
        "risk_style": style[0] if style else None,
    }
    storage.save_publication(record)
    return record
