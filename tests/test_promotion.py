from factory.models import BacktestMetrics, ValidationReport
from factory.promotion import evaluate_promotion


def _report(**overrides) -> ValidationReport:
    base = ValidationReport(
        strategy_id="s1",
        is_metrics=BacktestMetrics(),
        oos_metrics=BacktestMetrics(
            net_profit=1500.0,
            profit_factor=1.6,
            sharpe=1.2,
            max_dd_pct=8.0,
            trade_count=50,
        ),
        wfe=0.75,
        passed=True,
        stability_ratio=0.85,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_promotion_hits_edge_positive() -> None:
    decision = evaluate_promotion(_report(), holdout_passed=True)
    assert decision.hard_gates_passed is True
    assert decision.quality_score >= 65.0
    assert decision.promotion_state in {"edge_positive", "promoted_live_watchlist"}


def test_promotion_requires_holdout_for_edge() -> None:
    decision = evaluate_promotion(_report(), holdout_passed=False)
    assert decision.promotion_state == "validated"


def test_promotion_watchlist_requires_mt5() -> None:
    # High score + holdout but no MT5 → edge_positive, not watchlist.
    decision = evaluate_promotion(
        _report(),
        holdout_passed=True,
        mt5_confirmed=False,
        promote_threshold=50.0,
        edge_threshold=50.0,
    )
    assert decision.promotion_state == "edge_positive"

    confirmed = evaluate_promotion(
        _report(),
        holdout_passed=True,
        mt5_confirmed=True,
        promote_threshold=50.0,
        edge_threshold=50.0,
    )
    assert confirmed.promotion_state == "promoted_live_watchlist"


def test_promotion_fails_hard_gates_when_unprofitable() -> None:
    report = _report(oos_metrics=BacktestMetrics(net_profit=-50.0, profit_factor=0.9, sharpe=0.1, max_dd_pct=20.0))
    decision = evaluate_promotion(report, holdout_passed=True)
    assert decision.hard_gates_passed is False
    assert decision.promotion_state in {"candidate", "validated"}


def test_duplicate_penalty_reduces_score() -> None:
    report = _report()
    base = evaluate_promotion(report, holdout_passed=True)
    penalized = evaluate_promotion(report, duplicate_penalty=6.0, holdout_passed=True)
    assert penalized.quality_score < base.quality_score
    assert penalized.breakdown.get("duplicate_penalty") == -6.0


def test_quality_score_capped_at_100() -> None:
    from factory.models import MonteCarloResult

    report = _report(
        wfe=2.0,
        stability_ratio=1.5,
        degradation_pct=-50.0,
        oos_metrics=BacktestMetrics(
            net_profit=50_000.0,
            profit_factor=10.0,
            sharpe=5.0,
            max_dd_pct=0.0,
            trade_count=500,
        ),
        montecarlo=MonteCarloResult(robustness_score=100.0),
        best_params={},
    )
    decision = evaluate_promotion(report, holdout_passed=True, mt5_confirmed=True)
    assert decision.quality_score <= 100.0
    positive = sum(v for v in decision.breakdown.values() if v > 0)
    assert positive <= 100.0 + 1e-9
