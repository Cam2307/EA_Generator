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
    decision = evaluate_promotion(_report())
    assert decision.hard_gates_passed is True
    assert decision.quality_score >= 65.0
    assert decision.promotion_state in {"edge_positive", "promoted_live_watchlist"}


def test_promotion_fails_hard_gates_when_unprofitable() -> None:
    report = _report(oos_metrics=BacktestMetrics(net_profit=-50.0, profit_factor=0.9, sharpe=0.1, max_dd_pct=20.0))
    decision = evaluate_promotion(report)
    assert decision.hard_gates_passed is False
    assert decision.promotion_state in {"candidate", "validated"}


def test_duplicate_penalty_reduces_score() -> None:
    report = _report()
    base = evaluate_promotion(report)
    penalized = evaluate_promotion(report, duplicate_penalty=6.0)
    assert penalized.quality_score < base.quality_score
    assert penalized.breakdown.get("duplicate_penalty") == -6.0
