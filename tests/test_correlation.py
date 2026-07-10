"""Return-stream correlation for curation (factory.correlation)."""
import numpy as np

from factory.correlation import (
    daily_returns, duplicate_penalty_from_corr, max_correlation,
    return_correlation,
)
from factory.models import BacktestMetrics, ValidationReport


def _metrics(daily_pnl, start_day=19_700, samples_per_day=4, deposit=10_000.0):
    """Equity curve with several intraday samples and a known daily PnL."""
    ts, eq = [], []
    equity = deposit
    for d, pnl in enumerate(daily_pnl):
        day0 = (start_day + d) * 86400.0
        for k in range(samples_per_day):
            ts.append(day0 + k * 3600.0)
            eq.append(equity + pnl * (k + 1) / samples_per_day)
        equity += pnl
    return BacktestMetrics(equity_ts=ts, equity=eq)


def test_daily_returns_uses_last_sample_per_day():
    m = _metrics([100.0, -50.0, 25.0])
    dr = daily_returns(m)
    assert len(dr) == 2                     # first day is the base
    vals = [dr[k] for k in sorted(dr)]
    assert vals[0] < 0 < vals[1]            # -50 then +25 day


def test_return_correlation_identical_and_inverse():
    rng = np.random.default_rng(3)
    pnl = rng.normal(0, 50, 60).tolist()
    a = daily_returns(_metrics(pnl))
    b = daily_returns(_metrics(pnl))
    inv = daily_returns(_metrics([-x for x in pnl]))
    assert return_correlation(a, b) > 0.99
    assert return_correlation(a, inv) < -0.95
    # too little overlap -> None
    short = daily_returns(_metrics(pnl[:5]))
    assert return_correlation(a, short) is None


def _report(sid, pnl):
    return ValidationReport(strategy_id=sid, is_metrics=BacktestMetrics(),
                            oos_metrics=_metrics(pnl))


def test_max_correlation_finds_the_twin():
    rng = np.random.default_rng(9)
    pnl = rng.normal(10, 50, 60).tolist()
    other_pnl = rng.normal(10, 50, 60).tolist()
    me = _report("me", pnl)
    twin = _report("twin", [x * 1.01 for x in pnl])
    stranger = _report("stranger", other_pnl)
    corr, cid = max_correlation(me, [stranger, twin])
    assert cid == "twin"
    assert corr > 0.95
    # self is skipped
    corr2, cid2 = max_correlation(me, [me])
    assert corr2 is None and cid2 is None


def test_novelty_score():
    from factory.correlation import novelty_score

    rng = np.random.default_rng(21)
    pnl = rng.normal(0, 50, 60).tolist()
    fp = daily_returns(_metrics(pnl))
    twin = daily_returns(_metrics([x * 1.01 for x in pnl]))
    other = daily_returns(_metrics(rng.normal(0, 50, 60).tolist()))

    assert novelty_score(fp, []) == 1.0            # empty reservoir
    assert novelty_score({}, [twin]) == 1.0        # unmeasurable fingerprint
    assert novelty_score(fp, [twin]) < 0.05        # duplicate: near-zero novelty
    assert novelty_score(fp, [other]) > 0.5        # independent: mostly novel
    # inverse duplicates are still duplicates
    inv = daily_returns(_metrics([-x for x in pnl]))
    assert novelty_score(fp, [inv]) < 0.1


def test_duplicate_penalty_curve():
    assert duplicate_penalty_from_corr(None) == 0.0
    assert duplicate_penalty_from_corr(0.4) == 0.0
    assert duplicate_penalty_from_corr(0.6) == 0.0
    mid = duplicate_penalty_from_corr(0.8)
    full = duplicate_penalty_from_corr(1.0)
    assert 0.0 < mid < full == 15.0
    assert duplicate_penalty_from_corr(-0.9) > 0.0   # inverse dupes count too


def test_sync_promotion_scores_penalizes_correlated_candidate(tmp_path, monkeypatch):
    from factory.agent_alerts import sync_promotion_scores
    from factory.models import (
        EntryFilter, EntryFilterType, ExecutionMechanic, ExecutionMechanicType,
        StrategyDefinition,
    )
    from factory.storage import Storage

    storage = Storage(tmp_path / "corr.db")
    rng = np.random.default_rng(5)
    pnl = rng.normal(20, 40, 60).tolist()

    def _strategy(sid):
        return StrategyDefinition(
            id=sid, symbol="EURUSD", timeframe="M15",
            entry_filters=[EntryFilter(
                type=EntryFilterType.RSI_REVERSION,
                params={"rsi_period": 14, "oversold": 30, "overbought": 70})],
            mechanic=ExecutionMechanic(type=ExecutionMechanicType.STANDARD_SLTP,
                                       params={"sl_points": 300, "tp_points": 300}))

    strong = dict(net_profit=1500.0, profit_factor=1.6, sharpe=1.2,
                  max_dd_pct=8.0, trade_count=60)

    # an already-promoted strategy with this return stream
    promoted = _report("promoted", pnl)
    promoted.oos_metrics = promoted.oos_metrics.model_copy(update=strong)
    promoted.passed = True
    promoted.wfe = 0.8
    storage.save_strategy(_strategy("promoted"))
    storage.save_validation(promoted)
    storage.update_validation_promotion(
        "promoted", promotion_state="promoted_live_watchlist",
        quality_score=85.0, hard_gates_passed=True, quality_breakdown={})

    # a new candidate with a near-identical stream
    cand = _report("cand", [x * 1.02 for x in pnl])
    cand.oos_metrics = cand.oos_metrics.model_copy(update=strong)
    cand.passed = True
    cand.wfe = 0.8
    storage.save_strategy(_strategy("cand"))
    storage.save_validation(cand)

    processed = sync_promotion_scores(storage)
    assert processed >= 1
    updated = storage.get_validation("cand")
    assert updated.quality_breakdown.get("duplicate_penalty", 0.0) < 0.0
