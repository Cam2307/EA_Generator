"""Publication pipeline (factory.publication)."""
import numpy as np
import pytest

from factory.models import (
    BacktestMetrics, EntryFilter, EntryFilterType, ExecutionMechanic,
    ExecutionMechanicType, MonteCarloResult, RegimeStats, StrategyDefinition,
    ValidationReport,
)
from factory.publication import evaluate_publication, publish, risk_style
from factory.storage import Storage


def _strategy(sid="p1", mech=ExecutionMechanicType.STANDARD_SLTP,
              params=None):
    return StrategyDefinition(
        id=sid, name=f"Pub {sid}", symbol="EURUSD", timeframe="M15",
        entry_filters=[EntryFilter(
            type=EntryFilterType.RSI_REVERSION,
            params={"rsi_period": 14, "oversold": 30, "overbought": 70})],
        mechanic=ExecutionMechanic(
            type=mech, params=params or {"sl_points": 300, "tp_points": 300}))


def _publishable_report(sid="p1", **overrides):
    """A report that clears every publication check."""
    ts, eq = [], []
    equity = 10_000.0
    rng = np.random.default_rng(int(sid[-1], 36) if sid[-1].isdigit() else 7)
    for d in range(90):
        equity += float(rng.normal(20, 40))
        ts.append((19_700 + d) * 86400.0 + 43200.0)
        eq.append(equity)
    base = ValidationReport(
        strategy_id=sid, is_metrics=BacktestMetrics(),
        oos_metrics=BacktestMetrics(
            net_profit=2000.0, profit_factor=1.6, sharpe=1.2, max_dd_pct=8.0,
            trade_count=250, r_squared=0.85, equity_ts=ts, equity=eq),
        passed=True, wfe=0.85, dsr=0.97, n_trials=500,
        data_source="mt5",
        montecarlo=MonteCarloResult(n_runs=20, robustness_score=90.0,
                                    passed=True),
        regime_stats=[
            RegimeStats(code=0, name="quiet range", trades=80, net_profit=600),
            RegimeStats(code=1, name="quiet trend", trades=120, net_profit=1100),
            RegimeStats(code=3, name="volatile trend", trades=50, net_profit=300),
        ])
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _holdout_pass(sid="p1"):
    return {"strategy_id": sid, "passed": True, "net_profit": 400.0,
            "trade_count": 20, "max_dd_pct": 6.0, "error": None,
            "evaluated_at": 1.0}


def _seed(tmp_path, sid="p1", with_holdout=True, **report_overrides):
    storage = Storage(tmp_path / "pub.db")
    storage.save_strategy(_strategy(sid))
    storage.save_validation(_publishable_report(sid, **report_overrides))
    if with_holdout:
        storage.save_holdout_result(_holdout_pass(sid))
    return storage


def test_ready_when_all_checks_pass(tmp_path):
    storage = _seed(tmp_path)
    d = evaluate_publication(storage, "p1")
    assert d.ready, d.reasons
    assert all(d.checks.values())
    assert d.warnings == []              # plain SL/TP: no risk flag


def test_each_gate_blocks(tmp_path):
    cases = {
        "oos_trades": {"oos_metrics": BacktestMetrics(
            net_profit=2000, profit_factor=1.6, trade_count=50)},
        "dsr": {"dsr": 0.6},
        "wfe": {"wfe": 0.5},
        "real_data": {"data_source": "synthetic"},
        "montecarlo": {"montecarlo": None},
        "regimes": {"regime_stats": [RegimeStats(
            code=1, name="quiet trend", trades=200, net_profit=2000)]},
    }
    for i, (check, overrides) in enumerate(cases.items()):
        storage = _seed(tmp_path / str(i), **overrides)
        d = evaluate_publication(storage, "p1")
        assert not d.ready
        assert d.checks.get(check) is False, (check, d.checks)


def test_holdout_required(tmp_path):
    storage = _seed(tmp_path, with_holdout=False)
    d = evaluate_publication(storage, "p1")
    assert not d.ready and d.checks["holdout"] is False
    assert any("holdout" in r for r in d.reasons)


def test_correlation_blocks_near_duplicate_of_published(tmp_path):
    storage = _seed(tmp_path, sid="p1")
    publish(storage, "p1")
    # p2: essentially the same return stream
    storage.save_strategy(_strategy("p2"))
    rep1 = storage.get_validation("p1")
    rep2 = _publishable_report("p2")
    rep2.oos_metrics = rep2.oos_metrics.model_copy(update={
        "equity_ts": rep1.oos_metrics.equity_ts,
        "equity": [e * 1.01 for e in rep1.oos_metrics.equity]})
    storage.save_validation(rep2)
    storage.save_holdout_result(_holdout_pass("p2"))
    d = evaluate_publication(storage, "p2")
    assert d.checks["uncorrelated"] is False
    assert not d.ready


def test_risk_style_labels_and_warning(tmp_path):
    plain = _strategy("s")
    assert risk_style(plain) is None
    mart = _strategy("m", ExecutionMechanicType.DCA_GRID,
                     {"grid_step_points": 100, "lot_multiplier": 1.6,
                      "max_levels": 4, "basket_tp_points": 100})
    assert risk_style(mart) == ("Martingale grid", "red")
    flat = _strategy("f", ExecutionMechanicType.DCA_GRID,
                     {"grid_step_points": 100, "lot_multiplier": 1.0,
                      "max_levels": 4, "basket_tp_points": 100})
    assert risk_style(flat) == ("DCA grid", "amber")
    hedge = _strategy("h", ExecutionMechanicType.HEDGE_LAYER,
                      {"sl_points": 300, "tp_points": 300,
                       "hedge_trigger_points": 150, "hedge_ratio": 1.0})
    assert risk_style(hedge) == ("Hedge recovery", "violet")

    # warning (not a blocker) on the publication decision
    storage = Storage(tmp_path / "risk.db")
    storage.save_strategy(mart)
    storage.save_validation(_publishable_report("m"))
    storage.save_holdout_result(_holdout_pass("m"))
    d = evaluate_publication(storage, "m")
    assert d.ready
    assert any("Martingale" in w for w in d.warnings)


def test_publish_records_and_refuses_unready(tmp_path):
    storage = _seed(tmp_path)
    record = publish(storage, "p1", version="1.2.0")
    assert record["version"] == "1.2.0" and not record["forced"]
    stored = storage.get_publication("p1")
    assert stored["strategy_name"] == "Pub p1"
    assert (tmp_path / "pub.db").exists()

    bad = _seed(tmp_path / "bad", with_holdout=False)
    with pytest.raises(ValueError, match="not publication-ready"):
        publish(bad, "p1")
    forced = publish(bad, "p1", force=True)
    assert forced["forced"] is True


def test_risk_badge_helper():
    from app.components.strategy_card import risk_style_badge
    mart = _strategy("m", ExecutionMechanicType.DCA_GRID,
                     {"grid_step_points": 100, "lot_multiplier": 1.5,
                      "max_levels": 4, "basket_tp_points": 100})
    badge = risk_style_badge(mart)
    assert "Martingale grid" in badge and badge.startswith(":red-badge")
    assert risk_style_badge(_strategy("s")) == ""
