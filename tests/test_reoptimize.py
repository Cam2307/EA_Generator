"""Online re-optimization service (factory.reoptimize)."""
from datetime import datetime, timezone

from factory.backtest.base import BacktestEngine
from factory.models import (
    BacktestMetrics, EntryFilter, EntryFilterType, ExecutionMechanic,
    ExecutionMechanicType, ParamRange, StrategyDefinition, ValidationReport,
)
from factory.reoptimize import (
    format_reopt_report, reoptimize_promoted, reoptimize_strategy,
)
from factory.storage import Storage

NOW = datetime(2026, 7, 1, tzinfo=timezone.utc)


def _strategy(sid="reopt1"):
    return StrategyDefinition(
        id=sid, name=f"Reopt {sid}", symbol="EURUSD", timeframe="M15",
        entry_filters=[EntryFilter(
            type=EntryFilterType.RSI_REVERSION,
            params={"rsi_period": 14, "oversold": 30, "overbought": 70},
            ranges={"oversold": ParamRange(min=20, max=40, step=5)})],
        mechanic=ExecutionMechanic(
            type=ExecutionMechanicType.STANDARD_SLTP,
            params={"sl_points": 300, "tp_points": 300},
            ranges={"sl_points": ParamRange(min=100, max=600, step=100)}))


class PeakedEngine(BacktestEngine):
    """Fitness peaks at a configurable sl_points value — moving the peak
    between validation time and now simulates market drift."""
    name = "stub"

    def __init__(self, best_sl: float):
        self.best_sl = best_sl

    def run(self, strategy, start, end, params_override=None, deposit=10_000.0):
        if params_override:
            strategy = strategy.apply_flat_params(params_override)
        sl = strategy.mechanic.params.get("sl_points", 300.0)
        profit = 2000.0 - 3.0 * abs(sl - self.best_sl)
        return BacktestMetrics(
            net_profit=profit, initial_deposit=deposit, trade_count=40,
            max_dd_pct=8.0, profit_factor=1.5, r_squared=0.8,
            start_ts=start.timestamp(), end_ts=end.timestamp())


def _seed_storage(tmp_path, incumbent_sl=300.0, state="promoted_live_watchlist"):
    storage = Storage(tmp_path / "reopt.db")
    strat = _strategy()
    storage.save_strategy(strat)
    report = ValidationReport(
        strategy_id=strat.id, is_metrics=BacktestMetrics(),
        oos_metrics=BacktestMetrics(net_profit=1500, trade_count=40),
        passed=True, wfe=0.8,
        best_params={"M_STANDARD_SLTP_sl_points": incumbent_sl})
    storage.save_validation(report)
    storage.update_validation_promotion(
        strat.id, promotion_state=state, quality_score=85.0,
        hard_gates_passed=True, quality_breakdown={})
    return storage, strat


def test_no_drift_when_incumbent_still_optimal(tmp_path):
    storage, strat = _seed_storage(tmp_path, incumbent_sl=300.0)
    res = reoptimize_strategy(storage, strat.id, engine=PeakedEngine(300.0),
                              seed=1, now=NOW)
    assert res.error is None
    assert not res.drifted
    assert res.improvement <= 0.10


def test_drift_detected_and_set_written(tmp_path):
    # incumbent fit at 300, but the market's optimum moved to 600
    storage, strat = _seed_storage(tmp_path, incumbent_sl=300.0)
    res = reoptimize_strategy(storage, strat.id, engine=PeakedEngine(600.0),
                              seed=3, n_samples=40, now=NOW,
                              out_dir=tmp_path / "sets")
    assert res.error is None
    assert res.drifted, (res.improvement, res.new_params)
    assert res.improvement > 0.10
    assert res.new_params["M_STANDARD_SLTP_sl_points"] == 600.0
    assert "M_STANDARD_SLTP_sl_points" in res.changed_params
    assert res.set_path is not None and res.set_path.exists()
    content = res.set_path.read_text(encoding="utf-16")
    assert "Inp_M_sl_points=600" in content

    report = format_reopt_report([res])
    assert "[DRIFT]" in report and "1/1" in report


def test_reoptimize_promoted_filters_by_state(tmp_path):
    storage, strat = _seed_storage(tmp_path, state="validated")
    results = reoptimize_promoted(storage, engine=PeakedEngine(300.0), now=NOW)
    assert results == []                 # 'validated' is not a promoted state

    storage.update_validation_promotion(
        strat.id, promotion_state="edge_positive", quality_score=70.0,
        hard_gates_passed=True, quality_breakdown={})
    results = reoptimize_promoted(storage, engine=PeakedEngine(300.0),
                                  seed=1, now=NOW)
    assert len(results) == 1 and results[0].error is None


def test_missing_strategy_reports_error(tmp_path):
    storage = Storage(tmp_path / "empty.db")
    res = reoptimize_strategy(storage, "ghost", engine=PeakedEngine(300.0),
                              now=NOW)
    assert res.error == "strategy not found"
