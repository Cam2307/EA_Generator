"""Tests for empirical results review."""
from __future__ import annotations

import json
from pathlib import Path

from factory import results_review
from factory.models import BacktestMetrics, ValidationReport
from factory.generator import random_strategy
from factory import results_archive
import random


def test_analyze_results_separates_infra_and_tradeable(tmp_path: Path):
    job_id = "rev_job"
    results_archive.init_run(job_id, {"symbol": "EURUSD"}, root=tmp_path)
    rng = random.Random(1)
    good = random_strategy("EURUSD", "M15", rng)
    bad = random_strategy("EURUSD", "M15", rng)

    good_report = ValidationReport(
        strategy_id=good.id,
        is_metrics=BacktestMetrics(net_profit=100, trade_count=20),
        oos_metrics=BacktestMetrics(
            net_profit=50, trade_count=12, max_dd_pct=20.0,
            profit_factor=1.3, sharpe=0.5, r_squared=0.4),
        wfe=0.5, passed=True, highest_level_passed=3,
        reasons=[],
    )
    bad_report = ValidationReport(
        strategy_id=bad.id,
        is_metrics=BacktestMetrics(),
        oos_metrics=BacktestMetrics(),
        wfe=0.0, passed=False, highest_level_passed=0,
        reasons=["Validation did not complete: MT5RunnerError: terminal running"],
    )
    results_archive.write_candidate(job_id, good, good_report, root=tmp_path)
    results_archive.write_candidate(job_id, bad, bad_report, root=tmp_path)

    report = results_review.analyze_results(tmp_path, job_ids=[job_id])
    assert report["population"]["candidates_loaded"] == 2
    assert report["population"]["tradeable"] == 1
    assert report["population"]["infra_or_empty"] == 1
    assert report["distributions_tradeable"]["wfe"]["n"] == 1
    assert report["suggestions"]


def test_flag_wfe_zero_with_profitable_oos():
    rows = [{
        "job_id": "j", "strategy_id": "s", "tradeable": True,
        "wfe": 0.0, "oos_net_profit": 40.0, "oos_profit_factor": 1.2,
        "oos_sharpe": 0.5, "oos_trade_count": 10, "highest_level_passed": 0,
        "oos_max_dd_pct": 15.0,
    }]
    flags = results_review.flag_anomalies(rows)
    assert any(f["kind"] == "wfe_zero_with_profitable_oos" for f in flags)


def test_infra_failure_flag_and_prefix(tmp_path: Path):
    job_id = "rev_infra"
    results_archive.init_run(job_id, {"symbol": "EURUSD"}, root=tmp_path)
    rng = random.Random(2)
    strat = random_strategy("EURUSD", "M15", rng)
    report = ValidationReport(
        strategy_id=strat.id,
        is_metrics=BacktestMetrics(),
        oos_metrics=BacktestMetrics(),
        wfe=0.0, passed=False, highest_level_passed=0,
        infra_failure=True,
        reasons=["INFRA: Validation did not complete: MT5RunnerError: busy"],
    )
    results_archive.write_candidate(job_id, strat, report, root=tmp_path)
    out = results_review.analyze_results(tmp_path, job_ids=[job_id])
    assert out["population"]["infra_or_empty"] == 1
    assert out["population"]["tradeable"] == 0
