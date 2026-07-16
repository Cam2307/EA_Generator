"""Unit tests for the results/ filesystem archive."""
from __future__ import annotations

import json
from pathlib import Path

from factory import results_archive, validation_levels
from factory.models import (
    BacktestMetrics, ValidationReport,
)
from factory.generator import random_strategy
import random


def test_init_run_writes_standard_layout(tmp_path: Path):
    job_id = "disc_test_archive"
    payload = {"symbol": "EURUSD", "timeframe": "M15", "validation_level": 1}
    d = results_archive.init_run(
        job_id, payload, manifest={"job_id": job_id, "seed": 1},
        started_at=1_700_000_000.0, root=tmp_path)
    assert d == tmp_path / job_id
    assert (d / "config.json").is_file()
    assert (d / "levels.json").is_file()
    assert (d / "manifest.json").is_file()
    assert (d / "run.json").is_file()
    assert (d / "candidates").is_dir()

    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    assert cfg["payload"]["symbol"] == "EURUSD"
    levels = json.loads((d / "levels.json").read_text(encoding="utf-8"))
    assert levels["schema_version"] == validation_levels.LEVEL_SCHEMA_VERSION
    assert len(levels["levels"]) == 16


def test_screen_and_candidate_archive(tmp_path: Path):
    job_id = "disc_cand"
    results_archive.init_run(job_id, {"symbol": "XAUUSD"}, root=tmp_path)
    rng = random.Random(0)
    strat = random_strategy("XAUUSD", "M15", rng)
    results_archive.append_screen(
        job_id,
        results_archive.screen_record(
            strat, promising=True,
            metrics=BacktestMetrics(net_profit=10, trade_count=5,
                                    profit_factor=1.2),
            duration_ms=12.5, fitness=1.0),
        root=tmp_path,
    )
    report = ValidationReport(
        strategy_id=strat.id,
        is_metrics=BacktestMetrics(net_profit=20, trade_count=10),
        oos_metrics=BacktestMetrics(net_profit=8, trade_count=5,
                                    profit_factor=1.1, max_dd_pct=10),
        wfe=0.4, passed=True, highest_level_passed=2,
        levels_cleared={"1": True, "2": True, "3": False},
        duration_ms=1234.0,
    )
    results_archive.write_candidate(
        job_id, strat, report, metadata={"generation": 0}, root=tmp_path)
    results_archive.finalize_run(
        job_id, status="DONE", tested=1, promising=1, survivors=1,
        root=tmp_path)

    d = tmp_path / job_id
    screens = (d / "screens.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(screens) == 1
    assert json.loads(screens[0])["promising"] is True
    cand = json.loads((d / "candidates" / f"{strat.id}.json").read_text(
        encoding="utf-8"))
    assert cand["report"]["highest_level_passed"] == 2
    assert cand["report"]["duration_ms"] == 1234.0
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "DONE"
    assert run["duration_s"] is not None
    assert (d / "summary.csv").is_file()
    summary = (d / "summary.csv").read_text(encoding="utf-8")
    assert strat.id in summary
