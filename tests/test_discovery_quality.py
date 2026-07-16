"""Tests for survivor-biased mechanic weights and infra KPI counting."""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from factory.generator import (
    GenerationSettings, blend_family_weights, blend_filter_weights,
    blend_mechanic_weights, default_mechanic_weights, infer_hypothesis_family,
    random_strategy, _hypothesis_family_pool, _pick_filter_pack,
)
from factory.models import (
    BacktestMetrics, EntryFilterType, ExecutionMechanicType, ValidationReport,
)
from factory.storage import Storage
from factory.symbol_class import HYPOTHESIS_FAMILIES


def test_blend_mechanic_weights_boosts_clearing_styles():
    prior = default_mechanic_weights()
    blended = blend_mechanic_weights(
        {ExecutionMechanicType.PARTIAL_CLOSE.value: 3})
    assert blended[ExecutionMechanicType.PARTIAL_CLOSE] > prior[
        ExecutionMechanicType.PARTIAL_CLOSE]
    assert blended[ExecutionMechanicType.STANDARD_SLTP] == prior[
        ExecutionMechanicType.STANDARD_SLTP]


def test_infra_failure_excluded_from_tradeable_kpi(tmp_path: Path):
    storage = Storage(tmp_path / "kpi.db")
    ok = random_strategy("EURUSD", "H1", rng=random.Random(1))
    bad = random_strategy("EURUSD", "H1", rng=random.Random(2))
    storage.save_complete(
        ok,
        ValidationReport(
            strategy_id=ok.id,
            is_metrics=BacktestMetrics(trade_count=10, net_profit=50),
            oos_metrics=BacktestMetrics(trade_count=10, net_profit=40,
                                        profit_factor=1.2),
            passed=False,
            highest_level_passed=0,
            infra_failure=False,
            engine="simulator",
        ),
        job_id="j1",
    )
    storage.save_complete(
        bad,
        ValidationReport(
            strategy_id=bad.id,
            is_metrics=BacktestMetrics(),
            oos_metrics=BacktestMetrics(),
            passed=False,
            highest_level_passed=0,
            infra_failure=True,
            reasons=["INFRA: terminal busy"],
            engine="mt5",
        ),
        job_id="j1",
    )
    assert storage.count_validated(passed_only=None) == 2
    assert storage.count_validated(passed_only=None, exclude_infra=True) == 1
    assert storage.count_infra_failures() == 1
    prog = storage.run_progress_by_jobs(["j1"])["j1"]
    assert prog["total"] == 2
    assert prog["tradeable"] == 1
    assert prog["infra"] == 1


def test_list_cleared_strategies_filters_level(tmp_path: Path):
    storage = Storage(tmp_path / "elite.db")
    strat = random_strategy("USDJPY", "M15", rng=random.Random(3))
    storage.save_complete(
        strat,
        ValidationReport(
            strategy_id=strat.id,
            is_metrics=BacktestMetrics(trade_count=20, net_profit=100),
            oos_metrics=BacktestMetrics(
                trade_count=20, net_profit=80, profit_factor=1.3, sharpe=0.5),
            passed=True,
            highest_level_passed=5,
            wfe=0.5,
            engine="simulator",
        ),
        job_id="j2",
    )
    found = storage.list_cleared_strategies(
        symbol="USDJPY", timeframe="M15", min_level=4, limit=5)
    assert len(found) == 1
    assert found[0].id == strat.id
    assert storage.list_cleared_strategies(
        symbol="EURUSD", timeframe="M15", min_level=4) == []


def test_blend_family_and_filter_weights():
    fam = blend_family_weights({"trend_follow": 2})
    assert fam["trend_follow"] > fam["breakout_atr"]
    filt = blend_filter_weights({EntryFilterType.MA_CROSS.value: 3})
    assert filt[EntryFilterType.MA_CROSS.value] > filt[
        EntryFilterType.RSI_REVERSION.value]


def test_hypothesis_family_pool_respects_weights():
    compatible = list(EntryFilterType)
    # Heavy weight on mean_reversion_pct only.
    weights = {n: 0.01 for n in HYPOTHESIS_FAMILIES}
    weights["mean_reversion_pct"] = 1000.0
    picks = []
    for i in range(40):
        pool = _hypothesis_family_pool(
            compatible, random.Random(i), family_weights=weights)
        # Infer which family was chosen by membership.
        vals = {ft.value for ft in pool}
        best = max(
            HYPOTHESIS_FAMILIES,
            key=lambda n: len(vals & set(HYPOTHESIS_FAMILIES[n])),
        )
        picks.append(best)
    assert Counter(picks)["mean_reversion_pct"] >= 30


def test_pick_filter_pack_prefers_weighted_filters():
    settings = GenerationSettings(
        hypothesis_families=False,
        filter_weights={
            EntryFilterType.RSI_REVERSION.value: 1000.0,
            EntryFilterType.MA_CROSS.value: 0.01,
        },
    )
    # Compatible set that includes both.
    compatible = [
        EntryFilterType.RSI_REVERSION, EntryFilterType.MA_CROSS,
        EntryFilterType.BOLLINGER_FADE,
    ]
    hits = 0
    for i in range(40):
        pack = _pick_filter_pack(compatible, random.Random(i), settings)
        if EntryFilterType.RSI_REVERSION in pack:
            hits += 1
    assert hits >= 25


def test_family_filter_clear_counts_and_class_elite(tmp_path: Path):
    storage = Storage(tmp_path / "priors.db")
    # Build a trend-follow-ish EURUSD elite and a GBPUSD peer.
    eurusd = random_strategy("EURUSD", "H1", rng=random.Random(10))
    gbpusd = random_strategy("GBPUSD", "H1", rng=random.Random(11))
    for strat, jid in ((eurusd, "ja"), (gbpusd, "jb")):
        storage.save_complete(
            strat,
            ValidationReport(
                strategy_id=strat.id,
                is_metrics=BacktestMetrics(trade_count=20, net_profit=100),
                oos_metrics=BacktestMetrics(
                    trade_count=20, net_profit=80, profit_factor=1.3),
                passed=True,
                highest_level_passed=4,
                engine="simulator",
            ),
            job_id=jid,
        )
    fam_counts = storage.family_clear_counts(min_level=4)
    filt_counts = storage.filter_clear_counts(min_level=4)
    assert sum(fam_counts.values()) >= 1
    assert sum(filt_counts.values()) >= 1
    # Same-class transfer: ask for FX class on H1 without pinning EURUSD.
    peers = storage.list_cleared_strategies(
        timeframe="H1", symbol_class="fx", min_level=4, limit=10)
    assert len(peers) >= 2
    assert storage.job_max_level_passed("ja") == 4


def test_infer_hypothesis_family_matches_filters():
    strat = random_strategy(
        "EURUSD", "M15", rng=random.Random(42),
        generation_settings=GenerationSettings(hypothesis_families=True),
    )
    fam = infer_hypothesis_family(strat)
    assert fam in HYPOTHESIS_FAMILIES or fam is None


def test_discovery_yield_defaults_raised():
    from config import settings

    assert int(settings.DISCOVERY_ELITE_SEED_COUNT) >= 8
    assert int(settings.DISCOVERY_MAX_EDGE_VARIANTS) >= 8


def test_adaptive_fresh_blood_ratio():
    """Mirror worker policy: 50% when thin parents / no survivors, else 25%."""
    remaining = 64

    def n_fresh(breedable_n: int, survivors: int) -> int:
        if breedable_n < 20 or survivors == 0:
            return max(1, remaining // 2)
        return max(1, remaining // 4)

    assert n_fresh(5, 0) == 32
    assert n_fresh(50, 0) == 32
    assert n_fresh(50, 1) == 16
