"""Unit tests for multi-level population scoring."""
from __future__ import annotations

from factory.models import BacktestMetrics, MonteCarloResult
from factory import validation_levels


def _oos(**kwargs) -> BacktestMetrics:
    base = dict(
        net_profit=500.0,
        initial_deposit=10_000.0,
        max_dd_pct=12.0,
        trade_count=40,
        profit_factor=1.6,
        sharpe=1.1,
        r_squared=0.82,
        max_consecutive_losses=4,
    )
    base.update(kwargs)
    return BacktestMetrics(**base)


def _mc(**kwargs) -> MonteCarloResult:
    base = dict(
        n_runs=50,
        pct_profitable=0.90,
        dd_p95=12.0,
        resample_dd_p95=11.0,
        passed=True,
    )
    base.update(kwargs)
    return MonteCarloResult(**base)


def test_level_table_shape():
    assert len(validation_levels.VALIDATION_LEVELS) == 16
    assert validation_levels.MIN_LEVEL == 1
    assert validation_levels.MAX_LEVEL == 16
    assert validation_levels.MC_UNLOCK_LEVEL == 7
    assert validation_levels.MC_PRE_LEVEL == 6
    assert validation_levels.HONESTY_UNLOCK_LEVEL == 7
    assert validation_levels.DEFAULT_LEVEL == 4
    assert validation_levels.GALLERY_DEFAULT_MIN_LEVEL == 7
    assert validation_levels.LEVEL_SCHEMA_VERSION == 2
    assert validation_levels.LEGACY_LEVEL_MAP == {
        1: 1, 2: 4, 3: 7, 4: 10, 5: 13, 6: 16,
    }
    assert validation_levels.KPI_ANCHORS == (4, 7, 10)


def test_display_label_and_band():
    assert validation_levels.display_label(7) == "7 · Standard·A"
    assert validation_levels.band_for_level(7) == "Standard"
    assert validation_levels.band_for_level(16) == "Elite"
    assert validation_levels.badge_color_for_level(1) == "blue"
    assert validation_levels.badge_color_for_level(13) == "orange"
    assert validation_levels.badge_color_for_level(16) == "red"


def test_remap_legacy_level():
    assert validation_levels.remap_legacy_level(0) == 0
    assert validation_levels.remap_legacy_level(3, schema_version=1) == 7
    assert validation_levels.remap_legacy_level(6, schema_version=1) == 16
    # Already on fine schema — clamp only, do not re-map band anchors.
    assert validation_levels.remap_legacy_level(3, schema_version=2) == 3
    assert validation_levels.remap_legacy_level(16, schema_version=2) == 16


def test_highest_level_cleared_none():
    oos = _oos(net_profit=-10.0, profit_factor=0.5, trade_count=2)
    assert validation_levels.highest_level_cleared(oos, wfe=-0.1) == 0


def test_highest_level_cleared_l1_only():
    # Clears Screener·A but not Screener·B (WFE / trades / PF).
    oos = _oos(
        max_dd_pct=68.0, trade_count=3, profit_factor=0.97,
        sharpe=0.0, r_squared=0.0)
    assert validation_levels.highest_level_cleared(oos, wfe=0.05) == 1


def test_highest_level_cleared_standard_with_mc():
    # Metrics clear Standard·A (L7); not Standard·B (Sharpe / MC).
    oos = _oos(
        max_dd_pct=41.0, trade_count=15, profit_factor=1.16,
        sharpe=0.28, r_squared=0.46, max_consecutive_losses=5)
    mc = _mc(
        pct_profitable=0.66, dd_p95=40.0, resample_dd_p95=39.0, n_runs=20,
        path_runs=10, path_pct_profitable=0.55)
    honesty = validation_levels.HonestySignals(
        p_oos_loss=0.50, dsr=0.9, stability_ratio=0.9)
    assert validation_levels.highest_level_cleared(
        oos, wfe=0.51, montecarlo=mc, honesty=honesty) == 7


def test_l7_gate_values():
    """Phase-2 modest L7 easing (search-space expansion companion)."""
    l7 = validation_levels.get_level(7)
    assert l7.criteria.max_dd_pct == 48.0
    assert l7.criteria.min_sharpe == 0.15
    assert l7.criteria.min_r_squared == 0.35
    assert l7.mc_min_profitable == 0.60
    assert l7.mc_max_dd_p95 == 52.0
    assert l7.max_p_oos_loss == 0.65
    assert l7.mc_min_path_profitable == 0.50


def test_highest_level_cleared_robust_with_strong_mc():
    oos = _oos(
        max_dd_pct=34.0, trade_count=24, profit_factor=1.30,
        sharpe=0.65, r_squared=0.62, max_consecutive_losses=5)
    mc = _mc(
        pct_profitable=0.73, dd_p95=36.0, resample_dd_p95=35.0, n_runs=40,
        path_runs=10, path_pct_profitable=0.60)
    honesty = validation_levels.HonestySignals(
        p_oos_loss=0.30, dsr=0.9, stability_ratio=0.70)
    assert validation_levels.highest_level_cleared(
        oos, wfe=0.63, montecarlo=mc, honesty=honesty) == 10


def test_honesty_wfo_blocks_l7():
    oos = _oos(
        max_dd_pct=41.0, trade_count=15, profit_factor=1.16,
        sharpe=0.28, r_squared=0.46, max_consecutive_losses=5)
    mc = _mc(
        pct_profitable=0.66, dd_p95=40.0, resample_dd_p95=39.0, n_runs=20,
        path_runs=10, path_pct_profitable=0.55)
    bad = validation_levels.HonestySignals(
        p_oos_loss=0.90, dsr=0.9, stability_ratio=0.9)
    assert validation_levels.highest_level_cleared(
        oos, wfe=0.51, montecarlo=mc, honesty=bad) == 6


def test_path_mc_blocks_l7_when_path_runs_present():
    oos = _oos(
        max_dd_pct=41.0, trade_count=15, profit_factor=1.16,
        sharpe=0.28, r_squared=0.46, max_consecutive_losses=5)
    mc = _mc(
        pct_profitable=0.66, dd_p95=40.0, resample_dd_p95=39.0, n_runs=20,
        path_runs=10, path_pct_profitable=0.20)
    honesty = validation_levels.HonestySignals(
        p_oos_loss=0.40, dsr=0.9, stability_ratio=0.9)
    assert validation_levels.highest_level_cleared(
        oos, wfe=0.51, montecarlo=mc, honesty=honesty) == 6


def test_dsr_blocks_elite():
    oos = _oos()
    mc = _mc(
        pct_profitable=0.95, dd_p95=8.0, resample_dd_p95=8.0, n_runs=100,
        path_runs=10, path_pct_profitable=0.90)
    # Strong metrics, but DSR too low for Strict·B (needs 0.55).
    honesty = validation_levels.HonestySignals(
        p_oos_loss=0.10, dsr=0.50, stability_ratio=0.90)
    hi = validation_levels.highest_level_cleared(
        oos, wfe=0.90, montecarlo=mc, honesty=honesty)
    assert hi == 13  # Strict·A clears (min_dsr=0.45); Strict·B needs 0.55


def test_highest_level_cleared_respects_ceiling():
    oos = _oos()
    mc = _mc(pct_profitable=0.95, dd_p95=8.0, resample_dd_p95=8.0, n_runs=100)
    # Strong enough for high tiers, but ceiling caps the award.
    assert validation_levels.highest_level_cleared(
        oos, wfe=0.90, montecarlo=mc, ceiling=4) == 4


def test_mc_required_blocks_l7_without_mc():
    oos = _oos(
        max_dd_pct=18.0, trade_count=20, profit_factor=1.25,
        sharpe=0.4, r_squared=0.55)
    # Metrics clear Basic·C (L6); without MC cannot award Standard·A (L7).
    assert validation_levels.highest_level_cleared(oos, wfe=0.60) == 6


def test_levels_cleared_map():
    oos = _oos(
        max_dd_pct=68.0, trade_count=3, profit_factor=0.97,
        sharpe=0.0, r_squared=0.0)
    cleared = validation_levels.levels_cleared_map(oos, wfe=0.05)
    assert cleared["1"] is True
    assert cleared["2"] is False
    assert len(cleared) == 16


def test_l1_clears_with_zero_wfe_when_oos_profitable():
    """L1 min_wfe=0 disables the WFE gate — OOS-profitable still clears."""
    oos = _oos(
        max_dd_pct=68.0, trade_count=3, profit_factor=0.97,
        sharpe=0.0, r_squared=0.0, net_profit=50.0)
    assert validation_levels.level_clears(
        validation_levels.get_level(1), oos, wfe=0.0)
    assert validation_levels.highest_level_cleared(oos, wfe=0.0) == 1


def test_screener_soft_wfe_when_oos_profitable():
    """L2–L3 soft-waive WFE when OOS is profitable and other gates pass."""
    # Meets L2 DD/trades/PF but fails hard WFE (0.05 < 0.10).
    oos = _oos(
        max_dd_pct=60.0, trade_count=5, profit_factor=1.05,
        sharpe=0.0, r_squared=0.0, net_profit=80.0)
    assert validation_levels.level_clears(
        validation_levels.get_level(2), oos, wfe=0.05)
    # Soft WFE does not apply at Basic (L4+): same pattern, hard WFE fails.
    basic_oos = _oos(
        max_dd_pct=50.0, trade_count=10, profit_factor=1.10,
        sharpe=0.0, r_squared=0.0, net_profit=80.0)
    assert not validation_levels.level_clears(
        validation_levels.get_level(4), basic_oos, wfe=0.05)


def test_screener_soft_wfe_requires_profitable_oos():
    oos = _oos(
        max_dd_pct=60.0, trade_count=5, profit_factor=1.05,
        sharpe=0.0, r_squared=0.0, net_profit=-10.0)
    assert not validation_levels.level_clears(
        validation_levels.get_level(2), oos, wfe=0.05)


def test_human_gates_l1_skips_wfe_requirement():
    gates = validation_levels.get_level(1).human_gates()
    assert any("not required" in g.lower() for g in gates)
    gates_l2 = validation_levels.get_level(2).human_gates()
    assert any("soft" in g.lower() for g in gates_l2)


def test_reasons_for_next_level():
    oos = _oos(max_dd_pct=68.0, trade_count=3, profit_factor=0.97)
    reasons = validation_levels.reasons_for_next_level(
        oos, wfe=0.05, highest=1, ceiling=16)
    assert reasons
    assert any("L2" in r for r in reasons)


def test_mc_unlock_level_has_montecarlo():
    assert not validation_levels.get_level(
        validation_levels.MC_PRE_LEVEL).montecarlo
    assert validation_levels.get_level(
        validation_levels.MC_UNLOCK_LEVEL).montecarlo


def test_storage_lazy_remaps_legacy_levels(tmp_path):
    """Opening Storage remaps legacy L1–L6 columns without re-backtesting."""
    import json
    import time

    from factory.storage import Storage

    db = tmp_path / "legacy_levels.db"
    st = Storage(db)
    with st.connection() as con:
        con.execute(
            "INSERT OR REPLACE INTO app_settings (key, value, updated_at) "
            "VALUES (?, ?, ?)",
            ("validation_level_schema_version", json.dumps(1), time.time()),
        )
        con.execute(
            "INSERT OR REPLACE INTO app_settings (key, value, updated_at) "
            "VALUES (?, ?, ?)",
            ("discovery_validation_level", json.dumps(6), time.time()),
        )
        for i, lvl in enumerate([1, 2, 3, 4, 5, 6]):
            con.execute(
                "INSERT OR REPLACE INTO validations "
                "(strategy_id, passed, wfe, body, updated_at, highest_level_passed) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (f"s{i}", 1, 0.5, "{}", time.time(), lvl),
            )

    # Re-open applies the schema migration.
    st2 = Storage(db)
    with st2.connection() as con:
        levels = [
            r["highest_level_passed"]
            for r in con.execute(
                "SELECT highest_level_passed FROM validations "
                "ORDER BY strategy_id"
            )
        ]
        schema = json.loads(
            con.execute(
                "SELECT value FROM app_settings WHERE key=?",
                ("validation_level_schema_version",),
            ).fetchone()["value"]
        )
        dial = json.loads(
            con.execute(
                "SELECT value FROM app_settings WHERE key=?",
                ("discovery_validation_level",),
            ).fetchone()["value"]
        )
    assert levels == [1, 4, 7, 10, 13, 16]
    assert schema == validation_levels.LEVEL_SCHEMA_VERSION
    assert dial == 16
