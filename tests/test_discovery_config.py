from datetime import datetime, timezone

import pytest

from factory.discovery_config import (
    DiscoverySettings,
    build_discovery_payload,
    derive_wfo_from_duration,
    effective_validation_level,
    history_start_end,
    settings_from_app,
    settings_to_app,
)
from factory import validation_levels
from jobs.sweep import plan_sweeps


def test_plan_sweeps_builds_symbol_timeframe_grid() -> None:
    plans = plan_sweeps(
        symbols=["EURUSD", "GBPUSD"],
        timeframes=["M15", "H1"],
        months=6,
        base_seed=100,
    )
    assert len(plans) == 4
    assert plans[0].seed == 100
    assert plans[-1].seed == 103
    assert plans[0].payload_patch["symbol"] == "EURUSD"
    assert plans[0].payload_patch["timeframe"] == "M15"
    assert "strictness_profile" not in plans[0].payload_patch


def test_settings_from_app_falls_back_to_agent_keys() -> None:
    cfg = settings_from_app(
        {
            "agent_symbols": ["GBPUSD"],
            "agent_timeframes": ["M30"],
            "agent_history_months": 9,
            "agent_batch_size": 50,
        }
    )
    assert cfg.symbols == ["GBPUSD"]
    assert cfg.timeframes == ["M30"]
    assert cfg.months == 9
    assert cfg.batch_size == 50
    # WFO is derived from months, not legacy stored WFO keys
    assert cfg.wfo_train_months == derive_wfo_from_duration(9)[0]


def test_build_discovery_payload_includes_full_manual_fields() -> None:
    cfg = DiscoverySettings(
        engine="mt5",
        deposit=25_000.0,
        leverage=200,
        mechanics=["standard_sltp"],
        tm_features=["trailing"],
        validation_level=7,
        months=6,
    )
    payload = build_discovery_payload(cfg, symbol="EURUSD", timeframe="H1", seed=42)
    assert payload["engine"] == "mt5"
    assert payload["deposit"] == 25_000.0
    assert payload["leverage"] == 200
    assert payload["mechanics"] == ["standard_sltp"]
    assert payload["tm_features"] == ["trailing"]
    # Default progressive_strictness=True → floor starts at L1; score ceiling is always L16.
    assert payload["validation_level"] == validation_levels.MIN_LEVEL
    assert payload["validation_level_floor"] == validation_levels.MIN_LEVEL
    assert payload["validation_level_ceiling"] == validation_levels.MAX_LEVEL
    assert payload["validation_level_target"] == 7
    assert payload.get("progressive_strictness") is True
    assert payload["progressive_step"] == validation_levels.DEFAULT_PROGRESSIVE_STEP
    assert payload["seed"] == 42
    assert payload["symbol"] == "EURUSD"
    assert payload["test_duration_months"] == 6
    assert "data_source" in payload
    train, test, windows = derive_wfo_from_duration(6)
    assert payload["wfo_train_months"] == train
    assert payload["wfo_test_months"] == test
    assert payload["wfo_windows"] == windows


def test_build_discovery_payload_custom_gates() -> None:
    cfg = DiscoverySettings(
        use_custom=True,
        custom_criteria={"min_wfe": 0.8, "max_dd_pct": 10.0},
        custom_montecarlo=True,
        custom_mc_runs=30,
    )
    payload = build_discovery_payload(cfg, symbol="GBPUSD", timeframe="M15")
    assert "validation_level" not in payload
    assert payload["criteria"]["min_wfe"] == 0.8
    assert payload["montecarlo"] is True
    assert payload["mc_runs"] == 30


def test_build_discovery_payload_auto_symbol_economics() -> None:
    cfg = DiscoverySettings(
        auto_symbol_economics=True,
        contract_size=100_000.0,
        spread_points=15.0,
        slippage_points=9.0,
    )
    gold = build_discovery_payload(cfg, symbol="XAUUSD", timeframe="H1")
    assert gold["auto_symbol_economics"] is True
    assert gold["contract_size"] == 100.0
    assert gold["spread_points"] == 25.0
    assert gold["slippage_points"] == 5.0  # per-symbol, not UI override
    assert gold["edge_first"] is True

    fx = build_discovery_payload(cfg, symbol="EURUSD", timeframe="H1")
    assert fx["contract_size"] == 100_000.0
    assert fx["spread_points"] == 12.0
    assert fx["slippage_points"] == 2.0

    manual = DiscoverySettings(
        auto_symbol_economics=False,
        contract_size=50_000.0,
        spread_points=7.0,
        slippage_points=4.0,
        edge_first=False,
    )
    forced = build_discovery_payload(manual, symbol="XAUUSD", timeframe="H1")
    assert forced["auto_symbol_economics"] is False
    assert forced["contract_size"] == 50_000.0
    assert forced["spread_points"] == 7.0
    assert forced["slippage_points"] == 4.0
    assert forced["edge_first"] is False


def test_settings_round_trip_edge_first() -> None:
    original = DiscoverySettings(edge_first=False, symbols=["EURUSD"])
    restored = settings_from_app(settings_to_app(original))
    assert restored.edge_first is False


def test_settings_round_trip() -> None:
    original = DiscoverySettings(
        symbols=["USDJPY"], timeframes=["D1"], months=18,
        auto_symbol_economics=False, contract_size=50_000.0)
    restored = settings_from_app(settings_to_app(original))
    assert restored.symbols == ["USDJPY"]
    assert restored.timeframes == ["D1"]
    assert restored.months == 18
    assert restored.auto_symbol_economics is False
    assert restored.contract_size == 50_000.0
    assert restored.wfo_train_months == derive_wfo_from_duration(18)[0]


def test_derive_wfo_scales_with_duration() -> None:
    assert derive_wfo_from_duration(6) == (2, 1, 2)
    assert derive_wfo_from_duration(12) == (4, 2, 4)
    short = derive_wfo_from_duration(3)
    assert short[0] + short[1] <= 3 or short == (1, 1, 1)
    long_train, long_test, long_windows = derive_wfo_from_duration(24)
    short_train, short_test, short_windows = derive_wfo_from_duration(6)
    assert (long_train, long_test, long_windows) != (short_train, short_test, short_windows)


def test_history_start_end_respects_months(monkeypatch) -> None:
    from datetime import date

    from config import settings

    monkeypatch.setattr(settings, "HOLDOUT_ENABLED", False, raising=False)
    today = date(2026, 7, 10)
    start_6, end_6 = history_start_end(6, today=today)
    start_12, end_12 = history_start_end(12, today=today)
    assert end_6 == end_12
    assert start_12 < start_6
    span_6 = (end_6 - start_6).total_seconds()
    span_12 = (end_12 - start_12).total_seconds()
    assert span_12 == pytest.approx(span_6 * 2, rel=0.01)


def test_history_start_end_ends_at_holdout_boundary(monkeypatch) -> None:
    """12m history + 12m holdout must still yield ~12 months of usable data."""
    from datetime import date

    from config import settings
    from factory.holdout import holdout_boundary

    monkeypatch.setattr(settings, "HOLDOUT_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "HOLDOUT_MONTHS", 12, raising=False)
    today = date(2026, 7, 12)
    start, end = history_start_end(12, today=today)
    boundary = holdout_boundary(
        datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)
    )
    assert end.date() == boundary.date()
    span_days = (end - start).total_seconds() / 86400.0
    assert span_days == pytest.approx(12 * settings.DAYS_PER_MONTH, rel=0.02)
    assert start.year == 2024


def test_payload_duration_changes_start_and_wfo() -> None:
    short = build_discovery_payload(
        DiscoverySettings(months=3), symbol="EURUSD", timeframe="M15"
    )
    long = build_discovery_payload(
        DiscoverySettings(months=18), symbol="EURUSD", timeframe="M15"
    )
    assert short["start"] > long["start"]
    assert short["end"] == long["end"]
    assert short["wfo_train_months"] != long["wfo_train_months"] or (
        short["wfo_windows"] != long["wfo_windows"]
    )


def test_effective_validation_level_fixed() -> None:
    cfg = DiscoverySettings(validation_level=3, progressive_strictness=False)
    assert effective_validation_level(cfg, sweep_index=0, sweep_total=4) == 3
    assert effective_validation_level(cfg, sweep_index=9, sweep_total=4) == 3


def test_effective_validation_level_progressive() -> None:
    cfg = DiscoverySettings(
        validation_level=7,
        progressive_strictness=True,
        validation_level_start=1,
        progressive_step=2,
    )
    # cycle 0 → 1, cycle 1 → 3, cycle 2 → 5, cycle 3 → 7 (capped)
    assert effective_validation_level(cfg, sweep_index=0, sweep_total=2) == 1
    assert effective_validation_level(cfg, sweep_index=1, sweep_total=2) == 1
    assert effective_validation_level(cfg, sweep_index=2, sweep_total=2) == 3
    assert effective_validation_level(cfg, sweep_index=3, sweep_total=2) == 3
    assert effective_validation_level(cfg, sweep_index=4, sweep_total=2) == 5
    assert effective_validation_level(cfg, sweep_index=6, sweep_total=2) == 7
    assert effective_validation_level(cfg, sweep_index=99, sweep_total=2) == 7


def test_effective_validation_level_progressive_step_one() -> None:
    cfg = DiscoverySettings(
        validation_level=3,
        progressive_strictness=True,
        validation_level_start=1,
        progressive_step=1,
    )
    assert effective_validation_level(cfg, sweep_index=0, sweep_total=2) == 1
    assert effective_validation_level(cfg, sweep_index=2, sweep_total=2) == 2
    assert effective_validation_level(cfg, sweep_index=4, sweep_total=2) == 3


def test_build_discovery_payload_fixed_floor() -> None:
    cfg = DiscoverySettings(
        validation_level=5,
        progressive_strictness=False,
    )
    payload = build_discovery_payload(cfg, symbol="EURUSD", timeframe="H1")
    assert payload["validation_level"] == 5
    assert payload["validation_level_floor"] == 5
    assert payload["validation_level_ceiling"] == validation_levels.MAX_LEVEL
    assert payload["validation_level_target"] == 5
    assert "progressive_strictness" not in payload


def test_build_discovery_payload_progressive_override() -> None:
    cfg = DiscoverySettings(
        validation_level=10,
        progressive_strictness=True,
        progressive_step=2,
    )
    payload = build_discovery_payload(
        cfg, symbol="EURUSD", timeframe="H1", validation_level=4
    )
    assert payload["validation_level"] == 4
    assert payload["validation_level_floor"] == 4
    assert payload["progressive_strictness"] is True
    assert payload["validation_level_ceiling"] == validation_levels.MAX_LEVEL
    assert payload["validation_level_target"] == 10
    assert payload["validation_level_start"] == validation_levels.MIN_LEVEL
    assert payload["progressive_step"] == 2


def test_settings_round_trip_progressive() -> None:
    original = DiscoverySettings(
        validation_level=10,
        progressive_strictness=True,
        validation_level_start=1,
        progressive_step=2,
    )
    restored = settings_from_app(settings_to_app(original))
    assert restored.validation_level == 10
    assert restored.progressive_strictness is True
    assert restored.validation_level_start == 1
    assert restored.progressive_step == 2
    assert settings_to_app(original)["validation_level_schema_version"] == (
        validation_levels.LEVEL_SCHEMA_VERSION
    )


def test_settings_from_app_remaps_legacy_levels() -> None:
    cfg = settings_from_app(
        {
            "discovery_validation_level": 3,
            "discovery_validation_level_start": 2,
            "validation_level_schema_version": 1,
        }
    )
    assert cfg.validation_level == 7
    assert cfg.validation_level_start == 4
