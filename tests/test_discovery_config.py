import pytest

from factory.discovery_config import (
    DiscoverySettings,
    build_discovery_payload,
    derive_wfo_from_duration,
    history_start_end,
    settings_from_app,
    settings_to_app,
)
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
        validation_level=3,
        months=6,
    )
    payload = build_discovery_payload(cfg, symbol="EURUSD", timeframe="H1", seed=42)
    assert payload["engine"] == "mt5"
    assert payload["deposit"] == 25_000.0
    assert payload["leverage"] == 200
    assert payload["mechanics"] == ["standard_sltp"]
    assert payload["tm_features"] == ["trailing"]
    assert payload["validation_level"] == 3
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


def test_settings_round_trip() -> None:
    original = DiscoverySettings(symbols=["USDJPY"], timeframes=["D1"], months=18)
    restored = settings_from_app(settings_to_app(original))
    assert restored.symbols == ["USDJPY"]
    assert restored.timeframes == ["D1"]
    assert restored.months == 18
    assert restored.wfo_train_months == derive_wfo_from_duration(18)[0]


def test_derive_wfo_scales_with_duration() -> None:
    assert derive_wfo_from_duration(6) == (2, 1, 2)
    assert derive_wfo_from_duration(12) == (4, 2, 4)
    short = derive_wfo_from_duration(3)
    assert short[0] + short[1] <= 3 or short == (1, 1, 1)
    long_train, long_test, long_windows = derive_wfo_from_duration(24)
    short_train, short_test, short_windows = derive_wfo_from_duration(6)
    assert (long_train, long_test, long_windows) != (short_train, short_test, short_windows)


def test_history_start_end_respects_months() -> None:
    from datetime import date

    today = date(2026, 7, 10)
    start_6, end_6 = history_start_end(6, today=today)
    start_12, end_12 = history_start_end(12, today=today)
    assert end_6 == end_12
    assert start_12 < start_6
    span_6 = (end_6 - start_6).total_seconds()
    span_12 = (end_12 - start_12).total_seconds()
    assert span_12 == pytest.approx(span_6 * 2, rel=0.01)


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
