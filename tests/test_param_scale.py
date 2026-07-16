"""Tests for base × scale point-distance search dims."""
from __future__ import annotations

import random

import pytest

from factory.backtest.validation import _suggest_params
from factory.generator import (
    MECHANIC_PARAM_SPECS, TM_PARAM_SPECS, random_strategy, random_trade_mgmt,
)
from factory.models import (
    EntryFilter, EntryFilterType, ExecutionMechanic, ExecutionMechanicType,
    ParamRange, StrategyDefinition, TrailMode,
)
from factory.mql5.renderer import mql5_inputs_for, render_ea
from factory.param_scale import (
    POINT_DISTANCE_PARAM_NAMES, SCALE_RANGE, collapse_scaled_point_params,
    sample_log_uniform, scale_key_for,
)


def _std_strategy(**mech_params) -> StrategyDefinition:
    return StrategyDefinition(
        name="ScaleTest",
        mechanic=ExecutionMechanic(
            type=ExecutionMechanicType.STANDARD_SLTP,
            params=dict(mech_params) if mech_params else {
                "sl_points": 100.0, "tp_points": 200.0,
            },
            ranges=dict(MECHANIC_PARAM_SPECS[ExecutionMechanicType.STANDARD_SLTP]),
        ),
    )


def test_scale_key_naming():
    assert scale_key_for("sl_points") == "sl_scale"
    assert scale_key_for("trail_start_points") == "trail_start_scale"
    assert scale_key_for("buffer_points") == "buffer_scale"
    assert scale_key_for("grid_step_points") == "grid_step_scale"


def test_sample_log_uniform_covers_range_and_prefers_mid():
    rng = random.Random(0)
    samples = [sample_log_uniform(SCALE_RANGE, rng) for _ in range(500)]
    assert min(samples) >= SCALE_RANGE.min
    assert max(samples) <= SCALE_RANGE.max
    # Log-uniform puts more mass below the arithmetic midpoint than uniform.
    mid = (SCALE_RANGE.min + SCALE_RANGE.max) / 2.0
    below = sum(1 for s in samples if s <= mid)
    assert below / len(samples) >= 0.55


def test_random_strategy_uses_log_biased_scales():
    """Generated strategies should include *_scale params in ranges."""
    s = random_strategy(
        "EURUSD", "M15", random.Random(9),
        allowed_mechanics=[ExecutionMechanicType.STANDARD_SLTP])
    assert "sl_scale" in s.mechanic.params
    assert 1.0 <= s.mechanic.params["sl_scale"] <= 20.0


def test_collapse_apply_flat_params():
    s = _std_strategy(sl_points=500.0, sl_scale=10.0, tp_points=100.0, tp_scale=1.0)
    # Direct collapse (params already local, not via flat prefixes)
    out = collapse_scaled_point_params(s)
    assert out.mechanic.params["sl_points"] == 5000.0
    assert "sl_scale" not in out.mechanic.params
    assert out.mechanic.params["tp_points"] == 100.0
    assert "tp_scale" not in out.mechanic.params
    # Ranges still expose scale dims for Optuna.
    assert "sl_scale" in out.mechanic.ranges


def test_apply_flat_params_collapses_prefixed_keys():
    s = _std_strategy(sl_points=80.0, tp_points=80.0)
    tuned = s.apply_flat_params({
        "M_STANDARD_SLTP_sl_points": 500.0,
        "M_STANDARD_SLTP_sl_scale": 10.0,
        "M_STANDARD_SLTP_tp_points": 200.0,
        "M_STANDARD_SLTP_tp_scale": 5.0,
    })
    assert tuned.mechanic.params["sl_points"] == 5000.0
    assert tuned.mechanic.params["tp_points"] == 1000.0
    assert "sl_scale" not in tuned.mechanic.params
    assert "tp_scale" not in tuned.mechanic.params


def test_effective_ceiling_reaches_base_max_times_scale():
    specs = MECHANIC_PARAM_SPECS[ExecutionMechanicType.STANDARD_SLTP]
    assert specs["sl_points"].max * SCALE_RANGE.max >= 16_000
    assert specs["tp_points"].max * SCALE_RANGE.max >= 20_000
    assert specs["sl_scale"].max == 20
    assert specs["tp_scale"].min == 1


def test_all_point_distance_bases_have_scale_specs():
    for base in POINT_DISTANCE_PARAM_NAMES:
        sk = scale_key_for(base)
        found = False
        for block in MECHANIC_PARAM_SPECS.values():
            if base in block:
                assert sk in block, f"{sk} missing next to {base}"
                assert block[sk].min == 1 and block[sk].max == 20
                found = True
        if base in TM_PARAM_SPECS:
            assert sk in TM_PARAM_SPECS
            found = True
        # filter bases live in FILTER_PARAM_SPECS
        from factory.generator import FILTER_PARAM_SPECS
        for block in FILTER_PARAM_SPECS.values():
            if base in block:
                assert sk in block
                found = True
        assert found, f"{base} not found in any spec dict"


def test_suggest_params_includes_scale_keys():
    s = random_strategy(
        "EURUSD", "M15", random.Random(42),
        allowed_mechanics=[ExecutionMechanicType.STANDARD_SLTP])
    ranges = s.all_ranges()
    assert any(k.endswith("_sl_scale") or k.endswith("sl_scale") for k in ranges)
    assert "M_STANDARD_SLTP_sl_scale" in ranges
    assert "M_STANDARD_SLTP_tp_scale" in ranges

    class _Trial:
        def __init__(self):
            self.n = 0

        def suggest_float(self, name, low, high, step=None):
            self.n += 1
            return low

        def suggest_int(self, name, low, high, step=1):
            self.n += 1
            return low

    suggested = _suggest_params(_Trial(), ranges)
    assert "M_STANDARD_SLTP_sl_scale" in suggested
    assert suggested["M_STANDARD_SLTP_sl_scale"] >= 1.0


def test_lot_multiplier_unlocked_for_dca():
    r = MECHANIC_PARAM_SPECS[ExecutionMechanicType.DCA_GRID]["lot_multiplier"]
    assert r.min == 1.0
    assert r.max == 2.0
    assert r.step == pytest.approx(0.1)

    seen = set()
    for seed in range(40):
        s = random_strategy(
            "EURUSD", "M15", random.Random(seed),
            allowed_mechanics=[ExecutionMechanicType.DCA_GRID])
        lm = s.mechanic.params["lot_multiplier"]
        assert 1.0 <= lm <= 2.0
        seen.add(round(lm, 1))
    # With unlocked range, sampling should not be stuck at 1.0 only.
    assert max(seen) > 1.0


def test_tm_sampling_attaches_trail_scales():
    for seed in range(30):
        tm = random_trade_mgmt(
            ExecutionMechanicType.STANDARD_SLTP, random.Random(seed),
            allowed=["trailing"])
        if tm.trail_mode == TrailMode.FIXED:
            assert "trail_start_points" in tm.params
            assert "trail_start_scale" in tm.params
            assert "trail_distance_points" in tm.params
            assert "trail_distance_scale" in tm.params
            assert "trail_step_scale" in tm.params
            return
    pytest.fail("never sampled FIXED trailing")


def test_renderer_exports_effective_points_no_scale_inputs():
    s = StrategyDefinition(
        name="ScaleEA",
        entry_filters=[EntryFilter(
            type=EntryFilterType.RSI_REVERSION,
            params={"rsi_period": 14, "oversold": 30, "overbought": 70},
            ranges={"rsi_period": ParamRange(min=7, max=21, step=7)},
        )],
        mechanic=ExecutionMechanic(
            type=ExecutionMechanicType.STANDARD_SLTP,
            params={"sl_points": 500.0, "sl_scale": 10.0,
                    "tp_points": 200.0, "tp_scale": 5.0},
            ranges=dict(MECHANIC_PARAM_SPECS[ExecutionMechanicType.STANDARD_SLTP]),
        ),
    )
    ea = render_ea(s)
    assert "input double Inp_M_sl_points" in ea
    assert "= 5000;" in ea or "= 5000" in ea
    assert "Inp_M_sl_scale" not in ea
    assert "Inp_M_tp_scale" not in ea

    inputs, ranges = mql5_inputs_for(s)
    assert inputs["Inp_M_sl_points"] == 5000.0
    assert inputs["Inp_M_tp_points"] == 1000.0
    assert "Inp_M_sl_scale" not in inputs
    assert "Inp_M_sl_scale" not in ranges


def test_widened_multiplier_ranges():
    assert MECHANIC_PARAM_SPECS[ExecutionMechanicType.HEDGE_LAYER]["hedge_ratio"].max == 2.0
    assert MECHANIC_PARAM_SPECS[ExecutionMechanicType.PARTIAL_CLOSE]["partial_fraction"].min == 0.20
    assert TM_PARAM_SPECS["atr_sl_mult"].max == 8.0
    assert TM_PARAM_SPECS["tp_rr"].max == 6.0
    assert TM_PARAM_SPECS["risk_percent"].max == 3.0
