from jobs.sweep import plan_sweeps


def test_plan_sweeps_builds_symbol_timeframe_profile_grid() -> None:
    plans = plan_sweeps(
        symbols=["EURUSD", "GBPUSD"],
        timeframes=["M15", "H1"],
        strictness_profiles=["easy", "normal", "hard", "custom"],
        months=6,
        base_seed=100,
        custom_criteria={"min_wfe": 0.7},
    )
    assert len(plans) == 2 * 2 * 4
    assert plans[0].seed == 100
    assert plans[-1].seed == 100 + len(plans) - 1
    custom = [p for p in plans if p.strictness_profile == "custom"][0]
    assert "criteria" in custom.payload_patch
    assert "validation_level" not in custom.payload_patch
