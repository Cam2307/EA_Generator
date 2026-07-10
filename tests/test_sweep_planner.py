from jobs.sweep import plan_sweeps


def test_plan_sweeps_builds_symbol_timeframe_grid() -> None:
    plans = plan_sweeps(
        symbols=["EURUSD", "GBPUSD"],
        timeframes=["M15", "H1"],
        months=6,
        base_seed=100,
    )
    assert len(plans) == 2 * 2
    assert plans[0].seed == 100
    assert plans[-1].seed == 100 + len(plans) - 1
    assert plans[0].payload_patch["symbol"] == "EURUSD"
    assert plans[0].payload_patch["timeframe"] == "M15"
