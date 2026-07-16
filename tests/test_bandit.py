"""Thompson-sampling sweep arm selection."""
import random

from jobs.bandit import (
    arm_key, default_arm, dump_bandit_stats, load_bandit_stats, outcome_weight,
    record_outcome, record_pull, select_plan,
)
from jobs.sweep import SweepPlan


def _plans():
    return [
        SweepPlan("EURUSD", "M15", 1, {}),
        SweepPlan("GBPUSD", "M15", 2, {}),
        SweepPlan("USDJPY", "H1", 3, {}),
    ]


def test_bandit_roundtrip_stats():
    stats = {}
    record_pull(stats, "EURUSD", "M15")
    record_outcome(stats, "EURUSD", "M15", success=True)
    raw = dump_bandit_stats(stats)
    loaded = load_bandit_stats(raw)
    assert loaded[arm_key("EURUSD", "M15")]["successes"] == 1
    assert loaded[arm_key("EURUSD", "M15")]["alpha"] == 2.0


def test_select_plan_prefers_successful_arm():
    plans = _plans()
    stats = {arm_key(p.symbol, p.timeframe): default_arm() for p in plans}
    # Saturate EURUSD as a winner.
    for _ in range(40):
        record_outcome(stats, "EURUSD", "M15", success=True)
        record_pull(stats, "EURUSD", "M15")
        record_pull(stats, "GBPUSD", "M15")
        record_pull(stats, "USDJPY", "H1")
        record_outcome(stats, "GBPUSD", "M15", success=False)
        record_outcome(stats, "USDJPY", "H1", success=False)
    picks = []
    rng = random.Random(0)
    for i in range(50):
        _, plan = select_plan(plans, stats, rng=rng, exploration_floor=0.0,
                              min_pulls=0)
        picks.append(arm_key(plan.symbol, plan.timeframe))
    assert picks.count(arm_key("EURUSD", "M15")) > picks.count(
        arm_key("GBPUSD", "M15"))


def test_outcome_weight_soft_counts_l4():
    ok, w = outcome_weight(survivors=0, max_level=0)
    assert ok is False
    ok, w = outcome_weight(survivors=1, max_level=2, floor_level=4)
    assert ok is True and w == 1.0
    ok, w = outcome_weight(survivors=1, max_level=4, floor_level=4)
    assert ok is True and w == 2.0
    ok, w = outcome_weight(survivors=0, edges_found=1, max_level=0)
    assert ok is True and w == 2.0


def test_record_outcome_respects_weight():
    stats = {}
    record_outcome(stats, "EURUSD", "M15", success=True, weight=2.0)
    arm = stats[arm_key("EURUSD", "M15")]
    assert arm["alpha"] == 3.0  # prior 1.0 + 2.0
    assert arm["successes"] == 1
