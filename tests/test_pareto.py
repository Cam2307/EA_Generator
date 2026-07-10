"""NSGA-II primitives + Pareto evolution (factory.pareto, generator.evolve_pareto)."""
import random

from factory.generator import evolve_pareto, random_strategy
from factory.models import BacktestMetrics
from factory.pareto import (
    crowding_distances, dominates, fast_non_dominated_sort, nsga2_rank,
    objectives_from_metrics, pareto_front,
)


def test_dominates():
    assert dominates((2, 2), (1, 2))
    assert not dominates((1, 2), (2, 1))       # trade-off: neither dominates
    assert not dominates((2, 1), (1, 2))
    assert not dominates((1, 1), (1, 1))       # equal never dominates


def test_non_dominated_sort_fronts():
    objs = [
        (10, 10),   # front 0
        (12, 8),    # front 0 (trade-off with the above)
        (9, 9),     # dominated by (10,10) -> front 1
        (1, 1),     # dominated by everything -> front 2
    ]
    fronts = fast_non_dominated_sort(objs)
    assert fronts[0] == [0, 1]
    assert fronts[1] == [2]
    assert fronts[2] == [3]
    assert pareto_front(objs) == [0, 1]


def test_crowding_extremes_are_infinite():
    objs = [(0, 10), (5, 5), (10, 0), (5.1, 4.9)]
    front = [0, 1, 2, 3]
    d = crowding_distances(objs, front)
    assert d[0] == float("inf") and d[2] == float("inf")
    assert d[1] > d[3] or d[3] > 0            # interior points get finite crowding


def test_nsga2_rank_orders_fronts():
    objs = [(10, 10), (9, 9), (12, 8)]
    ranks = nsga2_rank(objs)
    assert ranks[0][0] == 0 and ranks[2][0] == 0
    assert ranks[1][0] == 1


def test_objectives_from_metrics():
    m = BacktestMetrics(net_profit=500.0, max_dd_pct=12.0, r_squared=0.8,
                        trade_count=40)
    assert objectives_from_metrics(m) == (500.0, -12.0, 0.8, 40.0)
    # trade-count cap
    m2 = m.model_copy(update={"trade_count": 5000})
    assert objectives_from_metrics(m2)[3] == 100.0
    # statistically empty runs are pushed to a dominated corner
    empty = BacktestMetrics(net_profit=9999.0, trade_count=2)
    assert objectives_from_metrics(empty)[0] == -1e9


def test_evolve_pareto_produces_offspring_from_front():
    rng = random.Random(42)
    pop = []
    for k in range(8):
        s = random_strategy("EURUSD", "M15", rng=rng)
        m = BacktestMetrics(net_profit=100.0 * k, max_dd_pct=30.0 - 3 * k,
                            r_squared=0.1 * k, trade_count=10 + k)
        pop.append((s, objectives_from_metrics(m)))
    children = evolve_pareto(pop, 10, rng)
    assert len(children) == 10
    parent_ids = {s.id for s, _ in pop}
    for child in children:
        assert child.id not in parent_ids
        assert child.lineage.parents          # lineage tracked
    assert evolve_pareto([], 5, rng) == []
