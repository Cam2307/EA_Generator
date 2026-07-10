"""NSGA-II multi-objective primitives for the genetic search.

The scalar fitness ``profit / (1 + dd) * smoothness`` collapses conflicting
goals into one number, so the search discards strategies that are excellent
on two axes but mediocre on the third. NSGA-II (Deb et al. 2002) instead
ranks candidates by Pareto dominance and keeps diversity along the front via
crowding distance — the population converges toward the whole profit/risk/
stability frontier rather than a single compromise point.

All objectives are expressed as MAXIMIZE (negate anything to minimize).
Pure Python + stdlib on purpose: no heavyweight optimizer dependency for
~100-500 candidates per generation.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from factory.models import BacktestMetrics

Objectives = Tuple[float, ...]


def objectives_from_metrics(m: BacktestMetrics) -> Objectives:
    """Screening objectives: (net profit, -max DD%, equity R², trade count).

    Trade count is capped so statistical significance is rewarded without
    the front filling up with hyperactive overtraders; statistically empty
    runs (< 3 trades) are pushed to a dominated corner outright.
    """
    if m.trade_count < 3:
        return (-1e9, -1e9, 0.0, 0.0)
    return (
        float(m.net_profit),
        -float(m.max_dd_pct),
        float(m.r_squared),
        float(min(m.trade_count, 100)),
    )


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """True when ``a`` Pareto-dominates ``b`` (>= everywhere, > somewhere)."""
    better_somewhere = False
    for x, y in zip(a, b):
        if x < y:
            return False
        if x > y:
            better_somewhere = True
    return better_somewhere


def fast_non_dominated_sort(objs: Sequence[Sequence[float]]) -> List[List[int]]:
    """Indices grouped into fronts; front 0 is the non-dominated set."""
    n = len(objs)
    dominated_by: List[List[int]] = [[] for _ in range(n)]
    domination_count = [0] * n
    fronts: List[List[int]] = [[]]
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(objs[i], objs[j]):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif dominates(objs[j], objs[i]):
                dominated_by[j].append(i)
                domination_count[i] += 1
    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt: List[int] = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    fronts.pop()                       # last front is always empty
    return fronts


def crowding_distances(objs: Sequence[Sequence[float]],
                       front: Sequence[int]) -> dict:
    """Crowding distance per index within one front (inf at the extremes)."""
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: float("inf") for i in front}
    n_obj = len(objs[front[0]])
    for k in range(n_obj):
        ordered = sorted(front, key=lambda i: objs[i][k])
        lo, hi = objs[ordered[0]][k], objs[ordered[-1]][k]
        dist[ordered[0]] = dist[ordered[-1]] = float("inf")
        span = hi - lo
        if span <= 0:
            continue
        for pos in range(1, len(ordered) - 1):
            gap = objs[ordered[pos + 1]][k] - objs[ordered[pos - 1]][k]
            if dist[ordered[pos]] != float("inf"):
                dist[ordered[pos]] += gap / span
    return dist


def nsga2_rank(objs: Sequence[Sequence[float]]) -> List[Tuple[int, float]]:
    """Per-index ``(front_rank, crowding_distance)`` — lower rank is better,
    larger crowding is better within a rank."""
    out: List[Tuple[int, float]] = [(0, 0.0)] * len(objs)
    for rank, front in enumerate(fast_non_dominated_sort(objs)):
        dists = crowding_distances(objs, front)
        for i in front:
            out[i] = (rank, dists[i])
    return out


def pareto_front(objs: Sequence[Sequence[float]]) -> List[int]:
    """Indices of the non-dominated set (front 0)."""
    if not objs:
        return []
    return fast_non_dominated_sort(objs)[0]
