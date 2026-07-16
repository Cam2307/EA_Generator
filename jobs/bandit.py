"""Thompson-sampling arm selection over symbol × timeframe sweep pairs.

Tracks Beta(α, β) posteriors per pair from live promotion outcomes
(``edge_positive``+ counts as a success). Guarantees a minimum exploration
floor so no selected pair is starved, and down-weights arms whose recent
survivors are highly correlated with the already-promoted portfolio niche.
"""
from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Sequence, Tuple

from jobs.sweep import SweepPlan

# Minimum pulls before an arm may be fully deprioritized by the bandit.
MIN_EXPLORATION_PULLS = 1
# Fraction of selections forced to the least-pulled eligible arm.
EXPLORATION_FLOOR = 0.15
# Multiply Thompson sample by this when mean |corr| vs promoted is high.
CORR_DOWNWEIGHT = 0.35


def arm_key(symbol: str, timeframe: str) -> str:
    return f"{symbol.strip().upper()}|{timeframe}"


def default_arm() -> dict:
    return {"alpha": 1.0, "beta": 1.0, "pulls": 0, "successes": 0}


def load_bandit_stats(raw: Optional[str]) -> Dict[str, dict]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, dict] = {}
    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        arm = default_arm()
        arm.update({
            "alpha": float(val.get("alpha", 1.0) or 1.0),
            "beta": float(val.get("beta", 1.0) or 1.0),
            "pulls": int(val.get("pulls", 0) or 0),
            "successes": int(val.get("successes", 0) or 0),
        })
        out[str(key)] = arm
    return out


def dump_bandit_stats(stats: Dict[str, dict]) -> str:
    return json.dumps(stats, separators=(",", ":"))


def record_pull(stats: Dict[str, dict], symbol: str, timeframe: str) -> None:
    key = arm_key(symbol, timeframe)
    arm = stats.setdefault(key, default_arm())
    arm["pulls"] = int(arm.get("pulls", 0)) + 1


def record_outcome(stats: Dict[str, dict], symbol: str, timeframe: str,
                   *, success: bool, weight: float = 1.0) -> None:
    """Update Beta posterior for an arm.

    ``weight`` soft-counts successes (e.g. 2.0 when a job cleared L4+) so
    productive niches get stronger Thompson pull without changing validation.
    """
    key = arm_key(symbol, timeframe)
    arm = stats.setdefault(key, default_arm())
    w = max(0.0, float(weight))
    if success and w > 0.0:
        arm["alpha"] = float(arm.get("alpha", 1.0)) + w
        arm["successes"] = int(arm.get("successes", 0)) + 1
    else:
        arm["beta"] = float(arm.get("beta", 1.0)) + 1.0


def outcome_weight(
    *,
    survivors: int = 0,
    max_level: int = 0,
    edges_found: int = 0,
    floor_level: int = 4,
) -> Tuple[bool, float]:
    """Map job results to (success, soft weight) for :func:`record_outcome`.

    Any survivor or edge counts as success (+1). Clearing ``floor_level``
    (default L4) or finding edges soft-boosts to +2.
    """
    surv = int(survivors or 0)
    lvl = int(max_level or 0)
    edges = int(edges_found or 0)
    floor = int(floor_level or 4)
    if surv <= 0 and edges <= 0:
        return False, 1.0
    if lvl >= floor or edges > 0:
        return True, 2.0
    return True, 1.0


def select_plan(
    plans: Sequence[SweepPlan],
    stats: Dict[str, dict],
    *,
    rng: Optional[random.Random] = None,
    corr_penalty: Optional[Dict[str, float]] = None,
    exploration_floor: float = EXPLORATION_FLOOR,
    min_pulls: int = MIN_EXPLORATION_PULLS,
) -> Tuple[int, SweepPlan]:
    """Pick the next sweep plan index via Thompson sampling + exploration floor.

    ``corr_penalty`` maps arm keys to [0, 1] (higher = more correlated niche).
    """
    if not plans:
        raise ValueError("no sweep plans")
    rng = rng or random.Random()
    corr_penalty = corr_penalty or {}

    # Forced exploration: occasionally pick the least-pulled arm.
    if rng.random() < max(0.0, min(1.0, exploration_floor)):
        least_i = min(
            range(len(plans)),
            key=lambda i: int(
                stats.get(arm_key(plans[i].symbol, plans[i].timeframe),
                          default_arm()).get("pulls", 0)),
        )
        return least_i, plans[least_i]

    # Prefer arms below the minimum pull floor until everyone has been seen.
    under = [
        i for i, p in enumerate(plans)
        if int(stats.get(arm_key(p.symbol, p.timeframe),
                         default_arm()).get("pulls", 0)) < min_pulls
    ]
    candidates = under if under else list(range(len(plans)))

    best_i = candidates[0]
    best_score = float("-inf")
    for i in candidates:
        p = plans[i]
        key = arm_key(p.symbol, p.timeframe)
        arm = stats.get(key, default_arm())
        sample = rng.betavariate(
            max(1e-3, float(arm.get("alpha", 1.0))),
            max(1e-3, float(arm.get("beta", 1.0))),
        )
        penalty = max(0.0, min(1.0, float(corr_penalty.get(key, 0.0))))
        score = sample * (1.0 - CORR_DOWNWEIGHT * penalty)
        if score > best_score:
            best_score = score
            best_i = i
    return best_i, plans[best_i]


def corr_penalties_for_plans(
    plans: Sequence[SweepPlan],
    *,
    storage,
) -> Dict[str, float]:
    """Cheap per-arm saturation signal for the bandit (must stay sub-second).

    The previous implementation deserialized the full validations table and
    called ``get_strategy`` per (plan × recent report) — with a multi-GB DB
    and ~100 symbol×TF plans that wedged the orchestrator for many minutes,
    leaving the UI stuck on "Submitted sweep".

    Soft penalty from ``strategy_metadata`` row counts only: arms that have
    already produced many candidates are gently down-weighted so under-explored
    pairs get more pulls. Returns {} when metadata is unavailable.
    """
    if not plans:
        return {}
    try:
        with storage.connection() as con:
            rows = con.execute(
                "SELECT sweep_symbol, sweep_timeframe, COUNT(*) AS n "
                "FROM strategy_metadata "
                "WHERE sweep_symbol IS NOT NULL AND sweep_symbol != '' "
                "GROUP BY sweep_symbol, sweep_timeframe"
            ).fetchall()
    except Exception:
        return {}

    counts: Dict[Tuple[str, str], int] = {}
    for row in rows:
        sym = str(row[0] or "").strip().upper()
        tf = str(row[1] or "").strip()
        if sym and tf:
            counts[(sym, tf)] = int(row[2] or 0)

    # ~20 prior candidates → mild penalty; ~60+ → full soft penalty.
    out: Dict[str, float] = {}
    for plan in plans:
        n = counts.get((plan.symbol.strip().upper(), plan.timeframe), 0)
        out[arm_key(plan.symbol, plan.timeframe)] = max(
            0.0, min(1.0, (n - 10) / 50.0))
    return out
