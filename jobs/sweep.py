"""Sweep planner for symbol x timeframe x strictness exploration."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone


@dataclass(frozen=True)
class SweepPlan:
    symbol: str
    timeframe: str
    strictness_profile: str
    seed: int
    payload_patch: dict


def plan_sweeps(
    *,
    symbols: list[str],
    timeframes: list[str],
    strictness_profiles: list[str],
    months: int,
    base_seed: int,
    custom_criteria: dict | None = None,
) -> list[SweepPlan]:
    """Build deterministic sweep combinations for the orchestrator."""
    end = date.today()
    start = end - timedelta(days=max(1, months) * 30)
    out: list[SweepPlan] = []
    seed = int(base_seed)
    for symbol in symbols:
        for timeframe in timeframes:
            for profile in strictness_profiles:
                patch: dict = {
                    "symbol": symbol.strip().upper(),
                    "timeframe": timeframe,
                    "start": datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).isoformat(),
                    "end": datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc).isoformat(),
                    "strictness_profile": profile,
                    "validation_level": _profile_to_level(profile),
                }
                if profile == "custom":
                    patch["criteria"] = custom_criteria or {}
                    patch.pop("validation_level", None)
                out.append(
                    SweepPlan(
                        symbol=symbol,
                        timeframe=timeframe,
                        strictness_profile=profile,
                        seed=seed,
                        payload_patch=patch,
                    )
                )
                seed += 1
    return out


def _profile_to_level(profile: str) -> int:
    mapping = {"easy": 1, "normal": 3, "hard": 5}
    return mapping.get(profile, 3)
