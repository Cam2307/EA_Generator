"""Sweep planner for symbol x timeframe exploration."""
from __future__ import annotations

from dataclasses import dataclass

from factory.discovery_config import history_start_end


@dataclass(frozen=True)
class SweepPlan:
    symbol: str
    timeframe: str
    seed: int
    payload_patch: dict


def plan_sweeps(
    *,
    symbols: list[str],
    timeframes: list[str],
    months: int,
    base_seed: int,
) -> list[SweepPlan]:
    """Build deterministic sweep combinations for the orchestrator."""
    start_dt, end_dt = history_start_end(months)
    out: list[SweepPlan] = []
    seed = int(base_seed)
    for symbol in symbols:
        for timeframe in timeframes:
            patch: dict = {
                "symbol": symbol.strip().upper(),
                "timeframe": timeframe,
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "test_duration_months": int(months),
            }
            out.append(
                SweepPlan(
                    symbol=symbol,
                    timeframe=timeframe,
                    seed=seed,
                    payload_patch=patch,
                )
            )
            seed += 1
    return out
