"""Point-distance base × scale search dims for Optuna.

Optuna searches ``base`` + ``*_scale`` (1–20) as separate ParamRanges; at
apply/export we collapse onto the existing key so simulator and MQL5 only
ever see effective points (``effective = max(0, base * scale)``).
"""
from __future__ import annotations

import math
import random
from typing import Dict

from factory.models import ParamRange, StrategyDefinition

# Every distance-style param that gets a companion ``*_scale`` search dim.
POINT_DISTANCE_PARAM_NAMES: frozenset[str] = frozenset({
    # Mechanics
    "sl_points",
    "tp_points",
    "grid_step_points",
    "basket_tp_points",
    "basket_sl_points",
    "hedge_trigger_points",
    "partial_tp_points",
    # Trade management
    "trail_start_points",
    "trail_distance_points",
    "trail_step_points",
    "be_trigger_points",
    "be_offset_points",
    # Filters
    "buffer_points",
    "zone_points",
})

SCALE_RANGE = ParamRange(min=1, max=20, step=1)


def scale_key_for(base: str) -> str:
    """``sl_points`` → ``sl_scale``, ``trail_start_points`` → ``trail_start_scale``."""
    if base.endswith("_points"):
        return base[: -len("_points")] + "_scale"
    return f"{base}_scale"


def is_scale_key(name: str) -> bool:
    return name.endswith("_scale")


def sample_log_uniform(r: ParamRange, rng: random.Random) -> float:
    """Draw from ``r`` with log-uniform density, snapped to the step grid.

    Puts more mass on mid-range scales than a flat draw over 1..20, while
    still covering the full ``SCALE_RANGE``.
    """
    n_steps = int(round((r.max - r.min) / r.step)) if r.step > 0 else 0
    if n_steps <= 0:
        return float(r.min)
    lo = math.log(max(float(r.min), 1e-12))
    hi = math.log(max(float(r.max), float(r.min) + 1e-12))
    if hi <= lo:
        return float(r.min)
    raw = math.exp(rng.uniform(lo, hi))
    idx = int(round((raw - r.min) / r.step))
    idx = max(0, min(n_steps, idx))
    return float(r.min + idx * r.step)


def _collapse_params_dict(params: Dict[str, float]) -> bool:
    """In-place collapse. Returns True if any scale key was consumed."""
    changed = False
    for base in POINT_DISTANCE_PARAM_NAMES:
        sk = scale_key_for(base)
        if sk not in params:
            continue
        scale = float(params.pop(sk))
        changed = True
        if base in params:
            params[base] = max(0.0, float(params[base]) * scale)
    # Strip any leftover unknown *_scale keys so they never reach sim/MQL5.
    orphans = [k for k in params if is_scale_key(k)]
    for k in orphans:
        del params[k]
        changed = True
    return changed


def collapse_scaled_point_params(strategy: StrategyDefinition) -> StrategyDefinition:
    """Collapse base×scale into effective values on original keys.

    ``ranges`` are left intact so Optuna / neighbors still see scale dims.
    Returns ``strategy`` unchanged when no scale keys are present in params.
    """
    probe_blocks = [f.params for f in strategy.entry_filters]
    probe_blocks.append(strategy.mechanic.params)
    probe_blocks.append(strategy.trade_mgmt.params)
    if not any(is_scale_key(k) for params in probe_blocks for k in params):
        return strategy

    clone = strategy.model_copy(deep=True)
    for f in clone.entry_filters:
        _collapse_params_dict(f.params)
    _collapse_params_dict(clone.mechanic.params)
    _collapse_params_dict(clone.trade_mgmt.params)
    return clone
