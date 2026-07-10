"""Session-aware dynamic execution-cost model for the simulator.

The flat ``spread_points`` / ``slippage_points`` constants in
:class:`~factory.backtest.simulator.SymbolSpec` are systematically optimistic:
real FX spreads widen 2-4x around the daily rollover, stay elevated through
the thin Asian session, and are tightest during the London/New York overlap.
Slippage grows with volatility. Strategies that happen to trade the cheap
hours are fine either way; strategies whose "edge" lives in rollover spikes
or news candles get flattered badly by a flat cost model — exactly the kind
of candidate the pre-filter exists to kill.

This module precomputes per-bar spread and slippage arrays from the bar
timestamps and realized ranges. The user's configured ``spread_points`` /
``slippage_points`` remain the *typical* (London-session) cost; the model
scales them per bar. All hours are UTC (matching factory.data timestamps).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

# Spread multiplier by UTC hour. 1.0 = the user's configured spread
# (calibrated to the liquid London morning). Shape follows well-documented
# intraday FX spread seasonality: thin Asian session, tight London/NY
# overlap, and a sharp rollover spike around 21-22 UTC (5pm New York).
HOUR_SPREAD_MULT = np.array([
    1.35, 1.35, 1.30, 1.25, 1.20, 1.15,   # 00-05 Asian session
    1.10, 1.05,                            # 06-07 pre-London
    1.00, 1.00, 1.00, 1.00,                # 08-11 London morning
    0.90, 0.90, 0.90, 0.90, 0.95,          # 12-16 London/NY overlap
    1.00, 1.05, 1.10, 1.20,                # 17-20 NY afternoon fade
    2.50, 2.00, 1.50,                      # 21-23 rollover spike + recovery
])

# Extra widening around the weekly liquidity gaps.
FRIDAY_LATE_MULT = 1.40      # Friday from 20:00 UTC (pre-weekend de-risking)
SUNDAY_OPEN_MULT = 1.60      # Sunday session (thin re-open)

# Slippage scales with the bar's true range relative to the recent median
# range, clamped so a data glitch can't produce absurd fills.
SLIPPAGE_VOL_MIN = 0.5
SLIPPAGE_VOL_MAX = 3.0
VOL_BASELINE_WINDOW = 200    # bars used for the rolling median range


def spread_multipliers(hours: np.ndarray, weekdays: np.ndarray) -> np.ndarray:
    """Per-bar spread multiplier from UTC hour-of-day and weekday (Mon=0)."""
    mult = HOUR_SPREAD_MULT[hours].astype(float)
    mult = np.where((weekdays == 4) & (hours >= 20),
                    np.maximum(mult, FRIDAY_LATE_MULT), mult)
    mult = np.where(weekdays == 6, np.maximum(mult, SUNDAY_OPEN_MULT), mult)
    return mult


def slippage_multipliers(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Per-bar slippage multiplier: bar range / rolling median range, clamped."""
    rng = pd.Series(np.maximum(high - low, 0.0))
    baseline = rng.rolling(VOL_BASELINE_WINDOW, min_periods=10).median()
    ratio = (rng / baseline.replace(0.0, np.nan)).to_numpy()
    ratio = np.nan_to_num(ratio, nan=1.0, posinf=SLIPPAGE_VOL_MAX)
    return np.clip(ratio, SLIPPAGE_VOL_MIN, SLIPPAGE_VOL_MAX)


def build_cost_arrays(df: pd.DataFrame, spread_points: float,
                      slippage_points: float
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-bar ``(spread_points, slippage_points)`` arrays for ``df``.

    ``spread_points`` / ``slippage_points`` are the user's base (typical)
    costs; the arrays scale them by session and realized volatility.
    """
    t = pd.to_datetime(df["time"], utc=True)
    hours = t.dt.hour.to_numpy()
    weekdays = t.dt.weekday.to_numpy()
    spread = spread_points * spread_multipliers(hours, weekdays)
    slip = slippage_points * slippage_multipliers(
        df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float))
    return spread, slip
