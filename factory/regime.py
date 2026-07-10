"""Market-regime classification and per-regime performance breakdown.

Every bar is labeled with one of four regimes built from two observable,
MQL5-replicable proxies (no lookahead — both use only completed-bar data):

- **trend vs range**: Wilder ADX above/below a threshold;
- **volatile vs quiet**: ATR above/below a multiple of its rolling median.

    code 0  quiet range      code 1  quiet trend
    code 2  volatile range   code 3  volatile trend

A strategy's validation then reports how its OOS trades performed *per
regime*, so "profitable overall" can be decomposed into "earns in trends,
bleeds in ranges" — the information needed both for honest curation and for
regime-gated EAs. An optional acceptance gate rejects strategies whose worst
regime loses more than a configured fraction of the deposit.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from factory.models import RegimeStats

REGIME_NAMES = {
    0: "quiet range",
    1: "quiet trend",
    2: "volatile range",
    3: "volatile trend",
}

ADX_TREND_THRESHOLD = 25.0
VOL_ATR_MULT = 1.25          # ATR above 1.25x its rolling median = volatile
VOL_BASELINE_WINDOW = 200

# The tradable regime *filter* (TradeManagement.regime_filter) uses a
# mean-of-ATR baseline over a fixed window, because a rolling mean is what
# the generated EA can replicate exactly and cheaply from CopyBuffer data.
# The reporting classifier above keeps its more robust rolling median.
REGIME_FILTER_BASELINE_BARS = 100


def classify_regimes(df: pd.DataFrame, adx_period: int = 14,
                     atr_period: int = 14) -> np.ndarray:
    """Per-bar regime codes (0..3) for a bar DataFrame."""
    from factory.backtest.simulator import _adx, _atr

    adx, _, _ = _adx(df, adx_period)
    atr = _atr(df, atr_period)
    baseline = (pd.Series(atr)
                .rolling(VOL_BASELINE_WINDOW, min_periods=atr_period)
                .median().to_numpy())
    trending = np.nan_to_num(adx, nan=0.0) > ADX_TREND_THRESHOLD
    volatile = np.nan_to_num(atr, nan=0.0) > VOL_ATR_MULT * np.nan_to_num(
        baseline, nan=np.inf)
    return (trending.astype(int) + 2 * volatile.astype(int)).astype(np.int8)


def classify_regimes_filter(df: pd.DataFrame, adx_period: int,
                            atr_period: int, adx_min: float,
                            atr_mult: float) -> np.ndarray:
    """Per-bar regime codes for the *tradable* regime gate.

    Identical structure to :func:`classify_regimes` but fully parameterized
    (thresholds are optimizable strategy inputs) and using a rolling-**mean**
    ATR baseline over :data:`REGIME_FILTER_BASELINE_BARS` bars so the
    generated EA can reproduce the exact same classification in MQL5.
    """
    from factory.backtest.simulator import _adx, _atr

    adx, _, _ = _adx(df, int(adx_period))
    atr = _atr(df, int(atr_period))
    baseline = (pd.Series(atr)
                .rolling(REGIME_FILTER_BASELINE_BARS,
                         min_periods=int(atr_period))
                .mean().to_numpy())
    trending = np.nan_to_num(adx, nan=0.0) > float(adx_min)
    volatile = np.nan_to_num(atr, nan=0.0) > float(atr_mult) * np.nan_to_num(
        baseline, nan=np.inf)
    return (trending.astype(int) + 2 * volatile.astype(int)).astype(np.int8)


def allowed_by_mask(codes: np.ndarray, mask: int) -> np.ndarray:
    """Boolean array: True where the regime code's bit is set in ``mask``."""
    mask = int(mask)
    lut = np.array([(mask >> c) & 1 for c in range(4)], dtype=bool)
    return lut[codes]


def regime_stats_from_trades(bar_times: np.ndarray, regimes: np.ndarray,
                             open_times: List[float],
                             profits: List[float]) -> List[RegimeStats]:
    """Aggregate closed-trade PnL by the regime in force at trade open."""
    stats: List[RegimeStats] = []
    if len(bar_times) == 0:
        return stats
    opens = np.asarray(open_times, dtype=float)
    prof = np.asarray(profits, dtype=float)
    # regime at the last completed bar at/before the trade's open time
    idx = np.clip(np.searchsorted(bar_times, opens, side="right") - 1,
                  0, len(regimes) - 1)
    trade_regimes = regimes[idx]
    bar_share = {code: float((regimes == code).mean())
                 for code in REGIME_NAMES}
    for code, name in REGIME_NAMES.items():
        mask = trade_regimes == code
        n = int(mask.sum())
        if n == 0:
            stats.append(RegimeStats(code=code, name=name,
                                     bar_share=round(bar_share[code], 4)))
            continue
        p = prof[mask]
        gross_win = float(p[p > 0].sum())
        gross_loss = float(-p[p < 0].sum())
        stats.append(RegimeStats(
            code=code, name=name, trades=n,
            net_profit=round(float(p.sum()), 2),
            profit_factor=round(gross_win / gross_loss, 3) if gross_loss > 0
            else (999.0 if gross_win > 0 else 0.0),
            win_rate=round(float((p > 0).mean()), 4),
            bar_share=round(bar_share[code], 4),
        ))
    return stats


def regime_stats_from_book(df: pd.DataFrame, book) -> List[RegimeStats]:
    """Per-regime breakdown for a finished simulator PositionBook."""
    if not book.closed:
        return []
    t = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    bar_times = t.to_numpy().astype("datetime64[s]").astype("int64").astype(float)
    regimes = classify_regimes(df)
    return regime_stats_from_trades(
        bar_times, regimes,
        [tr.open_time for tr in book.closed],
        [tr.profit for tr in book.closed])


def worst_regime_net(stats: List[RegimeStats]) -> float:
    """Most negative per-regime net profit (0 when nothing loses)."""
    if not stats:
        return 0.0
    return min(0.0, min(s.net_profit for s in stats))
