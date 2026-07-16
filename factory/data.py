"""OHLC data access: MT5 -> parquet cache -> synthetic fallback.

The simulator needs bar data even on machines without MetaTrader 5 installed,
so the loader degrades to a deterministic synthetic random-walk series
(clearly a development aid — documented in README).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config import settings

TIMEFRAME_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}

# In-process OHLC memo: discovery re-runs the same bar series dozens of times
# per strategy (IS optimization, WFO windows, Monte Carlo). Re-reading parquet
# each time dominates runtime, so cache the parsed DataFrame in memory. Callers
# treat the frame as read-only.
_MEM_CACHE: dict = {}
_MEM_CACHE_MAX = 32

# Full-range bars keyed by (symbol, timeframe). WFO/IS windows slice from this
# instead of reloading parquet or re-querying MT5 for every sub-range.
_RANGE_CACHE: dict[tuple[str, str], pd.DataFrame] = {}

# Negative cache for ranges that already failed to load (e.g. no EURGBP M1).
# Without this, every Optuna trial re-initializes MT5 and waits for an empty
# copy_rates_range — the dominant "stuck on Submitted sweep" cost when M1
# intrabar mode is on but M1 history is missing.
_MISS_CACHE: set[tuple] = set()


def _cache_path(symbol: str, timeframe: str, start: datetime, end: datetime):
    key = f"{symbol}_{timeframe}_{start:%Y%m%d}_{end:%Y%m%d}"
    return settings.DATA_DIR / f"ohlc_{key}.parquet"


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _range_key(symbol: str, timeframe: str, start: datetime, end: datetime
               ) -> tuple:
    return (symbol.upper(), timeframe.upper(),
            _utc(start).isoformat(), _utc(end).isoformat())


def register_range_cache(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    """Pin the full discovery range for fast in-memory slicing."""
    key = (symbol.upper(), timeframe)
    _RANGE_CACHE[key] = df
    # A successful pin supersedes any prior miss for this symbol/TF.
    stale = [k for k in _MISS_CACHE if k[0] == key[0] and k[1] == key[1]]
    for k in stale:
        _MISS_CACHE.discard(k)


def get_range_cache(symbol: str, timeframe: str,
                    start: Optional[datetime] = None,
                    end: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """Return a pinned discovery range (optionally sliced), or ``None``."""
    key = (symbol.upper(), timeframe.upper() if timeframe else timeframe)
    # Keys are stored as (symbol, timeframe) without upper on TF in register —
    # try both exact and uppercased TF.
    df = _RANGE_CACHE.get(key)
    if df is None:
        df = _RANGE_CACHE.get((symbol.upper(), timeframe))
    if df is None:
        return None
    if start is not None and end is not None:
        return _slice_df(df, start, end)
    return df


def mark_unavailable(symbol: str, timeframe: str, start: datetime,
                     end: datetime) -> None:
    """Remember that this exact range cannot be loaded (skip MT5 next time)."""
    _MISS_CACHE.add(_range_key(symbol, timeframe, start, end))


def clear_range_cache() -> None:
    _RANGE_CACHE.clear()
    _MISS_CACHE.clear()
    _MEM_CACHE.clear()


def _slice_df(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    start, end = _utc(start), _utc(end)
    times = pd.to_datetime(df["time"], utc=True)
    mask = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
    out = df.loc[mask]
    if out.empty:
        return out
    sliced = out.copy()
    sliced.attrs = dict(getattr(df, "attrs", {}))
    return sliced


def _try_mt5(symbol: str, timeframe: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    """Pull OHLC via the MetaTrader5 Python API.

    ``mt5.initialize()`` starts an interactive ``terminal64.exe`` when none is
    running, and ``shutdown()`` does **not** close it. That leftover terminal
    then blocks headless Strategy Tester runs. If we started the terminal,
    kill it after the pull so discovery/MT5 testing can proceed.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return None
    from factory.backtest.mt5_runner import (
        interactive_terminal_running,
        kill_stray_terminals,
    )
    already_running = False
    started_by_us = False
    try:
        already_running = interactive_terminal_running()
        if not mt5.initialize():
            return None
        started_by_us = not already_running
        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        rates = mt5.copy_rates_range(symbol, tf_map[timeframe], start, end)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        return df[["time", "open", "high", "low", "close", "volume"]]
    except Exception:
        return None
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass
        if started_by_us:
            kill_stray_terminals()


def synthetic_ohlc(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Deterministic random-walk OHLC (seeded by symbol + range) for offline development."""
    minutes = TIMEFRAME_MINUTES[timeframe]
    start, end = _utc(start), _utc(end)
    idx = pd.date_range(start, end, freq=f"{minutes}min", tz="utc")
    n = len(idx)
    if n < 2:
        raise ValueError("Date range too short for the requested timeframe")

    # Include the date window in the seed so different test durations produce
    # distinct synthetic paths (not just a longer prefix of the same walk).
    seed_key = f"{symbol}_{timeframe}_{start:%Y%m%d}_{end:%Y%m%d}"
    seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    base = 1.1000 if symbol.upper().endswith("USD") else 100.0
    vol = 0.0004 * base
    drift = np.zeros(n)
    regime_len = max(200, n // 12)
    for i in range(0, n, regime_len):
        drift[i:i + regime_len] = rng.normal(0, 0.00003 * base)
    steps = rng.normal(0, vol, n) + drift
    close = base + np.cumsum(steps)
    close = np.maximum(close, base * 0.2)

    spread_noise = np.abs(rng.normal(0, vol, n))
    open_ = np.roll(close, 1)
    open_[0] = base
    high = np.maximum(open_, close) + spread_noise
    low = np.minimum(open_, close) - spread_noise
    volume = rng.integers(50, 500, n).astype(float)

    return pd.DataFrame({
        "time": idx, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


def load_ohlc(symbol: str, timeframe: str, start: datetime, end: datetime,
              allow_synthetic: bool = True) -> pd.DataFrame:
    """Load bars, preferring range cache slices, in-memory memo, parquet, MT5,
    then synthetic."""
    settings.ensure_dirs()
    mem_key = (symbol, timeframe, _utc(start).isoformat(), _utc(end).isoformat())
    cached = _MEM_CACHE.get(mem_key)
    if cached is not None:
        if "source" not in cached.attrs:
            cached.attrs["source"] = "cache"
        return cached

    parent = _RANGE_CACHE.get((symbol.upper(), timeframe))
    if parent is not None:
        sliced = _slice_df(parent, start, end)
        if len(sliced) >= 2:
            _remember(mem_key, sliced)
            return sliced

    miss_key = _range_key(symbol, timeframe, start, end)
    if miss_key in _MISS_CACHE and not allow_synthetic:
        raise RuntimeError(
            f"No OHLC data available for {symbol} {timeframe}: previously "
            "unavailable (cached miss)."
        )

    cache = _cache_path(symbol, timeframe, start, end)
    if cache.exists():
        df = pd.read_parquet(cache)
        df.attrs["source"] = "cache"
        _remember(mem_key, df)
        return df

    df = _try_mt5(symbol, timeframe, start, end)
    source = "mt5"
    if df is None:
        if not allow_synthetic:
            mark_unavailable(symbol, timeframe, start, end)
            raise RuntimeError(
                f"No OHLC data available for {symbol} {timeframe}: MT5 unavailable "
                "and synthetic data disabled."
            )
        df = synthetic_ohlc(symbol, timeframe, start, end)
        source = "synthetic"
    df.attrs["source"] = source
    try:
        df.to_parquet(cache)
    except Exception:
        pass  # cache write is best-effort
    _remember(mem_key, df)
    return df


def peek_source(symbol: str, timeframe: str, start: datetime, end: datetime) -> str:
    """Return the OHLC provenance label without a full MT5 round-trip when possible."""
    parent = _RANGE_CACHE.get((symbol.upper(), timeframe))
    if parent is not None and "source" in parent.attrs:
        return str(parent.attrs["source"])
    cache = _cache_path(symbol, timeframe, start, end)
    if cache.exists():
        return "cache"
    # Avoid loading (and possibly synthesizing) bars just to label provenance
    # during payload build — that doubles startup I/O before the worker begins.
    try:
        import MetaTrader5 as mt5  # noqa: F401
        return "mt5"
    except ImportError:
        return "synthetic"


def _remember(key, df: pd.DataFrame) -> None:
    if len(_MEM_CACHE) >= _MEM_CACHE_MAX:
        _MEM_CACHE.pop(next(iter(_MEM_CACHE)), None)
    _MEM_CACHE[key] = df
