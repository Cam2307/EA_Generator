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


def _cache_path(symbol: str, timeframe: str, start: datetime, end: datetime):
    key = f"{symbol}_{timeframe}_{start:%Y%m%d}_{end:%Y%m%d}"
    return settings.DATA_DIR / f"ohlc_{key}.parquet"


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def register_range_cache(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    """Pin the full discovery range for fast in-memory slicing."""
    _RANGE_CACHE[(symbol.upper(), timeframe)] = df


def clear_range_cache() -> None:
    _RANGE_CACHE.clear()


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
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return None
    try:
        if not mt5.initialize():
            return None
        tf_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        rates = mt5.copy_rates_range(symbol, tf_map[timeframe], start, end)
        mt5.shutdown()
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        return df[["time", "open", "high", "low", "close", "volume"]]
    except Exception:
        return None


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
    """Return the OHLC provenance label without mutating caller state."""
    parent = _RANGE_CACHE.get((symbol.upper(), timeframe))
    if parent is not None and "source" in parent.attrs:
        return str(parent.attrs["source"])
    df = load_ohlc(symbol, timeframe, start, end)
    return str(df.attrs.get("source", "unknown"))


def _remember(key, df: pd.DataFrame) -> None:
    if len(_MEM_CACHE) >= _MEM_CACHE_MAX:
        _MEM_CACHE.pop(next(iter(_MEM_CACHE)), None)
    _MEM_CACHE[key] = df
