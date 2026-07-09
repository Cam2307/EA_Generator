"""Event-driven fallback backtest engine.

This is intentionally NOT a vectorized backtester. DCA/grid/hedge mechanics
are path-dependent and stateful (sequential grid fills, floating margin,
partial closes), so the engine runs a bar-by-bar loop over a stateful
``PositionBook``. numpy/pandas vectorization is used only to precompute
indicator/signal arrays before the loop.

The simulator is a *pre-filter*: every surviving strategy must still pass a
real MT5 Strategy Tester run before export (see README).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from factory import data as data_mod
from factory.backtest.base import BacktestEngine
from factory.models import (
    BacktestMetrics, EntryFilterType, ExecutionMechanicType, JobCancelled,
    LotMode, StopLossMode, StrategyDefinition, TakeProfitMode, TrailMode,
)

# Cooperative cancellation probe (see factory.backtest.validation). Checked
# periodically inside the bar loop so even a single long backtest aborts
# promptly instead of running to completion after a cancel.
CancelCheck = Optional[Callable[[], bool]]

# How often (in bars) the bar loop polls the cancel probe. The probe itself is
# throttled upstream, so this is just a cheap "am I still wanted?" heartbeat.
_CANCEL_CHECK_EVERY_BARS = 4096


# ---------------------------------------------------------------------------
# Symbol economics
# ---------------------------------------------------------------------------

@dataclass
class SymbolSpec:
    point: float = 0.00001            # minimal price increment
    contract_size: float = 100_000.0
    leverage: float = 100.0
    spread_points: float = 15.0       # applied on every entry fill
    slippage_points: float = 2.0      # adverse slippage per fill

    @property
    def point_value(self) -> float:
        """Account-currency value of one point for 1.0 lot."""
        return self.contract_size * self.point

    # Account/execution economics a user may override from the UI. ``point``
    # is deliberately excluded: it is a property of the price scale and is
    # always inferred from the data, never taken from user input.
    OVERRIDABLE = ("contract_size", "leverage", "spread_points",
                   "slippage_points")

    @classmethod
    def infer(cls, price: float,
              overrides: Optional[Mapping[str, float]] = None) -> "SymbolSpec":
        """Infer the price scale from ``price`` and apply user overrides.

        ``point`` (and the scale-dependent defaults) are always inferred from
        the first bar's price. Any economics supplied in ``overrides`` — the
        user's chosen leverage/spread/slippage/contract size — then win over
        the inferred defaults, so ``infer`` never clobbers user-provided
        values while still auto-detecting the point size.
        """
        if price < 20:                # 5-digit FX
            spec = cls(point=0.00001)
        else:
            spec = cls(point=0.01, contract_size=100.0, spread_points=20.0)
        if overrides:
            for name in cls.OVERRIDABLE:
                value = overrides.get(name)
                if value is not None:
                    setattr(spec, name, float(value))
        return spec


@dataclass
class Position:
    direction: int                    # +1 buy, -1 sell
    lots: float
    entry_price: float                # actual fill (incl. spread/slippage)
    open_time: float
    sl: float = 0.0                   # absolute price, 0 = none
    tp: float = 0.0
    is_hedge: bool = False
    partial_done: bool = False
    grid_level: int = 0

    def pnl_at(self, price: float, spec: SymbolSpec) -> float:
        return self.direction * (price - self.entry_price) / spec.point * spec.point_value * self.lots


@dataclass
class ClosedTrade:
    direction: int
    lots: float
    entry_price: float
    exit_price: float
    open_time: float
    close_time: float
    profit: float


@dataclass
class PositionBook:
    """Stateful book of open positions + realized balance."""
    spec: SymbolSpec
    balance: float
    positions: List[Position] = field(default_factory=list)
    closed: List[ClosedTrade] = field(default_factory=list)

    # -- costs ---------------------------------------------------------
    def fill_price(self, direction: int, bid: float) -> float:
        """Entry fill: buys pay spread + slippage above bid; sells slip below."""
        cost_points = (self.spec.spread_points if direction > 0 else 0.0) + self.spec.slippage_points
        return bid + direction * cost_points * self.spec.point

    # -- lifecycle ------------------------------------------------------
    def open(self, direction: int, lots: float, bid: float, time_s: float,
             sl_points: float = 0.0, tp_points: float = 0.0,
             is_hedge: bool = False, grid_level: int = 0) -> Optional[Position]:
        price = self.fill_price(direction, bid)
        margin_needed = lots * self.spec.contract_size * price / self.spec.leverage
        if self.equity(bid) - self.margin_used(bid) < margin_needed:
            return None                                   # margin refusal
        pos = Position(
            direction=direction, lots=lots, entry_price=price, open_time=time_s,
            sl=price - direction * sl_points * self.spec.point if sl_points > 0 else 0.0,
            tp=price + direction * tp_points * self.spec.point if tp_points > 0 else 0.0,
            is_hedge=is_hedge, grid_level=grid_level,
        )
        self.positions.append(pos)
        return pos

    def close(self, pos: Position, price: float, time_s: float,
              lots: Optional[float] = None) -> float:
        """Close fully or partially; returns realized profit."""
        close_lots = min(lots, pos.lots) if lots is not None else pos.lots
        profit = pos.direction * (price - pos.entry_price) / self.spec.point \
            * self.spec.point_value * close_lots
        self.balance += profit
        self.closed.append(ClosedTrade(
            direction=pos.direction, lots=close_lots, entry_price=pos.entry_price,
            exit_price=price, open_time=pos.open_time, close_time=time_s, profit=profit,
        ))
        pos.lots -= close_lots
        if pos.lots <= 1e-9:
            self.positions.remove(pos)
        return profit

    def close_all(self, price: float, time_s: float) -> None:
        for pos in list(self.positions):
            self.close(pos, price, time_s)

    # -- accounting -----------------------------------------------------
    def floating_pnl(self, bid: float) -> float:
        return sum(p.pnl_at(bid, self.spec) for p in self.positions)

    def worst_case_floating(self, low: float, high: float) -> float:
        """Most adverse intrabar floating PnL given bar extremes."""
        return sum(p.pnl_at(low if p.direction > 0 else high, self.spec)
                   for p in self.positions)

    def equity(self, bid: float) -> float:
        return self.balance + self.floating_pnl(bid)

    def margin_used(self, bid: float) -> float:
        return sum(p.lots * self.spec.contract_size * bid / self.spec.leverage
                   for p in self.positions)

    def total_lots(self, hedges: bool = False) -> float:
        return sum(p.lots for p in self.positions if p.is_hedge == hedges)


# ---------------------------------------------------------------------------
# Vectorized indicator / signal precompute
# ---------------------------------------------------------------------------

def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(arr).rolling(period).mean().to_numpy()


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50).to_numpy()


def _atr(df: pd.DataFrame, period: int) -> np.ndarray:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().to_numpy()


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=period, adjust=False).mean().to_numpy()


def _macd(close: np.ndarray, fast: int, slow: int, signal: int):
    """Return (macd_line, signal_line) matching MT5 iMACD (EMA-based)."""
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line


def _stochastic_k(df: pd.DataFrame, k_period: int) -> np.ndarray:
    """Fast %K in 0..100 (MT5 iStochastic main buffer, LOWHIGH price field)."""
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    rng = (high_max - low_min).replace(0, np.nan)
    return (100.0 * (df["close"] - low_min) / rng).fillna(50.0).to_numpy()


def _cci(df: pd.DataFrame, period: int) -> np.ndarray:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(period).mean()
    mad = (tp - sma).abs().rolling(period).mean()
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    return cci.fillna(0.0).to_numpy()


def _momentum(close: np.ndarray, period: int) -> np.ndarray:
    """MT5 iMomentum: close / close[period] * 100 (centered near 100)."""
    s = pd.Series(close)
    return (s / s.shift(period) * 100.0).fillna(100.0).to_numpy()


def _williams_r(df: pd.DataFrame, period: int) -> np.ndarray:
    """MT5 iWPR: -100 .. 0."""
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    rng = (high_max - low_min).replace(0, np.nan)
    return (-100.0 * (high_max - df["close"]) / rng).fillna(-50.0).to_numpy()


def _adx(df: pd.DataFrame, period: int):
    """Wilder's ADX with +DI / -DI (matches MT5 iADX buffers 0/1/2)."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return (adx.fillna(0.0).to_numpy(),
            plus_di.fillna(0.0).to_numpy(),
            minus_di.fillna(0.0).to_numpy())


def _dema(arr: np.ndarray, period: int) -> np.ndarray:
    """Double EMA (matches MT5 iDEMA): 2*EMA - EMA(EMA)."""
    ema1 = _ema(arr, period)
    ema2 = _ema(ema1, period)
    return 2.0 * ema1 - ema2


def _parabolic_sar(df: pd.DataFrame, step: float, max_step: float) -> np.ndarray:
    """Classic Wilder Parabolic SAR (iterative)."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    sar = np.zeros(n)
    if n == 0:
        return sar
    trend_up = True
    af = step
    ep = high[0]
    sar[0] = low[0]
    for i in range(1, n):
        prev = sar[i - 1]
        cur = prev + af * (ep - prev)
        if trend_up:
            lo_prev = low[i - 1] if i >= 1 else low[i]
            lo_prev2 = low[i - 2] if i >= 2 else lo_prev
            cur = min(cur, lo_prev, lo_prev2)
            if low[i] < cur:
                trend_up = False
                cur = ep
                ep = low[i]
                af = step
            elif high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
        else:
            hi_prev = high[i - 1] if i >= 1 else high[i]
            hi_prev2 = high[i - 2] if i >= 2 else hi_prev
            cur = max(cur, hi_prev, hi_prev2)
            if high[i] > cur:
                trend_up = True
                cur = ep
                ep = high[i]
                af = step
            elif low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)
        sar[i] = cur
    return sar


def _demarker(df: pd.DataFrame, period: int) -> np.ndarray:
    high, low = df["high"], df["low"]
    de_max = (high.diff()).clip(lower=0.0)
    de_min = (-low.diff()).clip(lower=0.0)
    sma_max = de_max.rolling(period).mean()
    sma_min = de_min.rolling(period).mean()
    dem = sma_max / (sma_max + sma_min).replace(0, np.nan)
    return dem.fillna(0.5).to_numpy()


def _awesome(df: pd.DataFrame) -> np.ndarray:
    median = (df["high"] + df["low"]) / 2.0
    return (_sma(median.to_numpy(), 5) - _sma(median.to_numpy(), 34))


def _force_index(df: pd.DataFrame, period: int) -> np.ndarray:
    raw = df["volume"].astype(float) * df["close"].diff()
    return raw.ewm(span=period, adjust=False).mean().fillna(0.0).to_numpy()


def _mfi(df: pd.DataFrame, period: int) -> np.ndarray:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mf = tp * df["volume"].astype(float)
    delta = tp.diff()
    pos = mf.where(delta > 0, 0.0).rolling(period).sum()
    neg = mf.where(delta < 0, 0.0).rolling(period).sum()
    ratio = pos / neg.replace(0, np.nan)
    mfi = 100.0 - 100.0 / (1.0 + ratio)
    return mfi.fillna(50.0).to_numpy()


def _rvi(df: pd.DataFrame, period: int):
    """Relative Vigor Index main + signal (MT5 iRVI, 4-bar symmetric weights)."""
    co = df["close"] - df["open"]
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    num = (co + 2.0 * co.shift(1) + 2.0 * co.shift(2) + co.shift(3)) / 6.0
    den = (hl + 2.0 * hl.shift(1) + 2.0 * hl.shift(2) + hl.shift(3)) / 6.0
    rvi = (num.rolling(period).mean()
           / den.rolling(period).mean().replace(0, np.nan))
    rvi = rvi.fillna(0.0)
    signal = (rvi + 2.0 * rvi.shift(1) + 2.0 * rvi.shift(2) + rvi.shift(3)) / 6.0
    return rvi.to_numpy(), signal.fillna(0.0).to_numpy()


def compute_signals(df: pd.DataFrame, strategy: StrategyDefinition,
                    spec: SymbolSpec) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (long_signal, short_signal, warmup_bars). Pure precompute."""
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    long_ok = np.ones(n, dtype=bool)
    short_ok = np.ones(n, dtype=bool)
    warmup = 1

    for f in strategy.entry_filters:
        p = f.params
        if f.type == EntryFilterType.PRICE_ACTION_BREAKOUT:
            lb = int(p["lookback"])
            buf = p["buffer_points"] * spec.point
            hh = pd.Series(high).rolling(lb).max().shift(1).to_numpy()
            ll = pd.Series(low).rolling(lb).min().shift(1).to_numpy()
            long_ok &= close > np.nan_to_num(hh, nan=np.inf) + buf
            short_ok &= close < np.nan_to_num(ll, nan=-np.inf) - buf
            warmup = max(warmup, lb + 1)

        elif f.type == EntryFilterType.MTF_VOLATILITY:
            period = int(p["atr_period"])
            atr = _atr(df, period)
            baseline = pd.Series(atr).rolling(100, min_periods=period).median().to_numpy()
            gate = atr > p["atr_mult_min"] * np.nan_to_num(baseline, nan=np.inf)
            trend = _sma(close, period * 2)
            long_ok &= gate & (close > np.nan_to_num(trend, nan=np.inf))
            short_ok &= gate & (close < np.nan_to_num(trend, nan=-np.inf))
            warmup = max(warmup, period * 2 + 1)

        elif f.type == EntryFilterType.LIQUIDITY_ZONE:
            lb = int(p["zone_lookback"])
            zone = p["zone_points"] * spec.point
            ll = pd.Series(low).rolling(lb).min().shift(1).to_numpy()
            hh = pd.Series(high).rolling(lb).max().shift(1).to_numpy()
            long_ok &= np.abs(close - np.nan_to_num(ll, nan=-np.inf)) <= zone
            short_ok &= np.abs(np.nan_to_num(hh, nan=np.inf) - close) <= zone
            warmup = max(warmup, lb + 1)

        elif f.type == EntryFilterType.RSI_REVERSION:
            rsi = _rsi(close, int(p["rsi_period"]))
            long_ok &= rsi < p["oversold"]
            short_ok &= rsi > p["overbought"]
            warmup = max(warmup, int(p["rsi_period"]) * 3)

        elif f.type == EntryFilterType.MA_CROSS:
            fast = _sma(close, int(p["fast_period"]))
            slow = _sma(close, int(p["slow_period"]))
            with np.errstate(invalid="ignore"):
                above = fast > slow
            prev = np.roll(above, 1)
            prev[0] = above[0]
            long_ok &= above & ~prev
            short_ok &= ~above & prev
            warmup = max(warmup, int(p["slow_period"]) + 1)

        elif f.type == EntryFilterType.BOLLINGER_FADE:
            period, dev = int(p["bb_period"]), p["bb_dev"]
            mid = _sma(close, period)
            std = pd.Series(close).rolling(period).std().to_numpy()
            upper = mid + dev * std
            lower = mid - dev * std
            long_ok &= close < np.nan_to_num(lower, nan=-np.inf)
            short_ok &= close > np.nan_to_num(upper, nan=np.inf)
            warmup = max(warmup, period + 1)

        elif f.type == EntryFilterType.MACD_CROSS:
            fast, slow = int(p["fast_ema"]), int(p["slow_ema"])
            macd_line, signal_line = _macd(close, fast, slow,
                                           int(p["signal_period"]))
            prev_macd = np.roll(macd_line, 1)
            prev_sig = np.roll(signal_line, 1)
            prev_macd[0], prev_sig[0] = macd_line[0], signal_line[0]
            long_ok &= (macd_line > signal_line) & (prev_macd <= prev_sig)
            short_ok &= (macd_line < signal_line) & (prev_macd >= prev_sig)
            warmup = max(warmup, slow + int(p["signal_period"]) + 1)

        elif f.type == EntryFilterType.STOCHASTIC:
            k_period = int(p["k_period"])
            k = _stochastic_k(df, k_period)
            long_ok &= k < p["oversold"]
            short_ok &= k > p["overbought"]
            warmup = max(warmup, k_period + 1)

        elif f.type == EntryFilterType.ADX_TREND:
            period = int(p["adx_period"])
            adx, plus_di, minus_di = _adx(df, period)
            trend = adx > p["adx_min"]
            long_ok &= trend & (plus_di > minus_di)
            short_ok &= trend & (minus_di > plus_di)
            warmup = max(warmup, period * 3 + 1)

        elif f.type == EntryFilterType.CCI_REVERSION:
            period = int(p["cci_period"])
            cci = _cci(df, period)
            long_ok &= cci < -p["cci_level"]
            short_ok &= cci > p["cci_level"]
            warmup = max(warmup, period + 1)

        elif f.type == EntryFilterType.MOMENTUM:
            period = int(p["mom_period"])
            mom = _momentum(close, period)
            long_ok &= mom > (100.0 + p["mom_threshold"])
            short_ok &= mom < (100.0 - p["mom_threshold"])
            warmup = max(warmup, period + 1)

        elif f.type == EntryFilterType.WILLIAMS_R:
            period = int(p["wpr_period"])
            wpr = _williams_r(df, period)
            long_ok &= wpr < p["wpr_oversold"]
            short_ok &= wpr > p["wpr_overbought"]
            warmup = max(warmup, period + 1)

        elif f.type == EntryFilterType.VOLUME_SURGE:
            period = int(p["vol_period"])
            vol = df["volume"].astype(float)
            avg = vol.rolling(period).mean().to_numpy()
            surge = vol.to_numpy() > p["vol_mult"] * np.nan_to_num(avg, nan=np.inf)
            long_ok &= surge
            short_ok &= surge
            warmup = max(warmup, period + 1)

        elif f.type == EntryFilterType.PARABOLIC_SAR:
            sar = _parabolic_sar(df, p["sar_step"], p["sar_max"])
            long_ok &= close > sar
            short_ok &= close < sar
            warmup = max(warmup, 5)

        elif f.type == EntryFilterType.ICHIMOKU:
            tenkan_n, kijun_n = int(p["tenkan"]), int(p["kijun"])
            tenkan = ((pd.Series(high).rolling(tenkan_n).max()
                       + pd.Series(low).rolling(tenkan_n).min()) / 2.0).to_numpy()
            kijun = ((pd.Series(high).rolling(kijun_n).max()
                      + pd.Series(low).rolling(kijun_n).min()) / 2.0).to_numpy()
            above = tenkan > kijun
            prev = np.roll(above, 1)
            prev[0] = above[0]
            long_ok &= above & ~prev
            short_ok &= ~above & prev
            warmup = max(warmup, int(p["senkou"]) + 1)

        elif f.type == EntryFilterType.DEMARKER:
            period = int(p["dem_period"])
            dem = _demarker(df, period)
            long_ok &= dem < p["dem_oversold"]
            short_ok &= dem > p["dem_overbought"]
            warmup = max(warmup, period + 1)

        elif f.type == EntryFilterType.AWESOME:
            ao = _awesome(df)
            thr = p.get("ao_threshold", 0.0)
            prev = np.roll(ao, 1)
            prev[0] = ao[0]
            long_ok &= (ao > thr) & (prev <= thr)
            short_ok &= (ao < -thr) & (prev >= -thr)
            warmup = max(warmup, 35)

        elif f.type == EntryFilterType.FORCE_INDEX:
            period = int(p["force_period"])
            fi = _force_index(df, period)
            prev = np.roll(fi, 1)
            prev[0] = fi[0]
            long_ok &= (fi > 0) & (prev <= 0)
            short_ok &= (fi < 0) & (prev >= 0)
            warmup = max(warmup, period + 1)

        elif f.type == EntryFilterType.STDDEV_REGIME:
            period = int(p["std_period"])
            std = pd.Series(close).rolling(period).std()
            avg = std.rolling(period).mean().to_numpy()
            gate = std.to_numpy() > p["std_mult"] * np.nan_to_num(avg, nan=np.inf)
            long_ok &= gate
            short_ok &= gate
            warmup = max(warmup, period * 2 + 1)

        elif f.type == EntryFilterType.ENVELOPES:
            period = int(p["env_period"])
            mid = _sma(close, period)
            band = mid * (p["env_deviation"] / 100.0)
            long_ok &= close < np.nan_to_num(mid - band, nan=-np.inf)
            short_ok &= close > np.nan_to_num(mid + band, nan=np.inf)
            warmup = max(warmup, period + 1)

        elif f.type == EntryFilterType.MFI:
            period = int(p["mfi_period"])
            mfi = _mfi(df, period)
            long_ok &= mfi < p["mfi_oversold"]
            short_ok &= mfi > p["mfi_overbought"]
            warmup = max(warmup, period + 1)

        elif f.type == EntryFilterType.RVI:
            period = int(p["rvi_period"])
            rvi, signal = _rvi(df, period)
            above = rvi > signal
            prev = np.roll(above, 1)
            prev[0] = above[0]
            long_ok &= above & ~prev
            short_ok &= ~above & prev
            warmup = max(warmup, period + 4)

        elif f.type == EntryFilterType.DEMA_CROSS:
            fast_n, slow_n = int(p["dema_fast"]), int(p["dema_slow"])
            fast = _dema(close, fast_n)
            slow = _dema(close, slow_n)
            above = fast > slow
            prev = np.roll(above, 1)
            prev[0] = above[0]
            long_ok &= above & ~prev
            short_ok &= ~above & prev
            warmup = max(warmup, slow_n * 2 + 1)

    long_ok[:warmup] = False
    short_ok[:warmup] = False
    return long_ok, short_ok, warmup


# ---------------------------------------------------------------------------
# Trade-management (exit / risk overlay) helpers
# ---------------------------------------------------------------------------

def _trail_stop(tm, tp, d: int, price: float, highs: np.ndarray,
                lows: np.ndarray, i: int, atr_price: float,
                spec: SymbolSpec) -> float:
    """Proposed trailing-stop price for a directional position (0 = skip)."""
    if tm.trail_mode == TrailMode.FIXED:
        return price - d * tp.get("trail_distance_points", 0.0) * spec.point
    if tm.trail_mode == TrailMode.ATR:
        if atr_price <= 0.0:
            return 0.0
        return price - d * atr_price * tp.get("trail_atr_mult", 2.0)
    if tm.trail_mode == TrailMode.CHANDELIER:
        if atr_price <= 0.0:
            return 0.0
        lb = int(tp.get("chandelier_lookback", 22))
        lo = max(0, i - lb + 1)
        mult = tp.get("trail_atr_mult", 3.0)
        if d > 0:
            return float(np.max(highs[lo:i + 1])) - atr_price * mult
        return float(np.min(lows[lo:i + 1])) + atr_price * mult
    return 0.0


def _tm_entry_allowed(tm, tp, hour: int, trades_today: int,
                      day_start_balance: float, equity: float, i: int,
                      cooldown_until: int) -> bool:
    """Account-level entry gates: session / max-per-day / daily-loss / cooldown."""
    if tm.time_filter:
        sh, eh = int(tp.get("start_hour", 0)), int(tp.get("end_hour", 24))
        inside = (sh <= hour < eh) if sh <= eh else (hour >= sh or hour < eh)
        if not inside:
            return False
    if tm.limit_trades_per_day and trades_today >= int(tp.get("max_trades_per_day", 1e9)):
        return False
    if tm.daily_loss_enabled and day_start_balance > 0:
        dd = (day_start_balance - equity) / day_start_balance * 100.0
        if dd >= tp.get("daily_loss_pct", 1e18):
            return False
    if tm.cooldown_enabled and i < cooldown_until:
        return False
    return True


def _entry_sizing(tm, tp, mech_type, mp, directional: bool, atr_price: float,
                  spec: SymbolSpec, base_lots: float, equity: float,
                  max_open_lots: float) -> Tuple[float, float, float]:
    """Return ``(sl_points, tp_points, lots)`` for a new entry under the overlay."""
    if not directional:
        if mech_type == ExecutionMechanicType.DCA_GRID:
            return 0.0, 0.0, base_lots       # basket-managed: no broker SL/TP
        return mp.get("sl_points", 0.0), mp.get("tp_points", 0.0), base_lots

    if tm.sl_mode == StopLossMode.OFF:
        sl_pts = 0.0
    elif tm.sl_mode == StopLossMode.ATR and atr_price > 0.0:
        sl_pts = atr_price / spec.point * tp.get("atr_sl_mult", 2.0)
    else:                                     # FIXED (or ATR warmup fallback)
        sl_pts = mp.get("sl_points", 0.0)

    if tm.tp_mode == TakeProfitMode.OFF:
        tp_pts = 0.0
    elif tm.tp_mode == TakeProfitMode.RR and sl_pts > 0.0:
        tp_pts = sl_pts * tp.get("tp_rr", 2.0)
    else:
        tp_pts = mp.get("tp_points", 0.0)

    if tm.lot_mode == LotMode.RISK_PERCENT and sl_pts > 0.0 and spec.point_value > 0:
        risk_money = equity * tp.get("risk_percent", 1.0) / 100.0
        lots = min(max(0.01, round(risk_money / (sl_pts * spec.point_value), 2)),
                   max_open_lots)
    else:
        lots = base_lots
    return sl_pts, tp_pts, lots


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class SimulatorEngine(BacktestEngine):
    name = "simulator"

    def __init__(self, spec: Optional[SymbolSpec] = None,
                 ohlc: Optional[pd.DataFrame] = None,
                 spec_overrides: Optional[Mapping[str, float]] = None):
        """``ohlc`` may be injected for tests; otherwise loaded via factory.data.

        ``spec`` fully replaces the inferred symbol economics (used by tests).
        ``spec_overrides`` instead supplies just the user-chosen account /
        execution economics (leverage, spread, slippage, contract size) that
        win over the inferred defaults while ``point`` stays auto-inferred.
        """
        self._spec_override = spec
        self._ohlc_override = ohlc
        self._spec_overrides = dict(spec_overrides) if spec_overrides else None
        # Optional cancel probe set by the worker; every backtest this engine
        # runs (IS optimization, OOS, walk-forward) then honours it mid-loop.
        self._cancel_check: CancelCheck = None

    def run(self, strategy: StrategyDefinition, start: datetime, end: datetime,
            params_override: Optional[Dict[str, float]] = None,
            deposit: float = 10_000.0) -> BacktestMetrics:
        if params_override:
            strategy = strategy.apply_flat_params(params_override)

        if self._ohlc_override is not None:
            df = self._ohlc_override
        else:
            df = data_mod.load_ohlc(strategy.symbol, strategy.timeframe, start, end)
        if len(df) < 10:
            raise ValueError("Not enough bars for a backtest")

        spec = self._spec_override or SymbolSpec.infer(
            float(df["close"].iloc[0]), self._spec_overrides)
        return simulate(df, strategy, spec, deposit,
                        cancel_check=self._cancel_check)


def simulate(df: pd.DataFrame, strategy: StrategyDefinition, spec: SymbolSpec,
             deposit: float, cancel_check: CancelCheck = None) -> BacktestMetrics:
    metrics, _ = run_simulation(df, strategy, spec, deposit,
                                cancel_check=cancel_check)
    return metrics


def run_simulation(df: pd.DataFrame, strategy: StrategyDefinition,
                   spec: SymbolSpec, deposit: float,
                   entry_mask: Optional[np.ndarray] = None,
                   cancel_check: CancelCheck = None
                   ) -> Tuple[BacktestMetrics, PositionBook]:
    """Bar-by-bar event loop. Only signal arrays are precomputed.

    ``entry_mask`` (bool array, len == bars) optionally suppresses entry
    signals on masked-out bars — used by the Monte Carlo module for random
    entry skipping and first-bar jitter.

    ``cancel_check`` is polled every few thousand bars so a cancelled job
    aborts a long backtest promptly (raising :class:`JobCancelled`) instead of
    grinding to the end of the data first.

    Returns the final PositionBook too so tests can assert fill sequences.
    """
    long_sig, short_sig, _ = compute_signals(df, strategy, spec)
    if entry_mask is not None:
        long_sig = long_sig & entry_mask
        short_sig = short_sig & entry_mask

    # Robust unix-seconds regardless of the datetime resolution pandas chose
    # (pandas 3.0 uses us/ms, not ns — a hardcoded //10**9 corrupts timestamps).
    _t = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    times = _t.to_numpy().astype("datetime64[s]").astype("int64")
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(df)

    book = PositionBook(spec=spec, balance=deposit)
    mech = strategy.mechanic
    mp = mech.params
    max_open_lots = strategy.risk.max_open_lots
    base_lots = strategy.risk.fixed_lots

    # -- trade-management (exit/risk) overlay precompute + state -----------
    tm = strategy.trade_mgmt
    tp = tm.params
    tm_directional = mech.type in (ExecutionMechanicType.STANDARD_SLTP,
                                   ExecutionMechanicType.PARTIAL_CLOSE)
    atr_arr = _atr(df, int(tp.get("atr_period", 14))) if tm.uses_atr() else None
    day_idx = times // 86400                      # integer trading-day index
    hour_of_day = ((times % 86400) // 3600).astype("int64")
    cur_day = -1
    trades_today = 0
    day_start_balance = deposit
    cooldown_until = -1
    closed_seen = 0                               # trades realized so far

    equity_ts: List[float] = []
    equity_curve: List[float] = []
    peak = deposit
    max_dd_money = 0.0
    max_dd_pct = 0.0

    for i in range(n):
        if (cancel_check is not None and i % _CANCEL_CHECK_EVERY_BARS == 0
                and cancel_check()):
            raise JobCancelled()
        t, o, h, l, c = times[i], opens[i], highs[i], lows[i], closes[i]

        # new trading day: reset per-day counters (session/loss/trade caps)
        if day_idx[i] != cur_day:
            cur_day = day_idx[i]
            trades_today = 0
            day_start_balance = book.balance

        # -- 1. intrabar SL/TP exits (conservative: SL checked before TP) --
        for pos in list(book.positions):
            hit_price = None
            if pos.sl > 0 and ((pos.direction > 0 and l <= pos.sl)
                               or (pos.direction < 0 and h >= pos.sl)):
                hit_price = pos.sl
            elif pos.tp > 0 and ((pos.direction > 0 and h >= pos.tp)
                                 or (pos.direction < 0 and l <= pos.tp)):
                hit_price = pos.tp
            if hit_price is not None:
                was_primary = not pos.is_hedge
                book.close(pos, hit_price, t)
                # a hedge exists only to offset the primary: close it too
                if was_primary and mech.type == ExecutionMechanicType.HEDGE_LAYER:
                    for hp in [p for p in book.positions if p.is_hedge]:
                        book.close(hp, c, t)

        # -- 2. mechanic-specific management at bar close ------------------
        if mech.type == ExecutionMechanicType.DCA_GRID and book.positions:
            primaries = [p for p in book.positions if not p.is_hedge]
            if primaries:
                d = primaries[0].direction
                last_entry = min((p.entry_price for p in primaries), default=c) if d > 0 \
                    else max((p.entry_price for p in primaries), default=c)
                adverse_pts = (last_entry - c) / spec.point if d > 0 \
                    else (c - last_entry) / spec.point
                if (adverse_pts >= mp["grid_step_points"]
                        and len(primaries) < int(mp["max_levels"])):
                    lots = base_lots * (mp["lot_multiplier"] ** len(primaries))
                    if book.total_lots() + lots <= max_open_lots:
                        book.open(d, round(lots, 2), c, t, grid_level=len(primaries))
                # basket TP off volume-weighted average price
                primaries = [p for p in book.positions if not p.is_hedge]
                if primaries:
                    tot = sum(p.lots for p in primaries)
                    avg = sum(p.entry_price * p.lots for p in primaries) / tot
                    target = avg + d * mp["basket_tp_points"] * spec.point
                    if (d > 0 and h >= target) or (d < 0 and l <= target):
                        for p in list(primaries):
                            book.close(p, target, t)

        elif mech.type == ExecutionMechanicType.HEDGE_LAYER and book.positions:
            primaries = [p for p in book.positions if not p.is_hedge]
            hedges = [p for p in book.positions if p.is_hedge]
            if primaries and not hedges:
                p0 = primaries[0]
                adverse_pts = -p0.direction * (c - p0.entry_price) / spec.point
                if adverse_pts >= mp["hedge_trigger_points"]:
                    book.open(-p0.direction, round(p0.lots * mp["hedge_ratio"], 2),
                              c, t, is_hedge=True)
            elif primaries and hedges and book.floating_pnl(c) >= 0:
                book.close_all(c, t)      # basket recovered to breakeven

        elif mech.type == ExecutionMechanicType.PARTIAL_CLOSE:
            for pos in [p for p in book.positions if not p.partial_done]:
                gain_pts = pos.direction * (c - pos.entry_price) / spec.point
                if gain_pts >= mp["partial_tp_points"]:
                    book.close(pos, c, t, lots=round(pos.lots * mp["partial_fraction"], 2))
                    if pos in book.positions:
                        pos.partial_done = True
                        pos.sl = pos.entry_price      # breakeven stop

        # -- 2b. trade-management overlay (breakeven + trailing) -----------
        # Directional mechanics only; grid/hedge keep their own basket logic.
        if tm_directional and book.positions:
            atr_price = (float(atr_arr[i]) if atr_arr is not None
                         and not np.isnan(atr_arr[i]) else 0.0)
            for pos in book.positions:
                if pos.is_hedge:
                    continue
                d = pos.direction
                gain_pts = d * (c - pos.entry_price) / spec.point

                if tm.breakeven and gain_pts >= tp.get("be_trigger_points", 1e18):
                    be = pos.entry_price + d * tp.get("be_offset_points", 0.0) * spec.point
                    pos.sl = be if pos.sl == 0.0 else (
                        max(pos.sl, be) if d > 0 else min(pos.sl, be))

                if (tm.trail_mode != TrailMode.OFF
                        and gain_pts >= tp.get("trail_start_points", 1e18)):
                    new_sl = _trail_stop(tm, tp, d, c, highs, lows, i,
                                         atr_price, spec)
                    if new_sl > 0.0:
                        step = tp.get("trail_step_points", 0.0) * spec.point
                        if d > 0 and new_sl < c and new_sl > (
                                pos.sl if pos.sl > 0 else -1e18) + step:
                            pos.sl = new_sl
                        elif d < 0 and new_sl > c and (
                                pos.sl == 0.0 or new_sl < pos.sl - step):
                            pos.sl = new_sl

        # cooldown: remember when a losing trade just closed
        if tm.cooldown_enabled and len(book.closed) > closed_seen:
            for tr in book.closed[closed_seen:]:
                if tr.profit < 0.0:
                    cooldown_until = i + int(tp.get("cooldown_bars", 0))
        closed_seen = len(book.closed)

        # -- 3. entries on bar close ---------------------------------------
        if not book.positions and _tm_entry_allowed(
                tm, tp, hour_of_day[i], trades_today, day_start_balance,
                book.equity(c), i, cooldown_until):
            direction = 1 if long_sig[i] else (-1 if short_sig[i] else 0)
            if direction != 0:
                atr_price = (float(atr_arr[i]) if atr_arr is not None
                             and not np.isnan(atr_arr[i]) else 0.0)
                sl_pts, tp_pts, lots = _entry_sizing(
                    tm, tp, mech.type, mp, tm_directional, atr_price, spec,
                    base_lots, book.equity(c), max_open_lots)
                if book.open(direction, lots, c, t,
                             sl_points=sl_pts, tp_points=tp_pts) is not None:
                    trades_today += 1

        # -- 4. mark to market ----------------------------------------------
        eq_close = book.equity(c)
        eq_worst = book.balance + book.worst_case_floating(l, h)
        peak = max(peak, eq_close)
        dd_money = peak - min(eq_close, eq_worst)
        if dd_money > max_dd_money:
            max_dd_money = dd_money
            max_dd_pct = dd_money / peak * 100 if peak > 0 else 0.0
        equity_ts.append(float(t))
        equity_curve.append(float(eq_close))

    book.close_all(float(closes[-1]), float(times[-1]))
    equity_curve[-1] = book.balance

    metrics = _metrics_from_book(book, deposit, equity_ts, equity_curve,
                                 max_dd_money, max_dd_pct, df)
    return metrics, book


def equity_r_squared(equity: List[float]) -> float:
    """R-squared of a linear fit to the equity curve (stability metric).

    A perfectly straight rising (or falling) equity line scores 1.0; a
    choppy, stagnating curve scores near 0. Flat curves score 0.
    """
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 3 or float(np.std(eq)) < 1e-12:
        return 0.0
    x = np.arange(len(eq), dtype=float)
    corr = float(np.corrcoef(x, eq)[0, 1])
    return round(corr * corr, 4)


def max_drawdown_pct(equity: List[float]) -> float:
    """Worst peak-to-trough decline of an equity curve, as a percentage.

    Tracks the running high-water mark and returns the largest observed
    ``(peak - value) / peak`` in percent. Returns 0 for an empty/flat curve.
    """
    peak = float("-inf")
    worst = 0.0
    for v in equity:
        if v > peak:
            peak = v
        if peak > 0.0:
            dd = (peak - v) / peak * 100.0
            if dd > worst:
                worst = dd
    return worst


def max_consecutive_losses(profits: List[float]) -> int:
    """Longest run of consecutive losing trades (chronological order)."""
    worst = current = 0
    for p in profits:
        if p < 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


def _metrics_from_book(book: PositionBook, deposit: float,
                       equity_ts: List[float], equity_curve: List[float],
                       max_dd_money: float, max_dd_pct: float,
                       df: pd.DataFrame) -> BacktestMetrics:
    gross_profit = sum(tr.profit for tr in book.closed if tr.profit > 0)
    gross_loss = -sum(tr.profit for tr in book.closed if tr.profit < 0)
    net = book.balance - deposit
    eq = np.asarray(equity_curve, dtype=float)
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
    bar_seconds = float(np.median(np.diff(equity_ts))) if len(equity_ts) > 1 else 60.0
    bars_per_year = 365.25 * 86400 / max(bar_seconds, 1.0)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(bars_per_year)) \
        if len(rets) > 1 and rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = (float(rets.mean() / downside_std * np.sqrt(bars_per_year))
               if downside_std > 0 else 0.0)

    chronological = sorted(book.closed, key=lambda tr: tr.close_time)
    consec_losses = max_consecutive_losses([tr.profit for tr in chronological])

    # thin the stored equity curve to keep DB rows small
    stride = max(1, len(equity_curve) // 2000)

    return BacktestMetrics(
        net_profit=round(net, 2),
        gross_profit=round(gross_profit, 2),
        gross_loss=round(gross_loss, 2),
        profit_factor=round(gross_profit / gross_loss, 3) if gross_loss > 0
        else (999.0 if gross_profit > 0 else 0.0),
        recovery_factor=round(net / max_dd_money, 3) if max_dd_money > 0 else 0.0,
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        max_dd_pct=round(max_dd_pct, 3),
        max_dd_money=round(max_dd_money, 2),
        trade_count=len(book.closed),
        r_squared=equity_r_squared(equity_curve),
        max_consecutive_losses=consec_losses,
        initial_deposit=deposit,
        start_ts=float(equity_ts[0]) if equity_ts else 0.0,
        end_ts=float(equity_ts[-1]) if equity_ts else 0.0,
        equity_ts=[float(x) for x in equity_ts[::stride]],
        equity=[float(x) for x in equity_curve[::stride]],
    )
