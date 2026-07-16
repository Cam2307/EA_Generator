"""Numba-accelerated simulator core for Standard SL/TP + Partial close.

The Python :func:`factory.backtest.simulator.run_simulation` remains the
source of truth for DCA/grid, hedge, M1 intrabar replay, and rich
trade-management overlays. When ``SIMULATOR_NUMBA`` is enabled and the
strategy is eligible (default-enabled mechanics, path/conservative
intrabar, no regime overlays), discovery dispatches here for a flat-array
``@njit`` bar loop.

Regression: keep ``scripts/reconcile_engines.py`` / unit parity tests green
before trusting production runs on the JIT path.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from config import settings
from factory.models import (
    BacktestMetrics, ExecutionMechanicType, StrategyDefinition, TrailMode,
)

try:
    from numba import njit
    _NUMBA_OK = True
except Exception:  # pragma: no cover
    _NUMBA_OK = False

    def njit(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return deco


def numba_available() -> bool:
    return bool(_NUMBA_OK and getattr(settings, "SIMULATOR_NUMBA", False))


def strategy_numba_eligible(strategy: StrategyDefinition,
                            *, intrabar_mode: str) -> bool:
    """True when the JIT core can stand in for the Python bar loop."""
    if not numba_available():
        return False
    if strategy.mechanic.type not in (
            ExecutionMechanicType.STANDARD_SLTP,
            ExecutionMechanicType.PARTIAL_CLOSE):
        return False
    if intrabar_mode not in ("path", "conservative"):
        return False
    tm = strategy.trade_mgmt
    if getattr(tm, "regime_filter", False) or getattr(tm, "regime_sizing", False):
        return False
    if (getattr(tm, "hmm_regime_filter", False)
            or getattr(tm, "hmm_regime_sizing", False)):
        return False
    # Account-level gates live only in the Python PositionBook loop.
    if (tm.time_filter or tm.limit_trades_per_day
            or tm.daily_loss_enabled or tm.cooldown_enabled):
        return False
    if tm.trail_mode not in (TrailMode.OFF, TrailMode.FIXED):
        return False
    # ATR / percent stops need entry-time price — Python PositionBook path.
    if tm.uses_dynamic_stops():
        return False
    return True


@njit(cache=True)
def _path4(o, h, l, c):
    if c >= o:
        return o, l, h, c
    return o, h, l, c


@njit(cache=True)
def _first_touch(d, sl, tp, p0, p1, p2, p3):
    """Return exit price or NaN. Matches simulator._first_touch_exit."""
    points = (p0, p1, p2, p3)
    for i in range(3):
        a = points[i]
        b = points[i + 1]
        lo = a if a <= b else b
        hi = b if a <= b else a
        best_dist = 1e300
        best_prio = 99
        best_price = np.nan
        if sl > 0.0 and ((d > 0 and lo <= sl) or (d < 0 and hi >= sl)):
            already = (a <= sl) if d > 0 else (a >= sl)
            dist = 0.0 if already else abs(a - sl)
            if dist < best_dist or (dist == best_dist and 0 < best_prio):
                best_dist = dist
                best_prio = 0
                best_price = sl
        if tp > 0.0 and ((d > 0 and hi >= tp) or (d < 0 and lo <= tp)):
            already = (a >= tp) if d > 0 else (a <= tp)
            dist = 0.0 if already else abs(a - tp)
            if dist < best_dist or (dist == best_dist and 1 < best_prio):
                best_dist = dist
                best_prio = 1
                best_price = tp
        if not np.isnan(best_price):
            return best_price
    return np.nan


@njit(cache=True)
def _cash_pnl(price_diff, lots, mark_price, contract_size, pnl_div):
    """Account-currency PnL; ``pnl_div`` 1.0 divides by mark (JPY-style)."""
    raw = price_diff * contract_size * lots
    if pnl_div > 0.0 and mark_price > 0.0:
        return raw / mark_price
    return raw


@njit(cache=True)
def _run_core(
    times, opens, highs, lows, closes,
    long_sig, short_sig,
    point, point_value, contract_size, leverage,
    base_spread, base_slip,
    deposit, base_lots, max_open_lots,
    sl_points, tp_points,
    partial_enabled, partial_tp_points, partial_fraction,
    use_path,
    be_enabled, be_trigger, be_offset,
    trail_enabled, trail_start, trail_distance,
    max_spread_gate,
    spread_arr, slip_arr,  # empty length-0 arrays => static costs
    pnl_div,               # 1.0 => quote→account via /mark_price
):
    """Flat-array Standard SL/TP (+ optional partial) bar loop.

    Returns:
        balance, equity_curve, equity_ts,
        trade_dirs, trade_lots, trade_entry, trade_exit, trade_open_t,
        trade_close_t, trade_profit  (variable-length via n_trades)
    """
    n = times.shape[0]
    equity = np.empty(n, dtype=np.float64)
    eq_ts = np.empty(n, dtype=np.float64)

    max_trades = n  # upper bound
    t_dir = np.empty(max_trades, dtype=np.int64)
    t_lots = np.empty(max_trades, dtype=np.float64)
    t_entry = np.empty(max_trades, dtype=np.float64)
    t_exit = np.empty(max_trades, dtype=np.float64)
    t_ot = np.empty(max_trades, dtype=np.float64)
    t_ct = np.empty(max_trades, dtype=np.float64)
    t_profit = np.empty(max_trades, dtype=np.float64)
    n_trades = 0

    balance = deposit
    peak = deposit
    max_dd_money = 0.0
    max_dd_pct = 0.0

    # Open position state (at most one primary)
    has_pos = False
    p_dir = 0
    p_lots = 0.0
    p_entry = 0.0
    p_open_t = 0.0
    p_sl = 0.0
    p_tp = 0.0
    p_partial_done = False

    for i in range(n):
        t = times[i]
        o = opens[i]
        h = highs[i]
        l = lows[i]
        c = closes[i]

        if has_pos:
            hit = np.nan
            if use_path:
                p0, p1, p2, p3 = _path4(o, h, l, c)
                hit = _first_touch(p_dir, p_sl, p_tp, p0, p1, p2, p3)
            else:
                if p_sl > 0.0 and ((p_dir > 0 and l <= p_sl)
                                   or (p_dir < 0 and h >= p_sl)):
                    hit = p_sl
                elif p_tp > 0.0 and ((p_dir > 0 and h >= p_tp)
                                     or (p_dir < 0 and l <= p_tp)):
                    hit = p_tp
            if not np.isnan(hit):
                profit = _cash_pnl(p_dir * (hit - p_entry), p_lots, hit,
                                   contract_size, pnl_div)
                balance += profit
                t_dir[n_trades] = p_dir
                t_lots[n_trades] = p_lots
                t_entry[n_trades] = p_entry
                t_exit[n_trades] = hit
                t_ot[n_trades] = p_open_t
                t_ct[n_trades] = t
                t_profit[n_trades] = profit
                n_trades += 1
                has_pos = False

        if has_pos and partial_enabled and not p_partial_done:
            gain_pts = p_dir * (c - p_entry) / point
            if gain_pts >= partial_tp_points:
                close_lots = round(p_lots * partial_fraction * 100.0) / 100.0
                if close_lots > 1e-9 and close_lots < p_lots:
                    profit = _cash_pnl(p_dir * (c - p_entry), close_lots, c,
                                       contract_size, pnl_div)
                    balance += profit
                    t_dir[n_trades] = p_dir
                    t_lots[n_trades] = close_lots
                    t_entry[n_trades] = p_entry
                    t_exit[n_trades] = c
                    t_ot[n_trades] = p_open_t
                    t_ct[n_trades] = t
                    t_profit[n_trades] = profit
                    n_trades += 1
                    p_lots -= close_lots
                    p_partial_done = True
                    p_sl = p_entry  # breakeven after partial

        if has_pos:
            gain_pts = p_dir * (c - p_entry) / point
            if be_enabled and gain_pts >= be_trigger:
                be = p_entry + p_dir * be_offset * point
                if p_sl == 0.0:
                    p_sl = be
                elif p_dir > 0:
                    p_sl = be if be > p_sl else p_sl
                else:
                    p_sl = be if be < p_sl else p_sl
            if trail_enabled and gain_pts >= trail_start:
                new_sl = c - p_dir * trail_distance * point
                if p_dir > 0 and new_sl < c and new_sl > (
                        p_sl if p_sl > 0 else -1e300):
                    p_sl = new_sl
                elif p_dir < 0 and new_sl > c and (
                        p_sl == 0.0 or new_sl < p_sl):
                    p_sl = new_sl

        if not has_pos:
            direction = 1 if long_sig[i] else (-1 if short_sig[i] else 0)
            spread = base_spread
            slip = base_slip
            if spread_arr.shape[0] == n:
                spread = spread_arr[i]
                slip = slip_arr[i]
            if direction != 0 and max_spread_gate > 0 and spread > max_spread_gate:
                direction = 0
            if direction != 0:
                lots = base_lots
                if lots > max_open_lots:
                    lots = max_open_lots
                cost_pts = (spread if direction > 0 else 0.0) + slip
                price = c + direction * cost_pts * point
                if pnl_div > 0.0:
                    margin = lots * contract_size / leverage
                else:
                    margin = lots * contract_size * price / leverage
                # equity ≈ balance when flat
                if balance - 0.0 >= margin and lots > 0:
                    p_dir = direction
                    p_lots = lots
                    p_entry = price
                    p_open_t = t
                    p_sl = (price - direction * sl_points * point
                            if sl_points > 0 else 0.0)
                    p_tp = (price + direction * tp_points * point
                            if tp_points > 0 else 0.0)
                    p_partial_done = False
                    has_pos = True

        # Mark to market
        float_pnl = 0.0
        worst = 0.0
        if has_pos:
            float_pnl = _cash_pnl(p_dir * (c - p_entry), p_lots, c,
                                  contract_size, pnl_div)
            adv = l if p_dir > 0 else h
            worst = _cash_pnl(p_dir * (adv - p_entry), p_lots, adv,
                              contract_size, pnl_div)
        eq_close = balance + float_pnl
        eq_worst = balance + worst
        if eq_close > peak:
            peak = eq_close
        dd_money = peak - (eq_close if eq_close < eq_worst else eq_worst)
        if dd_money > max_dd_money:
            max_dd_money = dd_money
            max_dd_pct = dd_money / peak * 100.0 if peak > 0 else 0.0
        equity[i] = eq_close
        eq_ts[i] = t

    # Force-close remainder
    if has_pos and n > 0:
        c = closes[n - 1]
        t = times[n - 1]
        profit = _cash_pnl(p_dir * (c - p_entry), p_lots, c,
                           contract_size, pnl_div)
        balance += profit
        t_dir[n_trades] = p_dir
        t_lots[n_trades] = p_lots
        t_entry[n_trades] = p_entry
        t_exit[n_trades] = c
        t_ot[n_trades] = p_open_t
        t_ct[n_trades] = t
        t_profit[n_trades] = profit
        n_trades += 1
        equity[n - 1] = balance

    return (balance, equity, eq_ts, max_dd_money, max_dd_pct, n_trades,
            t_dir, t_lots, t_entry, t_exit, t_ot, t_ct, t_profit)


def run_simulation_numba(
    df,
    strategy: StrategyDefinition,
    spec,  # SymbolSpec
    deposit: float,
    entry_mask: Optional[np.ndarray] = None,
    cancel_check=None,
    intrabar_mode: str = "path",
) -> Tuple[BacktestMetrics, object]:
    """Numba fast path — same return shape as :func:`run_simulation`."""
    # Lazy imports avoid a circular import with simulator.py.
    from factory.backtest.simulator import (
        ClosedTrade, PositionBook, _metrics_from_book, compute_signals,
    )

    long_sig, short_sig, _ = compute_signals(df, strategy, spec)
    if entry_mask is not None:
        long_sig = long_sig & entry_mask
        short_sig = short_sig & entry_mask

    import pandas as pd
    _t = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    times = _t.to_numpy().astype("datetime64[s]").astype(np.int64)
    opens = df["open"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)

    mech = strategy.mechanic
    mp = mech.params
    tm = strategy.trade_mgmt
    tp = tm.params

    sl_points = float(mp.get("sl_points", 0.0) or 0.0)
    tp_points = float(mp.get("tp_points", 0.0) or 0.0)
    # Trade-mgmt FIXED stop / RR may override — keep mechanic defaults for JIT.
    if tm.sl_mode.value == "FIXED" and "sl_points" in tp:
        sl_points = float(tp.get("sl_points", sl_points))
    if tm.tp_mode.value == "FIXED" and "tp_points" in tp:
        tp_points = float(tp.get("tp_points", tp_points))

    partial = mech.type == ExecutionMechanicType.PARTIAL_CLOSE
    empty = np.empty(0, dtype=np.float64)
    spread_arr = empty
    slip_arr = empty
    if spec.dynamic_costs:
        from factory.backtest.costs import build_cost_arrays
        sa, sla = build_cost_arrays(df, spec.spread_points, spec.slippage_points)
        spread_arr = np.asarray(sa, dtype=np.float64)
        slip_arr = np.asarray(sla, dtype=np.float64)

    (balance, equity, eq_ts, max_dd_money, max_dd_pct, n_trades,
     t_dir, t_lots, t_entry, t_exit, t_ot, t_ct, t_profit) = _run_core(
        times, opens, highs, lows, closes,
        long_sig.astype(np.bool_), short_sig.astype(np.bool_),
        float(spec.point), float(spec.point_value),
        float(spec.contract_size), float(spec.leverage),
        float(spec.spread_points), float(spec.slippage_points),
        float(deposit), float(strategy.risk.fixed_lots),
        float(strategy.risk.max_open_lots),
        sl_points, tp_points,
        partial,
        float(mp.get("partial_tp_points", 0.0) or 0.0),
        float(mp.get("partial_fraction", 0.5) or 0.5),
        1 if intrabar_mode in ("path", "m1") else 0,
        1 if tm.breakeven else 0,
        float(tp.get("be_trigger_points", 1e18)),
        float(tp.get("be_offset_points", 0.0)),
        1 if tm.trail_mode == TrailMode.FIXED else 0,
        float(tp.get("trail_start_points", 1e18)),
        float(tp.get("trail_distance_points", 0.0)),
        float(strategy.risk.max_spread_points),
        spread_arr, slip_arr,
        1.0 if getattr(spec, "pnl_divide_by_price", False) else 0.0,
    )

    book = PositionBook(spec=spec, balance=float(balance))
    for k in range(int(n_trades)):
        book.closed.append(ClosedTrade(
            direction=int(t_dir[k]),
            lots=float(t_lots[k]),
            entry_price=float(t_entry[k]),
            exit_price=float(t_exit[k]),
            open_time=float(t_ot[k]),
            close_time=float(t_ct[k]),
            profit=float(t_profit[k]),
        ))
    equity_list = [float(x) for x in equity]
    ts_list = [float(x) for x in eq_ts]
    metrics = _metrics_from_book(
        book, deposit, ts_list, equity_list,
        float(max_dd_money), float(max_dd_pct), df)
    return metrics, book
