"""Intrabar SL/TP exit resolution (path heuristic + M1 replay)."""
import numpy as np
import pandas as pd

from factory.backtest.simulator import (
    Position, SymbolSpec, _first_touch_exit, _path_points, run_simulation,
)
from factory.models import (
    EntryFilter, EntryFilterType, ExecutionMechanic, ExecutionMechanicType,
    RiskBlock, StrategyDefinition,
)


def test_path_points_orientation():
    assert _path_points(1.0, 1.2, 0.9, 1.1) == (1.0, 0.9, 1.2, 1.1)  # bullish
    assert _path_points(1.1, 1.2, 0.9, 1.0) == (1.1, 1.2, 0.9, 1.0)  # bearish


def test_first_touch_bearish_bar_hits_tp_before_sl_for_long():
    # long from 1.1000, sl 1.0950, tp 1.1050; bearish bar touches both.
    pos = Position(direction=1, lots=0.1, entry_price=1.1000, open_time=0.0,
                   sl=1.0950, tp=1.1050)
    path = _path_points(1.1000, 1.1060, 1.0940, 1.0960)   # o -> h -> l -> c
    assert _first_touch_exit(pos, path) == 1.1050          # TP first

    bullish = _path_points(1.1000, 1.1060, 1.0940, 1.1040)  # o -> l -> h -> c
    assert _first_touch_exit(pos, bullish) == 1.0950       # SL first


def test_first_touch_single_level():
    pos = Position(direction=-1, lots=0.1, entry_price=1.1000, open_time=0.0,
                   sl=1.1050, tp=0.0)
    path = _path_points(1.1000, 1.1060, 1.0990, 1.1020)
    assert _first_touch_exit(pos, path) == 1.1050
    calm = _path_points(1.1000, 1.1010, 1.0990, 1.1000)
    assert _first_touch_exit(pos, calm) is None


def _sltp_strategy(sl_points=50.0, tp_points=50.0) -> StrategyDefinition:
    return StrategyDefinition(
        symbol="TEST", timeframe="H1",
        entry_filters=[EntryFilter(
            type=EntryFilterType.RSI_REVERSION,
            params={"rsi_period": 5, "oversold": 30, "overbought": 70})],
        mechanic=ExecutionMechanic(
            type=ExecutionMechanicType.STANDARD_SLTP,
            params={"sl_points": sl_points, "tp_points": tp_points}),
        risk=RiskBlock(fixed_lots=0.10, max_open_lots=5.0,
                       max_spread_points=0.0),
    )


def _ambiguous_bar_df():
    """Steady decline (RSI pinned oversold -> long entry), then one wide
    bearish bar that touches both the stop and the target."""
    n = 24
    idx = pd.date_range("2024-01-02 00:00", periods=n, freq="1h", tz="utc")
    close = 1.1200 - np.arange(n) * 0.0010
    open_ = np.roll(close, 1)
    open_[0] = close[0] + 0.0010
    high = np.maximum(open_, close) + 0.0001
    low = np.minimum(open_, close) - 0.0001

    # entry fires at the first post-warmup bar (i=15, warmup = period*3);
    # bar 16 is the ambiguous one. Entry fill = close[15] + slippage.
    entry = close[15] + 2 * 0.0001          # slippage_points=2 on the spec
    open_[16] = close[15]
    high[16] = entry + 0.0060               # touches TP (+50 pts = 0.0050)
    low[16] = entry - 0.0060                # touches SL too
    close[16] = open_[16] - 0.0005          # bearish -> path o->h->l->c
    # keep the rest of the series calm and out of oversold so nothing else fires
    for j in range(17, n):
        close[j] = close[16]
        open_[j] = close[16]
        high[j] = close[16] + 0.0001
        low[j] = close[16] - 0.0001

    return pd.DataFrame({"time": idx, "open": open_, "high": high,
                         "low": low, "close": close, "volume": 100.0})


_SPEC = SymbolSpec(point=0.0001, contract_size=100_000, leverage=100,
                   spread_points=0.0, slippage_points=2.0)


def test_path_mode_vs_conservative_disagree_on_ambiguous_bar():
    df = _ambiguous_bar_df()
    strat = _sltp_strategy()

    _, book_cons = run_simulation(df, strat, _SPEC, 10_000.0,
                                  intrabar_mode="conservative")
    _, book_path = run_simulation(df, strat, _SPEC, 10_000.0,
                                  intrabar_mode="path")
    assert book_cons.closed[0].profit < 0    # legacy: SL assumed first
    assert book_path.closed[0].profit > 0    # bearish path reaches TP first


def test_m1_replay_overrides_path_heuristic():
    df = _ambiguous_bar_df()
    strat = _sltp_strategy()
    entry = df["close"].iloc[15] + 2 * 0.0001

    # M1 sub-bars inside strategy bar 16: price dives to the SL *first*,
    # then rallies through the TP — the opposite of the bearish-bar path
    # heuristic (which assumes high before low).
    bar_t = df["time"].iloc[16]
    m1_idx = pd.date_range(bar_t, periods=60, freq="1min", tz="utc")
    m1_close = np.full(60, df["open"].iloc[16])
    m1_close[5] = entry - 0.0058             # near the low, past the SL
    m1_close[30] = entry + 0.0058            # later: past the TP
    m1_open = np.roll(m1_close, 1)
    m1_open[0] = df["open"].iloc[16]
    m1 = pd.DataFrame({
        "time": m1_idx, "open": m1_open,
        "high": np.maximum(m1_open, m1_close),
        "low": np.minimum(m1_open, m1_close),
        "close": m1_close, "volume": 10.0,
    })

    _, book = run_simulation(df, strat, _SPEC, 10_000.0,
                             intrabar_mode="m1", intrabar_df=m1)
    assert book.closed[0].profit < 0         # M1 path: SL genuinely hit first
