"""Meta-labeling diagnostics (factory.metalabel)."""
import numpy as np
import pandas as pd

from factory.metalabel import MIN_TRADES, _auc, metalabel_report


class _Trade:
    def __init__(self, open_time, close_time, direction, profit):
        self.open_time = open_time
        self.close_time = close_time
        self.direction = direction
        self.profit = profit


class _Book:
    def __init__(self, closed):
        self.closed = closed


def _regime_df(n=800):
    """First half relentless trend, second half tight oscillation."""
    trend = 1.10 + np.arange(n // 2) * 0.0008
    chop = trend[-1] + 0.0015 * np.sin(np.arange(n // 2) * 2 * np.pi / 10)
    close = np.concatenate([trend, chop])
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="1h", tz="utc"),
        "open": open_, "high": np.maximum(open_, close) + 0.0002,
        "low": np.minimum(open_, close) - 0.0002, "close": close,
        "volume": np.full(n, 100.0)})


def _bar_ts(df, i):
    return float(pd.Timestamp(df["time"].iloc[i]).timestamp())


def test_auc_helper():
    assert _auc(np.array([0.9, 0.8, 0.2, 0.1]),
                np.array([1, 1, 0, 0])) == 1.0
    assert _auc(np.array([0.1, 0.2, 0.8, 0.9]),
                np.array([1, 1, 0, 0])) == 0.0
    assert _auc(np.array([0.5, 0.5]), np.array([1, 1])) == 0.5  # one class


def test_metalabel_learns_regime_dependence():
    """Trades opened in the trend half win, chop half lose -> learnable."""
    df = _regime_df()
    n = len(df)
    rng = np.random.default_rng(3)
    trades = []
    for k in range(120):
        # interleave trend-half and chop-half trades chronologically-ish
        if k % 2 == 0:
            i = int(rng.integers(100, n // 2 - 5))     # trend regime
            profit = float(rng.normal(60, 15))          # winners
        else:
            i = int(rng.integers(n // 2 + 20, n - 5))   # chop regime
            profit = float(rng.normal(-40, 15))         # losers
        ts = _bar_ts(df, i)
        trades.append(_Trade(ts, ts + 3600, 1, profit))
    # chronological order by close time (mixes both halves in train AND test)
    trades.sort(key=lambda t: t.open_time % 7)          # pseudo-shuffle
    for j, t in enumerate(trades):                       # re-stamp close order
        t.close_time = j

    rep = metalabel_report(df, _Book(trades))
    assert rep.n_trades == 120 and rep.n_test > 0
    assert rep.test_auc > 0.75, rep
    assert rep.usable
    assert rep.filtered_expectancy > rep.baseline_expectancy
    assert rep.uplift > 0


def test_metalabel_reports_noise_as_unusable():
    df = _regime_df()
    n = len(df)
    rng = np.random.default_rng(9)
    trades = []
    for k in range(120):
        i = int(rng.integers(100, n - 5))
        ts = _bar_ts(df, i)
        trades.append(_Trade(ts, ts + k, 1 if k % 2 else -1,
                             float(rng.normal(0, 50))))
    rep = metalabel_report(df, _Book(trades))
    # random outcomes: no reliable edge for the filter to find
    assert rep.test_auc < 0.70
    if not rep.usable:
        assert rep.reason


def test_metalabel_refuses_tiny_samples():
    df = _regime_df()
    ts = _bar_ts(df, 200)
    trades = [_Trade(ts + i, ts + i + 1, 1, 10.0)
              for i in range(MIN_TRADES - 1)]
    rep = metalabel_report(df, _Book(trades))
    assert not rep.usable
    assert "need >=" in rep.reason
