"""Meta-labeling diagnostics: can a secondary model predict which trades win?

Meta-labeling (Lopez de Prado) keeps the strategy's entry logic as-is and
trains a classifier on *when its trades succeed* — regime, session, and
direction context. This module is the honest first stage: a pure-numpy
logistic regression with a strict chronological train/test split, reporting
out-of-sample AUC and the expectancy uplift a probability-threshold filter
would have delivered. A strategy whose winners are predictable from context
is a candidate for a premium "AI-filtered" variant; one whose AUC sits at
0.5 should not grow a filter (it would only be curve-fit).

Diagnostics only — nothing here changes validation gates or exports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

MIN_TRADES = 40                # below this, any fit is noise
TRAIN_FRACTION = 0.7
DEFAULT_THRESHOLD = 0.5


@dataclass
class MetaLabelReport:
    n_trades: int = 0
    n_train: int = 0
    n_test: int = 0
    test_auc: float = 0.0
    baseline_expectancy: float = 0.0   # mean profit/trade, test window
    filtered_expectancy: float = 0.0   # mean profit of model-approved trades
    coverage: float = 0.0              # fraction of test trades approved
    uplift: float = 0.0                # filtered - baseline expectancy
    threshold: float = DEFAULT_THRESHOLD
    usable: bool = False
    reason: str = ""
    weights: List[float] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)


_FEATURES = ["bias", "regime_quiet_trend", "regime_vol_range",
             "regime_vol_trend", "hour_sin", "hour_cos", "direction"]


def _trade_features(df: pd.DataFrame, open_times: np.ndarray,
                    directions: np.ndarray) -> np.ndarray:
    """Feature matrix per trade: regime one-hots, session encoding, side."""
    from factory.regime import classify_regimes

    t = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    bar_times = t.to_numpy().astype("datetime64[s]").astype("int64").astype(float)
    regimes = classify_regimes(df)
    idx = np.clip(np.searchsorted(bar_times, open_times, side="right") - 1,
                  0, len(regimes) - 1)
    code = regimes[idx].astype(int)
    hours = ((open_times % 86400) // 3600).astype(float)
    ang = 2.0 * np.pi * hours / 24.0
    n = len(open_times)
    X = np.column_stack([
        np.ones(n),                       # bias
        (code == 1).astype(float),        # quiet trend (quiet range = base)
        (code == 2).astype(float),        # volatile range
        (code == 3).astype(float),        # volatile trend
        np.sin(ang), np.cos(ang),
        directions.astype(float),
    ])
    return X


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 0.1,
                  lr: float = 0.5, epochs: int = 400) -> np.ndarray:
    """Gradient-descent logistic regression (bias column unregularized)."""
    w = np.zeros(X.shape[1])
    n = len(y)
    reg = np.full(X.shape[1], l2)
    reg[0] = 0.0
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        grad = X.T @ (p - y) / n + reg * w
        w -= lr * grad
    return w


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney); 0.5 when one class is absent."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def metalabel_report(df: pd.DataFrame, book,
                     threshold: float = DEFAULT_THRESHOLD) -> MetaLabelReport:
    """Fit and evaluate a meta-labeler on a finished simulator book.

    Chronological split: the model is fit on the first 70% of trades and
    every reported number comes from the untouched last 30%.
    """
    trades = sorted(book.closed, key=lambda tr: tr.close_time)
    rep = MetaLabelReport(n_trades=len(trades), threshold=threshold,
                          feature_names=list(_FEATURES))
    if len(trades) < MIN_TRADES:
        rep.reason = (f"only {len(trades)} trades; need >= {MIN_TRADES} "
                      "for a meaningful fit")
        return rep

    open_times = np.asarray([tr.open_time for tr in trades], dtype=float)
    directions = np.asarray([tr.direction for tr in trades], dtype=float)
    profits = np.asarray([tr.profit for tr in trades], dtype=float)
    y = (profits > 0).astype(float)

    X = _trade_features(df, open_times, directions)
    split = int(len(trades) * TRAIN_FRACTION)
    rep.n_train, rep.n_test = split, len(trades) - split
    if y[:split].min() == y[:split].max():
        rep.reason = "training window has only one outcome class"
        return rep

    w = _fit_logistic(X[:split], y[:split])
    rep.weights = [round(float(x), 4) for x in w]

    scores = 1.0 / (1.0 + np.exp(-np.clip(X[split:] @ w, -30, 30)))
    y_test = y[split:]
    p_test = profits[split:]
    rep.test_auc = round(_auc(scores, y_test), 4)
    rep.baseline_expectancy = round(float(p_test.mean()), 2)
    approved = scores >= threshold
    rep.coverage = round(float(approved.mean()), 4)
    if approved.any():
        rep.filtered_expectancy = round(float(p_test[approved].mean()), 2)
    rep.uplift = round(rep.filtered_expectancy - rep.baseline_expectancy, 2)
    rep.usable = rep.test_auc >= 0.55 and rep.coverage >= 0.2
    if not rep.usable:
        rep.reason = ("context does not predict trade outcomes"
                      if rep.test_auc < 0.55
                      else "filter approves too few trades to trust")
    return rep


def metalabel_for_strategy(storage, strategy_id: str, *,
                           deposit: float = 10_000.0,
                           now: Optional[object] = None) -> MetaLabelReport:
    """Convenience wrapper: re-run the strategy's OOS zone and analyze it."""
    from datetime import datetime, timezone

    from factory.backtest.simulator import SimulatorEngine

    strategy = storage.get_strategy(strategy_id)
    report = storage.get_validation(strategy_id)
    if strategy is None or report is None:
        out = MetaLabelReport()
        out.reason = "strategy or validation record missing"
        return out
    start = datetime.fromtimestamp(report.oos_range[0], tz=timezone.utc)
    end = datetime.fromtimestamp(report.oos_range[1], tz=timezone.utc)
    engine = SimulatorEngine()
    strategy = strategy.apply_flat_params(report.best_params)
    _, book, df = engine.run_with_trades(strategy, start, end,
                                         deposit=deposit)
    return metalabel_report(df, book)
