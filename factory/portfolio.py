"""Portfolio construction over validated strategies (HRP).

Turns the gallery's daily-return fingerprints (factory/correlation.py) into a
risk-managed *portfolio*: a correlation matrix, Hierarchical Risk Parity
weights (Lopez de Prado 2016 — robust, needs no expected-return estimates and
no matrix inversion), and combined-portfolio metrics so "run these five
together" can be evaluated as one product instead of five hopeful charts.

Pure Python + numpy: single-linkage clustering and recursive bisection are
implemented directly (the strategy count is small, so O(N^3) is irrelevant).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from factory.correlation import daily_returns
from factory.models import ValidationReport

TRADING_DAYS_PER_YEAR = 252.0


# ---------------------------------------------------------------------------
# Return matrix
# ---------------------------------------------------------------------------

def build_return_matrix(reports: Sequence[ValidationReport]
                        ) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Align OOS daily returns on the union of trading days.

    Returns ``(strategy_ids, day_indices, matrix)`` where ``matrix`` is
    (days x strategies); days a strategy was flat/absent contribute 0.
    Strategies without any usable fingerprint are dropped.
    """
    series: List[Tuple[str, Dict[int, float]]] = []
    for rep in reports:
        dr = daily_returns(rep.oos_metrics)
        if dr:
            series.append((rep.strategy_id, dr))
    if not series:
        return [], np.empty(0, dtype=np.int64), np.empty((0, 0))
    all_days = sorted(set().union(*(set(dr) for _, dr in series)))
    matrix = np.zeros((len(all_days), len(series)))
    for j, (_sid, dr) in enumerate(series):
        for i, d in enumerate(all_days):
            matrix[i, j] = dr.get(d, 0.0)
    ids = [sid for sid, _ in series]
    return ids, np.asarray(all_days, dtype=np.int64), matrix


def correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Pearson correlation between strategy return columns (safe on zeros)."""
    n = matrix.shape[1]
    if n == 0:
        return np.empty((0, 0))
    stds = matrix.std(axis=0)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if stds[i] > 1e-15 and stds[j] > 1e-15:
                c = float(np.corrcoef(matrix[:, i], matrix[:, j])[0, 1])
            else:
                c = 0.0
            corr[i, j] = corr[j, i] = c
    return corr


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity
# ---------------------------------------------------------------------------

def _seriation_order(dist: np.ndarray) -> List[int]:
    """Leaf order from single-linkage agglomerative clustering.

    Groups similar strategies adjacently so recursive bisection splits
    between clusters rather than through them.
    """
    n = dist.shape[0]
    clusters: List[List[int]] = [[i] for i in range(n)]
    while len(clusters) > 1:
        best = (np.inf, 0, 1)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = min(dist[i, j] for i in clusters[a] for j in clusters[b])
                if d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return clusters[0]


def _best_split(cluster: Sequence[int], dist: np.ndarray) -> int:
    """Split index maximizing the single-linkage distance between halves.

    Vanilla HRP bisects at the midpoint, which can cut straight through a
    correlated pair when the cluster count is odd; splitting at the widest
    gap keeps near-duplicates on the same side of the bisection.
    """
    best_k, best_d = len(cluster) // 2, -1.0
    for k in range(1, len(cluster)):
        d = min(dist[i, j] for i in cluster[:k] for j in cluster[k:])
        if d > best_d:
            best_d, best_k = d, k
    return best_k


def _cluster_variance(cov: np.ndarray, idx: Sequence[int]) -> float:
    """Variance of a cluster under inverse-variance intra-cluster weights."""
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.maximum(np.diag(sub), 1e-18)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(matrix: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity weights over the return matrix columns."""
    n = matrix.shape[1]
    if n == 0:
        return np.empty(0)
    if n == 1:
        return np.array([1.0])
    cov = np.cov(matrix, rowvar=False)
    cov = np.atleast_2d(cov)
    corr = correlation_matrix(matrix)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    order = _seriation_order(dist)

    weights = np.ones(n)
    stack: List[List[int]] = [order]
    while stack:
        cluster = stack.pop()
        if len(cluster) <= 1:
            continue
        mid = _best_split(cluster, dist)
        left, right = cluster[:mid], cluster[mid:]
        var_l = _cluster_variance(cov, left)
        var_r = _cluster_variance(cov, right)
        total = var_l + var_r
        alpha = 0.5 if total <= 0 else 1.0 - var_l / total
        weights[left] *= alpha
        weights[right] *= (1.0 - alpha)
        stack.extend([left, right])
    weights = np.maximum(weights, 0.0)
    s = weights.sum()
    return weights / s if s > 0 else np.full(n, 1.0 / n)


# ---------------------------------------------------------------------------
# Portfolio evaluation
# ---------------------------------------------------------------------------

@dataclass
class PortfolioReport:
    strategy_ids: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    corr: List[List[float]] = field(default_factory=list)
    avg_pairwise_corr: float = 0.0
    max_pairwise_corr: float = 0.0
    # combined daily-return stream statistics
    ann_return: float = 0.0            # annualized mean daily return
    ann_sharpe: float = 0.0
    max_dd_pct: float = 0.0
    days: int = 0
    equity: List[float] = field(default_factory=list)   # growth of 1.0
    # diversification: portfolio vol / weighted avg standalone vol (<1 = good)
    diversification_ratio: float = 0.0


def portfolio_metrics(matrix: np.ndarray, weights: np.ndarray
                      ) -> Tuple[float, float, float, List[float], float]:
    """(ann_return, ann_sharpe, max_dd_pct, equity_curve, diversification)."""
    port = matrix @ weights
    equity = np.cumprod(1.0 + port)
    peak = np.maximum.accumulate(equity)
    dd = float(((peak - equity) / np.maximum(peak, 1e-12)).max() * 100.0)
    mu, sd = float(port.mean()), float(port.std())
    ann_ret = mu * TRADING_DAYS_PER_YEAR
    sharpe = mu / sd * np.sqrt(TRADING_DAYS_PER_YEAR) if sd > 0 else 0.0
    stand_vols = matrix.std(axis=0)
    weighted_vol = float(weights @ stand_vols)
    div_ratio = sd / weighted_vol if weighted_vol > 0 else 0.0
    return ann_ret, float(sharpe), dd, [float(x) for x in equity], div_ratio


def build_portfolio(reports: Sequence[ValidationReport]) -> PortfolioReport:
    """Full HRP portfolio over the given validated strategies."""
    ids, _days, matrix = build_return_matrix(reports)
    if not ids:
        return PortfolioReport()
    w = hrp_weights(matrix)
    corr = correlation_matrix(matrix)
    n = len(ids)
    off_diag = [abs(corr[i, j]) for i in range(n) for j in range(i + 1, n)]
    ann_ret, sharpe, dd, equity, div = portfolio_metrics(matrix, w)
    return PortfolioReport(
        strategy_ids=ids,
        weights={sid: round(float(x), 4) for sid, x in zip(ids, w)},
        corr=[[round(float(x), 4) for x in row] for row in corr],
        avg_pairwise_corr=round(float(np.mean(off_diag)), 4) if off_diag else 0.0,
        max_pairwise_corr=round(float(np.max(off_diag)), 4) if off_diag else 0.0,
        ann_return=round(ann_ret, 4),
        ann_sharpe=round(sharpe, 3),
        max_dd_pct=round(dd, 3),
        days=matrix.shape[0],
        equity=equity,
        diversification_ratio=round(div, 4),
    )
