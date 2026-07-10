"""Selection-bias statistics: Deflated Sharpe Ratio and OOS-loss probability.

A discovery run evaluates hundreds or thousands of candidates and keeps the
best-looking ones. Under that much selection pressure, *some* strategy will
always look great by chance. The Deflated Sharpe Ratio (Bailey & Lopez de
Prado, 2014) answers the honest question: "given how many things we tried,
what is the probability this Sharpe is better than a zero-skill benchmark?"

- ``expected_max_sharpe(n_trials, sr_var)``: the Sharpe the *best of N*
  zero-skill strategies is expected to show (via extreme-value theory).
- ``deflated_sharpe_ratio(...)``: the Probabilistic Sharpe Ratio evaluated
  against that expected-max benchmark, adjusted for the return series'
  skewness and kurtosis. DSR near 1.0 = the edge very likely survives the
  multiple-testing haircut; DSR near 0.5 or below = plausibly pure selection.

Pure stdlib/numpy (normal CDF via ``math.erf``, PPF via Acklam's rational
approximation) — no scipy dependency.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

from factory.models import BacktestMetrics, WFOWindowResult

_EULER_GAMMA = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's approximation, |rel err| < 1.15e-9)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - p_low:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """Expected maximum Sharpe among ``n_trials`` zero-skill strategies.

    ``sr_var`` is the cross-trial variance of Sharpe estimates; with only one
    trial (or no dispersion) the benchmark is zero.
    """
    if n_trials <= 1 or sr_var <= 0.0:
        return 0.0
    z1 = _norm_ppf(1.0 - 1.0 / n_trials)
    z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_var) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


def deflated_sharpe_ratio(sr: float, n_obs: int, skew: float, kurt: float,
                          n_trials: int,
                          sr_var: Optional[float] = None) -> float:
    """DSR = probability the observed (per-period) Sharpe beats the expected
    best-of-N zero-skill Sharpe, given ``n_obs`` return observations.

    ``sr_var`` defaults to the estimator variance of the Sharpe itself —
    a conservative stand-in when the cross-trial Sharpe dispersion was not
    recorded. ``kurt`` is *raw* kurtosis (normal = 3).
    """
    if n_obs < 3:
        return 0.0
    adj = max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr)
    if sr_var is None:
        sr_var = adj / max(n_obs - 1, 1)
    sr0 = expected_max_sharpe(n_trials, sr_var)
    z = (sr - sr0) * math.sqrt(n_obs - 1.0) / math.sqrt(adj)
    return round(_norm_cdf(z), 4)


def _per_period_returns(metrics: BacktestMetrics) -> np.ndarray:
    eq = np.asarray(metrics.equity, dtype=float)
    if len(eq) < 4:
        return np.empty(0)
    return np.diff(eq) / np.maximum(eq[:-1], 1e-9)


def dsr_from_metrics(metrics: BacktestMetrics, n_trials: int) -> float:
    """Deflated Sharpe Ratio from a backtest's stored (thinned) equity curve."""
    rets = _per_period_returns(metrics)
    if len(rets) < 3 or float(rets.std()) <= 0.0:
        return 0.0
    sr = float(rets.mean() / rets.std())
    mu, sd = rets.mean(), rets.std()
    z = (rets - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    return deflated_sharpe_ratio(sr, len(rets), skew, kurt,
                                 max(1, int(n_trials)))


def p_oos_loss(windows: Sequence[WFOWindowResult]) -> float:
    """Fraction of walk-forward OOS windows that ended unprofitable.

    A blunt but honest dispersion statistic: 0.0 = every OOS slice made
    money, 0.5 = a coin flip. Reported, not gated, by default.
    """
    if not windows:
        return 0.0
    losses = sum(1 for w in windows if w.oos_metrics.net_profit <= 0.0)
    return round(losses / len(windows), 4)
