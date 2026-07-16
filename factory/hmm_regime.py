"""Causal 2-state Gaussian HMM regime filter (simulator + MQL5 parity).

Online forward filtering of log returns — no lookahead, no sklearn. Emission
and transition parameters are strategy inputs so the generated EA can run the
exact same recurrence bar-by-bar (see ``TM_HmmUpdate`` in ``base.mq5``).

States
------
code 0 / 1 — latent regimes (typically low-vol vs high-vol depending on the
optimised ``sigma`` / ``mu``). Hard classification is ``argmax`` of the
filtered posterior; soft gating uses ``hmm_min_prob``.
"""
from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd

HMM_STATE_NAMES = {0: "hmm state 0", 1: "hmm state 1"}
HMM_WARMUP_BARS = 2          # need at least one return before filtering
_EPS = 1e-12
_SIGMA_FLOOR = 1e-8
_P_LO, _P_HI = 0.01, 0.99


def sanitize_hmm_params(
    mu0: float, mu1: float,
    sigma0: float, sigma1: float,
    p00: float, p11: float,
    pi0: float = 0.5,
) -> Tuple[float, float, float, float, float, float, float]:
    """Clamp / floor parameters so both Python and MQL5 stay well-defined."""
    s0 = max(float(sigma0), _SIGMA_FLOOR)
    s1 = max(float(sigma1), _SIGMA_FLOOR)
    a00 = float(np.clip(p00, _P_LO, _P_HI))
    a11 = float(np.clip(p11, _P_LO, _P_HI))
    prior = float(np.clip(pi0, _P_LO, _P_HI))
    return float(mu0), float(mu1), s0, s1, a00, a11, prior


def log_returns(close: np.ndarray) -> np.ndarray:
    """Per-bar log returns; index 0 is 0 (no prior close)."""
    c = np.asarray(close, dtype=float)
    out = np.zeros(len(c), dtype=float)
    if len(c) < 2:
        return out
    prev = np.maximum(c[:-1], _EPS)
    out[1:] = np.log(np.maximum(c[1:], _EPS) / prev)
    return out


def _gauss_pdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return float(np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi) + _EPS))


def forward_step(
    alpha0: float, alpha1: float, ret: float,
    mu0: float, mu1: float, sigma0: float, sigma1: float,
    p00: float, p11: float,
) -> Tuple[float, float]:
    """One causal forward update. Returns renormalised posteriors (α₀, α₁).

    Mirrors ``TM_HmmUpdate`` in the MQL5 template exactly (probability space,
    not log-space) so EA and simulator stay bit-compatible within float noise.
    """
    mu0, mu1, sigma0, sigma1, p00, p11, _ = sanitize_hmm_params(
        mu0, mu1, sigma0, sigma1, p00, p11)
    p01 = 1.0 - p00
    p10 = 1.0 - p11
    e0 = max(_gauss_pdf(ret, mu0, sigma0), _EPS)
    e1 = max(_gauss_pdf(ret, mu1, sigma1), _EPS)
    pred0 = alpha0 * p00 + alpha1 * p10
    pred1 = alpha0 * p01 + alpha1 * p11
    un0 = pred0 * e0
    un1 = pred1 * e1
    total = un0 + un1
    if total <= _EPS:
        return 0.5, 0.5
    return un0 / total, un1 / total


def forward_filter(
    returns: np.ndarray,
    mu0: float = 0.0, mu1: float = 0.0,
    sigma0: float = 0.0005, sigma1: float = 0.002,
    p00: float = 0.95, p11: float = 0.90,
    pi0: float = 0.5,
) -> np.ndarray:
    """Filtered posteriors shape ``(N, 2)`` for each bar.

    Bar 0 has no return — priors are stored. From bar 1 onward each step
    consumes ``returns[t]`` (log return from t-1 → t).
    """
    mu0, mu1, sigma0, sigma1, p00, p11, pi0 = sanitize_hmm_params(
        mu0, mu1, sigma0, sigma1, p00, p11, pi0)
    r = np.asarray(returns, dtype=float)
    n = len(r)
    post = np.empty((n, 2), dtype=float)
    a0, a1 = pi0, 1.0 - pi0
    post[0, 0] = a0
    post[0, 1] = a1
    for t in range(1, n):
        a0, a1 = forward_step(a0, a1, float(r[t]),
                              mu0, mu1, sigma0, sigma1, p00, p11)
        post[t, 0] = a0
        post[t, 1] = a1
    return post


def params_from_mapping(params: Optional[Mapping[str, float]] = None
                        ) -> dict:
    """Pull HMM knobs from a TradeManagement.params-style mapping."""
    p = params or {}
    return {
        "mu0": float(p.get("hmm_mu0", 0.0)),
        "mu1": float(p.get("hmm_mu1", 0.0)),
        "sigma0": float(p.get("hmm_sigma0", 0.0005)),
        "sigma1": float(p.get("hmm_sigma1", 0.002)),
        "p00": float(p.get("hmm_p00", 0.95)),
        "p11": float(p.get("hmm_p11", 0.90)),
        "pi0": float(p.get("hmm_pi0", 0.5)),
        "min_prob": float(p.get("hmm_min_prob", 0.55)),
        "allow_mask": int(p.get("hmm_allow_mask", 3)),
        "size0": float(p.get("hmm_size_state0", 1.0)),
        "size1": float(p.get("hmm_size_state1", 1.0)),
    }


def classify_hmm_filter(
    df: pd.DataFrame,
    params: Optional[Mapping[str, float]] = None,
    *,
    mu0: Optional[float] = None, mu1: Optional[float] = None,
    sigma0: Optional[float] = None, sigma1: Optional[float] = None,
    p00: Optional[float] = None, p11: Optional[float] = None,
    pi0: Optional[float] = None,
    min_prob: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(codes, posteriors)`` for ``df`` closes.

    ``codes`` is int8 in {0, 1}. When the winning state's posterior is below
    ``min_prob``, the code is still the argmax (sizing uses it); the soft gate
    is applied separately via :func:`allowed_by_hmm`.
    """
    cfg = params_from_mapping(params)
    if mu0 is not None:
        cfg["mu0"] = float(mu0)
    if mu1 is not None:
        cfg["mu1"] = float(mu1)
    if sigma0 is not None:
        cfg["sigma0"] = float(sigma0)
    if sigma1 is not None:
        cfg["sigma1"] = float(sigma1)
    if p00 is not None:
        cfg["p00"] = float(p00)
    if p11 is not None:
        cfg["p11"] = float(p11)
    if pi0 is not None:
        cfg["pi0"] = float(pi0)
    if min_prob is not None:
        cfg["min_prob"] = float(min_prob)

    close = df["close"].to_numpy(dtype=float)
    rets = log_returns(close)
    post = forward_filter(
        rets, cfg["mu0"], cfg["mu1"], cfg["sigma0"], cfg["sigma1"],
        cfg["p00"], cfg["p11"], cfg["pi0"])
    codes = np.argmax(post, axis=1).astype(np.int8)
    return codes, post


def allowed_by_hmm(
    codes: np.ndarray, posteriors: np.ndarray,
    mask: int, min_prob: float,
) -> np.ndarray:
    """Boolean gate: mask bit set AND winning-state posterior ≥ ``min_prob``."""
    mask = int(mask)
    min_prob = float(min_prob)
    lut = np.array([(mask >> c) & 1 for c in range(2)], dtype=bool)
    in_mask = lut[codes]
    conf = posteriors[np.arange(len(codes)), codes] >= min_prob
    return in_mask & conf


def lot_mult_from_codes(
    codes: np.ndarray, size0: float, size1: float,
) -> np.ndarray:
    """Per-bar lot multipliers for HMM sizing."""
    mults = np.array([float(size0), float(size1)], dtype=float)
    return mults[codes]
