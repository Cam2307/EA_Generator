"""Canonical metric display helpers shared by gates, cards, charts, and exports.

Drawdown semantics
------------------
Acceptance gates and primary UI labels use :attr:`BacktestMetrics.max_dd_pct`
from the simulator — intrabar worst-case floating drawdown (conservative).

Equity-curve peak-to-trough on thinned close-equity samples
(:func:`equity_curve_drawdown_pct`) is for diagnostics only and is often
lower than ``max_dd_pct`` because it does not see intrabar extremes.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from factory.models import BacktestMetrics, ValidationReport, WFOWindowResult


def sortino_ratio(metrics: BacktestMetrics) -> float:
    """Downside-deviation risk-adjusted return for sorting and display.

    Derived from the thinned equity curve when available; falls back to the
    stored ``sortino`` field for rows without curve data.
    """
    if len(metrics.equity) >= 2 and len(metrics.equity_ts) >= 2:
        eq = np.asarray(metrics.equity, dtype=float)
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
        if len(rets) >= 2:
            downside = rets[rets < 0]
            downside_std = float(downside.std()) if len(downside) > 1 else 0.0
            if downside_std > 0:
                bar_seconds = float(np.median(np.diff(metrics.equity_ts)))
                bars_per_year = 365.25 * 86400 / max(bar_seconds, 1.0)
                return float(rets.mean() / downside_std * np.sqrt(bars_per_year))
    return metrics.sortino


def gate_drawdown_pct(metrics: BacktestMetrics) -> float:
    """Authoritative drawdown % — matches acceptance-criteria gates."""
    return metrics.max_dd_pct


def zone_drawdown_label(zone: str = "OOS") -> str:
    """Metric label for strategy cards and charts."""
    return f"{zone} Max DD %"


def report_zone_drawdown(report: ValidationReport, zone: str = "OOS") -> float:
    """Drawdown for a validation zone (``IS`` or ``OOS``)."""
    metrics = report.is_metrics if zone.upper() == "IS" else report.oos_metrics
    return gate_drawdown_pct(metrics)


def equity_curve_drawdown_pct(equity: List[float]) -> float:
    """Peak-to-trough on a thinned close-equity curve (diagnostic only)."""
    from factory.backtest.simulator import max_drawdown_pct
    return max_drawdown_pct(equity)


def data_source_label(source: str) -> str:
    """Human-readable badge text for OHLC provenance."""
    labels = {
        "mt5": "MT5 live",
        "cache": "Parquet cache",
        "synthetic": "Synthetic (dev)",
    }
    return labels.get(source, source or "unknown")


def data_source_badge(source: str) -> str:
    """Streamlit markdown badge for OHLC provenance."""
    colors = {"mt5": "green", "cache": "blue", "synthetic": "orange"}
    color = colors.get(source, "gray")
    return f":{color}[{data_source_label(source)}]"


def dsr_label(dsr: float, n_trials: int) -> str:
    """Human-readable Deflated Sharpe Ratio badge for strategy cards.

    DSR is the probability the OOS Sharpe beats the expected best-of-N
    zero-skill Sharpe (N = candidates tried by the run).
    """
    if dsr >= 0.95:
        verdict = "very likely real"
    elif dsr >= 0.80:
        verdict = "probably real"
    elif dsr >= 0.50:
        verdict = "uncertain"
    else:
        verdict = "likely selection luck"
    return f"DSR {dsr:.2f} over {n_trials} trials — {verdict}"


def dsr_badge(dsr: float) -> str:
    """Streamlit markdown badge color-coding the DSR verdict."""
    if dsr >= 0.95:
        color = "green"
    elif dsr >= 0.80:
        color = "blue"
    elif dsr >= 0.50:
        color = "orange"
    else:
        color = "red"
    return f":{color}[DSR {dsr:.2f}]"


def wfo_summary(windows: List[WFOWindowResult],
                mode: Optional[str] = None) -> str:
    """One-line summary of walk-forward window results."""
    subset = [w for w in windows if mode is None or w.mode == mode]
    if not subset:
        return "no windows"
    wfes = [w.wfe for w in subset]
    avg = sum(wfes) / len(wfes)
    label = mode or "all"
    return f"{len(subset)} {label} window(s), avg WFE {avg:.2f}"
