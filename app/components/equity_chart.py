"""Plotly equity curves with shaded IS/OOS regions and WFO window markers."""
from __future__ import annotations

from datetime import datetime, timezone

import plotly.graph_objects as go

from factory.metrics_display import report_zone_drawdown, zone_drawdown_label
from factory.models import ValidationReport


def _ts_to_dt(ts: float) -> datetime:
    # Guard against corrupt/degenerate timestamps so the x-axis never
    # collapses to the 1970 epoch.
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return datetime.fromtimestamp(0, tz=timezone.utc)


def build_equity_figure(report: ValidationReport) -> go.Figure:
    """IS and OOS equity as independent backtests (each starts at deposit).

    Drawdown stats shown in the title use simulator ``max_dd_pct`` per zone,
    matching acceptance gates — not a combined carry-forward curve.
    """
    fig = go.Figure()

    is_m, oos_m = report.is_metrics, report.oos_metrics
    deposit = is_m.initial_deposit or oos_m.initial_deposit or 10_000.0

    if is_m.equity:
        fig.add_trace(go.Scatter(
            x=[_ts_to_dt(t) for t in is_m.equity_ts], y=is_m.equity,
            mode="lines", name="In-sample equity",
            line=dict(color="#4f8bf9", width=1.6),
        ))
    if oos_m.equity:
        fig.add_trace(go.Scatter(
            x=[_ts_to_dt(t) for t in oos_m.equity_ts], y=oos_m.equity,
            mode="lines", name="Out-of-sample equity",
            line=dict(color="#f9a84f", width=1.6),
        ))

    # shaded IS / OOS regions
    if report.is_range[1] > report.is_range[0]:
        fig.add_vrect(x0=_ts_to_dt(report.is_range[0]),
                      x1=_ts_to_dt(report.is_range[1]),
                      fillcolor="#4f8bf9", opacity=0.07, line_width=0,
                      annotation_text="IS", annotation_position="top left")
    if report.oos_range[1] > report.oos_range[0]:
        fig.add_vrect(x0=_ts_to_dt(report.oos_range[0]),
                      x1=_ts_to_dt(report.oos_range[1]),
                      fillcolor="#f9a84f", opacity=0.09, line_width=0,
                      annotation_text="OOS", annotation_position="top left")

    # visual discontinuity at the IS/OOS split — independent runs
    if report.is_range[1] > 0:
        split_x = _ts_to_dt(report.is_range[1])
        fig.add_vline(x=split_x, line_width=2, line_dash="dash",
                      line_color="#ffffff", opacity=0.45)
        fig.add_annotation(
            x=split_x, y=1.0, yref="paper", xref="x",
            text="Independent runs — OOS restarts at deposit",
            showarrow=False, font=dict(size=10, color="#cccccc"),
            yanchor="bottom", yshift=6,
        )
        if oos_m.equity:
            fig.add_hline(y=deposit, line_width=1, line_dash="dot",
                          line_color="#f9a84f", opacity=0.5,
                          annotation_text=f"OOS start ${deposit:,.0f}",
                          annotation_position="bottom right")

    # WFO window boundary markers
    for w in report.wfo_windows:
        fig.add_vline(x=_ts_to_dt(w.oos_start_ts), line_width=1,
                      line_dash="dot",
                      line_color="#888" if w.mode == "anchored" else "#bbb")

    is_dd = report_zone_drawdown(report, "IS")
    oos_dd = report_zone_drawdown(report, "OOS")
    # Reserve a taller top margin so the three stacked elements above the plot
    # (title → legend → "Independent runs" annotation) each get their own row
    # and never overlap the chart.
    fig.update_layout(
        height=350, margin=dict(l=10, r=10, t=92, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.15),
        xaxis_title=None, yaxis_title="Equity",
        template="plotly_dark",
        title=dict(
            text=(f"{zone_drawdown_label('IS')}: {is_dd:.1f}% · "
                  f"{zone_drawdown_label('OOS')}: {oos_dd:.1f}% "
                  f"(intrabar, per-zone)"),
            font=dict(size=12),
            y=0.99, yanchor="top", pad=dict(b=6),
        ),
    )
    return fig
