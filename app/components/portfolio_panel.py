"""HRP portfolio view: weights, correlation heatmap, combined equity."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.components import theme
from factory.portfolio import build_portfolio
from factory.storage import Storage

_PROMOTED_STATES = ("edge_positive", "promoted_live_watchlist")


def render_portfolio_panel(storage: Storage) -> None:
    theme.section(
        "Portfolio construction",
        "Hierarchical Risk Parity over the selected strategies' OOS "
        "daily-return streams — no expected-return estimates, no matrix "
        "inversion, robust to near-duplicate edges.")

    reports = storage.list_validated(passed_only=True)
    if not reports:
        st.info("No passing strategies in the library yet — run a discovery "
                "sweep first.")
        return

    strategy_index = {s.id: s for s in storage.list_strategy_summaries()}

    def _label(rep) -> str:
        s = strategy_index.get(rep.strategy_id)
        name = s.name if s else rep.strategy_id[:8]
        sym = f" · {s.symbol} {s.timeframe}" if s else ""
        return f"{name}{sym}"

    labels = {_label(r): r for r in reports}
    promoted_labels = [_label(r) for r in reports
                       if r.promotion_state in _PROMOTED_STATES]
    default = promoted_labels or list(labels)[: min(5, len(labels))]

    chosen = st.multiselect(
        "Strategies to combine",
        options=list(labels),
        default=default,
        help="Defaults to promoted / edge-positive strategies. Weights are "
             "computed from OOS daily returns; overlapping history is "
             "required for correlations to be meaningful.")
    if len(chosen) < 2:
        st.info("Pick at least two strategies to build a portfolio.")
        return

    selected = [labels[c] for c in chosen]
    port = build_portfolio(selected)
    if len(port.strategy_ids) < 2:
        st.warning("Fewer than two of the selected strategies have usable "
                   "OOS equity curves — cannot build a portfolio.")
        return

    corr_tone = ("good" if port.avg_pairwise_corr < 0.3
                 else "" if port.avg_pairwise_corr < 0.6 else "bad")
    theme.kpi_row(
        [
            ("Ann. Sharpe", f"{port.ann_sharpe:.2f}",
             "good" if port.ann_sharpe > 1 else ""),
            ("Ann. return", f"{port.ann_return:+.1%}",
             "good" if port.ann_return > 0 else "bad"),
            ("Max drawdown", f"{port.max_dd_pct:.1f}%",
             "bad" if port.max_dd_pct > 20 else ""),
            ("Avg pairwise corr", f"{port.avg_pairwise_corr:.2f}", corr_tone),
            ("Diversification", f"{port.diversification_ratio:.2f}",
             "good" if port.diversification_ratio < 0.85 else ""),
        ],
        deltas=["OOS daily returns", f"{port.days} common days",
                "combined stream", f"max {port.max_pairwise_corr:.2f}",
                "port vol / avg vol"],
    )

    id_to_label = {r.strategy_id: lbl for lbl, r in labels.items()}
    names = [id_to_label.get(sid, sid[:8]) for sid in port.strategy_ids]

    col_l, col_r = st.columns([1, 1])
    with col_l:
        theme.section("HRP weights", "risk-balanced allocation")
        weights = [port.weights[sid] for sid in port.strategy_ids]
        bar = go.Figure(go.Bar(
            x=weights, y=names, orientation="h",
            marker=dict(color="#D4A853",
                        line=dict(color="#2A3441", width=1)),
            text=[f"{w:.0%}" for w in weights], textposition="outside",
        ))
        bar.update_layout(
            height=max(180, 44 * len(names)),
            margin=dict(l=6, r=30, t=8, b=6),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, max(weights) * 1.25], showgrid=False,
                       tickformat=".0%"),
            yaxis=dict(autorange="reversed"),
            font=dict(color="#8B95A5", size=12),
        )
        st.plotly_chart(bar, width="stretch", key="portfolio_weights")

    with col_r:
        theme.section("Return correlation", "OOS daily-return overlap")
        heat = go.Figure(go.Heatmap(
            z=port.corr, x=names, y=names,
            zmin=-1, zmax=1,
            colorscale=[[0.0, "#38BDF8"], [0.5, "#141A22"], [1.0, "#F87171"]],
            text=[[f"{v:.2f}" for v in row] for row in port.corr],
            texttemplate="%{text}", textfont=dict(size=11),
            showscale=False,
        ))
        heat.update_layout(
            height=max(180, 44 * len(names)),
            margin=dict(l=6, r=6, t=8, b=6),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8B95A5", size=11),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(heat, width="stretch", key="portfolio_corr")

    theme.section("Combined equity", "growth of 1.0 at HRP weights")
    eq = go.Figure(go.Scatter(
        y=port.equity, mode="lines",
        line=dict(color="#2DD4BF", width=2),
        fill="tozeroy", fillcolor="rgba(45,212,191,0.06)",
    ))
    eq.update_layout(
        height=280, margin=dict(l=6, r=6, t=8, b=6),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="trading day", showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1E2733"),
        font=dict(color="#8B95A5", size=12),
    )
    st.plotly_chart(eq, width="stretch", key="portfolio_equity")

    st.caption(
        "Robustness ≠ profitability: HRP balances *risk*, it cannot create "
        "edge. Validate the combined basket in the MT5 Strategy Tester "
        "before running it live.")
