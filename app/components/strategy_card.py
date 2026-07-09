"""Gallery of surviving strategies: metric cards + equity chart + rule math.

The same card renderer powers both the full-size gallery grid and the compact
per-run grid shown on the Discovery tab, via a ``compact`` flag — so the two
views stay visually consistent without duplicated markup.
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime

import streamlit as st

from app.components.equity_chart import build_equity_figure
from factory.metrics_display import (
    data_source_badge, gate_drawdown_pct, sortino_ratio, wfo_summary,
    zone_drawdown_label,
)
from factory.assets.exporter import export_marketplace_package
from factory.models import (
    LotMode, StopLossMode, StrategyDefinition, TakeProfitMode, TrailMode,
    ValidationReport,
)
from factory.storage import Storage


def trade_mgmt_badges(strategy: StrategyDefinition) -> str:
    """Compact colored chips summarizing a strategy's exit/risk overlays."""
    tm = strategy.trade_mgmt
    chips = []
    if tm.sl_mode == StopLossMode.ATR:
        chips.append("ATR stop")
    if tm.tp_mode == TakeProfitMode.RR:
        chips.append("R:R TP")
    if tm.trail_mode == TrailMode.FIXED:
        chips.append("Trailing")
    elif tm.trail_mode == TrailMode.ATR:
        chips.append("ATR trail")
    elif tm.trail_mode == TrailMode.CHANDELIER:
        chips.append("Chandelier")
    if tm.breakeven:
        chips.append("Breakeven")
    if tm.lot_mode == LotMode.RISK_PERCENT:
        chips.append("Risk %")
    if tm.time_filter:
        chips.append("Session")
    if tm.limit_trades_per_day:
        chips.append("Max/day")
    if tm.daily_loss_enabled:
        chips.append("Daily-loss")
    if tm.cooldown_enabled:
        chips.append("Cooldown")
    return " ".join(f":violet-badge[{c}]" for c in chips)


def format_run_label(run_id: str | None, runs: dict | None = None) -> str:
    """Human label for the discovery run a result came from.

    ``run_id`` is the discovery job id (e.g. ``disc_ab12cd34ef``). When the
    matching :class:`~factory.models.Job` is available in ``runs`` its start
    time is appended, giving e.g. ``disc_ab12cd34ef · Jul 08, 18:27``.
    """
    if not run_id:
        return "unknown run"
    job = (runs or {}).get(run_id)
    created = getattr(job, "created_at", None) if job else None
    if created:
        try:
            when = datetime.fromtimestamp(created).strftime("%b %d, %H:%M")
            return f"{run_id} · {when}"
        except (OverflowError, OSError, ValueError):
            pass
    return run_id


def _zip_dir(folder) -> bytes:
    """Zip an exported package folder into downloadable bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(folder.parent))
    buf.seek(0)
    return buf.getvalue()


# Compact modal grid: two columns × four rows keeps Plotly work bounded.
_MODAL_PAGE_SIZE = 8


def _run_created_at(report: ValidationReport, runs: dict | None) -> float:
    job = (runs or {}).get(report.run_id or "")
    return float(getattr(job, "created_at", 0) or 0)


# Sort options: label -> (key function(report, runs), reverse).
_SORT_OPTIONS: dict = {
    "WFE (high → low)": (lambda r, runs: r.wfe, True),
    "OOS net profit (high → low)": (lambda r, runs: r.oos_metrics.net_profit, True),
    "OOS profit factor (high → low)": (lambda r, runs: r.oos_metrics.profit_factor, True),
    "OOS Sharpe (high → low)": (lambda r, runs: r.oos_metrics.sharpe, True),
    "OOS Sortino (high → low)": (lambda r, runs: sortino_ratio(r.oos_metrics), True),
    "Run (newest → oldest)": (_run_created_at, True),
    "Run (oldest → newest)": (_run_created_at, False),
    "Equity R² (high → low)": (lambda r, runs: r.oos_metrics.r_squared, True),
    "MC robustness (high → low)": (
        lambda r, runs: r.montecarlo.robustness_score if r.montecarlo else 0.0, True),
    "OOS max drawdown (low → high)": (lambda r, runs: gate_drawdown_pct(r.oos_metrics), False),
    "Degradation (low → high)": (lambda r, runs: r.degradation_pct, False),
    "Stability (high → low)": (lambda r, runs: r.stability_ratio, True),
}

SORT_OPTION_LABELS: tuple[str, ...] = tuple(_SORT_OPTIONS.keys())


def sort_reports(reports: list, sort_by: str,
                 runs: dict | None = None) -> list:
    """Return a copy of ``reports`` ordered by the chosen metric."""
    spec = _SORT_OPTIONS.get(sort_by)
    if spec is None:
        return sorted(reports, key=lambda r: r.wfe, reverse=True)
    key_fn, reverse = spec
    return sorted(reports, key=lambda r: key_fn(r, runs), reverse=reverse)


def render_sort_selectbox(key: str, *, index: int = 0,
                          page_key: str | None = None) -> str:
    """Shared sort-by selectbox for gallery and modal result grids."""
    kwargs: dict = {}
    if page_key is not None:
        kwargs["on_change"] = _reset_page
        kwargs["args"] = (page_key,)
    return st.selectbox(
        "Sort by", list(SORT_OPTION_LABELS), index=index,
        key=key, help="Order the result cards below by any metric.",
        **kwargs)


def _reset_page(page_key: str) -> None:
    st.session_state[page_key] = 0


def _page_slice(reports: list, page_key: str,
                page_size: int) -> tuple[list, int, int]:
    """Return ``(slice, page_index, total_pages)`` for paginated grids."""
    total_pages = max(1, (len(reports) + page_size - 1) // page_size)
    page = min(int(st.session_state.get(page_key, 0)), total_pages - 1)
    st.session_state[page_key] = page
    start = page * page_size
    return reports[start:start + page_size], page, total_pages


def render_page_controls(page_key: str, page: int, total_pages: int,
                         total_items: int, page_size: int,
                         *, control_key: str) -> None:
    """Previous / next controls for paginated result grids."""
    if total_pages <= 1:
        return
    first = page * page_size + 1
    last = min((page + 1) * page_size, total_items)
    nav_l, nav_m, nav_r = st.columns([1, 2, 1])
    with nav_l:
        if st.button("Previous page", key=f"{control_key}_prev",
                     disabled=page <= 0, width="stretch"):
            st.session_state[page_key] = page - 1
            st.rerun()
    with nav_m:
        st.caption(f"Showing {first}–{last} of {total_items} · "
                   f"page {page + 1} of {total_pages}")
    with nav_r:
        if st.button("Next page", key=f"{control_key}_next",
                     disabled=page >= total_pages - 1, width="stretch"):
            st.session_state[page_key] = page + 1
            st.rerun()


@st.cache_data(max_entries=256, show_spinner=False)
def _cached_compact_figure(chart_key: str, report_json: str):
    """Cache compact equity thumbnails — building Plotly figures is costly."""
    report = ValidationReport.model_validate_json(report_json)
    fig = build_equity_figure(report)
    fig.update_layout(height=180, showlegend=False, title=None,
                      margin=dict(l=6, r=6, t=10, b=6))
    return fig


def _chart_cache_key(report: ValidationReport) -> str:
    m = report.oos_metrics
    return (f"{report.strategy_id}:{report.wfe}:"
            f"{len(m.equity)}:{m.equity[-1] if m.equity else 0}")


def render_gallery(storage: Storage) -> None:
    st.subheader("Surviving strategies")

    # Load once: a strategy map avoids a per-card DB round-trip and powers the
    # symbol filter without extra queries. The run map lets each card show which
    # discovery run produced it (id + start time).
    strategies = {s.id: s for s in storage.list_strategies()}
    runs = {j.id: j for j in storage.list_jobs("discovery")}

    with st.container(border=True):
        top_l, top_r = st.columns([3, 1], vertical_alignment="bottom")
        with top_l:
            default_sort_idx = SORT_OPTION_LABELS.index("Equity R² (high → low)")
            sort_by = render_sort_selectbox("gallery_sort", index=default_sort_idx)
        with top_r:
            show_all = st.toggle(
                "Include failed", value=True,
                help="Show candidates that did not clear every gate.")

        reports = storage.list_validated(passed_only=not show_all)

        # Build filter option lists from what's actually present.
        symbols = sorted({strategies[r.strategy_id].symbol
                          for r in reports if r.strategy_id in strategies})
        engines = sorted({r.engine for r in reports if r.engine})
        sources = sorted({r.data_source for r in reports if r.data_source})

        f1, f2, f3 = st.columns(3)
        with f1:
            pick_symbols = st.multiselect(
                "Symbols", symbols, default=[],
                help="Leave empty to include every symbol.")
        with f2:
            pick_engines = st.multiselect("Engines", engines, default=[])
        with f3:
            pick_sources = st.multiselect("Data source", sources, default=[])

        g1, g2, g3, g4 = st.columns(4)
        with g1:
            min_profit = st.number_input("Min OOS net profit", value=0.0,
                                         step=100.0, format="%.0f")
        with g2:
            min_pf = st.number_input("Min profit factor", 0.0, 100.0, 0.0, step=0.1)
        with g3:
            min_sharpe = st.number_input("Min Sharpe", 0.0, 100.0, 0.0, step=0.1)
        with g4:
            min_wfe = st.number_input("Min WFE", 0.0, 5.0, 0.0, step=0.05)

    def _keep(r) -> bool:
        strat = strategies.get(r.strategy_id)
        if pick_symbols and (strat is None or strat.symbol not in pick_symbols):
            return False
        if pick_engines and r.engine not in pick_engines:
            return False
        if pick_sources and r.data_source not in pick_sources:
            return False
        m = r.oos_metrics
        return (m.net_profit >= min_profit and m.profit_factor >= min_pf
                and m.sharpe >= min_sharpe and r.wfe >= min_wfe)

    filtered = [r for r in reports if _keep(r)]
    total = len(reports)
    filtered = sort_reports(filtered, sort_by, runs=runs)

    if not reports:
        st.info("Nothing here yet. Run a discovery batch first — survivors "
                "appear automatically.")
        return
    if not filtered:
        st.warning("No results match the current filters. Loosen them to see "
                   f"more (of {total} total).")
        return

    st.caption(f"Showing {len(filtered)} of {total} result"
               f"{'s' if total != 1 else ''}.")
    render_report_grid(filtered, strategies, runs=runs)


def render_report_grid(reports: list, strategies: dict, runs: dict | None = None,
                       key_prefix: str = "", compact: bool = False,
                       storage: Storage | None = None) -> None:
    """Grid of strategy cards. ``strategies`` maps id -> definition.

    ``compact`` renders smaller cards in a tighter grid (used by the Discovery
    tab's per-run results). ``runs`` maps run id -> Job; when provided, each
    card shows which run produced it.

    When ``storage`` is given with ``compact=True``, strategies are loaded
    lazily for the current page only (faster modal open).
    """
    n_cols = 2 if compact else 3
    for i in range(0, len(reports), n_cols):
        cols = st.columns(n_cols, gap="small" if compact else "medium")
        row_reports = reports[i:i + n_cols]
        if compact and storage is not None:
            row_strategies = _strategies_for_reports(storage, row_reports, strategies)
        else:
            row_strategies = strategies
        for col, report in zip(cols, row_reports):
            strategy = row_strategies.get(report.strategy_id)
            if strategy is None:
                continue
            run_label = (format_run_label(report.run_id, runs)
                         if runs is not None else None)
            with col:
                render_strategy_card(strategy, report, run_label=run_label,
                                     key_prefix=key_prefix, compact=compact)


def _strategies_for_reports(storage: Storage, reports: list,
                            strategies: dict) -> dict:
    """Resolve strategy definitions for one page, reusing a pre-built map."""
    if strategies:
        return strategies
    out: dict = {}
    for report in reports:
        sid = report.strategy_id
        if sid in out:
            continue
        strat = storage.get_strategy(sid)
        if strat is not None:
            out[sid] = strat
    return out


def _sort_reports(reports: list, sort_by: str) -> list:
    """Backward-compatible alias — prefer :func:`sort_reports`."""
    return sort_reports(reports, sort_by)


def render_strategy_card(strategy: StrategyDefinition,
                         report: ValidationReport,
                         run_label: str | None = None,
                         key_prefix: str = "",
                         compact: bool = False) -> None:
    """Render one strategy result card.

    ``compact`` produces a smaller card (tighter header, fewer headline
    metrics, a shorter chart, no select/export controls or expanders) for the
    Discovery tab's per-run grid, while the full card is used in the gallery.
    ``run_label`` (when given) shows which discovery run produced the result.
    """
    if compact:
        _render_compact_card(strategy, report, run_label=run_label,
                             key_prefix=key_prefix)
        return

    oos = report.oos_metrics
    badge = (":green-badge[:material/check_circle: PASS]" if report.passed
             else ":red-badge[:material/cancel: FAIL]")
    with st.container(border=True):
        head_l, head_r = st.columns([5, 1], vertical_alignment="center")
        with head_l:
            st.markdown(f"#### {strategy.name}")
        with head_r:
            st.checkbox("Select", key=f"{key_prefix}select_{strategy.id}",
                        label_visibility="collapsed",
                        help="Selected strategies can be exported from the "
                             "Export tab.")

        src_badge = data_source_badge(report.data_source)
        st.markdown(f"{badge} &nbsp; {src_badge}")
        st.caption(
            f"Promotion: {report.promotion_state} · quality score {report.quality_score:.1f}"
        )
        if run_label:
            st.caption(f":material/tag: Run {run_label}")
        st.caption(
            f"{strategy.symbol} · {strategy.timeframe} · "
            f"{report.engine} · gen {strategy.lineage.generation} · "
            f"magic {strategy.magic_number}")

        tm_badges = trade_mgmt_badges(strategy)
        if tm_badges:
            st.markdown(tm_badges)

        oos_dd = gate_drawdown_pct(oos)
        is_dd = gate_drawdown_pct(report.is_metrics)

        # Headline metrics in a compact 2-wide grid that reads well at a third
        # of page width. R² (equity smoothness) is surfaced since the search
        # now targets a smooth, steadily-rising curve.
        r1a, r1b = st.columns(2)
        r1a.metric("OOS net profit", f"{oos.net_profit:,.0f}")
        r1b.metric("Profit factor", f"{oos.profit_factor:.2f}")
        r2a, r2b = st.columns(2)
        r2a.metric("Equity R²", f"{oos.r_squared:.2f}",
                   help="Straightness of the equity curve (1.0 = perfectly "
                        "smooth rising line).")
        r2b.metric("WFE", f"{report.wfe:.2f}")
        r3a, r3b = st.columns(2)
        r3a.metric(zone_drawdown_label("OOS"), f"{oos_dd:.1f}%")
        if report.montecarlo:
            r3b.metric("MC robustness", f"{report.montecarlo.robustness_score:.0f}")
        else:
            r3b.metric("MC robustness", "—")

        st.caption(
            f"Sharpe {oos.sharpe:.2f} · degradation {report.degradation_pct:.0f}% "
            f"· stability {report.stability_ratio:.2f} · IS "
            f"{zone_drawdown_label('IS')} {is_dd:.1f}%")

        st.plotly_chart(build_equity_figure(report),
                        width="stretch",
                        key=f"{key_prefix}chart_{strategy.id}")

        if report.montecarlo:
            mc = report.montecarlo
            st.caption(
                f"Monte Carlo ({mc.n_runs} runs): "
                f"{mc.pct_profitable:.0%} profitable · "
                f"profit P05/P50/P95 = {mc.profit_p05:,.0f} / "
                f"{mc.profit_p50:,.0f} / {mc.profit_p95:,.0f} · "
                f"95%-worst DD = {mc.dd_p95:.1f}%"
                + (" · :green[MC PASS]" if mc.passed else " · :red[MC FAIL]"))

        if report.wfo_windows:
            rolling = [w for w in report.wfo_windows if w.mode == "rolling"]
            anchored = [w for w in report.wfo_windows if w.mode == "anchored"]
            train_m = report.wfo_train_months
            test_m = report.wfo_test_months
            period_note = (f" ({train_m}m train / {test_m}m test)"
                           if train_m and test_m else "")
            with st.expander(f"Walk-forward windows{period_note}"):
                if rolling:
                    st.markdown(f"**Rolling:** {wfo_summary(rolling, 'rolling')}")
                    for w in rolling:
                        st.caption(
                            f"  #{w.index + 1}: WFE {w.wfe:.2f} · "
                            f"OOS DD {gate_drawdown_pct(w.oos_metrics):.1f}% · "
                            f"OOS profit {w.oos_metrics.net_profit:,.0f}")
                if anchored:
                    st.markdown(f"**Anchored:** {wfo_summary(anchored, 'anchored')}")
                    for w in anchored:
                        st.caption(
                            f"  #{w.index + 1}: WFE {w.wfe:.2f} · "
                            f"OOS DD {gate_drawdown_pct(w.oos_metrics):.1f}% · "
                            f"OOS profit {w.oos_metrics.net_profit:,.0f}")

        with st.expander("Exact entry/exit rule math"):
            st.code(strategy.rule_description or "(no description)")
            if report.best_params:
                st.markdown("**Validated best parameters:**")
                st.json(report.best_params, expanded=False)
        if not report.passed and report.reasons:
            st.warning(" · ".join(report.reasons))

        _export_controls(strategy, report, key_prefix=key_prefix)


def _render_compact_card(strategy: StrategyDefinition,
                         report: ValidationReport,
                         run_label: str | None = None,
                         key_prefix: str = "") -> None:
    """Small version of the gallery card for the per-run results grid.

    Keeps the same visual language — border, pass/fail badge, headline metrics
    and equity curve — but drops the select/export controls, Monte Carlo and
    walk-forward expanders, and shrinks the chart so several fit on the page.
    Open the Strategy Gallery for the full card.
    """
    oos = report.oos_metrics
    badge = (":green-badge[:material/check_circle: PASS]" if report.passed
             else ":red-badge[:material/cancel: FAIL]")
    with st.container(border=True):
        st.markdown(f"##### {strategy.name}")
        st.markdown(f"{badge} &nbsp; {data_source_badge(report.data_source)}")
        st.caption(f"Promotion: {report.promotion_state} · score {report.quality_score:.1f}")
        if run_label:
            st.caption(f":material/tag: Run {run_label}")
        st.caption(f"{strategy.symbol} · {strategy.timeframe} · {report.engine} · "
                   f"gen {strategy.lineage.generation}")

        c1, c2 = st.columns(2)
        c1.metric("OOS net profit", f"{oos.net_profit:,.0f}")
        c2.metric("Profit factor", f"{oos.profit_factor:.2f}")
        c3, c4 = st.columns(2)
        c3.metric("Equity R²", f"{oos.r_squared:.2f}")
        c4.metric("WFE", f"{report.wfe:.2f}")

        # Reuse the shared figure, shrunk to a thumbnail: no legend/title so the
        # curve reads cleanly at a fraction of the gallery size.
        fig = _cached_compact_figure(
            _chart_cache_key(report), report.model_dump_json())
        st.plotly_chart(fig, width="stretch",
                        key=f"{key_prefix}chart_{strategy.id}")

        if not report.passed and report.reasons:
            st.caption(":red[" + " · ".join(report.reasons) + "]")


def _export_controls(strategy: StrategyDefinition,
                     report: ValidationReport, key_prefix: str = "") -> None:
    """One-click MQL5 export: build the package and offer an instant download."""
    zip_key = f"{key_prefix}zip_{strategy.id}"
    exp_col, dl_col = st.columns([1, 1])
    with exp_col:
        if st.button("Convert to MQL5", key=f"{key_prefix}export_{strategy.id}",
                     type="primary", width="stretch",
                     help="Render the validator-proof .mq5 + .set + .md "
                          "marketplace package."):
            try:
                out_dir = export_marketplace_package(strategy, report)
                st.session_state[zip_key] = {
                    "bytes": _zip_dir(out_dir),
                    "name": out_dir.name,
                    "path": str(out_dir),
                }
                st.toast(f"Exported to {out_dir}")
            except Exception as exc:
                st.error(f"Export failed: {exc}")
    with dl_col:
        pkg = st.session_state.get(zip_key)
        if pkg:
            st.download_button(
                "Download package (.zip)", data=pkg["bytes"],
                file_name=f"{pkg['name']}.zip", mime="application/zip",
                key=f"{key_prefix}dl_{strategy.id}", width="stretch")
            st.caption(f"Saved to `{pkg['path']}`")
