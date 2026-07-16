"""Gallery of surviving strategies: metric cards + equity chart + rule math.

The same card renderer powers both the full-size gallery grid and the compact
per-run grid shown on the Discovery tab, via a ``compact`` flag — so the two
views stay visually consistent without duplicated markup.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime

import streamlit as st

from app.components.equity_chart import build_equity_figure
from factory.metrics_display import (
    data_source_badge, dsr_badge, dsr_label, gate_drawdown_pct, sortino_ratio,
    wfo_summary, zone_drawdown_label,
)
from factory.assets.exporter import export_marketplace_package
from factory import validation_levels
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
# Full gallery grid: three columns × four rows; full bodies load for this page only.
_GALLERY_PAGE_SIZE = 12
# Safety cap when listing summaries (filters/sort still apply in Python after).
_GALLERY_SUMMARY_CAP = 500


def _hydrate_reports(storage: Storage, summaries: list) -> list:
    """Load full ValidationReport bodies for the visible page only."""
    ids = [s.strategy_id for s in summaries]
    full = storage.get_validations(ids)
    # Preserve summary order.
    return [full[sid] for sid in ids if sid in full]


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
    "DSR (high → low)": (lambda r, runs: getattr(r, "dsr", 0.0), True),
}

SORT_OPTION_LABELS: tuple[str, ...] = tuple(_SORT_OPTIONS.keys())



_RISK_BADGE_COLORS = {"red": "red", "amber": "orange", "violet": "violet"}


def risk_style_badge(strategy) -> str:
    """Streamlit badge for aggressive-recovery execution styles ('' if none).

    Martingale grids and hedge-recovery layers produce the prettiest
    backtests and the harshest live drawdowns — the label follows the
    strategy everywhere so that trade-off is never invisible.
    """
    from factory.publication import risk_style
    style = risk_style(strategy)
    if style is None:
        return ""
    label, tone = style
    color = _RISK_BADGE_COLORS.get(tone, "orange")
    return f":{color}-badge[:material/warning: {label}]"


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
    from app.components import theme
    theme.section(
        "Strategy library",
        "Browse all validated strategies, filter by level, and build portfolios.")

    view = st.segmented_control(
        "View",
        options=["Library", "Portfolio"],
        default="Library",
        key="gallery_view_mode",
        help="Library shows all survivors with filters. Portfolio builds an HRP "
             "allocation over selected strategies. Per-run progress lives on the Runs tab.",
    )
    if view == "Portfolio":
        from app.components.portfolio_panel import render_portfolio_panel
        render_portfolio_panel(storage)
        return

    st.markdown("#### Surviving strategies")

    # Lightweight index only — never deserialize full strategy/validation bodies
    # for the whole library up front.
    strategy_index = {s.id: s for s in storage.list_strategy_summaries()}
    runs = {j.id: j for j in storage.list_jobs("discovery")}
    page_key = "gallery_lib_page"

    from app.components.run_view import (
        format_tier_caption, render_min_level_filter,
    )

    with st.container(border=True):
        top_l, top_r = st.columns([3, 1], vertical_alignment="bottom")
        with top_l:
            default_sort_idx = SORT_OPTION_LABELS.index("Equity R² (high → low)")
            sort_by = render_sort_selectbox(
                "gallery_sort", index=default_sort_idx, page_key=page_key)
        with top_r:
            min_level = render_min_level_filter(
                "gallery_min_level", page_key=page_key,
                default=validation_levels.GALLERY_DEFAULT_MIN_LEVEL)

        summaries = storage.list_validation_summaries(
            passed_only=None,
            min_level=min_level,
            limit=_GALLERY_SUMMARY_CAP,
        )

        # Build filter option lists from what's actually present.
        symbols = sorted({strategy_index[r.strategy_id].symbol
                          for r in summaries if r.strategy_id in strategy_index
                          and strategy_index[r.strategy_id].symbol})
        engines = sorted({r.engine for r in summaries if r.engine})
        sources = sorted({r.data_source for r in summaries if r.data_source})

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
        strat = strategy_index.get(r.strategy_id)
        if pick_symbols and (strat is None or strat.symbol not in pick_symbols):
            return False
        if pick_engines and r.engine not in pick_engines:
            return False
        if pick_sources and r.data_source not in pick_sources:
            return False
        m = r.oos_metrics
        return (m.net_profit >= min_profit and m.profit_factor >= min_pf
                and m.sharpe >= min_sharpe and r.wfe >= min_wfe)

    filtered = [r for r in summaries if _keep(r)]
    total = len(summaries)
    filtered = sort_reports(filtered, sort_by, runs=runs)

    if not summaries:
        st.info("Nothing here yet. Run a discovery batch first — survivors "
                "appear automatically.")
        return
    if not filtered:
        st.warning("No results match the current filters. Loosen them to see "
                   f"more (of {total} total).")
        return

    tier_hist = storage.level_counts()
    if total >= _GALLERY_SUMMARY_CAP:
        st.caption(
            f"Showing top {_GALLERY_SUMMARY_CAP} by level/WFE"
            + (f" (≥ L{min_level})" if min_level else " (full population)")
            + f". Tiers: {format_tier_caption(tier_hist)}. "
            f"Use filters or Results per run for narrower sets.")
    else:
        st.caption(
            f"Showing {len(filtered)} of {total} result"
            f"{'s' if total != 1 else ''}"
            + (f" (≥ L{min_level})" if min_level else "")
            + f". Tiers: {format_tier_caption(tier_hist)}.")

    page_summaries, page, total_pages = _page_slice(
        filtered, page_key, _GALLERY_PAGE_SIZE)
    render_page_controls(
        page_key, page, total_pages, len(filtered), _GALLERY_PAGE_SIZE,
        control_key="gallery_lib_nav")

    page_reports = _hydrate_reports(storage, page_summaries)
    page_strategies = _strategies_for_reports(storage, page_reports, {})
    render_report_grid(page_reports, page_strategies, runs=runs,
                       storage=storage)

    _render_insights_section(storage, pick_symbols)


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
                                     key_prefix=key_prefix, compact=compact,
                                     storage=storage)


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


def _truncate_id(sid: str, n: int = 8) -> str:
    return sid if len(sid) <= n else sid[:n]


def _lineage_caption(strategy: StrategyDefinition) -> str:
    """Compact lineage line: parents, operation, mutations."""
    lin = strategy.lineage
    parents = list(lin.parents or [])
    mutations = list(lin.mutations or [])
    if len(parents) > 1:
        operation = "crossover"
    elif parents:
        operation = "mutate"
    else:
        operation = "random"
    parts = [f"op {operation}"]
    if parents:
        parts.append("parents " + ", ".join(_truncate_id(p) for p in parents[:3]))
        if len(parents) > 3:
            parts[-1] += f" +{len(parents) - 3}"
    if mutations:
        shown = ", ".join(mutations[:4])
        if len(mutations) > 4:
            shown += f" +{len(mutations) - 4}"
        parts.append(f"Δ {shown}")
    return " · ".join(parts)


def _dsr_display(report: ValidationReport) -> str | None:
    """Markdown DSR badge + label when DSR or trial count is available."""
    dsr = float(getattr(report, "dsr", 0.0) or 0.0)
    n_trials = int(getattr(report, "n_trials", 0) or 0)
    if dsr <= 0.0 and n_trials <= 0:
        return None
    if dsr > 0.0:
        return f"{dsr_badge(dsr)} · {dsr_label(dsr, max(n_trials, 1))}"
    return f"DSR n/a · {n_trials} trials"


def render_strategy_card(strategy: StrategyDefinition,
                         report: ValidationReport,
                         run_label: str | None = None,
                         key_prefix: str = "",
                         compact: bool = False,
                         storage: Storage | None = None) -> None:
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
    from app.components.run_view import level_badge
    tier = level_badge(getattr(report, "highest_level_passed", 0) or 0)
    badge = (":green-badge[:material/check_circle: ≥ floor]" if report.passed
             else ":gray-badge[:material/remove: below floor]")
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
        risk_badge = risk_style_badge(strategy)
        line = f"{tier} &nbsp; {badge} &nbsp; {src_badge}"
        if risk_badge:
            line += f" &nbsp; {risk_badge}"
        st.markdown(line)
        st.caption(
            f"Promotion: {report.promotion_state} · quality score {report.quality_score:.1f}"
        )
        if run_label:
            st.caption(f":material/tag: Run {run_label}")
        st.caption(
            f"{strategy.symbol} · {strategy.timeframe} · "
            f"{report.engine} · gen {strategy.lineage.generation} · "
            f"magic {strategy.magic_number}")
        st.caption(_lineage_caption(strategy))

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

        dsr_md = _dsr_display(report)
        if dsr_md:
            st.markdown(dsr_md)
        is_trials = int(getattr(report, "is_trials", 0) or 0)
        extra = ""
        if is_trials > 0:
            extra = f" · IS trials {is_trials}"
        st.caption(
            f"Sharpe {oos.sharpe:.2f} · degradation {report.degradation_pct:.0f}% "
            f"· stability {report.stability_ratio:.2f} · IS "
            f"{zone_drawdown_label('IS')} {is_dd:.1f}%" + extra)

        if getattr(report, "regime_stats", None):
            traded = [s for s in report.regime_stats if s.trades > 0]
            if traded:
                st.caption("Regimes: " + " · ".join(
                    f"{s.name} {s.net_profit:+.0f} ({s.trades}t, PF {s.profit_factor:.2f})"
                    for s in traded))

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

        trace = getattr(report, "param_search_trace", None) or []
        if trace:
            with st.expander(f"Param search trace ({len(trace)} top trials)"):
                rows = []
                for i, trial in enumerate(trace):
                    params = trial.get("params") or {}
                    value = trial.get("value")
                    preview = ", ".join(
                        f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in list(params.items())[:6])
                    if len(params) > 6:
                        preview += f" …(+{len(params) - 6})"
                    rows.append({
                        "#": i + 1,
                        "value": (f"{value:.4f}" if isinstance(value, (int, float))
                                  else str(value)),
                        "params": preview,
                    })
                st.dataframe(rows, hide_index=True, width="stretch")

        with st.expander("Exact entry/exit rule math"):
            st.code(strategy.rule_description or "(no description)")
            if report.best_params:
                st.markdown("**Validated best parameters:**")
                st.json(report.best_params, expanded=False)
        if report.reasons:
            if report.passed:
                st.caption("Next tier: " + " · ".join(report.reasons[:3]))
            else:
                st.warning(" · ".join(report.reasons))

        if report.run_id:
            _render_manifest_controls(report.run_id, strategy.id,
                                      key_prefix=key_prefix, storage=storage)

        _export_controls(strategy, report, key_prefix=key_prefix)


def _render_manifest_controls(run_id: str, strategy_id: str, *,
                              key_prefix: str = "",
                              storage: Storage | None = None) -> None:
    """View / download the discovery-run reproducibility manifest."""
    with st.expander("View manifest"):
        store = storage or Storage()
        manifest = store.get_run_manifest(run_id)
        if not manifest:
            st.caption(f"No manifest stored for run `{run_id}`.")
            return
        text = json.dumps(manifest, indent=2, sort_keys=True, default=str)
        st.download_button(
            "Download manifest JSON",
            data=text,
            file_name=f"manifest_{run_id}.json",
            mime="application/json",
            key=f"{key_prefix}manifest_dl_{strategy_id}",
        )
        st.json(manifest, expanded=False)


def _render_insights_section(storage: Storage,
                             pick_symbols: list | None = None) -> None:
    """Gallery Insights: offline parameter importance for a symbol/TF."""
    st.markdown("#### Insights")
    st.caption(
        "Permutation importance of validated parameters vs OOS profit factor "
        "(or WFE). Needs enough strategies with parameter snapshots.")

    summaries = storage.list_strategy_summaries()
    symbols = sorted({s.symbol for s in summaries if s.symbol})
    timeframes = sorted({s.timeframe for s in summaries if s.timeframe})
    if not symbols:
        st.caption("No strategies yet.")
        return

    default_sym = (pick_symbols[0] if pick_symbols else None)
    c1, c2, c3 = st.columns([2, 2, 2], vertical_alignment="bottom")
    with c1:
        sym_idx = (symbols.index(default_sym) if default_sym in symbols else 0)
        sym = st.selectbox("Symbol", symbols, index=sym_idx,
                           key="insights_symbol")
    with c2:
        tf_opts = ["(all)"] + timeframes
        tf = st.selectbox("Timeframe", tf_opts, index=0,
                          key="insights_timeframe")
    with c3:
        run = st.button("Compute parameter importance",
                        key="insights_compute", width="stretch")

    if not run and "insights_importance" not in st.session_state:
        return

    if run:
        from scripts.parameter_importance import compute_parameter_importance
        tf_arg = None if tf == "(all)" else tf
        with st.spinner("Fitting RandomForest…"):
            rows = compute_parameter_importance(
                storage, symbol=sym, timeframe=tf_arg)
        st.session_state["insights_importance"] = {
            "symbol": sym, "timeframe": tf_arg, "rows": rows,
        }

    cached = st.session_state.get("insights_importance") or {}
    rows = cached.get("rows") or []
    if not rows:
        st.info("Not enough overlapping parameter snapshots to rank features "
                "(need ≥5 strategies with shared numeric params).")
        return
    scope = cached.get("symbol", sym)
    tf_lab = cached.get("timeframe") or "all TF"
    st.caption(f"Top features for {scope} · {tf_lab}")
    st.dataframe(
        [{"feature": r["feature"], "importance": f"{r['importance']:.4f}"}
         for r in rows[:25]],
        hide_index=True, width="stretch",
    )


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
    from app.components.run_view import level_badge
    tier = level_badge(getattr(report, "highest_level_passed", 0) or 0)
    badge = (":green-badge[:material/check_circle: ≥ floor]" if report.passed
             else ":gray-badge[:material/remove: below floor]")
    with st.container(border=True):
        st.markdown(f"##### {strategy.name}")
        compact_line = (
            f"{tier} &nbsp; {badge} &nbsp; {data_source_badge(report.data_source)}"
        )
        compact_risk = risk_style_badge(strategy)
        if compact_risk:
            compact_line += f" &nbsp; {compact_risk}"
        st.markdown(compact_line)
        st.caption(f"Promotion: {report.promotion_state} · score {report.quality_score:.1f}")
        if run_label:
            st.caption(f":material/tag: Run {run_label}")
        st.caption(f"{strategy.symbol} · {strategy.timeframe} · {report.engine} · "
                   f"gen {strategy.lineage.generation}")
        st.caption(_lineage_caption(strategy))

        dsr = float(getattr(report, "dsr", 0.0) or 0.0)
        n_trials = int(getattr(report, "n_trials", 0) or 0)
        if dsr > 0.0:
            st.markdown(f"{dsr_badge(dsr)} · N={max(n_trials, 1)}")

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
