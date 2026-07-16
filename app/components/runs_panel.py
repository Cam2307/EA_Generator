"""Runs overview — primary dashboard view with per-run validation progress."""
from __future__ import annotations

import html as html_lib
import time
from datetime import datetime

import streamlit as st

from app.components import theme
from app.components.pie_progress import render_pie_progress
from app.components.run_view import job_summary, render_run_detail
from factory import validation_levels
from factory.models import JobStatus
from factory.storage import Storage
from jobs.orchestrator import (
    recover_stuck_starting_agent,
    stop_orchestrator_process,
    sync_agent_with_orchestrator_lock,
)

_ACTIVE = (JobStatus.PENDING, JobStatus.RUNNING)
_MAX_LEVEL = validation_levels.MAX_LEVEL

_STATUS_BADGE = {
    JobStatus.RUNNING: ("Running", "amber"),
    JobStatus.PENDING: ("Queued", "gray"),
    JobStatus.DONE: ("Done", "teal"),
    JobStatus.CANCELLED: ("Cancelled", "gray"),
    JobStatus.FAILED: ("Failed", "red"),
}


def _mt5_infra_banner() -> None:
    """Warn when the interactive MT5 terminal would block headless validation."""
    try:
        from factory.backtest.mt5_runner import interactive_terminal_running
        if interactive_terminal_running():
            st.warning(
                "**MetaTrader 5 is open interactively.** Discovery falls back to "
                "the simulator for Stage-2 while it stays open, so results are "
                "real quality scores instead of empty INFRA aborts. Close MT5 "
                "only if you need Strategy Tester confirmation.",
                icon=":material/warning:",
            )
    except Exception:
        pass


@st.cache_data(ttl=30, show_spinner=False)
def _factory_kpi_cached(db_path: str, anchors: tuple[int, ...]):
    from pathlib import Path
    storage = Storage(Path(db_path))
    tradeable = storage.count_validated(passed_only=None, exclude_infra=True)
    infra = storage.count_infra_failures()
    counts = {
        lvl: storage.count_validated(passed_only=None, min_level=lvl)
        for lvl in anchors
    }
    states = storage.promotion_state_counts()
    return tradeable, infra, counts, states


def _factory_kpi_strip(storage: Storage) -> None:
    try:
        tradeable, infra, counts, states = _factory_kpi_cached(
            str(storage.db_path),
            tuple(validation_levels.KPI_ANCHORS),
        )
        promoted = states.get("promoted_live_watchlist", 0)
        edge = states.get("edge_positive", 0)
        items = [
            ("Tradeable scored", f"{tradeable:,}", ""),
            ("Infra aborts", f"{infra:,}", "bad" if infra else ""),
        ]
        for lvl in validation_levels.KPI_ANCHORS:
            n = counts.get(lvl, 0)
            band = validation_levels.band_for_level(lvl)
            tone = "good" if lvl == validation_levels.KPI_ANCHORS[0] and n else (
                "accent" if n else "")
            items.append((f"Cleared ≥ L{lvl} {band}", f"{n:,}", tone))
        items.append(
            ("Edge / watchlist", f"{edge + promoted:,}",
             "accent" if (edge or promoted) else ""),
        )
        theme.kpi_row(items)
    except Exception:
        pass


def render_runs_panel(storage: Storage) -> None:
    """All discovery runs with total tests and L1–L16 pass counts."""
    sync_agent_with_orchestrator_lock(storage)
    recover_stuck_starting_agent(storage)

    _mt5_infra_banner()
    _factory_kpi_strip(storage)

    jobs = storage.list_jobs("discovery")
    active = [j for j in jobs if j.status in _ACTIVE]
    finished = [j for j in jobs if j.status not in _ACTIVE]

    theme.section(
        "Discovery runs",
        "Every batch shows how many candidates were tested and how many "
        "cleared each validation level (L1 = softest, L16 = strictest). "
        "Infra aborts (incomplete MT5) are excluded from tradeable KPIs.",
    )

    if active:
        _render_active_section(storage)
    else:
        agent = storage.get_agent_state()
        if int(agent.get("enabled", 0) or 0):
            st.info(str(agent.get("message") or "Agent enabled — waiting for next sweep…"))
        else:
            st.caption(
                "No runs in flight. Open **Discovery** to launch a search, "
                "or review finished runs below."
            )

    if not jobs:
        st.info("No discovery runs yet. Go to **Discovery** to start your first batch.")
        return

    _render_runs_table(storage, jobs)


@st.fragment(run_every="1s")
def _render_active_section(storage: Storage) -> None:
    """Live cards for in-flight runs — refreshes every second."""
    sync_agent_with_orchestrator_lock(storage)
    jobs = storage.list_jobs("discovery")
    active = [j for j in jobs if j.status in _ACTIVE]
    if not active:
        return

    theme.section("Active now", f"{len(active)} run(s) in progress")
    progress = storage.run_progress_by_jobs([j.id for j in active])

    state = storage.get_agent_state()
    agent_on = bool(int(state.get("enabled", 0) or 0))
    if agent_on:
        ctl_l, ctl_r = st.columns([4, 1], vertical_alignment="center")
        with ctl_l:
            mode = str(state.get("mode") or "continuous")
            sweep_total = int(state.get("sweep_total", 0) or 0)
            cursor = int(state.get("cursor", 0) or 0)
            if mode == "batch" and sweep_total > 0:
                idx = max(cursor - 1, 0) % sweep_total
                st.progress(
                    min((idx + 1) / sweep_total, 1.0),
                    text=f"Batch sweep {idx + 1} / {sweep_total}",
                )
            elif mode == "continuous":
                st.caption(
                    f"Continuous agent · {max(cursor, 0)} sweep(s) submitted · "
                    "runs until you stop"
                )
            msg = str(state.get("message") or "").strip()
            if msg:
                st.caption(msg)
        with ctl_r:
            if st.button(":material/stop: Stop all", key="runs_stop_agent", width="stretch"):
                stop_orchestrator_process()
                st.rerun()

    for job in active:
        _render_active_card(job, progress.get(job.id, {}))


def _render_active_card(job, stats: dict) -> None:
    max_candidates = int(job.payload.get("max_candidates", 0) or 0)
    target_survivors = int(job.payload.get("target_survivors", 0) or 0)
    tested_live = int(getattr(job, "tested", 0) or 0)
    survivors_live = int(getattr(job, "survivors", 0) or 0)
    generation = int(getattr(job, "generation", 0) or 0)

    db_total = int(stats.get("total", 0) or 0)
    tested = max(tested_live, db_total)
    passed = max(survivors_live, int(stats.get("passed", 0) or 0))
    level_passes = stats.get("level_passes") or {}

    label, tone = _STATUS_BADGE.get(job.status, (job.status.value, "gray"))
    pct = min(max(job.progress, 0.0), 1.0) * 100.0
    if tested <= 0:
        progress_label = (job.message or "starting…")[:42]
    else:
        progress_label = f"gen {generation} · {tested:,} tested"

    with st.container(border=True):
        head_l, head_r = st.columns([5, 1], vertical_alignment="center")
        with head_l:
            st.markdown(f"**`{job.id}`** · {theme.chip(label, tone, dot=True)}")
            st.caption(job_summary(job))
        with head_r:
            when = datetime.fromtimestamp(job.created_at).strftime("%b %d %H:%M")
            st.caption(when)

        pie_col, stats_col = st.columns([1, 3], vertical_alignment="center")
        with pie_col:
            render_pie_progress(pct, label=progress_label, size=120)
        with stats_col:
            m1, m2, m3, m4 = st.columns(4)
            tested_txt = f"{tested:,} / {max_candidates:,}" if max_candidates else f"{tested:,}"
            passed_txt = f"{passed:,} / {target_survivors:,}" if target_survivors else f"{passed:,}"
            m1.metric("Tested", tested_txt)
            m2.metric("Passed floor", passed_txt)
            m3.metric("Generation", generation)
            elapsed = max(time.time() - job.created_at, 0)
            active_elapsed = max(time.time() - float(getattr(job, "updated_at", 0) or job.created_at), 0)
            # Prefer recent activity window for rate so overnight sleep does not
            # inflate ETA; fall back to wall elapsed when freshly started.
            rate_window = elapsed if tested < 3 else max(elapsed - max(active_elapsed - 120, 0), 60.0)
            msg = (job.message or "").lower()
            if any(tok in msg for tok in ("confirm", "validat")) and tested > 0:
                eta_txt = "validating…"
            else:
                eta_txt = _estimate_eta(tested, survivors_live, max_candidates,
                                        target_survivors, rate_window)
            m4.metric("Est. to target", eta_txt)

            infra_n = int(stats.get("infra", 0) or 0)
            tradeable_n = int(stats.get("tradeable", 0) or 0)
            rate = tested / max(rate_window, 1e-6) * 60.0 if tested > 0 else 0.0
            detail_bits = [f"elapsed {_fmt_duration(elapsed)}"]
            if tested > 0:
                detail_bits.append(f"{rate:.0f} tested/min")
            if infra_n:
                detail_bits.append(f"infra {infra_n}")
            if tradeable_n:
                detail_bits.append(f"tradeable {tradeable_n}")
            if job.message:
                detail_bits.insert(0, job.message.strip()[:80])
            st.caption(" · ".join(detail_bits))

            _render_level_bar(tradeable_n or db_total or tested, level_passes,
                              highlight_active=True)

        if job.error:
            with st.expander("Issue"):
                st.code(job.error)


def _estimate_eta(
    tested: int,
    survivors: int,
    max_candidates: int,
    target_survivors: int,
    elapsed: float,
) -> str:
    etas: list[float] = []
    if max_candidates and tested > 0 and elapsed > 0:
        rate = tested / elapsed
        if rate > 1e-9:
            etas.append(max(max_candidates - tested, 0) / rate)
    if target_survivors and survivors > 0 and elapsed > 0 and target_survivors < 10**8:
        rate = survivors / elapsed
        if rate > 1e-9:
            etas.append(max(target_survivors - survivors, 0) / rate)
    if not etas:
        return "estimating…"
    best = min(etas)
    # Cap absurd extrapolations (sleep gaps / stalled counters).
    if best > 48 * 3600:
        return "slow — see rate"
    return "~" + _fmt_duration(best)


def _render_runs_table(storage: Storage, jobs: list) -> None:
    job_ids = [j.id for j in jobs]
    progress = storage.run_progress_by_jobs(job_ids)

    st.markdown("#### All runs")
    filter_col, sort_col = st.columns([2, 1], vertical_alignment="bottom")
    with filter_col:
        status_filter = st.multiselect(
            "Status",
            options=["Running", "Queued", "Done", "Failed", "Cancelled"],
            default=[],
            key="runs_status_filter",
            help="Leave empty to show every run.",
        )
    with sort_col:
        sort_by = st.selectbox(
            "Sort",
            ["Newest first", "Oldest first", "Most tested", "Most L1 passes"],
            key="runs_sort",
        )

    filtered = _apply_filters(jobs, status_filter)
    filtered = _apply_sort(filtered, progress, sort_by)

    if not filtered:
        st.info("No runs match the current filter.")
        return

    st.markdown(_runs_table_html(filtered, progress), unsafe_allow_html=True)

    st.caption(
        f"Showing {len(filtered)} of {len(jobs)} runs · "
        "Click a run below to inspect its strategies."
    )

    run_options = {j.id: j for j in filtered}
    default_id = st.session_state.get("runs_selected_id")
    if default_id not in run_options:
        default_id = filtered[0].id

    pick_l, pick_r = st.columns([3, 1], vertical_alignment="bottom")
    with pick_l:
        selected = st.selectbox(
            "Inspect run",
            options=list(run_options.keys()),
            index=list(run_options.keys()).index(default_id),
            format_func=lambda rid: _run_option_label(rid, run_options[rid], progress),
            key="runs_selected_id",
        )
    with pick_r:
        stats = progress.get(selected, {})
        st.metric("Total tested", f"{stats.get('total', 0):,}")

    render_run_detail(storage, run_options[selected])


def _apply_filters(jobs: list, status_filter: list[str]) -> list:
    if not status_filter:
        return list(jobs)
    label_map = {
        JobStatus.RUNNING: "Running",
        JobStatus.PENDING: "Queued",
        JobStatus.DONE: "Done",
        JobStatus.FAILED: "Failed",
        JobStatus.CANCELLED: "Cancelled",
    }
    allowed = set(status_filter)
    return [j for j in jobs if label_map.get(j.status, j.status.value) in allowed]


def _apply_sort(jobs: list, progress: dict, sort_by: str) -> list:
    out = list(jobs)
    if sort_by == "Oldest first":
        out.sort(key=lambda j: j.created_at)
    elif sort_by == "Most tested":
        out.sort(
            key=lambda j: progress.get(j.id, {}).get("total", 0),
            reverse=True,
        )
    elif sort_by == "Most L1 passes":
        out.sort(
            key=lambda j: (progress.get(j.id, {}).get("level_passes") or {}).get(1, 0),
            reverse=True,
        )
    else:
        out.sort(key=lambda j: j.created_at, reverse=True)
    return out


def _run_option_label(rid: str, job, progress: dict) -> str:
    stats = progress.get(rid, {})
    total = stats.get("total", 0)
    l1 = (stats.get("level_passes") or {}).get(1, 0)
    sym = job.payload.get("symbol", "?")
    tf = job.payload.get("timeframe", "?")
    return f"{rid} · {job.status.value} · {sym}/{tf} · {total:,} tested · L1+ {l1:,}"


def _runs_table_html(jobs: list, progress: dict) -> str:
    """Scrollable HTML table — level columns L1 through L16."""
    headers = (
        ["Run", "Status", "Market", "Started", "Tests", "Pass"]
        + [f"L{lvl}" for lvl in range(1, _MAX_LEVEL + 1)]
    )
    head_cells = "".join(f"<th>{html_lib.escape(h)}</th>" for h in headers)
    rows_html = []
    for job in jobs:
        stats = progress.get(job.id, {})
        total = int(stats.get("total", 0) or 0)
        passed = int(stats.get("passed", 0) or 0)
        level_passes = stats.get("level_passes") or {}
        if job.status in _ACTIVE:
            total = max(total, int(getattr(job, "tested", 0) or 0))
            passed = max(passed, int(getattr(job, "survivors", 0) or 0))

        label, tone = _STATUS_BADGE.get(job.status, (job.status.value, "gray"))
        fg, bg, border = theme.chip_style(tone)
        status_cell = (
            f'<span class="ea-run-status" style="color:{fg};background:{bg};'
            f'border:1px solid {border}">{html_lib.escape(label)}</span>'
        )
        sym = html_lib.escape(str(job.payload.get("symbol", "?")))
        tf = html_lib.escape(str(job.payload.get("timeframe", "?")))
        eng = html_lib.escape(str(job.payload.get("engine", "?")))
        when = datetime.fromtimestamp(job.created_at).strftime("%m/%d %H:%M")
        rid = html_lib.escape(job.id)

        cells = [
            f'<td class="ea-run-id"><code>{rid}</code></td>',
            f"<td>{status_cell}</td>",
            f"<td>{sym} · {tf} · {eng}</td>",
            f'<td class="ea-run-num">{when}</td>',
            f'<td class="ea-run-num">{total:,}</td>',
            f'<td class="ea-run-num ea-run-pass">{passed:,}</td>',
        ]
        for lvl in range(1, _MAX_LEVEL + 1):
            n = int(level_passes.get(lvl, 0) or 0)
            cls = "ea-run-lvl"
            if n > 0 and lvl in validation_levels.KPI_ANCHORS:
                cls += " ea-run-lvl-anchor"
            cells.append(f'<td class="{cls}">{n:,}</td>')

        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    return (
        '<div class="ea-runs-table-wrap"><table class="ea-runs-table">'
        f"<thead><tr>{head_cells}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table></div>"
    )


def _render_level_bar(
    total: int,
    level_passes: dict,
    *,
    highlight_active: bool = False,
) -> None:
    """Compact L1–L16 pass counts as a labeled bar row."""
    if total <= 0 and not any(level_passes.values()):
        st.caption("No validation results yet.")
        return

    parts = []
    for lvl in range(1, _MAX_LEVEL + 1):
        n = int(level_passes.get(lvl, 0) or 0)
        anchor = lvl in validation_levels.KPI_ANCHORS
        cls = "ea-lvl-chip anchor" if anchor else "ea-lvl-chip"
        if highlight_active and n > 0:
            cls += " live"
        parts.append(
            f'<span class="{cls}" title="≥ L{lvl}">'
            f"L{lvl}<b>{n:,}</b></span>"
        )
    st.markdown(
        f'<div class="ea-level-bar">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
