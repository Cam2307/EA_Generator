"""Shared run metadata helpers and per-run result browsing for the gallery."""
from __future__ import annotations

import streamlit as st

from factory import validation_levels
from factory.models import JobStatus
from factory.storage import Storage

_ACTIVE = (JobStatus.PENDING, JobStatus.RUNNING)

# Sorts that only need table columns (no json_extract over validation bodies).
_COLUMN_SORTS = frozenset({
    "WFE (high → low)",
    "Run (newest → oldest)",
    "Run (oldest → newest)",
})

_STATUS_STYLE = {
    JobStatus.RUNNING: (":orange-badge[:material/bolt: Running]", "orange"),
    JobStatus.PENDING: (":gray-badge[:material/schedule: Queued]", "gray"),
    JobStatus.DONE: (":green-badge[:material/check_circle: Done]", "green"),
    JobStatus.CANCELLED: (":orange-badge[:material/stop_circle: Cancelled]", "orange"),
    JobStatus.FAILED: (":red-badge[:material/error: Failed]", "red"),
}

# Min-level filter options: 0 = entire population, 1..MAX = cleared at least Ln.
_MIN_LEVEL_OPTIONS = [0] + list(range(
    validation_levels.MIN_LEVEL, validation_levels.MAX_LEVEL + 1))


def level_badge(highest: int) -> str:
    """Streamlit badge for a population tier (empty when uncleared)."""
    lvl = int(highest or 0)
    if lvl <= 0:
        return ":gray-badge[L0 none]"
    color = validation_levels.badge_color_for_level(lvl)
    label = validation_levels.display_label(lvl)
    return f":{color}-badge[{label}]"


def format_tier_caption(level_hist: dict[int, int]) -> str:
    """Compact population histogram, e.g. ``L1:12 · L2:4 · L3:1``."""
    parts = []
    for lvl in range(validation_levels.MAX_LEVEL, 0, -1):
        n = int(level_hist.get(lvl, 0) or 0)
        if n:
            parts.append(f"L{lvl}:{n}")
    none = int(level_hist.get(0, 0) or 0)
    if none:
        parts.append(f"L0:{none}")
    return " · ".join(parts) if parts else "no tiers yet"


def render_min_level_filter(
    key: str,
    *,
    page_key: str | None = None,
    default: int | None = None,
    label: str = "Min level",
) -> int | None:
    """Population filter: show strategies that cleared at least this level.

    Returns ``None`` when "All" (0) is selected — no min_level SQL filter.
    Defaults to Standard·A so soft Screener clears stay out of the default view.
    """
    if default is None:
        default = validation_levels.GALLERY_DEFAULT_MIN_LEVEL

    def _on_change() -> None:
        if page_key:
            st.session_state[page_key] = 0

    labels = {
        0: "All (population)",
        **{
            lvl: f"≥ {validation_levels.display_label(lvl)}"
            for lvl in range(validation_levels.MIN_LEVEL,
                             validation_levels.MAX_LEVEL + 1)
        },
    }
    choice = st.selectbox(
        label,
        options=_MIN_LEVEL_OPTIONS,
        index=_MIN_LEVEL_OPTIONS.index(default)
        if default in _MIN_LEVEL_OPTIONS else 1,
        format_func=lambda v: labels.get(v, str(v)),
        key=key,
        help=(
            "Filter the population by highest validation level cleared. "
            f"Each strategy is scored once against levels "
            f"{validation_levels.MIN_LEVEL}–{validation_levels.MAX_LEVEL}."
        ),
        on_change=_on_change if page_key else None,
    )
    return None if int(choice) <= 0 else int(choice)


def job_summary(job) -> str:
    floor = job.payload.get("validation_level_floor",
                            job.payload.get("validation_level"))
    ceiling = job.payload.get("validation_level_ceiling")
    if floor is not None and ceiling is not None and int(ceiling) != int(floor):
        gate = f"floor L{floor} · ceiling L{ceiling}"
        if job.payload.get("progressive_strictness"):
            gate += " (progressive)"
    elif floor is not None:
        gate = f"Level {floor}"
    else:
        gate = "custom gates"
    return (f"{job.payload.get('symbol', '?')} · "
            f"{job.payload.get('timeframe', '?')} · "
            f"{job.payload.get('engine', '?')} · {gate} · "
            f"data {job.payload.get('data_source', '?')} · "
            f"target {job.payload.get('target_survivors', '?')}")


def render_run_detail(storage: Storage, job) -> None:
    """Strategy list for one discovery run (used from Runs tab)."""
    from app.components.strategy_card import (
        SORT_OPTION_LABELS,
        _MODAL_PAGE_SIZE,
        _hydrate_reports,
        render_page_controls,
        render_report_grid,
        render_sort_selectbox,
        sort_reports,
        _page_slice,
    )

    page_key = f"run_detail_page_{job.id}"
    badge = _STATUS_STYLE.get(job.status, (job.status.value,))[0]
    progress = storage.run_progress_by_jobs([job.id]).get(job.id, {})
    total = int(progress.get("total", 0) or 0)
    passed = int(progress.get("passed", 0) or 0)
    level_passes = progress.get("level_passes") or {}

    with st.container(border=True):
        head_l, head_r = st.columns([4, 1], vertical_alignment="center")
        with head_l:
            st.markdown(f"**`{job.id}`**")
            st.caption(job_summary(job))
        with head_r:
            st.markdown(badge)

        if job.message:
            st.caption(job.message)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total tested", f"{total:,}")
        c2.metric("Passed floor", f"{passed:,}")
        l1 = int(level_passes.get(1, 0) or 0)
        l7 = int(level_passes.get(validation_levels.MC_UNLOCK_LEVEL, 0) or 0)
        c3.metric("L1+ / L7+", f"{l1:,} / {l7:,}")

        _render_run_level_summary(total, level_passes)

        if job.error:
            with st.expander("Run issue"):
                st.code(job.error)

    ctrl_l, ctrl_r = st.columns([3, 1], vertical_alignment="bottom")
    with ctrl_l:
        sort_by = render_sort_selectbox(
            f"run_detail_sort_{job.id}", page_key=page_key, index=0)
    with ctrl_r:
        min_level = render_min_level_filter(
            f"run_detail_level_{job.id}", page_key=page_key,
            default=validation_levels.GALLERY_DEFAULT_MIN_LEVEL)

    need_metrics = sort_by not in _COLUMN_SORTS
    if sort_by not in SORT_OPTION_LABELS:
        sort_by = SORT_OPTION_LABELS[0]
        need_metrics = False

    reports = storage.list_validation_summaries(
        passed_only=None,
        job_id=job.id,
        min_level=min_level,
        include_body_metrics=need_metrics,
    )
    if not reports and min_level:
        st.info(f"No strategies in this run cleared level {min_level}+. "
                "Lower the min level to see more of the population.")
        return
    if not reports:
        st.info("This run produced no evaluated strategies yet.")
        return

    runs = {job.id: job}
    shown = sort_reports(reports, sort_by, runs=runs)
    floor_n = sum(1 for r in shown if r.passed)

    st.caption(
        f"Showing {len(shown)}"
        + (f" (≥ L{min_level})" if min_level else " (full population)")
        + f" · :green[{floor_n} ≥ floor]")

    page_reports, page, total_pages = _page_slice(
        shown, page_key, _MODAL_PAGE_SIZE)
    render_page_controls(
        page_key, page, total_pages, len(shown), _MODAL_PAGE_SIZE,
        control_key=f"run_detail_nav_{job.id}")
    full_page = _hydrate_reports(storage, page_reports)
    render_report_grid(
        full_page, {}, storage=storage,
        key_prefix=f"run_detail_{job.id}_", compact=True, runs=runs)


def _render_run_level_summary(total: int, level_passes: dict) -> None:
    """One-line L1–L16 pass counts for a single run."""
    if total <= 0:
        return
    parts = []
    for lvl in range(1, validation_levels.MAX_LEVEL + 1):
        n = int(level_passes.get(lvl, 0) or 0)
        if n or lvl in validation_levels.KPI_ANCHORS:
            band = validation_levels.band_for_level(lvl) if lvl in validation_levels.KPI_ANCHORS else ""
            label = f"L{lvl}" + (f" ({band})" if band else "")
            parts.append(f"**{label}:** {n:,}")
    if parts:
        st.caption(" · ".join(parts))


def render_per_run_section(storage: Storage) -> None:
    """Browse finished discovery runs — delegates to :func:`render_run_detail`."""
    finished = [j for j in storage.list_jobs("discovery")
                if j.status not in _ACTIVE]
    if not finished:
        st.info("No finished runs yet. Start a discovery batch on the "
                "Runs or Discovery tab — results appear when a run completes.")
        return

    st.markdown("#### :material/history: Results per run")
    progress = storage.run_progress_by_jobs([j.id for j in finished])
    run_options = {j.id: j for j in finished}
    labels = {
        jid: _run_select_label(jid, job, progress.get(jid, {}))
        for jid, job in run_options.items()
    }

    selected = st.selectbox(
        "Discovery run",
        options=list(run_options.keys()),
        format_func=lambda rid: labels.get(rid, rid),
        key="gallery_selected_run",
    )
    render_run_detail(storage, run_options[selected])


def _run_select_label(rid: str, job, stats: dict) -> str:
    total = int(stats.get("total", 0) or 0)
    passed = int(stats.get("passed", 0) or 0)
    l1 = int((stats.get("level_passes") or {}).get(1, 0) or 0)
    return (
        f"{rid} · {job.status.value} · {total:,} tested · "
        f"{passed:,} pass · L1+ {l1:,}"
    )
