"""Shared run metadata helpers and per-run result browsing for the gallery."""
from __future__ import annotations

import streamlit as st

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


def job_summary(job) -> str:
    lvl = job.payload.get("validation_level")
    gate = f"Level {lvl}" if lvl is not None else "custom gates"
    return (f"{job.payload.get('symbol', '?')} · "
            f"{job.payload.get('timeframe', '?')} · "
            f"{job.payload.get('engine', '?')} · {gate} · "
            f"data {job.payload.get('data_source', '?')} · "
            f"target {job.payload.get('target_survivors', '?')}")


def render_per_run_section(storage: Storage) -> None:
    """Browse finished discovery runs and inspect each run's strategies inline."""
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

    finished = [j for j in storage.list_jobs("discovery")
                if j.status not in _ACTIVE]
    if not finished:
        st.info("No finished runs yet. Start a discovery batch on the "
                "Discovery tab — results appear here when a run completes.")
        return

    st.markdown("#### :material/history: Results per run")
    st.caption(
        "Pick a discovery run to inspect every strategy it evaluated. "
        "Use the **Library** view above for cross-run filtering and export.")

    run_options = {j.id: j for j in finished}
    page_key = "gallery_run_page"

    # One GROUP BY for all finished runs — never N×COUNT in selectbox format_func.
    counts = storage.count_validated_by_jobs(list(run_options.keys()))
    labels = {
        jid: (f"{jid} · {job.status.value} · "
              f"{counts.get(jid, (0, 0))[0]}/{counts.get(jid, (0, 0))[1]} passed")
        for jid, job in run_options.items()
    }

    pick_l, pick_r = st.columns([3, 1], vertical_alignment="bottom")
    with pick_l:
        selected = st.selectbox(
            "Discovery run",
            options=list(run_options.keys()),
            format_func=lambda rid: labels.get(rid, rid),
            key="gallery_selected_run",
            help="Each option shows pass count, status, and run summary.",
        )
    with pick_r:
        only_passed = st.toggle(
            "Passed only",
            value=False,
            key="gallery_run_passed_only",
            on_change=_reset_gallery_run_page,
            args=(page_key,),
        )

    job = run_options[selected]
    passed, total = counts.get(job.id, (0, 0))
    badge = _STATUS_STYLE.get(job.status, (job.status.value,))[0]

    with st.container(border=True):
        head_l, head_r = st.columns([4, 1], vertical_alignment="center")
        with head_l:
            st.markdown(f"**`{job.id}`**")
            st.caption(job_summary(job))
        with head_r:
            st.markdown(badge)

        if job.message:
            st.caption(job.message)
        st.caption(f":green[{passed} passed] · {total} evaluated")
        if job.error:
            with st.expander("Run issue"):
                st.code(job.error)

    # Resolve sort before listing so column-only sorts skip body json_extract.
    sort_by = render_sort_selectbox(
        "gallery_run_sort", page_key=page_key, index=0)
    need_metrics = sort_by not in _COLUMN_SORTS
    # Session may still hold a prior choice before the widget returns.
    if sort_by not in SORT_OPTION_LABELS:
        sort_by = SORT_OPTION_LABELS[0]
        need_metrics = False

    reports = storage.list_validation_summaries(
        passed_only=None,
        job_id=job.id,
        include_body_metrics=need_metrics,
    )
    if not reports:
        st.info("This run produced no evaluated strategies.")
        return

    passed_reports = [r for r in reports if r.passed]
    runs = {job.id: job}

    shown = ([r for r in reports if r.passed] if only_passed else reports)
    shown = sort_reports(shown, sort_by, runs=runs)

    st.caption(
        f":green[{len(passed_reports)} passed] · "
        f"{len(reports) - len(passed_reports)} failed · "
        f"{len(reports)} evaluated · showing {len(shown)}")

    page_reports, page, total_pages = _page_slice(
        shown, page_key, _MODAL_PAGE_SIZE)
    render_page_controls(
        page_key, page, total_pages, len(shown), _MODAL_PAGE_SIZE,
        control_key=f"gallery_run_nav_{job.id}")
    full_page = _hydrate_reports(storage, page_reports)
    render_report_grid(
        full_page, {}, storage=storage,
        key_prefix=f"gallery_run_{job.id}_", compact=True, runs=runs)


def _reset_gallery_run_page(page_key: str) -> None:
    st.session_state[page_key] = 0
