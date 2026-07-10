"""MQL5 EA Factory & Curation Dashboard — Streamlit entry point."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config import settings
from factory.storage import Storage
from jobs import worker

st.set_page_config(
    page_title="MQL5 EA Factory",
    page_icon=":material/precision_manufacturing:",
    layout="wide",
)

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] {
        height: 0 !important;
        min-height: 0 !important;
        background: transparent !important;
        border: none !important;
    }
    header[data-testid="stHeader"] [data-testid="stToolbar"] {
        display: none !important;
    }
    .ea-page-title {
        margin: 0 0 0.2rem;
        font-size: 1.65rem;
        font-weight: 700;
        color: #E8EAED;
        letter-spacing: -0.02em;
    }
    .ea-page-sub {
        margin: 0 0 1.25rem;
        font-size: 0.9rem;
        color: #8B95A5;
        line-height: 1.55;
        max-width: 46rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_NAV_OPTIONS = (
    ":material/radar: Discovery",
    ":material/grid_view: Strategy gallery",
    ":material/download: Export",
)


@st.cache_resource
def get_queue() -> worker.JobQueue:
    """Process-wide singleton job queue — reruns never spawn duplicates."""
    return worker.get_job_queue()


@st.cache_resource
def get_storage() -> Storage:
    settings.ensure_dirs()
    return Storage()


def main() -> None:
    queue = get_queue()
    storage = get_storage()

    st.markdown(
        '<p class="ea-page-title">'
        ':material/precision_manufacturing: MQL5 EA Factory</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="ea-page-sub">Generate, backtest, validate, curate, and export '
        "MetaTrader 5 Expert Advisors. The simulator is a pre-filter — final "
        "verification belongs in the real MT5 Strategy Tester.</p>",
        unsafe_allow_html=True,
    )

    # Segmented control (not st.tabs): hidden tabs still execute their bodies
    # on every rerun, so Gallery/Export would deserialize the full validation
    # library whenever Discovery start/stop triggered a rerun.
    view = st.segmented_control(
        "Section",
        options=list(_NAV_OPTIONS),
        default=_NAV_OPTIONS[0],
        key="main_nav_section",
        label_visibility="collapsed",
    )

    if view == _NAV_OPTIONS[0] or view is None:
        from app.components.discovery_panel import render_discovery_panel
        render_discovery_panel(queue, storage)
    elif view == _NAV_OPTIONS[1]:
        from app.components.strategy_card import render_gallery
        render_gallery(storage)
    elif view == _NAV_OPTIONS[2]:
        from app.components.export_panel import render_export_panel
        render_export_panel(storage)


if __name__ == "__main__":
    main()
