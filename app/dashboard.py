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

# toolbarMode=minimal hides Deploy / Stop / menu; collapse the leftover bar.
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
    </style>
    """,
    unsafe_allow_html=True,
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

    st.title("MQL5 EA Factory & Curation Dashboard")
    st.caption(
        "Generate, backtest, validate, curate, and export MetaTrader 5 "
        "Expert Advisors. The simulator engine is a pre-filter; final "
        "verification always belongs to the real MT5 Strategy Tester."
    )

    from app.components.discovery_panel import render_discovery_panel
    from app.components.export_panel import render_export_panel
    from app.components.strategy_card import render_gallery

    tab_discover, tab_gallery, tab_export = st.tabs([
        ":material/search: Discovery",
        ":material/grid_view: Strategy gallery",
        ":material/download: Export",
    ])

    with tab_discover:
        render_discovery_panel(queue, storage)
    with tab_gallery:
        render_gallery(storage)
    with tab_export:
        render_export_panel(storage)


# Guard so process-pool workers (spawned for parallel backtests) never re-run
# the dashboard: a multiprocessing child imports this module as "__mp_main__",
# while Streamlit runs it as "__main__".
if __name__ == "__main__":
    main()
