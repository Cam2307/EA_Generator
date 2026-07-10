"""MQL5 EA Factory & Curation Dashboard — Streamlit entry point."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components import theme
from config import settings
from factory.storage import Storage
from jobs import worker

st.set_page_config(
    page_title="MQL5 EA Factory",
    page_icon=":material/precision_manufacturing:",
    layout="wide",
)

theme.inject_global_css()

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


def _status_chips(storage: Storage) -> list[str]:
    """Live agent + library status chips for the hero header."""
    chips: list[str] = []
    try:
        state = storage.get_agent_state()
        if state.get("status") == "running" or state.get("enabled"):
            chips.append(theme.chip("Discovery agent active", "teal"))
        else:
            chips.append(theme.chip("Agent idle", "gray"))
        active = [j for j in storage.list_jobs("discovery")
                  if j.status.value == "RUNNING"]
        if active:
            chips.append(theme.chip(
                f"{len(active)} job(s) in flight", "amber"))
    except Exception:
        pass
    return chips


def _kpi_strip(storage: Storage) -> None:
    """Factory-wide numbers, cheap single-row queries only."""
    try:
        total = storage.count_validated(passed_only=None)
        passing = storage.count_validated(passed_only=True)
        states = storage.promotion_state_counts()
        promoted = states.get("promoted_live_watchlist", 0)
        edge = states.get("edge_positive", 0)
        pass_rate = f"{passing / total:.0%}" if total else "—"
        theme.kpi_row(
            [
                ("Candidates validated", f"{total:,}", ""),
                ("Passing library", f"{passing:,}", "good" if passing else ""),
                ("Pass rate", pass_rate, ""),
                ("Edge positive", f"{edge:,}", "accent" if edge else ""),
                ("Promoted / watchlist", f"{promoted:,}",
                 "accent" if promoted else ""),
            ],
        )
    except Exception:
        pass


def main() -> None:
    queue = get_queue()
    storage = get_storage()

    theme.hero(
        "MQL5 EA Factory",
        "Generate, backtest, validate, curate, and export MetaTrader 5 "
        "Expert Advisors. The simulator is a pre-filter — final verification "
        "belongs in the real MT5 Strategy Tester.",
        chips=_status_chips(storage),
    )
    _kpi_strip(storage)

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
