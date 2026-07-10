"""Checkbox selection -> Export Marketplace Package action."""
from __future__ import annotations

import streamlit as st

from config import settings
from factory.assets.exporter import export_marketplace_package
from factory.storage import Storage


def _selected_strategy_ids() -> list[str]:
    """Collect gallery checkbox selections from session state (no DB load)."""
    ids: list[str] = []
    for key, value in st.session_state.items():
        if not value:
            continue
        if key.startswith("select_") and not key.startswith("select_gallery"):
            # Gallery keys are ``select_{strategy_id}`` (see strategy_card).
            sid = key[len("select_"):]
            if sid:
                ids.append(sid)
    return ids


def render_export_panel(storage: Storage) -> None:
    st.subheader("Export Marketplace Packages")
    st.caption(
        "Each package bundles the standalone validator-proof .mq5, the "
        "optimized .set file (every parameter with its optimization range), "
        "and the marketplace .md description into "
        f"`{settings.OUTPUT_DIR}`.")

    selected_ids = _selected_strategy_ids()

    if not selected_ids:
        st.info("Select strategies in the Strategy Gallery tab first.")
        return

    st.write(f"**{len(selected_ids)}** strategies selected.")
    if st.button("Export Marketplace Package(s)", type="primary"):
        exported = []
        errors = []
        for sid in selected_ids:
            strategy = storage.get_strategy(sid)
            report = storage.get_validation(sid)
            if strategy is None or report is None:
                errors.append(f"{sid}: missing strategy or validation record")
                continue
            try:
                out_dir = export_marketplace_package(strategy, report)
                exported.append(out_dir)
            except Exception as exc:
                errors.append(f"{strategy.name}: {exc}")
        for path in exported:
            st.success(f"Exported `{path}`")
        for err in errors:
            st.error(err)
