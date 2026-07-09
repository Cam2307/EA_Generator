"""Checkbox selection -> Export Marketplace Package action."""
from __future__ import annotations

import streamlit as st

from config import settings
from factory.assets.exporter import export_marketplace_package
from factory.storage import Storage


def render_export_panel(storage: Storage) -> None:
    st.subheader("Export Marketplace Packages")
    st.caption(
        "Each package bundles the standalone validator-proof .mq5, the "
        "optimized .set file (every parameter with its optimization range), "
        "and the marketplace .md description into "
        f"`{settings.OUTPUT_DIR}`.")

    reports = storage.list_validated(passed_only=False)
    selected_ids = [r.strategy_id for r in reports
                    if st.session_state.get(f"select_{r.strategy_id}")]

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
