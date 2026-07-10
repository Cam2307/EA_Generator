"""Marketplace-package builder: selection review -> export action."""
from __future__ import annotations

import streamlit as st

from app.components import theme
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


_PACKAGE_PARTS = (
    ("description", ".mq5 source",
     "Standalone, validator-proof Expert Advisor assembled from hardened "
     "templates — Market metadata, volume/margin preflight, stops/freeze "
     "handling, netting/hedging branches."),
    ("tune", ".set parameters",
     "Every optimized parameter with its full optimization range in the "
     "native Value||Start||Step||Stop||Y format — drives an MT5 Strategy "
     "Tester optimization directly."),
    ("article", ".md description",
     "Marketplace-ready listing text generated from the validation report: "
     "metrics, walk-forward summary, honest limitations."),
)


def render_export_panel(storage: Storage) -> None:
    theme.section(
        "Export Marketplace Packages",
        f"Bundles land in `{settings.OUTPUT_DIR}` — one folder per strategy.")

    cols = st.columns(3)
    for col, (_icon, title, desc) in zip(cols, _PACKAGE_PARTS):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(desc)

    selected_ids = _selected_strategy_ids()
    st.write("")

    if not selected_ids:
        st.info("Nothing selected yet — tick strategies in the **Strategy "
                "gallery** tab, then come back here to export them.")
        return

    # Review list: names + key numbers before the user commits.
    theme.section("Selected for export", f"{len(selected_ids)} strategies")
    reports = storage.get_validations(selected_ids)
    rows = []
    for sid in selected_ids:
        strategy = storage.get_strategy(sid)
        rep = reports.get(sid)
        if strategy is None:
            rows.append({"Strategy": sid, "Symbol": "?", "Status": "missing"})
            continue
        oos = rep.oos_metrics if rep else None
        rows.append({
            "Strategy": strategy.name,
            "Symbol": f"{strategy.symbol} {strategy.timeframe}",
            "Mechanic": strategy.mechanic.type.value,
            "OOS profit": f"{oos.net_profit:,.0f}" if oos else "—",
            "PF": f"{oos.profit_factor:.2f}" if oos else "—",
            "WFE": f"{rep.wfe:.2f}" if rep else "—",
            "State": rep.promotion_state if rep else "—",
        })
    st.dataframe(rows, width="stretch", hide_index=True)

    if st.button(f"Export {len(selected_ids)} Marketplace Package(s)",
                 type="primary", width="stretch"):
        exported = []
        errors = []
        progress = st.progress(0.0, text="Exporting…")
        for i, sid in enumerate(selected_ids):
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
            progress.progress((i + 1) / len(selected_ids),
                              text=f"Exporting… {i + 1}/{len(selected_ids)}")
        progress.empty()
        if exported:
            st.success(f"Exported {len(exported)} package(s):")
            for path in exported:
                st.code(str(path), language=None)
        for err in errors:
            st.error(err)
