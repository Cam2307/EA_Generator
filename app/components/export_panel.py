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


_CHECK_LABELS = {
    "validation_passed": "Validation gates",
    "real_data": "Real data",
    "oos_trades": "OOS sample size",
    "dsr": "Deflated Sharpe",
    "wfe": "Walk-forward efficiency",
    "montecarlo": "Monte Carlo robustness",
    "regimes": "Multi-regime edge",
    "uncorrelated": "Unique return stream",
    "holdout": "Untouched holdout",
}


def _render_publication_readiness(storage: Storage,
                                  selected_ids: list[str]) -> None:
    """Publication-tier checklist per selected strategy (see
    factory/publication.py). Exporting a package never requires these; a
    marketplace *publication* should."""
    from factory.holdout import evaluate_holdout, factory_hit_rate
    from factory.publication import evaluate_publication, publish

    theme.section(
        "Publication readiness",
        "The marketplace bar — far stricter than the discovery gates.")
    stats = factory_hit_rate(storage)
    if stats["evaluated"]:
        st.caption(
            f"Factory holdout hit rate: **{stats['hit_rate']:.0%}** over "
            f"{stats['evaluated']:.0f} one-shot evaluations.")

    for sid in selected_ids:
        strategy = storage.get_strategy(sid)
        if strategy is None:
            continue
        decision = evaluate_publication(storage, sid)
        n_ok = sum(1 for v in decision.checks.values() if v)
        icon = "✅" if decision.ready else "🚧"
        with st.expander(
                f"{icon} {strategy.name} — {n_ok}/{len(decision.checks)} "
                "checks", expanded=not decision.ready):
            chips = [
                theme.chip(_CHECK_LABELS.get(name, name),
                           "teal" if ok else "red")
                for name, ok in decision.checks.items()]
            st.markdown(" ".join(chips), unsafe_allow_html=True)
            for why in decision.reasons:
                st.caption(f":material/close: {why}")
            for warn in decision.warnings:
                st.warning(warn, icon=":material/warning:")

            btn_l, btn_r = st.columns(2)
            with btn_l:
                if not decision.checks.get("holdout", True):
                    if st.button("Run one-shot holdout", key=f"holdout_{sid}",
                                 help="Scores the strategy ONCE on the "
                                      "reserved recent window. This cannot "
                                      "be re-rolled — that's the point."):
                        res = evaluate_holdout(storage, sid)
                        (st.success if res.passed else st.error)(
                            f"Holdout net {res.net_profit:,.0f} · "
                            f"{res.trade_count} trades · "
                            f"DD {res.max_dd_pct:.1f}%"
                            + (f" · {res.error}" if res.error else ""))
                        st.rerun()
            with btn_r:
                if decision.ready and st.button(
                        "Publish", key=f"publish_{sid}", type="primary"):
                    record = publish(storage, sid)
                    st.success(f"Published v{record['version']} -> "
                               f"{record['package_dir']}")


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

    _render_publication_readiness(storage, selected_ids)

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
