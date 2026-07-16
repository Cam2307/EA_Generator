"""Unified discovery terminal — agent + single-run in one panel."""
from __future__ import annotations

# Bump when discovery UI changes — visible in the panel header for debugging.
DISCOVERY_UI_VERSION = "2026-07-16-edge-first"

import time
import uuid
from dataclasses import replace
from datetime import datetime

import streamlit as st

from app.components.discovery_styles import inject_discovery_styles
from config import settings
from factory import alerts as alerts_mod
from factory import data as data_mod
from factory import validation_levels
from factory.discovery_config import (
    DiscoverySettings,
    build_discovery_payload,
    derive_wfo_from_duration,
    history_start_end,
    settings_from_app,
    settings_to_app,
)
from factory.backtest.simulator import SymbolSpec
from factory.metrics_display import data_source_badge, data_source_label
from factory.models import ExecutionMechanicType, JobStatus
from factory.storage import Storage
from jobs.orchestrator import (
    recover_stuck_starting_agent,
    start_orchestrator_process,
    stop_orchestrator_process,
    sync_agent_with_orchestrator_lock,
)
from jobs.sweep import plan_sweeps
from jobs.worker import JobQueue

send_email = alerts_mod.send_email
smtp_diagnostics = getattr(
    alerts_mod, "smtp_diagnostics", lambda: type("Diag", (), {"configured": False})()
)
smtp_missing_message = getattr(
    alerts_mod,
    "smtp_missing_message",
    lambda _diag: "SMTP diagnostics unavailable in this build.",
)

_ACTIVE = (JobStatus.PENDING, JobStatus.RUNNING)
_RUN_MODES = ("Continuous agent", "Single run")

# Phase 4 discovery presets — pre-fill only; submit still uses the form.
_PRESET_QUICK = "quick"
_PRESET_OVERNIGHT = "overnight"
_PRESET_PUBLISH = "publish"
_PRESET_LABELS = {
    _PRESET_QUICK: "Quick explore",
    _PRESET_OVERNIGHT: "Overnight search",
    _PRESET_PUBLISH: "Publish-grade",
}
_TF_OPTIONS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

_MECHANIC_LABELS = {
    ExecutionMechanicType.STANDARD_SLTP: "Standard SL/TP",
    ExecutionMechanicType.DCA_GRID: "DCA / Grid",
    ExecutionMechanicType.HEDGE_LAYER: "Hedging",
    ExecutionMechanicType.PARTIAL_CLOSE: "Partial close",
}

_TM_FEATURE_LABELS = {
    "adaptive_sl": "Adaptive (ATR) stop loss",
    "risk_reward_tp": "Risk-reward take profit",
    "trailing": "Trailing stops (fixed / ATR / chandelier)",
    "breakeven": "Auto-breakeven",
    "risk_sizing": "Risk-% position sizing",
    "time_filter": "Session / time-of-day filter",
    "safeguards": "Daily loss limit + max trades/day",
    "cooldown": "Cooldown after a loss",
    "regime_filter": "ADX/ATR regime filter",
    "regime_sizing": "ADX/ATR regime lot sizing",
    "hmm_regime_filter": "HMM regime filter (2-state)",
    "hmm_regime_sizing": "HMM regime lot sizing",
}


def _quick_explore_symbols(base: DiscoverySettings) -> list[str]:
    """1–2 symbols: keep the first selected symbol(s), else EURUSD."""
    available = {str(s).strip().upper() for s in settings.SYMBOLS}
    picked: list[str] = []
    for sym in base.symbols or []:
        s = str(sym).strip().upper()
        if s in available and s not in picked:
            picked.append(s)
        if len(picked) >= 2:
            break
    if not picked:
        fallback = str(settings.DEFAULT_SYMBOL).strip().upper()
        picked = [fallback if fallback in available else "EURUSD"]
    return picked


def _preset_config(
    preset_id: str, base: DiscoverySettings
) -> tuple[DiscoverySettings, str]:
    """Return DiscoverySettings + run-mode label for a named preset."""
    if preset_id == _PRESET_QUICK:
        cfg = replace(
            base,
            symbols=_quick_explore_symbols(base),
            timeframes=["M15"],
            months=3,
            engine="simulator",
            validation_level=1,
            progressive_strictness=False,
            progressive_step=validation_levels.DEFAULT_PROGRESSIVE_STEP,
            batch_size=32,
            max_candidates=150,
            target_survivors=2,
            genetic=True,
        )
        cfg.sync_wfo_from_duration()
        return cfg, "Single run"

    if preset_id == _PRESET_OVERNIGHT:
        cfg = replace(
            base,
            symbols=list(settings.SYMBOLS),
            timeframes=["M15", "H1", "H4"],
            months=12,
            engine="simulator",
            validation_level=7,  # Standard·A (legacy L3 band anchor)
            progressive_strictness=True,
            validation_level_start=validation_levels.MIN_LEVEL,
            progressive_step=validation_levels.DEFAULT_PROGRESSIVE_STEP,
            batch_size=96,
            max_candidates=2_000,
            target_survivors=8,
            genetic=True,
            recipient_email=(
                str(base.recipient_email).strip()
                or settings.DEFAULT_ALERT_RECIPIENT
            ),
            alert_min_score=float(
                base.alert_min_score or settings.DEFAULT_ALERT_MIN_SCORE
            ),
            progress_email_hours=float(
                base.progress_email_hours
                or settings.DEFAULT_PROGRESS_EMAIL_HOURS
            ),
        )
        cfg.sync_wfo_from_duration()
        return cfg, "Continuous agent"

    if preset_id == _PRESET_PUBLISH:
        # Longer history → more WFO holdout folds; MT5 + high fixed ceiling.
        cfg = replace(
            base,
            symbols=_quick_explore_symbols(base),
            timeframes=["M15", "H1"],
            months=18,
            engine="mt5",
            validation_level=10,  # Robust·A (legacy L4 band anchor)
            progressive_strictness=False,
            progressive_step=validation_levels.DEFAULT_PROGRESSIVE_STEP,
            batch_size=64,
            max_candidates=1_000,
            target_survivors=5,
            genetic=True,
        )
        cfg.sync_wfo_from_duration()
        return cfg, "Single run"

    raise ValueError(f"Unknown discovery preset: {preset_id}")


def _apply_discovery_preset(
    preset_id: str, storage: Storage, base: DiscoverySettings
) -> None:
    """Persist preset into app settings + session state; remount the form."""
    cfg, run_mode = _preset_config(preset_id, base)
    storage.upsert_app_settings(settings_to_app(cfg))
    st.session_state["discovery_preset"] = preset_id
    st.session_state["discovery_run_mode"] = run_mode
    # Collapse Customize after a preset; user can expand to review / submit.
    st.session_state["discovery_customize_expanded"] = False
    st.session_state["discovery_form_nonce"] = (
        int(st.session_state.get("discovery_form_nonce", 0)) + 1
    )


def _render_discovery_presets(storage: Storage, saved: DiscoverySettings) -> None:
    st.markdown(
        '<p class="ea-section-label">Presets</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Apply a starting profile, then review and launch below. "
        "Presets only pre-fill settings — they do not start a run."
    )
    c1, c2, c3 = st.columns(3)
    active = st.session_state.get("discovery_preset")
    with c1:
        if st.button(
            ":material/bolt: Quick explore",
            key="discovery_preset_quick",
            width="stretch",
            type="primary" if active == _PRESET_QUICK else "secondary",
            help=(
                "1–2 symbols, M15, simulator, validation L1 Screener·A, short history, "
                "small search budget — single run."
            ),
        ):
            _apply_discovery_preset(_PRESET_QUICK, storage, saved)
            st.rerun()
    with c2:
        if st.button(
            ":material/nights_stay: Overnight search",
            key="discovery_preset_overnight",
            width="stretch",
            type="primary" if active == _PRESET_OVERNIGHT else "secondary",
            help=(
                "Broad symbols×timeframes, continuous agent, progressive "
                "strictness, alerts ready, larger budget — runs until Stop."
            ),
        ):
            _apply_discovery_preset(_PRESET_OVERNIGHT, storage, saved)
            st.rerun()
    with c3:
        if st.button(
            ":material/verified: Publish-grade",
            key="discovery_preset_publish",
            width="stretch",
            type="primary" if active == _PRESET_PUBLISH else "secondary",
            help=(
                "MT5 engine, validation L10 Robust·A, longer holdout-aware history — "
                "single run oriented."
            ),
        ):
            _apply_discovery_preset(_PRESET_PUBLISH, storage, saved)
            st.rerun()

    if active in _PRESET_LABELS:
        st.caption(
            f"Active preset: **{_PRESET_LABELS[active]}** — "
            "open **Customize** to review fields and start."
        )


def render_discovery_panel(queue: JobQueue, storage: Storage) -> None:
    inject_discovery_styles()
    saved = settings_from_app(storage.get_app_settings())
    agent_state = storage.get_agent_state()
    agent_running = bool(int(agent_state.get("enabled", 0) or 0))

    if agent_running and "discovery_run_mode" not in st.session_state:
        st.session_state.discovery_run_mode = "Continuous agent"

    st.session_state.setdefault("discovery_preset", None)
    st.session_state.setdefault("discovery_form_nonce", 0)
    if "discovery_customize_expanded" not in st.session_state:
        # Full form open until the user picks a preset.
        st.session_state.discovery_customize_expanded = (
            st.session_state.discovery_preset is None
        )

    st.markdown(
        f"""
        <div class="ea-terminal-shell">
          <div class="ea-terminal-head">
            <div>
              <p class="ea-terminal-title">Launch edge discovery</p>
              <p class="ea-terminal-sub">
                Find entry edges first (win rate vs R:R), then expand only
                validated edges into execution EA styles. Track progress on Runs.
              </p>
            </div>
            <span class="ea-terminal-chip">EA factory · UI {DISCOVERY_UI_VERSION}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _live_status(storage)

    from factory.discovery_config import DEFAULT_DISCOVERY_MECHANICS
    _mech_default = [
        m for m in _MECHANIC_LABELS if m.value in saved.mechanics
    ] or [
        m for m in _MECHANIC_LABELS
        if m.value in DEFAULT_DISCOVERY_MECHANICS
    ]
    _tm_default = [f for f in _TM_FEATURE_LABELS if f in saved.tm_features]
    if not _tm_default:
        _tm_default = list(_TM_FEATURE_LABELS.keys())
    _tf_options = list(_TF_OPTIONS)

    with st.container(border=True):
        _render_discovery_presets(storage, saved)

        mode = st.segmented_control(
            "Run mode",
            options=_RUN_MODES,
            key="discovery_run_mode",
            help=(
                "Continuous agent cycles symbol×timeframe sweeps until you stop. "
                "Single run executes once — one job for a single pair, or a "
                "batch pass for every selected combination."
            ),
        )
        is_agent = mode == "Continuous agent"

        # Open by default with no preset; collapse after applying one.
        # Remount via keyed container so `expanded=` takes effect after presets.
        if st.session_state.get("discovery_preset") is None:
            expand_customize = True
        else:
            expand_customize = bool(
                st.session_state.get("discovery_customize_expanded", False)
            )
        form_nonce = int(st.session_state.get("discovery_form_nonce", 0))

        with st.container(key=f"discovery_customize_wrap_{form_nonce}"):
            with st.expander(
                "Customize",
                expanded=expand_customize,
                icon=":material/tune:",
            ):
                with st.form(f"discovery_form_{form_nonce}"):
                    st.markdown(
                        '<p class="ea-section-label">Market scope</p>',
                        unsafe_allow_html=True,
                    )
                    s1, s2 = st.columns(2)
                    with s1:
                        symbols = st.multiselect(
                            "Symbols",
                            options=list(settings.SYMBOLS),
                            default=list(saved.symbols),
                            help="Every selected symbol is included in each sweep.",
                        )
                    with s2:
                        timeframes = st.multiselect(
                            "Timeframes",
                            options=_tf_options,
                            default=[
                                tf for tf in saved.timeframes if tf in _tf_options
                            ]
                            or ["M15", "H1"],
                            help="Every selected timeframe is included in each sweep.",
                        )

                    st.markdown(
                        '<p class="ea-section-label">Validation</p>',
                        unsafe_allow_html=True,
                    )
                    v1, v2 = st.columns([3, 2])
                    with v1:
                        level_label = (
                            "Survivor floor target (progressive ceiling)"
                            if is_agent
                            else "Survivor floor (minimum level to count as a pass)"
                        )
                        # Clamp saved level into the current fine range (legacy
                        # remaps happen in settings_from_app; this covers stale
                        # session values).
                        level_options = [
                            int(lv.level) for lv in validation_levels.VALIDATION_LEVELS
                        ]
                        slider_default = max(
                            validation_levels.MIN_LEVEL,
                            min(validation_levels.MAX_LEVEL,
                                int(saved.validation_level)),
                        )
                        if slider_default not in level_options:
                            slider_default = validation_levels.DEFAULT_LEVEL
                        # Stale widget state (e.g. after a schema remap) can
                        # leave a value that is no longer in options and crash
                        # select_slider — reset before rendering.
                        _slider_key = "discovery_level_slider"
                        if (
                            _slider_key in st.session_state
                            and st.session_state[_slider_key] not in level_options
                        ):
                            st.session_state[_slider_key] = slider_default
                        level_num = st.select_slider(
                            level_label,
                            options=level_options,
                            value=slider_default,
                            key=_slider_key,
                            format_func=lambda n: validation_levels.display_label(n),
                            help=(
                                "Each candidate is backtested once and scored against "
                                "all 16 levels. This dial sets the survivor floor "
                                "(what counts as a pass for the run). "
                                f"Monte Carlo unlocks at L{validation_levels.MC_UNLOCK_LEVEL}."
                                + (
                                    " With progressive strictness, the agent starts at L1 "
                                    "and raises the survivor floor toward this target "
                                    "over successive cycles."
                                    if is_agent
                                    else ""
                                )
                            ),
                        )
                    with v2:
                        if is_agent:
                            progressive_strictness = st.checkbox(
                                "Progressive strictness",
                                value=bool(saved.progressive_strictness),
                                help=(
                                    "Start each agent run at level "
                                    f"{validation_levels.MIN_LEVEL} and increase by "
                                    "the step below after every full pass through all "
                                    "symbol×timeframe combinations, up to the ceiling."
                                ),
                            )
                            progressive_step = st.number_input(
                                "Progressive step",
                                min_value=1,
                                max_value=max(1, validation_levels.MAX_LEVEL - 1),
                                value=max(
                                    1,
                                    int(getattr(
                                        saved,
                                        "progressive_step",
                                        validation_levels.DEFAULT_PROGRESSIVE_STEP,
                                    )),
                                ),
                                help=(
                                    "How many fine levels to climb after each full "
                                    f"symbol×timeframe cycle (default "
                                    f"{validation_levels.DEFAULT_PROGRESSIVE_STEP})."
                                ),
                            )
                        else:
                            progressive_strictness = False
                            progressive_step = validation_levels.DEFAULT_PROGRESSIVE_STEP
                    level = validation_levels.get_level(level_num)
                    mc_note = "Monte Carlo on" if level.montecarlo else "Monte Carlo off"
                    if is_agent and progressive_strictness and not saved.use_custom:
                        sweep_count = max(
                            1,
                            len(list(saved.symbols or settings.SYMBOLS[:2]))
                            * len(list(saved.timeframes or ["M15"])),
                        )
                        start_lvl = validation_levels.get_level(
                            validation_levels.MIN_LEVEL
                        )
                        step_n = max(1, int(progressive_step))
                        st.markdown(
                            f"""
                            <div class="ea-val-card">
                              <strong>Progressive — up to {validation_levels.display_label(level.level)}</strong>
                              &nbsp;·&nbsp; starts at {validation_levels.display_label(start_lvl.level)}
                              &nbsp;·&nbsp; step +{step_n}
                              <p>The survivor floor rises by {step_n} level{'s' if step_n != 1 else ''} after each full cycle of
                              {sweep_count} sweep{'s' if sweep_count != 1 else ''}, then
                              holds at this target. Every candidate is still scored against L1–L16.</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="ea-val-card">
                              <strong>{validation_levels.display_label(level.level)}</strong>
                              &nbsp;·&nbsp; {mc_note}
                              <p>{level.summary}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with st.expander("Exactly what this level checks", icon=":material/rule:"):
                        preview_level = level
                        if is_agent and progressive_strictness:
                            preview_level = validation_levels.get_level(
                                validation_levels.MIN_LEVEL
                            )
                            st.caption(
                                f"Gates below are for the **starting** level "
                                f"({validation_levels.display_label(preview_level.level)}). "
                                f"Later cycles use stricter gates "
                                f"up to {validation_levels.display_label(level_num)}."
                            )
                        for bullet in preview_level.human_gates():
                            st.markdown(f"- {bullet}")

                    st.markdown(
                        '<p class="ea-section-label">Run budget</p>',
                        unsafe_allow_html=True,
                    )
                    e1, e2, e3 = st.columns(3)
                    with e1:
                        target_survivors = st.number_input(
                            "Winning strategies to find", 1, 100, int(saved.target_survivors),
                            help="Discovery stops as soon as it has found this many "
                                 "strategies that pass the chosen validation level.",
                        )
                        max_candidates = st.number_input(
                            "Max candidates to test", 10, 100_000, int(saved.max_candidates),
                            step=100,
                            help="Upper bound on how many strategy/parameter combinations "
                                 "to screen before giving up.",
                        )
                    with e2:
                        months = st.slider(
                            "Test duration (months)", 1, 36, int(saved.months),
                            help="How much recent history each backtest covers. "
                                 "Walk-forward folds are sized automatically from this.",
                        )
                        _wfo_train, _wfo_test, _wfo_n = derive_wfo_from_duration(int(months))
                        st.caption(
                            f"Walk-forward auto: {_wfo_train}m train / {_wfo_test}m test "
                            f"× {_wfo_n} window{'s' if _wfo_n != 1 else ''}"
                        )
                        _engine_options = ["simulator", "mt5"]
                        _engine_default = (
                            saved.engine if saved.engine in _engine_options else "simulator"
                        )
                        engine = st.selectbox(
                            "Backtest engine", _engine_options,
                            index=_engine_options.index(_engine_default),
                            help="Simulator: fast in-app validation (recommended "
                                 "for continuous discovery). MT5: still screens "
                                 "with the simulator first, then runs the Strategy "
                                 "Tester on promising candidates — falls back to "
                                 "the simulator automatically if MT5 is open. "
                                 "Continuous agent always uses the simulator.",
                        )
                    with e3:
                        batch_size = st.number_input(
                            "Generation size", 10, 5000, int(saved.batch_size), step=10,
                            help="How many candidates are generated per evolution round.",
                        )
                        genetic = st.checkbox(
                            "Evolve toward winners", value=bool(saved.genetic),
                            help="Breed the best-screened candidates into each new "
                                 "generation instead of pure random search.",
                        )

                    _render_data_source_notice(symbols, timeframes, months)

                    st.markdown(
                        '<p class="ea-section-label">Edge → execution styles</p>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Discovery searches **entry edges** under a simple SL/TP + R:R "
                        "probe. Only after an edge clears your validation floor does it "
                        "build MQL5-oriented execution variants (partial close, trailing, "
                        "DCA, …) that reuse that edge — so you are not flooded with "
                        "unrelated EA files."
                    )
                    mechanics = st.multiselect(
                        "Execution styles after an edge is found",
                        options=list(_MECHANIC_LABELS.keys()),
                        default=_mech_default,
                        format_func=lambda m: _MECHANIC_LABELS[m],
                        help=(
                            "Mechanic variants tried on each validated edge. "
                            "Default is Standard SL/TP + Partial close (defined risk). "
                            "DCA / Grid and Hedging recover against the market — enable "
                            "only if you accept that risk profile."
                        ),
                    )
                    tm_features = st.multiselect(
                        "Trade-management overlays on edge EAs",
                        options=list(_TM_FEATURE_LABELS.keys()),
                        default=_tm_default,
                        format_func=lambda f: _TM_FEATURE_LABELS[f],
                        help=(
                            "Exit/risk overlays applied when expanding a validated edge "
                            "into EA variants (trailing, breakeven, regime filters, …). "
                            "Edge search itself uses a lean R:R probe only."
                        ),
                    )

                    if is_agent:
                        st.markdown(
                            '<p class="ea-section-label">Notifications</p>',
                            unsafe_allow_html=True,
                        )
                        n1, n2, n3 = st.columns(3)
                        with n1:
                            recipient = st.text_input(
                                "Alert recipient email",
                                value=str(saved.recipient_email),
                            )
                        with n2:
                            min_score = st.number_input(
                                "Minimum quality score to alert",
                                min_value=0.0, max_value=100.0,
                                value=float(saved.alert_min_score), step=1.0,
                            )
                        with n3:
                            progress_hours = st.number_input(
                                "Progress email interval (hours)",
                                min_value=0.25, max_value=24.0,
                                value=float(saved.progress_email_hours), step=0.25,
                            )
                    else:
                        recipient = str(saved.recipient_email)
                        min_score = float(saved.alert_min_score)
                        progress_hours = float(saved.progress_email_hours)

                    with st.expander("Account & execution economics", icon=":material/account_balance:"):
                        st.caption(
                            "Balance/leverage are yours; spread, slippage, and contract "
                            "size default per symbol so multi-symbol runs stay realistic."
                        )
                        ec1, ec2 = st.columns(2)
                        with ec1:
                            deposit = st.number_input(
                                "Starting balance", 100.0, 100_000_000.0,
                                value=float(saved.deposit), step=1000.0, format="%.2f")
                        with ec2:
                            leverage = st.number_input(
                                "Leverage (1:N)", 1, 1000,
                                value=int(saved.leverage), step=1)
                        auto_symbol_economics = st.checkbox(
                            "Auto spread, slippage & contract size per symbol",
                            value=bool(saved.auto_symbol_economics),
                            help=(
                                "Each sweep uses that instrument's typical London-session "
                                "spread/slippage and contract size (e.g. EURUSD ~12/2 pts, "
                                "XAUUSD ~25/5, BTCUSD ~500/80). Turn off only to force one "
                                "manual spread/slippage/size across every symbol."
                            ),
                        )

                        if auto_symbol_economics:
                            preview = list(symbols) or list(saved.symbols) or [
                                settings.DEFAULT_SYMBOL]
                            bits = []
                            for s in preview[:10]:
                                d = SymbolSpec.defaults_for_symbol(str(s))
                                bits.append(
                                    f"{s}: size {d.contract_size:g}, "
                                    f"spread {d.spread_points:g}, "
                                    f"slip {d.slippage_points:g}"
                                )
                            more = "" if len(preview) <= 10 else f" (+{len(preview) - 10} more)"
                            st.caption(
                                "Per-symbol defaults — " + "; ".join(bits) + more)
                            spread_points = float(saved.spread_points)
                            slippage_points = float(saved.slippage_points)
                            contract_size = float(saved.contract_size)
                        else:
                            m1, m2, m3 = st.columns(3)
                            with m1:
                                spread_points = st.number_input(
                                    "Spread (points)", 0.0, 500.0,
                                    value=float(saved.spread_points), step=1.0)
                            with m2:
                                slippage_points = st.number_input(
                                    "Slippage (points)", 0.0, 200.0,
                                    value=float(saved.slippage_points), step=1.0)
                            with m3:
                                contract_size = st.number_input(
                                    "Contract size", 1.0, 10_000_000.0,
                                    value=float(saved.contract_size),
                                    step=1000.0, format="%.2f")

                    with st.expander(
                        "Advanced (search behaviour + expert gate override)",
                        icon=":material/tune:",
                    ):
                        edge_first = st.checkbox(
                            "Edge-first discovery (recommended)",
                            value=bool(getattr(saved, "edge_first", True)),
                            help=(
                                "Search entry edges under a simple R:R probe first, "
                                "then expand only survivors into the execution styles "
                                "above. Turn off to sample mechanics in one mixed pool "
                                "(legacy behaviour)."
                            ),
                        )
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            advanced_mode = st.checkbox(
                                "Enable advanced mode", value=bool(saved.advanced_mode))
                        with c2:
                            complexity_cap = st.slider(
                                "Complexity cap", 2, 10, int(saved.complexity_cap))
                        with c3:
                            enable_regime_switching = st.checkbox(
                                "Enable regime switching",
                                value=bool(saved.enable_regime_switching))
                        enable_mtf_context = st.checkbox(
                            "Enable multi-timeframe context",
                            value=bool(saved.enable_mtf_context))
                        feature_toggles = st.multiselect(
                            "Feature families",
                            options=["momentum", "mean_reversion", "volatility", "market_structure"],
                            default=list(saved.feature_toggles),
                        )

                        use_custom = st.checkbox(
                            "Override the level with custom gates (expert)",
                            value=bool(saved.use_custom))
                        _crit = saved.custom_criteria or {}
                        ac1, ac2, ac3, ac4 = st.columns(4)
                        with ac1:
                            wfe_min = st.number_input(
                                "Min WFE", 0.0, 2.0,
                                value=float(_crit.get("min_wfe", settings.WFE_THRESHOLD)),
                                step=0.05)
                            max_dd = st.number_input(
                                "Max OOS DD %", 1.0, 50.0,
                                value=float(_crit.get("max_dd_pct", settings.OOS_MAX_DD_PCT)),
                                step=1.0)
                        with ac2:
                            min_trades = st.number_input(
                                "Min OOS trades", 1, 200,
                                value=int(_crit.get("min_trades", settings.MIN_OOS_TRADES)))
                            min_pf = st.number_input(
                                "Min profit factor (0=off)", 0.0, 5.0,
                                value=float(_crit.get("min_profit_factor", settings.MIN_PROFIT_FACTOR)),
                                step=0.1)
                        with ac3:
                            min_sharpe = st.number_input(
                                "Min Sharpe (0=off)", 0.0, 5.0,
                                value=float(_crit.get("min_sharpe", settings.MIN_SHARPE)),
                                step=0.1)
                            min_r2 = st.number_input(
                                "Min R-squared (0=off)", 0.0, 1.0,
                                value=float(_crit.get("min_r_squared", settings.MIN_R_SQUARED)),
                                step=0.05)
                        with ac4:
                            max_consec = st.number_input(
                                "Max consec. losses (0=off)", 0, 50,
                                value=int(_crit.get(
                                    "max_consecutive_losses", settings.MAX_CONSECUTIVE_LOSSES
                                )))
                            custom_mc = st.checkbox(
                                "Monte Carlo gate", value=bool(saved.custom_montecarlo))
                            custom_mc_runs = st.number_input(
                                "MC runs", 5, 200, int(saved.custom_mc_runs))

                        st.caption(
                            "Walk-forward fold sizes follow **Test duration** above "
                            f"(currently {_wfo_train}m / {_wfo_test}m × {_wfo_n})."
                        )

                    if is_agent:
                        submit_label = ":material/smart_toy: Start continuous agent"
                    else:
                        submit_label = ":material/play_arrow: Run once"
                    submitted = st.form_submit_button(
                        submit_label,
                        type="primary",
                        width="stretch",
                    )

        active_jobs = [j for j in storage.list_jobs("discovery") if j.status in _ACTIVE]

        if submitted:
            # Fresh seed each submit so re-runs explore different strategies
            # instead of replaying the same genetic trajectory.
            run_seed = (int(time.time()) ^ (saved.base_seed * 2654435761)) & 0x7FFFFFFF
            _wfo_train, _wfo_test, _wfo_n = derive_wfo_from_duration(int(months))
            discovery_cfg = _form_to_settings(
                level_num=int(level_num),
                progressive_strictness=bool(progressive_strictness),
                progressive_step=int(progressive_step),
                symbols=symbols,
                timeframes=timeframes,
                months=int(months),
                engine=engine,
                deposit=float(deposit),
                leverage=int(leverage),
                auto_symbol_economics=bool(auto_symbol_economics),
                spread_points=float(spread_points),
                slippage_points=float(slippage_points),
                contract_size=float(contract_size),
                batch_size=int(batch_size),
                target_survivors=int(target_survivors),
                max_candidates=int(max_candidates),
                genetic=bool(genetic),
                edge_first=bool(edge_first),
                mechanics=mechanics,
                tm_features=tm_features,
                wfo_train_months=int(_wfo_train),
                wfo_test_months=int(_wfo_test),
                wfo_windows=int(_wfo_n),
                advanced_mode=bool(advanced_mode),
                complexity_cap=int(complexity_cap),
                enable_regime_switching=bool(enable_regime_switching),
                enable_mtf_context=bool(enable_mtf_context),
                feature_toggles=feature_toggles,
                use_custom=bool(use_custom),
                wfe_min=wfe_min,
                max_dd=max_dd,
                min_trades=min_trades,
                min_pf=min_pf,
                min_sharpe=min_sharpe,
                min_r2=min_r2,
                max_consec=max_consec,
                custom_mc=custom_mc,
                custom_mc_runs=int(custom_mc_runs),
                recipient=recipient,
                min_score=float(min_score),
                progress_hours=float(progress_hours),
                base_seed=run_seed,
            )
            storage.upsert_app_settings(settings_to_app(discovery_cfg))

            if not symbols:
                st.warning("Pick at least one symbol.")
            elif not timeframes:
                st.warning("Pick at least one timeframe.")
            elif not mechanics:
                st.warning("Pick at least one execution style for post-edge EA variants.")
            elif active_jobs:
                st.warning("A discovery run is already going — stop it first "
                           "or wait for it to finish.")
            elif is_agent:
                sweep_count = len(plan_sweeps(
                    symbols=list(symbols),
                    timeframes=list(timeframes),
                    months=discovery_cfg.months,
                    base_seed=discovery_cfg.base_seed,
                ))
                if _start_orchestrator(storage, mode="continuous"):
                    st.success(
                        f"Agent started — cycling {sweep_count} sweeps "
                        f"({len(symbols)} symbols × {len(timeframes)} timeframes)."
                    )
                else:
                    st.error(storage.get_agent_state().get("message")
                             or "Could not start the discovery agent.")
                st.rerun()
            else:
                _submit_single_run(queue, storage, discovery_cfg)

        if is_agent:
            _render_email_tools(storage)


def _submit_single_run(
    queue: JobQueue,
    storage: Storage,
    cfg: DiscoverySettings,
) -> None:
    plans = plan_sweeps(
        symbols=list(cfg.symbols),
        timeframes=list(cfg.timeframes),
        months=cfg.months,
        base_seed=cfg.base_seed,
    )
    if len(plans) == 1:
        plan = plans[0]
        payload = build_discovery_payload(
            cfg,
            symbol=plan.symbol,
            timeframe=plan.timeframe,
            seed=plan.seed,
        )
        job_id = f"disc_{uuid.uuid4().hex[:10]}"
        if queue.submit_discovery(job_id, payload):
            st.success(
                f"Single run queued — {plan.symbol} · {plan.timeframe} · `{job_id}`"
            )
        else:
            st.info("This run is already queued (duplicate submit ignored).")
    else:
        if _start_orchestrator(storage, mode="batch"):
            st.success(
                f"Batch run started — {len(plans)} sweeps "
                f"({len(cfg.symbols)} symbols × {len(cfg.timeframes)} timeframes)."
            )
        else:
            st.error(storage.get_agent_state().get("message")
                     or "Could not start the batch agent.")
    st.rerun()


def _form_to_settings(
    *,
    level_num: int,
    progressive_strictness: bool,
    progressive_step: int,
    symbols: list[str],
    timeframes: list[str],
    months: int,
    engine: str,
    deposit: float,
    leverage: int,
    auto_symbol_economics: bool,
    spread_points: float,
    slippage_points: float,
    contract_size: float,
    batch_size: int,
    target_survivors: int,
    max_candidates: int,
    genetic: bool,
    edge_first: bool,
    mechanics: list[ExecutionMechanicType],
    tm_features: list[str],
    wfo_train_months: int,
    wfo_test_months: int,
    wfo_windows: int,
    advanced_mode: bool,
    complexity_cap: int,
    enable_regime_switching: bool,
    enable_mtf_context: bool,
    feature_toggles: list[str],
    use_custom: bool,
    wfe_min: float,
    max_dd: float,
    min_trades: int,
    min_pf: float,
    min_sharpe: float,
    min_r2: float,
    max_consec: int,
    custom_mc: bool,
    custom_mc_runs: int,
    recipient: str,
    min_score: float,
    progress_hours: float,
    base_seed: int,
) -> DiscoverySettings:
    custom_criteria = {
        "min_wfe": wfe_min,
        "max_dd_pct": max_dd,
        "min_trades": int(min_trades),
        "min_profit_factor": min_pf,
        "min_sharpe": min_sharpe,
        "min_r_squared": min_r2,
        "max_consecutive_losses": int(max_consec),
    }
    cfg = DiscoverySettings(
        symbols=list(symbols) or list(settings.SYMBOLS[:5]),
        timeframes=list(timeframes) or ["M15", "H1"],
        months=int(months),
        engine=str(engine),
        deposit=float(deposit),
        leverage=int(leverage),
        auto_symbol_economics=bool(auto_symbol_economics),
        spread_points=float(spread_points),
        slippage_points=float(slippage_points),
        contract_size=float(contract_size),
        batch_size=int(batch_size),
        target_survivors=int(target_survivors),
        max_candidates=int(max_candidates),
        genetic=bool(genetic),
        edge_first=bool(edge_first),
        mechanics=[m.value for m in mechanics],
        tm_features=list(tm_features),
        wfo_train_months=int(wfo_train_months),
        wfo_test_months=int(wfo_test_months),
        wfo_windows=int(wfo_windows),
        advanced_mode=bool(advanced_mode),
        complexity_cap=int(complexity_cap),
        enable_regime_switching=bool(enable_regime_switching),
        enable_mtf_context=bool(enable_mtf_context),
        feature_toggles=list(feature_toggles),
        validation_level=int(level_num),
        progressive_strictness=bool(progressive_strictness),
        validation_level_start=validation_levels.MIN_LEVEL,
        progressive_step=max(1, int(progressive_step)),
        use_custom=bool(use_custom),
        custom_criteria=custom_criteria,
        custom_montecarlo=bool(custom_mc),
        custom_mc_runs=int(custom_mc_runs),
        base_seed=int(base_seed),
        recipient_email=str(recipient).strip(),
        alert_min_score=float(min_score),
        progress_email_hours=float(progress_hours),
    )
    cfg.sync_wfo_from_duration()
    return cfg


def _start_orchestrator(storage: Storage, *, mode: str) -> bool:
    storage.update_agent_state(
        enabled=1,
        status="starting",
        mode=mode,
        cursor=0,
        pid=None,
        spawn_attempts=0,
        effective_validation_level=None,
        message="Starting discovery…",
    )
    if sync_agent_with_orchestrator_lock(storage):
        return True
    if start_orchestrator_process():
        sync_agent_with_orchestrator_lock(storage)
        return True
    if sync_agent_with_orchestrator_lock(storage):
        return True
    storage.update_agent_state(
        enabled=0,
        status="stopped",
        pid=None,
        message=(
            "Could not start discovery agent — another instance may "
            "already be running. Stop it or restart the dashboard."
        ),
    )
    return False


def _render_data_source_notice(
    symbols: list[str],
    timeframes: list[str],
    months: int,
) -> None:
    sym = str(symbols[0]).strip().upper() if symbols else settings.DEFAULT_SYMBOL
    tf = str(timeframes[0]) if timeframes else settings.DEFAULT_TIMEFRAME
    start_dt, end_dt = history_start_end(int(months))
    try:
        src = data_mod.peek_source(sym.strip().upper(), tf, start_dt, end_dt)
    except Exception:
        src = "unknown"
    badge = data_source_badge(src)
    span_label = (
        f"{start_dt.date().isoformat()} → {end_dt.date().isoformat()} "
        f"({int(months)} mo)"
    )
    if src == "synthetic":
        st.warning(
            f"**Synthetic market data** {badge} — {span_label}. "
            "Results are for development only. Install MetaTrader 5 and "
            "connect to a broker for real OHLC, or use a parquet cache in "
            "`data/`.")
    elif src == "cache":
        st.caption(
            f"Backtest data: {badge} ({data_source_label(src)}) · {span_label}"
        )
    else:
        st.caption(f"Backtest data: {badge} · {span_label}")


@st.fragment(run_every="1s")
def _live_status(storage: Storage) -> None:
    sync_agent_with_orchestrator_lock(storage)
    recover_stuck_starting_agent(storage)
    jobs = storage.list_jobs("discovery")
    active = [j for j in jobs if j.status in _ACTIVE]
    state = storage.get_agent_state()
    agent_enabled = bool(int(state.get("enabled", 0) or 0))
    agent_status = str(state.get("status", "stopped"))

    st.session_state["_active_run_count"] = len(active)

    if active or agent_enabled:
        st.caption(
            ":material/info: Live run progress (tests, L1–L16 passes) is on the "
            "**Runs** tab — this panel is for launching new searches."
        )
        _render_status_strip(state, agent_enabled, agent_status, active)
    else:
        mode_hint = st.session_state.get("discovery_run_mode", "Single run")
        if mode_hint == "Continuous agent":
            st.caption("Configure options below and start the continuous agent.")
        else:
            st.caption("Configure options below and press **Run once**.")


def _render_status_strip(
    state: dict,
    agent_enabled: bool,
    agent_status: str,
    active: list,
) -> None:
    hb = state.get("heartbeat_at")
    hb_txt = "never" if not hb else datetime.fromtimestamp(hb).strftime("%H:%M:%S")
    running = agent_enabled or bool(active)
    cols = st.columns([1, 1, 1, 1, 1, 1] if running else [1, 1, 1, 1, 1],
                       vertical_alignment="center")
    cols[0].metric("Agent status", agent_status, border=True)
    cols[1].metric("Heartbeat", hb_txt, border=True)
    cols[2].metric("Queue", int(state.get("queue_depth", 0) or 0), border=True)
    cols[3].metric("Sweeps", int(state.get("jobs_submitted", 0) or 0), border=True)
    eff_level = state.get("effective_validation_level")
    if eff_level is not None:
        try:
            lvl = validation_levels.get_level(int(eff_level))
            cols[4].metric(
                "Strictness",
                f"L{lvl.level} {lvl.name}",
                border=True,
                help="Current effective validation level for the agent run.",
            )
        except (TypeError, ValueError):
            cols[4].metric("Strictness", str(eff_level), border=True)
    elif len(cols) > 4:
        cols[4].metric("Strictness", "—", border=True)
    if running:
        with cols[5]:
            st.button(
                ":material/stop: Stop",
                key="discovery_stop_btn",
                width="stretch",
                on_click=_stop_agent,
            )

    message = str(state.get("message") or "").strip()
    mode = str(state.get("mode") or "continuous")
    hb_age = (time.time() - float(hb)) if hb else None
    if agent_enabled and hb_age is not None and hb_age > 45:
        st.warning(
            f"Agent heartbeat is stale ({int(hb_age)}s) — the orchestrator "
            "may have been stuck. Stop discovery and start it again if this "
            "persists."
        )
    if agent_enabled and agent_status in ("running", "starting"):
        sweep_total = int(state.get("sweep_total", 0) or 0)
        cursor = int(state.get("cursor", 0) or 0)
        if mode == "batch" and sweep_total > 0:
            sweep_idx = max(cursor - 1, 0) % sweep_total
            st.progress(
                min((sweep_idx + 1) / sweep_total, 1.0),
                text=f"Batch sweep — {sweep_idx + 1} / {sweep_total}",
            )
        elif mode == "continuous":
            cycles = max(cursor, 0)
            st.caption(
                f"Continuous agent — {cycles} sweep(s) submitted · "
                "runs until you press Stop"
            )
    if message:
        st.caption(message)
    elif agent_status == "stopping":
        st.caption("Stop requested — cancelling jobs and force-stopping the agent.")


def _stop_agent() -> None:
    stop_orchestrator_process()


def _render_email_tools(storage: Storage) -> None:
    app_cfg = storage.get_app_settings()
    with st.expander(":material/mail: Email delivery", expanded=False):
        smtp_diag_key = "smtp_diag_refresh_nonce"
        st.session_state.setdefault(smtp_diag_key, 0)
        smtp_diag = smtp_diagnostics()
        if smtp_diag.configured:
            st.success(
                "SMTP configured: "
                f"host `{smtp_diag.host}:{smtp_diag.port}` · "
                f"from `{smtp_diag.from_email or smtp_diag.username}` · "
                f"TLS `{'on' if smtp_diag.use_tls else 'off'}`"
            )
        else:
            st.warning(smtp_missing_message(smtp_diag))
        st.caption(
            f"SMTP auth user: `{smtp_diag.username or '(none)'}` · "
            f"password configured: `{'yes' if smtp_diag.has_password else 'no'}`"
        )
        if st.button(":material/refresh: Reload SMTP config", width="content"):
            st.session_state[smtp_diag_key] = int(st.session_state[smtp_diag_key]) + 1
            st.rerun()

        test_default_recipient = str(
            app_cfg.get("recipient_email", settings.DEFAULT_ALERT_RECIPIENT)
        ).strip()
        test_recipient_override = st.text_input(
            "Test email recipient (optional override)",
            value=test_default_recipient,
            key="test_email_recipient_override",
        )
        if st.button("Send test email", key="send_test_email"):
            recipient = test_recipient_override.strip() or test_default_recipient
            if not recipient:
                st.error("Set an alert recipient email first, then retry.")
            else:
                try:
                    send_email(
                        recipient,
                        "EA Generator test email",
                        "This is a test email from the EA Generator discovery agent.",
                    )
                    st.success(f"Test email sent to {recipient}.")
                except Exception as exc:
                    st.error(f"Test email failed: {exc}")
