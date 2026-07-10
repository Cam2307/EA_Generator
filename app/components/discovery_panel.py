"""Unified discovery terminal — agent + single-run in one panel."""
from __future__ import annotations

# Bump when discovery UI changes — visible in the panel header for debugging.
DISCOVERY_UI_VERSION = "2026-07-10-duration"

import time
import uuid
from datetime import datetime

import streamlit as st

from app.components.discovery_styles import inject_discovery_styles
from app.components.pie_progress import render_pie_progress
from app.components.run_view import job_summary
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
}

_STATUS_STYLE = {
    JobStatus.RUNNING: (":orange-badge[:material/bolt: Running]", "orange"),
    JobStatus.PENDING: (":gray-badge[:material/schedule: Queued]", "gray"),
    JobStatus.DONE: (":green-badge[:material/check_circle: Done]", "green"),
    JobStatus.CANCELLED: (":orange-badge[:material/stop_circle: Cancelled]", "orange"),
    JobStatus.FAILED: (":red-badge[:material/error: Failed]", "red"),
}


def render_discovery_panel(queue: JobQueue, storage: Storage) -> None:
    inject_discovery_styles()
    saved = settings_from_app(storage.get_app_settings())
    agent_state = storage.get_agent_state()
    agent_running = bool(int(agent_state.get("enabled", 0) or 0))

    if agent_running and "discovery_run_mode" not in st.session_state:
        st.session_state.discovery_run_mode = "Continuous agent"

    st.markdown(
        f"""
        <div class="ea-terminal-shell">
          <div class="ea-terminal-head">
            <div>
              <p class="ea-terminal-title">Discovery terminal</p>
              <p class="ea-terminal-sub">
                Configure markets, validation, and search — then launch a
                continuous agent or a one-shot sweep. Live status and controls
                share one workspace.
              </p>
            </div>
            <span class="ea-terminal-chip">EA factory · UI {DISCOVERY_UI_VERSION}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _live_status(queue, storage)

    _mech_default = [
        m for m in _MECHANIC_LABELS if m.value in saved.mechanics
    ] or list(_MECHANIC_LABELS.keys())
    _tm_default = [f for f in _TM_FEATURE_LABELS if f in saved.tm_features]
    if not _tm_default:
        _tm_default = list(_TM_FEATURE_LABELS.keys())
    _tf_options = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

    with st.container(border=True):
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

        with st.form("discovery_form"):
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
                    default=[tf for tf in saved.timeframes if tf in _tf_options]
                    or ["M15", "H1"],
                    help="Every selected timeframe is included in each sweep.",
                )

            st.markdown(
                '<p class="ea-section-label">Validation</p>',
                unsafe_allow_html=True,
            )
            level_num = st.select_slider(
                "How strict should a pass be?",
                options=[lv.level for lv in validation_levels.VALIDATION_LEVELS],
                value=int(saved.validation_level),
                format_func=lambda n: f"{n} · {validation_levels.get_level(n).name}",
                help="Higher levels apply every gate of the lower levels, only "
                     "stricter, and add heavier Monte Carlo robustness testing.",
            )
            level = validation_levels.get_level(level_num)
            mc_note = "Monte Carlo on" if level.montecarlo else "Monte Carlo off"
            st.markdown(
                f"""
                <div class="ea-val-card">
                  <strong>Level {level.level} — {level.name}</strong>
                  &nbsp;·&nbsp; {mc_note}
                  <p>{level.summary}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Exactly what this level checks", icon=":material/rule:"):
                for bullet in level.human_gates():
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
                    help="The simulator is a fast pre-filter. MT5 runs the real "
                         "Strategy Tester headlessly (requires an installed "
                         "terminal) and executes strictly sequentially.",
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
                '<p class="ea-section-label">Strategy universe</p>',
                unsafe_allow_html=True,
            )
            mechanics = st.multiselect(
                "Allowed strategy types",
                options=list(_MECHANIC_LABELS.keys()),
                default=_mech_default,
                format_func=lambda m: _MECHANIC_LABELS[m],
                help="Which trade-management styles the factory may generate and evolve.",
            )
            tm_features = st.multiselect(
                "Trade-management options to explore",
                options=list(_TM_FEATURE_LABELS.keys()),
                default=_tm_default,
                format_func=lambda f: _TM_FEATURE_LABELS[f],
                help="Advanced exit/risk overlays the optimizer may switch on and tune.",
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
                    "The account balance and market frictions the backtests trade with.")
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    deposit = st.number_input(
                        "Starting balance", 100.0, 100_000_000.0,
                        value=float(saved.deposit), step=1000.0, format="%.2f")
                    leverage = st.number_input(
                        "Leverage (1:N)", 1, 1000,
                        value=int(saved.leverage), step=1)
                with ec2:
                    spread_points = st.number_input(
                        "Spread (points)", 0.0, 500.0,
                        value=float(saved.spread_points), step=1.0)
                    slippage_points = st.number_input(
                        "Slippage (points)", 0.0, 200.0,
                        value=float(saved.slippage_points), step=1.0)
                with ec3:
                    contract_size = st.number_input(
                        "Contract size", 1.0, 10_000_000.0,
                        value=float(saved.contract_size), step=1000.0, format="%.2f")

            with st.expander(
                "Advanced (search behaviour + expert gate override)",
                icon=":material/tune:",
            ):
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
                symbols=symbols,
                timeframes=timeframes,
                months=int(months),
                engine=engine,
                deposit=float(deposit),
                leverage=int(leverage),
                spread_points=float(spread_points),
                slippage_points=float(slippage_points),
                contract_size=float(contract_size),
                batch_size=int(batch_size),
                target_survivors=int(target_survivors),
                max_candidates=int(max_candidates),
                genetic=bool(genetic),
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
                st.warning("Pick at least one strategy type to search.")
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

    _render_run_history(storage)


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
    symbols: list[str],
    timeframes: list[str],
    months: int,
    engine: str,
    deposit: float,
    leverage: int,
    spread_points: float,
    slippage_points: float,
    contract_size: float,
    batch_size: int,
    target_survivors: int,
    max_candidates: int,
    genetic: bool,
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
        spread_points=float(spread_points),
        slippage_points=float(slippage_points),
        contract_size=float(contract_size),
        batch_size=int(batch_size),
        target_survivors=int(target_survivors),
        max_candidates=int(max_candidates),
        genetic=bool(genetic),
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


@st.fragment(run_every="2s")
def _live_status(queue: JobQueue, storage: Storage) -> None:
    sync_agent_with_orchestrator_lock(storage)
    recover_stuck_starting_agent(storage)
    jobs = storage.list_jobs("discovery")
    active = [j for j in jobs if j.status in _ACTIVE]
    state = storage.get_agent_state()
    agent_enabled = bool(int(state.get("enabled", 0) or 0))
    agent_status = str(state.get("status", "stopped"))

    st.session_state["_active_run_count"] = len(active)

    passing = storage.count_validated(passed_only=True)
    tested_total = storage.count_strategies()

    st.markdown('<p class="ea-section-label">Live status</p>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    k1.metric("Winning strategies", passing, border=True,
              help="Passed the validation level — ready to export.")
    k2.metric("Total tested (library)", tested_total, border=True,
              help="Every candidate ever screened across all runs.")
    k3.metric("Active runs", len(active), border=True)

    _render_status_strip(state, agent_enabled, agent_status, active)

    if active:
        for job in active:
            _render_active_hero(job, queue, show_heading=False)
    elif agent_enabled and agent_status in ("running", "starting", "stopping"):
        _render_agent_waiting(state)
    elif not agent_enabled and not active:
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
    cols = st.columns([1, 1, 1, 1, 1] if running else [1, 1, 1, 1],
                       vertical_alignment="center")
    cols[0].metric("Agent status", agent_status, border=True)
    cols[1].metric("Heartbeat", hb_txt, border=True)
    cols[2].metric("Queue", int(state.get("queue_depth", 0) or 0), border=True)
    cols[3].metric("Sweeps", int(state.get("jobs_submitted", 0) or 0), border=True)
    if running:
        with cols[4]:
            st.button(
                ":material/stop: Stop",
                key="discovery_stop_btn",
                width="stretch",
                on_click=_stop_agent,
            )

    message = str(state.get("message") or "").strip()
    mode = str(state.get("mode") or "continuous")
    if agent_enabled and agent_status in ("running", "starting"):
        sweep_total = int(state.get("sweep_total", 0) or 0)
        if sweep_total > 0:
            cursor = int(state.get("cursor", 0) or 0)
            sweep_idx = max(cursor - 1, 0) % sweep_total
            label = (
                f"Batch sweep — {sweep_idx + 1} / {sweep_total}"
                if mode == "batch"
                else f"Cycle — {sweep_idx + 1} / {sweep_total}"
            )
            st.progress(
                min((sweep_idx + 1) / sweep_total, 1.0),
                text=label,
            )
    if message:
        st.caption(message)
    elif agent_status == "stopping":
        st.caption("Stop requested — cancelling jobs and force-stopping the agent.")


def _render_agent_waiting(state: dict) -> None:
    st.info(str(state.get("message") or "Agent running — waiting for the next sweep…"))


def _stop_agent() -> None:
    stop_orchestrator_process()


def _render_active_hero(job, queue: JobQueue, *, show_heading: bool = True) -> None:
    is_agent = str(job.id).startswith("auto_")
    with st.container(border=True):
        head_l, head_r = st.columns([4, 1], vertical_alignment="center")
        with head_l:
            if show_heading:
                title = (
                    ":material/smart_toy: Agent sweep"
                    if is_agent else ":material/bolt: Current run"
                )
                st.markdown(f"### {title} &nbsp; `{job.id}`")
            else:
                label = "Agent sweep" if is_agent else "Current run"
                st.markdown(f"**{label}** · `{job.id}`")
            st.caption(job_summary(job))
        with head_r:
            if job.cancel_requested:
                st.button("Cancelling…", key=f"cancel_{job.id}",
                          disabled=True, width="stretch")
            else:
                st.button(":material/stop: Stop", key=f"cancel_{job.id}",
                          type="secondary", width="stretch",
                          on_click=_request_cancel, args=(queue, job.id))
        _render_running_progress(job)
        if job.error:
            with st.expander("Last non-fatal issue"):
                st.code(job.error)


def _render_running_progress(job) -> None:
    max_candidates = int(job.payload.get("max_candidates", 0) or 0)
    target_survivors = int(job.payload.get("target_survivors", 0) or 0)

    if getattr(job, "cancel_requested", False):
        st.warning("Cancelling — stopping after the current backtest…")

    tested = int(getattr(job, "tested", 0) or 0)
    promising = int(getattr(job, "promising", 0) or 0)
    survivors = int(getattr(job, "survivors", 0) or 0)
    generation = int(getattr(job, "generation", 0) or 0)

    pct = min(max(job.progress, 0.0), 1.0) * 100.0
    if tested <= 0:
        progress_label = job.message or "starting…"
    else:
        progress_label = f"gen {generation} · {promising} promising"

    pie_col, stats_col = st.columns([1, 2], vertical_alignment="center")
    with pie_col:
        render_pie_progress(pct, label=progress_label, size=156)
    with stats_col:
        elapsed = max(time.time() - job.created_at, 1e-6)
        passed_txt = (f"{survivors} / {target_survivors}"
                      if target_survivors else str(survivors))
        tested_txt = (f"{tested} / {max_candidates}"
                      if max_candidates else str(tested))
        eta_txt = _estimate_eta(job, max_candidates, target_survivors, elapsed)

        m1, m2, m3 = st.columns(3)
        m1.metric("Passed", passed_txt,
                  help="Strategies that cleared every gate.")
        m2.metric("Tested", tested_txt,
                  help="Candidates screened so far.")
        m3.metric("Est. time to target", eta_txt,
                  help="Estimate for whichever limit is hit first — the "
                       "target number of winners or the max candidates.")

        rate = tested / elapsed * 60.0 if tested > 0 else 0.0
        if tested > 0:
            st.caption(f"elapsed {_fmt_duration(elapsed)} · "
                       f"{rate:.0f} tested/min")
        else:
            st.caption(job.message or "warming up…")


def _estimate_eta(job, max_candidates: int, target_survivors: int,
                  elapsed: float) -> str:
    tested = int(getattr(job, "tested", 0) or 0)
    survivors = int(getattr(job, "survivors", 0) or 0)
    etas: list[float] = []
    if max_candidates and tested > 0:
        rate = tested / elapsed
        if rate > 0:
            etas.append(max(max_candidates - tested, 0) / rate)
    if target_survivors and survivors > 0:
        rate = survivors / elapsed
        if rate > 0:
            etas.append(max(target_survivors - survivors, 0) / rate)

    if not etas:
        return "estimating…"
    return "~" + _fmt_duration(min(etas))


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def _request_cancel(queue: JobQueue, job_id: str) -> None:
    queue.cancel(job_id)
    st.toast(f"Cancellation requested for {job_id}")


def _render_run_history(storage: Storage) -> None:
    finished = [j for j in storage.list_jobs("discovery")
                if j.status not in _ACTIVE]
    if not finished:
        return

    recent = finished[:10]
    counts = storage.count_validated_by_jobs([j.id for j in recent])

    with st.expander(
        f":material/history: Recent runs ({len(finished)})",
        expanded=False,
    ):
        for job in recent:
            badge = _STATUS_STYLE.get(job.status, (job.status.value,))[0]
            passed, total = counts.get(job.id, (0, 0))
            with st.container(border=True):
                left, mid, right = st.columns([3, 4, 2], vertical_alignment="center")
                with left:
                    st.markdown(f"**`{job.id}`**")
                    st.caption(job_summary(job))
                with mid:
                    st.markdown(badge)
                    if job.message:
                        st.caption(job.message)
                    if total:
                        st.caption(f":green[{passed} passed] · {total} evaluated")
                with right:
                    if st.button(
                            f":material/insights: Results ({total})",
                            key=f"results_{job.id}", width="stretch",
                            disabled=total == 0,
                            help=("This run's strategies."
                                  if total else "No evaluated strategies.")):
                        _results_dialog(storage, job)
                    if job.error:
                        with st.expander("Issue"):
                            st.code(job.error)


@st.dialog("Run results", width="large")
def _results_dialog(storage: Storage, job) -> None:
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

    st.markdown(f"**`{job.id}`** · {job_summary(job)}")
    page_key = f"dlg_page_{job.id}"
    runs = {job.id: job}
    passed_n, _total_n = storage.count_validated(job.id)

    ctrl_l, ctrl_r = st.columns([3, 1], vertical_alignment="bottom")
    with ctrl_l:
        sort_by = render_sort_selectbox(f"dlg_sort_{job.id}", page_key=page_key)
    with ctrl_r:
        only_passed = st.toggle(
            "Passed only",
            value=passed_n > 0,
            key=f"dlg_pass_{job.id}",
            on_change=_reset_modal_page,
            args=(page_key,),
        )

    column_sorts = {
        "WFE (high → low)",
        "Run (newest → oldest)",
        "Run (oldest → newest)",
    }
    if sort_by not in SORT_OPTION_LABELS:
        sort_by = SORT_OPTION_LABELS[0]
    need_metrics = sort_by not in column_sorts

    reports = storage.list_validation_summaries(
        passed_only=None,
        job_id=job.id,
        include_body_metrics=need_metrics,
    )
    if not reports:
        st.info("This run produced no evaluated strategies.")
        return

    passed = [r for r in reports if r.passed]
    # Default "Passed only" on when the run has survivors; keep widget value.
    shown = [r for r in reports if r.passed] if only_passed else reports
    shown = sort_reports(shown, sort_by, runs=runs)

    st.caption(f":green[{len(passed)} passed] · {len(reports) - len(passed)} "
               f"failed · {len(reports)} evaluated")

    page_reports, page, total_pages = _page_slice(
        shown, page_key, _MODAL_PAGE_SIZE)
    render_page_controls(page_key, page, total_pages, len(shown),
                         _MODAL_PAGE_SIZE, control_key=f"dlg_nav_{job.id}")
    full_page = _hydrate_reports(storage, page_reports)
    render_report_grid(full_page, {}, storage=storage,
                       key_prefix=f"run_{job.id}_", compact=True)

    st.caption("Open the **Strategy gallery** tab for full cards, walk-forward "
               "detail and one-click MQL5 export.")


def _reset_modal_page(page_key: str) -> None:
    st.session_state[page_key] = 0


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
