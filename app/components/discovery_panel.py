"""Discovery control panel with live progress via st.fragment polling SQLite."""
from __future__ import annotations

import time
import uuid
from datetime import date, datetime, timedelta, timezone

import streamlit as st

from config import settings
from factory import data as data_mod
from factory import validation_levels
from factory.backtest.simulator import SymbolSpec
from factory.metrics_display import data_source_badge, data_source_label
from factory.models import ExecutionMechanicType, JobStatus
from factory.storage import Storage
from jobs.orchestrator import start_orchestrator_process, stop_orchestrator_process
from jobs.worker import JobQueue
from factory import alerts as alerts_mod

send_email = alerts_mod.send_email
# Backward-compat fallback for stale/older alert modules during hot reloads.
smtp_diagnostics = getattr(
    alerts_mod, "smtp_diagnostics", lambda: type("Diag", (), {"configured": False})()
)
smtp_missing_message = getattr(
    alerts_mod,
    "smtp_missing_message",
    lambda _diag: "SMTP diagnostics unavailable in this build.",
)

_ACTIVE = (JobStatus.PENDING, JobStatus.RUNNING)

# Friendly labels for the execution-mechanic picker (enum -> UI text).
_MECHANIC_LABELS = {
    ExecutionMechanicType.STANDARD_SLTP: "Standard SL/TP",
    ExecutionMechanicType.DCA_GRID: "DCA / Grid",
    ExecutionMechanicType.HEDGE_LAYER: "Hedging",
    ExecutionMechanicType.PARTIAL_CLOSE: "Partial close",
}

# Friendly labels for the trade-management overlay picker (feature key -> text).
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


def render_discovery_panel(queue: JobQueue, storage: Storage) -> None:
    _render_discovery_agent_panel(storage)
    # Live progress, KPIs and run history sit at the very top — this is the
    # screen users watch the longest while a run is in flight, so it gets the
    # most visual priority.
    _live_dashboard(queue, storage)
    _render_run_history(storage)

    st.divider()
    st.subheader(":material/rocket_launch: Start a new run")
    st.caption(
        "Set how many winning strategies you want, how hard they must be to "
        "earn a pass, and how many candidates the factory may test. It "
        "generates and screens thousands of strategy + parameter combinations, "
        "validates the promising ones, and keeps the survivors.")

    # --- Validation level (outside the form so the description updates live) --
    st.markdown("#### Validation level")
    level_num = st.select_slider(
        "How strict should a 'pass' be?",
        options=[lv.level for lv in validation_levels.VALIDATION_LEVELS],
        value=validation_levels.DEFAULT_LEVEL,
        format_func=lambda n: f"{n} · {validation_levels.get_level(n).name}",
        key="val_level",
        help="Higher levels apply every gate of the lower levels, only "
             "stricter, and add heavier Monte Carlo robustness testing.")
    level = validation_levels.get_level(level_num)
    mc_note = "Monte Carlo ON" if level.montecarlo else "Monte Carlo off"
    st.info(f"**Level {level.level} — {level.name}**  ·  {mc_note}\n\n{level.summary}")
    with st.expander("Exactly what this level checks"):
        for bullet in level.human_gates():
            st.markdown(f"- {bullet}")

    # Data-source probe (uses same loader as the backtest engine)
    _render_data_source_notice()

    with st.form("discovery_form"):
        e1, e2, e3 = st.columns(3)
        with e1:
            symbol_options = list(settings.SYMBOLS)
            default_symbol = settings.DEFAULT_SYMBOL
            if default_symbol not in symbol_options:
                symbol_options.insert(0, default_symbol)
            symbol = st.selectbox(
                "Symbol", symbol_options,
                index=symbol_options.index(default_symbol),
                key="disc_symbol",
                help="Instrument to search on. The simulator uses cached or "
                     "synthetic history; MT5 uses your terminal's symbols.")
            timeframe = st.selectbox(
                "Timeframe", ["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
                index=2, key="disc_tf")
        with e2:
            target_survivors = st.number_input(
                "Winning strategies to find", 1, 100, 5,
                help="Discovery stops as soon as it has found this many "
                     "strategies that pass the chosen validation level.")
            max_candidates = st.number_input(
                "Max candidates to test", 10, 100_000, 2000, step=100,
                help="Upper bound on how many strategy/parameter combinations "
                     "to screen before giving up.")
        with e3:
            months = st.slider(
                "History window (months)", 1, 36, 12, key="disc_months",
                help="How much history to backtest over. Shorter = faster "
                     "screening; longer = more robust.")
            engine = st.selectbox(
                "Backtest engine", ["simulator", "mt5"],
                help="The simulator is a fast pre-filter. MT5 runs the real "
                     "Strategy Tester headlessly (requires an installed "
                     "terminal) and executes strictly sequentially.")

        mechanics = st.multiselect(
            "Allowed strategy types",
            options=list(_MECHANIC_LABELS.keys()),
            default=list(_MECHANIC_LABELS.keys()),
            format_func=lambda m: _MECHANIC_LABELS[m],
            help="Which trade-management styles the factory may generate and "
                 "evolve. Pick any combination — e.g. only DCA/Grid + Hedging, "
                 "or leave all selected to search across every type.")

        tm_features = st.multiselect(
            "Trade-management options to explore",
            options=list(_TM_FEATURE_LABELS.keys()),
            default=list(_TM_FEATURE_LABELS.keys()),
            format_func=lambda f: _TM_FEATURE_LABELS[f],
            help="Advanced exit/risk overlays the optimizer may switch on and "
                 "tune per strategy: adaptive & trailing stops, breakeven, "
                 "risk-based sizing and session/loss safeguards. Clear all for "
                 "plain fixed SL/TP only. Trailing/adaptive-SL/risk sizing apply "
                 "to directional (Standard & Partial-close) strategies; the "
                 "filters apply to every type.")

        with st.expander("Account & execution economics"):
            st.caption(
                "The account balance and market frictions the backtests trade "
                "with. Defaults match the current engine settings, so leaving "
                "them untouched keeps behaviour unchanged. Starting balance and "
                "leverage apply to every engine; spread, slippage and contract "
                "size shape the simulator (in MT5 these come from the broker).")
            _spec = SymbolSpec()
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                deposit = st.number_input(
                    "Starting balance", 100.0, 100_000_000.0,
                    value=float(settings.DEFAULT_DEPOSIT), step=1000.0,
                    format="%.2f",
                    help="Initial account deposit for every backtest, in the "
                         "account currency.")
                leverage = st.number_input(
                    "Leverage (1:N)", 1, 1000,
                    value=int(settings.DEFAULT_LEVERAGE), step=1,
                    help="Account leverage. Higher leverage allows the same "
                         "balance to hold larger positions before a margin "
                         "refusal.")
            with ec2:
                spread_points = st.number_input(
                    "Spread (points)", 0.0, 500.0,
                    value=float(_spec.spread_points), step=1.0,
                    help="Spread charged on every entry fill (simulator).")
                slippage_points = st.number_input(
                    "Slippage (points)", 0.0, 200.0,
                    value=float(_spec.slippage_points), step=1.0,
                    help="Adverse slippage added to every fill (simulator).")
            with ec3:
                contract_size = st.number_input(
                    "Contract size", 1.0, 10_000_000.0,
                    value=float(_spec.contract_size), step=1000.0,
                    format="%.2f",
                    help="Units per 1.0 lot (e.g. 100000 for standard FX). "
                         "Drives point value and margin (simulator).")

        with st.expander("Advanced (search behaviour + expert gate override)"):
            b1, b2 = st.columns(2)
            with b1:
                batch_size = st.number_input(
                    "Generation size", 10, 5000, 100, step=10,
                    help="How many candidates are generated per evolution "
                         "round before the best are bred into the next round.")
            with b2:
                genetic = st.checkbox(
                    "Evolve toward winners", value=True,
                    help="Breed the best-screened candidates into each new "
                         "generation instead of pure random search.")
            st.markdown("**Advanced strategy generation**")
            c1, c2, c3 = st.columns(3)
            with c1:
                advanced_mode = st.checkbox(
                    "Enable advanced mode", value=False,
                    help="Generate richer multi-signal candidates with anti-bloat caps.")
            with c2:
                complexity_cap = st.slider(
                    "Complexity cap", 2, 10, 5,
                    help="Maximum complexity budget for entry logic; higher explores more expressive rules.")
            with c3:
                enable_regime_switching = st.checkbox(
                    "Enable regime switching", value=True,
                    help="Bias generation toward volatility/trend state-aware compositions.")
            enable_mtf_context = st.checkbox(
                "Enable multi-timeframe context", value=True,
                help="Inject higher-timeframe context filters when compatible.")
            feature_toggles = st.multiselect(
                "Feature families",
                options=["momentum", "mean_reversion", "volatility", "market_structure"],
                default=["momentum", "mean_reversion", "volatility", "market_structure"],
                help="Constrain advanced primitives to selected families.")

            use_custom = st.checkbox(
                "Override the level with custom gates (expert)", value=False,
                help="Ignore the validation level above and use the exact "
                     "numeric gates below instead.")
            st.caption("These only apply when the override box is ticked.")
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                wfe_min = st.number_input("Min WFE", 0.0, 2.0,
                                          value=float(settings.WFE_THRESHOLD), step=0.05)
                max_dd = st.number_input("Max OOS DD %", 1.0, 50.0,
                                         value=float(settings.OOS_MAX_DD_PCT), step=1.0)
            with ac2:
                min_trades = st.number_input("Min OOS trades", 1, 200,
                                             value=int(settings.MIN_OOS_TRADES))
                min_pf = st.number_input("Min profit factor (0=off)", 0.0, 5.0,
                                         value=float(settings.MIN_PROFIT_FACTOR), step=0.1)
            with ac3:
                min_sharpe = st.number_input("Min Sharpe (0=off)", 0.0, 5.0,
                                             value=float(settings.MIN_SHARPE), step=0.1)
                min_r2 = st.number_input("Min R-squared (0=off)", 0.0, 1.0,
                                         value=float(settings.MIN_R_SQUARED), step=0.05)
            with ac4:
                max_consec = st.number_input("Max consec. losses (0=off)", 0, 50,
                                             value=int(settings.MAX_CONSECUTIVE_LOSSES))
                custom_mc = st.checkbox("Monte Carlo gate", value=settings.MC_ENABLED)
                custom_mc_runs = st.number_input("MC runs", 5, 200,
                                                 int(settings.MC_RUNS))

            st.markdown("**Walk-forward windows**")
            wfo1, wfo2, wfo3 = st.columns(3)
            with wfo1:
                wfo_train_months = st.number_input(
                    "WFO train (months)", 1, 24, int(settings.WFO_TRAIN_MONTHS),
                    help="In-sample length for each rolling walk-forward window.")
            with wfo2:
                wfo_test_months = st.number_input(
                    "WFO test (months)", 1, 12, int(settings.WFO_TEST_MONTHS),
                    help="Out-of-sample length tested after each train window.")
            with wfo3:
                wfo_window_count = st.number_input(
                    "WFO windows per mode", 1, 12, int(settings.WFO_WINDOWS),
                    help="Number of anchored and rolling windows to compute.")

        submitted = st.form_submit_button("Start discovery", type="primary")

    active_jobs = [j for j in storage.list_jobs("discovery") if j.status in _ACTIVE]

    if submitted:
        if active_jobs:
            st.warning("A discovery run is already going — cancel it first "
                       "or wait for it to finish.")
        elif not mechanics:
            st.warning("Pick at least one strategy type to search.")
        else:
            end = date.today()
            start = end - timedelta(days=int(months) * 30)
            job_id = f"disc_{uuid.uuid4().hex[:10]}"
            payload = {
                "symbol": symbol.strip().upper(),
                "timeframe": timeframe,
                "start": datetime.combine(start, datetime.min.time(),
                                          tzinfo=timezone.utc).isoformat(),
                "end": datetime.combine(end, datetime.min.time(),
                                        tzinfo=timezone.utc).isoformat(),
                "engine": engine,
                "deposit": float(deposit),
                "leverage": int(leverage),
                "spread_points": float(spread_points),
                "slippage_points": float(slippage_points),
                "contract_size": float(contract_size),
                "batch_size": int(batch_size),
                "target_survivors": int(target_survivors),
                "max_candidates": int(max_candidates),
                "genetic": bool(genetic),
                "mechanics": [m.value for m in mechanics],
                "tm_features": list(tm_features),
                "wfo_train_months": int(wfo_train_months),
                "wfo_test_months": int(wfo_test_months),
                "wfo_windows": int(wfo_window_count),
                "advanced_mode": bool(advanced_mode),
                "complexity_cap": int(complexity_cap),
                "enable_regime_switching": bool(enable_regime_switching),
                "enable_mtf_context": bool(enable_mtf_context),
                "feature_toggles": list(feature_toggles),
            }
            try:
                payload["data_source"] = data_mod.peek_source(
                    payload["symbol"], payload["timeframe"],
                    datetime.fromisoformat(payload["start"]),
                    datetime.fromisoformat(payload["end"]),
                )
            except Exception:
                payload["data_source"] = "unknown"
            if use_custom:
                payload["montecarlo"] = custom_mc
                payload["mc_runs"] = int(custom_mc_runs)
                payload["criteria"] = {
                    "min_wfe": wfe_min,
                    "max_dd_pct": max_dd,
                    "min_trades": int(min_trades),
                    "min_profit_factor": min_pf,
                    "min_sharpe": min_sharpe,
                    "min_r_squared": min_r2,
                    "max_consecutive_losses": int(max_consec),
                }
                pass_desc = "custom expert gates"
            else:
                payload["validation_level"] = int(level_num)
                pass_desc = f"Level {level.level} ({level.name})"

            if queue.submit_discovery(job_id, payload):
                st.success(f"Discovery started — searching for "
                           f"{int(target_survivors)} strategies passing "
                           f"{pass_desc}.")
            else:
                st.info("This run is already queued (duplicate submit ignored).")
            st.rerun()


def _render_data_source_notice() -> None:
    """Show a prominent warning when backtests would use synthetic OHLC."""
    sym = st.session_state.get("disc_symbol", settings.DEFAULT_SYMBOL)
    tf = st.session_state.get("disc_tf", settings.DEFAULT_TIMEFRAME)
    months = st.session_state.get("disc_months", 12)
    end = date.today()
    start = end - timedelta(days=int(months) * 30)
    try:
        src = data_mod.peek_source(
            sym.strip().upper(), tf,
            datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc),
        )
    except Exception:
        src = "unknown"
    badge = data_source_badge(src)
    if src == "synthetic":
        st.warning(
            f"**Synthetic market data** {badge} — results are for development "
            "only. Install MetaTrader 5 and connect to a broker for real "
            "OHLC, or use a parquet cache in `data/`.")
    elif src == "cache":
        st.caption(f"Backtest data: {badge} ({data_source_label(src)})")
    else:
        st.caption(f"Backtest data: {badge}")


_STATUS_STYLE = {
    JobStatus.RUNNING: (":blue-badge[:material/bolt: Running]", "blue"),
    JobStatus.PENDING: (":gray-badge[:material/schedule: Queued]", "gray"),
    JobStatus.DONE: (":green-badge[:material/check_circle: Done]", "green"),
    JobStatus.CANCELLED: (":orange-badge[:material/stop_circle: Cancelled]", "orange"),
    JobStatus.FAILED: (":red-badge[:material/error: Failed]", "red"),
}


def _job_summary(job) -> str:
    lvl = job.payload.get("validation_level")
    gate = f"Level {lvl}" if lvl is not None else "custom gates"
    return (f"{job.payload.get('symbol', '?')} · "
            f"{job.payload.get('timeframe', '?')} · "
            f"{job.payload.get('engine', '?')} · {gate} · "
            f"data {job.payload.get('data_source', '?')} · "
            f"target {job.payload.get('target_survivors', '?')}")


@st.fragment(run_every="2s")
def _live_dashboard(queue: JobQueue, storage: Storage) -> None:
    """Top-of-page live view: KPIs plus the current-run hero.

    This 2-second fragment polls SQLite so the progress stays live without
    rerunning the (expensive) configuration form below it. Run history and the
    results modal live *outside* this fragment on purpose — a modal opened from
    inside a ``run_every`` fragment would be torn down on every tick.
    """
    jobs = storage.list_jobs("discovery")
    active = [j for j in jobs if j.status in _ACTIVE]

    # When the last active run finishes, do one full-app rerun so the run
    # history (rendered outside this fragment) picks up the newly-finished run
    # and its results. Only the >0 -> 0 edge triggers it, so we never loop.
    prev_active = st.session_state.get("_active_run_count")
    st.session_state["_active_run_count"] = len(active)
    if prev_active and len(active) == 0:
        st.rerun(scope="app")

    passing = len(storage.list_validated(passed_only=True))
    tested_total = len(storage.list_strategies())
    k1, k2, k3 = st.columns(3)
    k1.metric("Winning strategies", passing, border=True,
              help="Passed the validation level — ready to export.")
    k2.metric("Total tested (library)", tested_total, border=True,
              help="Every candidate ever screened across all runs.")
    k3.metric("Active runs", len(active), border=True)

    if active:
        for job in active:
            _render_active_hero(job, queue)
    else:
        st.info("No run in progress. Configure one below and press "
                "**Start discovery** — live progress will appear here.")


def _render_active_hero(job, queue: JobQueue) -> None:
    """Big, prominent progress panel for a run that is in flight."""
    with st.container(border=True):
        head_l, head_r = st.columns([4, 1], vertical_alignment="center")
        with head_l:
            st.markdown(f"### :material/bolt: Current run &nbsp; "
                        f"`{job.id}`")
            st.caption(_job_summary(job))
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


def _render_run_history(storage: Storage) -> None:
    """Run history + per-run results modal.

    Deliberately *not* inside the ``run_every`` fragment: the results dialog is
    opened with the button-return pattern, which only survives if no timer tick
    tears it down. Finished runs don't change, so no live refresh is needed.
    """
    finished = [j for j in storage.list_jobs("discovery")
                if j.status not in _ACTIVE]
    if not finished:
        return

    with st.expander(f":material/history: Run history ({len(finished)})",
                     expanded=True):
        for job in finished[:12]:
            badge = _STATUS_STYLE.get(job.status, (job.status.value,))[0]
            passed, total = storage.count_validated(job.id)
            with st.container(border=True):
                left, mid, right = st.columns([3, 4, 2],
                                              vertical_alignment="center")
                with left:
                    st.markdown(f"**`{job.id}`**")
                    st.caption(_job_summary(job))
                with mid:
                    st.markdown(badge)
                    if job.message:
                        st.caption(job.message)
                    if total:
                        st.caption(f":green[{passed} passed] · {total} evaluated")
                with right:
                    if st.button(
                            f":material/insights: Show results ({total})",
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
    """Modal showing every strategy a single run produced, as compact cards.

    Renders the *same* card component as the Strategy Gallery in ``compact``
    mode, so a run's results look consistent with the gallery. A per-run
    ``key_prefix`` keeps widget/chart keys distinct from the gallery's cards.
    This dialog lives outside the 2-second live fragment (opened from the run
    history), so its charts are not torn down on each refresh.

    Pagination and cached Plotly thumbnails keep the modal responsive when a
    run evaluated many candidates.
    """
    from app.components.strategy_card import (
        _MODAL_PAGE_SIZE, render_page_controls, render_report_grid,
        render_sort_selectbox, sort_reports, _page_slice,
    )

    st.markdown(f"**`{job.id}`** · {_job_summary(job)}")
    reports = storage.list_validated(passed_only=False, job_id=job.id)
    if not reports:
        st.info("This run produced no evaluated strategies.")
        return

    passed = [r for r in reports if r.passed]
    page_key = f"dlg_page_{job.id}"
    runs = {job.id: job}

    ctrl_l, ctrl_r = st.columns([3, 1], vertical_alignment="bottom")
    with ctrl_l:
        sort_by = render_sort_selectbox(f"dlg_sort_{job.id}", page_key=page_key)
    with ctrl_r:
        only_passed = st.toggle("Passed only", value=bool(passed),
                                key=f"dlg_pass_{job.id}",
                                on_change=_reset_modal_page,
                                args=(page_key,))
    shown = [r for r in reports if r.passed] if only_passed else reports
    shown = sort_reports(shown, sort_by, runs=runs)

    st.caption(f":green[{len(passed)} passed] · {len(reports) - len(passed)} "
               f"failed · {len(reports)} evaluated")

    page_reports, page, total_pages = _page_slice(
        shown, page_key, _MODAL_PAGE_SIZE)
    render_page_controls(page_key, page, total_pages, len(shown),
                         _MODAL_PAGE_SIZE, control_key=f"dlg_nav_{job.id}")
    render_report_grid(page_reports, {}, storage=storage,
                       key_prefix=f"run_{job.id}_", compact=True)

    st.caption("Open the **Strategy Gallery** tab for full cards, walk-forward "
               "detail and one-click MQL5 export.")


def _reset_modal_page(page_key: str) -> None:
    st.session_state[page_key] = 0


def _render_running_progress(job) -> None:
    """Live, determinate progress for an in-flight discovery run.

    Reads the counters the worker persists on each candidate (tested/promising/
    survivors/generation) so the 2s fragment refresh shows steady progress.
    Falls back to an indeterminate 'starting' state for the brief window before
    the first candidate has been screened.
    """
    max_candidates = int(job.payload.get("max_candidates", 0) or 0)
    target_survivors = int(job.payload.get("target_survivors", 0) or 0)

    if getattr(job, "cancel_requested", False):
        st.warning("Cancelling — stopping after the current backtest…")

    # Read live counters defensively: a Streamlit process that imported an
    # older Job model (before these fields shipped) would otherwise raise
    # AttributeError and crash the whole panel on every 2s refresh.
    tested = int(getattr(job, "tested", 0) or 0)
    promising = int(getattr(job, "promising", 0) or 0)
    survivors = int(getattr(job, "survivors", 0) or 0)
    generation = int(getattr(job, "generation", 0) or 0)

    if tested <= 0:
        st.progress(min(max(job.progress, 0.0), 1.0),
                    text=job.message or "starting…")
        return

    st.progress(min(max(job.progress, 0.0), 1.0),
                text=f"generation {generation} · {promising} promising")

    elapsed = max(time.time() - job.created_at, 1e-6)
    passed_txt = (f"{survivors} / {target_survivors}"
                  if target_survivors else str(survivors))
    tested_txt = (f"{tested} / {max_candidates}"
                  if max_candidates else str(tested))
    eta_txt = _estimate_eta(job, max_candidates, target_survivors, elapsed)

    m1, m2, m3 = st.columns(3)
    m1.metric("Passed", passed_txt, help="Strategies that cleared every gate.")
    m2.metric("Tested", tested_txt, help="Candidates screened so far.")
    m3.metric("Est. time to target", eta_txt,
              help="Estimate for whichever limit is hit first — the target "
                   "number of winners or the max candidates.")

    rate = tested / elapsed * 60.0
    st.caption(f"elapsed {_fmt_duration(elapsed)} · {rate:.0f} tested/min")


def _estimate_eta(job, max_candidates: int, target_survivors: int,
                  elapsed: float) -> str:
    """Estimate the time until the run finishes.

    A run stops when EITHER the target survivor count OR the max-candidate
    ceiling is reached, so the ETA is the sooner of the two projections based
    on the observed throughput. Returns a short human string (or a placeholder
    while there isn't enough signal to project yet).
    """
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
    """Cancel callback.

    Using ``on_click`` (instead of checking the button's return value) makes
    the click reliable inside the auto-rerunning ``run_every`` fragment: the
    callback is guaranteed to run on the interaction, with no race against the
    2-second refresh.
    """
    queue.cancel(job_id)
    st.toast(f"Cancellation requested for {job_id}")


def _render_discovery_agent_panel(storage: Storage) -> None:
    """Detached discovery-agent controls and alert settings."""
    state = storage.get_agent_state()
    app_cfg = storage.get_app_settings()
    st.subheader(":material/smart_toy: Discovery agent")
    with st.container(border=True):
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Status", str(state.get("status", "stopped")))
        hb = state.get("heartbeat_at")
        hb_txt = "never" if not hb else datetime.fromtimestamp(hb).strftime("%H:%M:%S")
        s2.metric("Heartbeat", hb_txt)
        s3.metric("Queue depth", int(state.get("queue_depth", 0) or 0))
        s4.metric("Sweeps submitted", int(state.get("jobs_submitted", 0) or 0))

        c1, c2 = st.columns(2)
        with c1:
            if st.button(":material/play_arrow: Start agent", width="stretch"):
                storage.update_agent_state(enabled=1, status="starting")
                started = start_orchestrator_process()
                if started:
                    st.success("Discovery agent service started.")
                else:
                    st.info("Discovery agent already running.")
                st.rerun()
        with c2:
            if st.button(":material/stop: Stop agent", width="stretch"):
                stop_orchestrator_process()
                st.warning("Stop requested for discovery agent.")
                st.rerun()

        st.markdown("#### Agent settings")
        st.caption(
            "Configure what the discovery agent searches, how aggressive each run is, "
            "and when it sends notifications."
        )
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

        with st.form("agent_settings_form"):
            st.markdown("##### :material/tune: Notifications")
            n1, n2, n3 = st.columns(3)
            with n1:
                recipient = st.text_input(
                    "Alert recipient email",
                    value=str(app_cfg.get("recipient_email", settings.DEFAULT_ALERT_RECIPIENT)),
                    help="Where pass alerts and promotion notifications are sent.",
                )
            with n2:
                min_score = st.number_input(
                    "Minimum quality score to alert",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(app_cfg.get("alert_min_score", 70.0)),
                    step=1.0,
                    help="Only strategies scoring at or above this value trigger an email.",
                )
            with n3:
                cooldown_min = st.number_input(
                    "Alert cooldown (minutes)",
                    min_value=1,
                    max_value=10_000,
                    value=int(app_cfg.get("alert_cooldown_minutes", 60)),
                    step=1,
                    help="Minimum time between alert emails.",
                )

            st.markdown("##### :material/travel_explore: What to search")
            s1, s2, s3 = st.columns(3)
            with s1:
                symbols = st.multiselect(
                    "Symbols",
                    options=list(settings.SYMBOLS),
                    default=list(app_cfg.get("agent_symbols", settings.SYMBOLS[:5])),
                    help="Instruments the agent will sweep.",
                )
            with s2:
                tf = st.multiselect(
                    "Timeframes",
                    options=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
                    default=list(app_cfg.get("agent_timeframes", ["M15", "H1"])),
                    help="Chart intervals included in each sweep.",
                )
            with s3:
                strictness = st.multiselect(
                    "Validation strictness",
                    options=["easy", "normal", "hard", "custom"],
                    default=list(app_cfg.get(
                        "agent_strictness_profiles",
                        ["easy", "normal", "hard", "custom"],
                    )),
                    help="Gate profiles used during validation.",
                )

            st.markdown("##### :material/schedule: Run budget")
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                months = st.number_input(
                    "History (months)",
                    min_value=1,
                    max_value=36,
                    value=int(app_cfg.get("agent_history_months", 12)),
                    step=1,
                    help="Backtest window length used by the agent.",
                )
            with b2:
                agent_batch = st.number_input(
                    "Batch size",
                    min_value=10,
                    max_value=5000,
                    value=int(app_cfg.get("agent_batch_size", 100)),
                    help="Candidates generated per generation.",
                )
            with b3:
                agent_max = st.number_input(
                    "Max candidates",
                    min_value=10,
                    max_value=100000,
                    value=int(app_cfg.get("agent_max_candidates", 1000)),
                    help="Hard cap per sweep.",
                )
            with b4:
                agent_target = st.number_input(
                    "Target survivors",
                    min_value=1,
                    max_value=100,
                    value=int(app_cfg.get("agent_target_survivors", 2)),
                    help="Stop once this many passing strategies are found.",
                )

            with st.expander(":material/science: Advanced generation options"):
                st.caption(
                    "These control search sophistication. Defaults are good for most runs."
                )
                ag1, ag2, ag3 = st.columns(3)
                with ag1:
                    agent_advanced_mode = st.checkbox(
                        "Enable advanced mode",
                        value=bool(app_cfg.get("agent_advanced_mode", True)),
                        help="Turns on richer building blocks and search operators.",
                    )
                with ag2:
                    agent_complexity_cap = st.slider(
                        "Complexity cap",
                        2,
                        10,
                        int(app_cfg.get("agent_complexity_cap", 6)),
                        help="Upper bound on strategy complexity.",
                    )
                with ag3:
                    agent_regime = st.checkbox(
                        "Enable regime switching",
                        value=bool(app_cfg.get("agent_enable_regime_switching", True)),
                        help="Allow market-regime-aware variants.",
                    )
                agent_mtf = st.checkbox(
                    "Enable multi-timeframe context",
                    value=bool(app_cfg.get("agent_enable_mtf_context", True)),
                    help="Use higher/lower timeframe context as additional signals.",
                )
                agent_feature_toggles = st.multiselect(
                    "Feature families",
                    options=["momentum", "mean_reversion", "volatility", "market_structure"],
                    default=list(app_cfg.get(
                        "agent_feature_toggles",
                        ["momentum", "mean_reversion", "volatility", "market_structure"],
                    )),
                    help="Signal families the generator may include.",
                )

            save = st.form_submit_button(
                ":material/save: Save agent settings",
                type="primary",
                width="stretch",
            )
            if save:
                storage.upsert_app_settings(
                    {
                        "recipient_email": recipient.strip(),
                        "alert_min_score": float(min_score),
                        "alert_cooldown_minutes": int(cooldown_min),
                        "agent_timeframes": list(tf) or ["M15", "H1"],
                        "agent_strictness_profiles": list(strictness) or ["normal"],
                        "agent_history_months": int(months),
                        "agent_symbols": list(symbols) or settings.SYMBOLS[:5],
                        "agent_batch_size": int(agent_batch),
                        "agent_max_candidates": int(agent_max),
                        "agent_target_survivors": int(agent_target),
                        "agent_advanced_mode": bool(agent_advanced_mode),
                        "agent_complexity_cap": int(agent_complexity_cap),
                        "agent_enable_regime_switching": bool(agent_regime),
                        "agent_enable_mtf_context": bool(agent_mtf),
                        "agent_feature_toggles": list(agent_feature_toggles)
                        or ["momentum", "mean_reversion", "volatility", "market_structure"],
                    }
                )
                st.success("Discovery agent settings saved.")
                st.rerun()
        test_default_recipient = str(app_cfg.get("recipient_email", settings.DEFAULT_ALERT_RECIPIENT)).strip()
        test_recipient_override = st.text_input(
            "Test email recipient (optional override)",
            value=test_default_recipient,
            key="test_email_recipient_override",
            help="Used only for the test send button below.",
        )
        if st.button("Send test email", key="send_test_email"):
            recipient = test_recipient_override.strip() or test_default_recipient
            if not recipient:
                st.error("Set an alert recipient email first, then retry.")
                return
            try:
                send_email(
                    recipient,
                    "EA Generator test email",
                    "This is a test email from the EA Generator discovery agent.",
                )
                st.success(f"Test email sent to {recipient}.")
            except Exception as exc:
                st.error(f"Test email failed: {exc}")
