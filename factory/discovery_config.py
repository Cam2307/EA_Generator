"""Shared discovery settings and job-payload builder.

Manual batch runs and the automated orchestrator both read/write the same
``DiscoverySettings`` so every sweep uses an identical discovery pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from config import settings
from factory import data as data_mod
from factory import validation_levels
from factory.backtest.simulator import SymbolSpec
from factory.models import ExecutionMechanicType

# Execution styles tried *after* a signal edge qualifies (edge-first mode).
# DCA / hedge recoveries pass loose gates easily and crowd out defined-risk
# EAs unless the user explicitly enables them.
DEFAULT_DISCOVERY_MECHANICS: list[str] = [
    ExecutionMechanicType.STANDARD_SLTP.value,
    ExecutionMechanicType.PARTIAL_CLOSE.value,
]


def _first(app: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in app and app[key] is not None:
            return app[key]
    return default


def derive_wfo_from_duration(months: int) -> tuple[int, int, int]:
    """Derive WFO train/test/window counts from a single test-duration value.

    Keeps folds proportional to the outer history window so changing
    ``months`` actually changes the walk-forward structure (not just the
    outer envelope while folds stay stuck at 2m/1m).
    """
    months = max(1, int(months))
    if months <= 2:
        train = 1
        test = 1
        windows = 1
    else:
        train = max(1, months // 3)
        test = max(1, months // 6)
        if train + test > months:
            test = 1
            train = max(1, months - test)
        max_fit = max(1, (months - train) // test)
        windows = min(max_fit, max(1, months // 3))
    return int(train), int(test), int(windows)


def history_start_end(months: int, *, today: date | None = None) -> tuple[datetime, datetime]:
    """Outer backtest envelope from a single test-duration (months).

    When the untouched holdout is enabled, the envelope ends at the holdout
    boundary (not "now"). Requesting N months then yields ~N months of usable
    discovery data *before* the reserved window, instead of collapsing to a
    few hours when ``months ≈ HOLDOUT_MONTHS``.
    """
    ref = today or date.today()
    end_dt = datetime.combine(ref, datetime.min.time(), tzinfo=timezone.utc)
    if getattr(settings, "HOLDOUT_ENABLED", True):
        from factory.holdout import holdout_boundary
        end_dt = holdout_boundary(end_dt)
    days = max(1, int(months)) * settings.DAYS_PER_MONTH
    start_dt = end_dt - timedelta(days=days)
    # Stable midnight UTC cache keys (boundary may carry a time-of-day).
    start_dt = datetime.combine(
        start_dt.date(), datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(
        end_dt.date(), datetime.min.time(), tzinfo=timezone.utc)
    if end_dt <= start_dt:
        # Pathological settings (e.g. holdout disabled mid-call) — keep a span.
        start_dt = end_dt - timedelta(days=days)
    return start_dt, end_dt


@dataclass
class DiscoverySettings:
    symbols: list[str] = field(default_factory=lambda: list(settings.SYMBOLS[:2]))
    timeframes: list[str] = field(default_factory=lambda: ["M15"])
    # Longer history → more OOS trades → fewer false kills on min-trade gates.
    months: int = 12
    engine: str = settings.DEFAULT_ENGINE
    deposit: float = settings.DEFAULT_DEPOSIT
    leverage: int = settings.DEFAULT_LEVERAGE
    # When True (default), each discovery sweep fills contract_size / spread /
    # slippage from SymbolSpec.defaults_for_symbol(symbol) so multi-symbol
    # runs stay economically correct. Manual values below are only used when False.
    auto_symbol_economics: bool = True
    spread_points: float = SymbolSpec().spread_points
    slippage_points: float = SymbolSpec().slippage_points
    contract_size: float = SymbolSpec().contract_size
    batch_size: int = 64
    target_survivors: int = 5
    max_candidates: int = 500
    genetic: bool = True
    # Edge-first: search STANDARD_SLTP entry probes first; only expand into
    # ``mechanics`` execution variants after an edge clears the floor.
    edge_first: bool = True
    mechanics: list[str] = field(
        default_factory=lambda: list(DEFAULT_DISCOVERY_MECHANICS)
    )
    tm_features: list[str] = field(default_factory=list)
    # WFO fields are derived from ``months`` at payload-build time; kept for
    # backward-compatible persistence / report display.
    wfo_train_months: int = settings.WFO_TRAIN_MONTHS
    wfo_test_months: int = settings.WFO_TEST_MONTHS
    wfo_windows: int = settings.WFO_WINDOWS
    advanced_mode: bool = False
    complexity_cap: int = 5
    enable_regime_switching: bool = True
    enable_mtf_context: bool = True
    feature_toggles: list[str] = field(
        default_factory=lambda: [
            "momentum",
            "mean_reversion",
            "volatility",
            "market_structure",
        ]
    )
    validation_level: int = validation_levels.DEFAULT_LEVEL
    # When enabled (continuous/batch agent), each full symbol×timeframe cycle
    # bumps the effective validation level by ``progressive_step`` until
    # ``validation_level``.
    progressive_strictness: bool = True
    validation_level_start: int = validation_levels.MIN_LEVEL
    progressive_step: int = validation_levels.DEFAULT_PROGRESSIVE_STEP
    use_custom: bool = False
    custom_criteria: dict = field(default_factory=dict)
    custom_montecarlo: bool = settings.MC_ENABLED
    custom_mc_runs: int = settings.MC_RUNS
    base_seed: int = 1337
    recipient_email: str = settings.DEFAULT_ALERT_RECIPIENT
    alert_min_score: float = settings.DEFAULT_ALERT_MIN_SCORE
    progress_email_hours: float = settings.DEFAULT_PROGRESS_EMAIL_HOURS
    # Optional daily wall-clock budget for the continuous agent (hours).
    # 0 = unlimited. When exhausted the orchestrator pauses until next UTC day.
    daily_budget_hours: float = 0.0

    def sync_wfo_from_duration(self) -> None:
        """Overwrite WFO fields from the single test-duration control."""
        train, test, windows = derive_wfo_from_duration(self.months)
        self.wfo_train_months = train
        self.wfo_test_months = test
        self.wfo_windows = windows


def settings_from_app(app: dict) -> DiscoverySettings:
    """Load discovery settings from SQLite ``app_settings`` with legacy fallbacks."""
    spec = SymbolSpec()
    default_tm = [
        "adaptive_sl",
        "risk_reward_tp",
        "percent_exits",
        "trailing",
        "breakeven",
    ]
    cfg = DiscoverySettings(
        symbols=list(
            _first(app, "discovery_symbols", "agent_symbols", default=settings.SYMBOLS[:2])
        ),
        timeframes=list(
            _first(app, "discovery_timeframes", "agent_timeframes", default=["M15"])
        ),
        months=int(
            _first(app, "discovery_months", "agent_history_months", default=12)
        ),
        engine=str(_first(app, "discovery_engine", default=settings.DEFAULT_ENGINE)),
        deposit=float(_first(app, "discovery_deposit", default=settings.DEFAULT_DEPOSIT)),
        leverage=int(_first(app, "discovery_leverage", default=settings.DEFAULT_LEVERAGE)),
        auto_symbol_economics=bool(
            _first(app, "discovery_auto_symbol_economics", default=True)
        ),
        spread_points=float(
            _first(app, "discovery_spread_points", default=spec.spread_points)
        ),
        slippage_points=float(
            _first(app, "discovery_slippage_points", default=spec.slippage_points)
        ),
        contract_size=float(
            _first(app, "discovery_contract_size", default=spec.contract_size)
        ),
        batch_size=int(
            _first(app, "discovery_batch_size", "agent_batch_size", default=64)
        ),
        target_survivors=int(
            _first(app, "discovery_target_survivors", "agent_target_survivors", default=5)
        ),
        max_candidates=int(
            _first(app, "discovery_max_candidates", "agent_max_candidates", default=500)
        ),
        genetic=bool(_first(app, "discovery_genetic", default=True)),
        edge_first=bool(_first(app, "discovery_edge_first", default=True)),
        mechanics=list(
            _first(
                app,
                "discovery_mechanics",
                default=list(DEFAULT_DISCOVERY_MECHANICS),
            )
        ),
        tm_features=list(_first(app, "discovery_tm_features", default=default_tm)),
        # WFO values are always re-derived from months below.
        wfo_train_months=settings.WFO_TRAIN_MONTHS,
        wfo_test_months=settings.WFO_TEST_MONTHS,
        wfo_windows=settings.WFO_WINDOWS,
        advanced_mode=bool(
            _first(app, "discovery_advanced_mode", "agent_advanced_mode", default=False)
        ),
        complexity_cap=int(
            _first(app, "discovery_complexity_cap", "agent_complexity_cap", default=5)
        ),
        enable_regime_switching=bool(
            _first(
                app,
                "discovery_enable_regime_switching",
                "agent_enable_regime_switching",
                default=True,
            )
        ),
        enable_mtf_context=bool(
            _first(
                app,
                "discovery_enable_mtf_context",
                "agent_enable_mtf_context",
                default=True,
            )
        ),
        feature_toggles=list(
            _first(
                app,
                "discovery_feature_toggles",
                "agent_feature_toggles",
                default=[
                    "momentum",
                    "mean_reversion",
                    "volatility",
                    "market_structure",
                ],
            )
        ),
        validation_level=_remap_persisted_level(
            app,
            "discovery_validation_level",
            default=validation_levels.DEFAULT_LEVEL,
        ),
        progressive_strictness=bool(
            _first(app, "discovery_progressive_strictness", default=True)
        ),
        validation_level_start=_remap_persisted_level(
            app,
            "discovery_validation_level_start",
            default=validation_levels.MIN_LEVEL,
        ),
        progressive_step=max(
            1,
            int(
                _first(
                    app,
                    "discovery_progressive_step",
                    default=validation_levels.DEFAULT_PROGRESSIVE_STEP,
                )
            ),
        ),
        use_custom=bool(_first(app, "discovery_use_custom", default=False)),
        custom_criteria=dict(
            _first(
                app,
                "discovery_custom_criteria",
                "agent_custom_criteria",
                default={},
            )
        ),
        custom_montecarlo=bool(
            _first(app, "discovery_custom_montecarlo", default=settings.MC_ENABLED)
        ),
        custom_mc_runs=int(
            _first(app, "discovery_custom_mc_runs", default=settings.MC_RUNS)
        ),
        base_seed=int(_first(app, "discovery_base_seed", "agent_base_seed", default=1337)),
        recipient_email=str(
            _first(
                app,
                "recipient_email",
                default=(
                    os.environ.get("EA_ALERT_RECIPIENT")
                    or settings.DEFAULT_ALERT_RECIPIENT
                ),
            )
        ),
        alert_min_score=float(
            _first(app, "alert_min_score", default=settings.DEFAULT_ALERT_MIN_SCORE)
        ),
        progress_email_hours=float(
            _first(
                app,
                "progress_email_hours",
                default=settings.DEFAULT_PROGRESS_EMAIL_HOURS,
            )
        ),
        daily_budget_hours=float(
            _first(app, "discovery_daily_budget_hours", default=0.0)
        ),
    )
    cfg.sync_wfo_from_duration()
    return cfg


def settings_to_app(cfg: DiscoverySettings) -> dict:
    """Serialize discovery settings for ``storage.upsert_app_settings``."""
    cfg.sync_wfo_from_duration()
    return {
        "discovery_symbols": list(cfg.symbols),
        "discovery_timeframes": list(cfg.timeframes),
        "discovery_months": int(cfg.months),
        "discovery_engine": str(cfg.engine),
        "discovery_deposit": float(cfg.deposit),
        "discovery_leverage": int(cfg.leverage),
        "discovery_auto_symbol_economics": bool(cfg.auto_symbol_economics),
        "discovery_spread_points": float(cfg.spread_points),
        "discovery_slippage_points": float(cfg.slippage_points),
        "discovery_contract_size": float(cfg.contract_size),
        "discovery_batch_size": int(cfg.batch_size),
        "discovery_target_survivors": int(cfg.target_survivors),
        "discovery_max_candidates": int(cfg.max_candidates),
        "discovery_genetic": bool(cfg.genetic),
        "discovery_edge_first": bool(cfg.edge_first),
        "discovery_mechanics": list(cfg.mechanics),
        "discovery_tm_features": list(cfg.tm_features),
        "discovery_wfo_train_months": int(cfg.wfo_train_months),
        "discovery_wfo_test_months": int(cfg.wfo_test_months),
        "discovery_wfo_windows": int(cfg.wfo_windows),
        "discovery_advanced_mode": bool(cfg.advanced_mode),
        "discovery_complexity_cap": int(cfg.complexity_cap),
        "discovery_enable_regime_switching": bool(cfg.enable_regime_switching),
        "discovery_enable_mtf_context": bool(cfg.enable_mtf_context),
        "discovery_feature_toggles": list(cfg.feature_toggles),
        "discovery_validation_level": int(cfg.validation_level),
        "discovery_progressive_strictness": bool(cfg.progressive_strictness),
        "discovery_validation_level_start": int(cfg.validation_level_start),
        "discovery_progressive_step": int(cfg.progressive_step),
        "validation_level_schema_version": int(
            validation_levels.LEVEL_SCHEMA_VERSION
        ),
        "discovery_use_custom": bool(cfg.use_custom),
        "discovery_custom_criteria": dict(cfg.custom_criteria),
        "discovery_custom_montecarlo": bool(cfg.custom_montecarlo),
        "discovery_custom_mc_runs": int(cfg.custom_mc_runs),
        "discovery_base_seed": int(cfg.base_seed),
        "recipient_email": str(cfg.recipient_email).strip(),
        "alert_min_score": float(cfg.alert_min_score),
        "progress_email_hours": float(cfg.progress_email_hours),
        "discovery_daily_budget_hours": float(cfg.daily_budget_hours),
    }


def _persisted_level_schema(app: dict) -> int:
    raw = _first(app, "validation_level_schema_version", default=1)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 1


def _remap_persisted_level(app: dict, *keys: str, default: int) -> int:
    """Load a stored level and remap when the app still has the legacy schema."""
    raw = int(_first(app, *keys, default=default))
    return validation_levels.remap_legacy_level(
        raw, schema_version=_persisted_level_schema(app)
    )


def effective_validation_level(
    cfg: DiscoverySettings,
    *,
    sweep_index: int,
    sweep_total: int,
) -> int:
    """Resolve the validation level for sweep ``sweep_index`` (orchestrator cursor).

    Fixed mode returns ``cfg.validation_level``. Progressive mode starts at
    ``validation_level_start`` and increases by ``progressive_step`` after each
    full cycle through all symbol×timeframe combinations, capped at
    ``validation_level``.
    """
    if cfg.use_custom or not cfg.progressive_strictness:
        return int(cfg.validation_level)
    start = max(
        validation_levels.MIN_LEVEL,
        min(int(cfg.validation_level_start), int(cfg.validation_level)),
    )
    ceiling = max(start, int(cfg.validation_level))
    if sweep_total <= 0:
        return ceiling
    step = max(1, int(cfg.progressive_step))
    cycle = int(sweep_index) // int(sweep_total)
    return min(start + cycle * step, ceiling)


def build_discovery_payload(
    cfg: DiscoverySettings,
    *,
    symbol: str,
    timeframe: str,
    seed: int | None = None,
    validation_level: int | None = None,
) -> dict:
    """Build a full discovery job payload for one symbol/timeframe sweep."""
    from factory.symbol_class import classify_symbol, recommended_history_months

    sym = symbol.strip().upper()
    # Crypto / metals / indices need longer history so trade-count and WFO
    # gates are reachable without loosening honesty.
    months = recommended_history_months(
        classify_symbol(sym), default=int(cfg.months))
    # Keep WFO structure aligned with the *effective* history window.
    train, test, windows = derive_wfo_from_duration(months)
    start_dt, end_dt = history_start_end(months)

    if cfg.auto_symbol_economics:
        econ = SymbolSpec.defaults_for_symbol(sym)
        spread_points = float(econ.spread_points)
        contract_size = float(econ.contract_size)
        slippage_points = float(econ.slippage_points)
    else:
        spread_points = float(cfg.spread_points)
        contract_size = float(cfg.contract_size)
        slippage_points = float(cfg.slippage_points)

    payload: dict = {
        "symbol": sym,
        "timeframe": timeframe,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "test_duration_months": int(months),
        "engine": cfg.engine,
        "deposit": float(cfg.deposit),
        "leverage": int(cfg.leverage),
        "auto_symbol_economics": bool(cfg.auto_symbol_economics),
        "spread_points": spread_points,
        "slippage_points": slippage_points,
        "contract_size": contract_size,
        "batch_size": int(cfg.batch_size),
        "target_survivors": int(cfg.target_survivors),
        "max_candidates": int(cfg.max_candidates),
        "genetic": bool(cfg.genetic),
        "edge_first": bool(cfg.edge_first),
        "mechanics": list(cfg.mechanics),
        "tm_features": list(cfg.tm_features),
        "wfo_train_months": int(train),
        "wfo_test_months": int(test),
        "wfo_windows": int(windows),
        "advanced_mode": bool(cfg.advanced_mode),
        "complexity_cap": int(cfg.complexity_cap),
        "enable_regime_switching": bool(cfg.enable_regime_switching),
        "enable_mtf_context": bool(cfg.enable_mtf_context),
        "feature_toggles": list(cfg.feature_toggles),
        "hypothesis_families": True,
        "mt5_confirm_survivors": bool(
            getattr(settings, "DISCOVERY_MT5_CONFIRM_SURVIVORS", True)),
    }
    if seed is not None:
        payload["seed"] = int(seed)

    if cfg.use_custom:
        payload["montecarlo"] = bool(cfg.custom_montecarlo)
        payload["mc_runs"] = int(cfg.custom_mc_runs)
        payload["criteria"] = dict(cfg.custom_criteria)
    else:
        # Floor = survivor stop condition (progressive target / fixed dial).
        # Scoring ceiling is always MAX_LEVEL — one backtest, score L1–L16.
        target = int(cfg.validation_level)
        score_ceiling = validation_levels.MAX_LEVEL
        if cfg.progressive_strictness:
            start = max(
                validation_levels.MIN_LEVEL,
                min(int(cfg.validation_level_start), target),
            )
            # Orchestrator passes the effective floor; without an override, start
            # at the progressive floor so a lone payload build isn't pinned to
            # the target.
            if validation_level is not None:
                floor = max(start, min(int(validation_level), target))
            else:
                floor = start
            payload["progressive_strictness"] = True
            payload["validation_level_start"] = int(cfg.validation_level_start)
            payload["progressive_step"] = max(1, int(cfg.progressive_step))
        else:
            # Fixed mode: survivor floor is the UI dial; still score L1–L16.
            level = (
                int(validation_level)
                if validation_level is not None
                else target
            )
            floor = max(
                validation_levels.MIN_LEVEL,
                min(validation_levels.MAX_LEVEL, level),
            )
        payload["validation_level"] = int(floor)
        payload["validation_level_ceiling"] = int(score_ceiling)
        payload["validation_level_floor"] = int(floor)
        payload["validation_level_target"] = int(target)

    try:
        payload["data_source"] = data_mod.peek_source(
            payload["symbol"],
            payload["timeframe"],
            start_dt,
            end_dt,
        )
    except Exception:
        payload["data_source"] = "unknown"

    return payload
