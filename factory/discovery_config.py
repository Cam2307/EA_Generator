"""Shared discovery settings and job-payload builder.

Manual batch runs and the automated orchestrator both read/write the same
``DiscoverySettings`` so every sweep uses an identical discovery pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from config import settings
from factory import data as data_mod
from factory import validation_levels
from factory.backtest.simulator import SymbolSpec
from factory.models import ExecutionMechanicType


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
    """Outer backtest envelope from a single test-duration (months)."""
    end = today or date.today()
    days = max(1, int(months)) * settings.DAYS_PER_MONTH
    start = end - timedelta(days=days)
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc)
    return start_dt, end_dt


@dataclass
class DiscoverySettings:
    symbols: list[str] = field(default_factory=lambda: list(settings.SYMBOLS[:2]))
    timeframes: list[str] = field(default_factory=lambda: ["M15"])
    months: int = 6
    engine: str = settings.DEFAULT_ENGINE
    deposit: float = settings.DEFAULT_DEPOSIT
    leverage: int = settings.DEFAULT_LEVERAGE
    spread_points: float = SymbolSpec().spread_points
    slippage_points: float = SymbolSpec().slippage_points
    contract_size: float = SymbolSpec().contract_size
    batch_size: int = 64
    target_survivors: int = 5
    max_candidates: int = 500
    genetic: bool = True
    mechanics: list[str] = field(
        default_factory=lambda: [m.value for m in ExecutionMechanicType]
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
    use_custom: bool = False
    custom_criteria: dict = field(default_factory=dict)
    custom_montecarlo: bool = settings.MC_ENABLED
    custom_mc_runs: int = settings.MC_RUNS
    base_seed: int = 1337
    recipient_email: str = settings.DEFAULT_ALERT_RECIPIENT
    alert_min_score: float = settings.DEFAULT_ALERT_MIN_SCORE
    progress_email_hours: float = settings.DEFAULT_PROGRESS_EMAIL_HOURS

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
            _first(app, "discovery_months", "agent_history_months", default=6)
        ),
        engine=str(_first(app, "discovery_engine", default=settings.DEFAULT_ENGINE)),
        deposit=float(_first(app, "discovery_deposit", default=settings.DEFAULT_DEPOSIT)),
        leverage=int(_first(app, "discovery_leverage", default=settings.DEFAULT_LEVERAGE)),
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
        mechanics=list(
            _first(
                app,
                "discovery_mechanics",
                default=[m.value for m in ExecutionMechanicType],
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
        validation_level=int(
            _first(app, "discovery_validation_level", default=validation_levels.DEFAULT_LEVEL)
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
            _first(app, "recipient_email", default=settings.DEFAULT_ALERT_RECIPIENT)
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
        "discovery_spread_points": float(cfg.spread_points),
        "discovery_slippage_points": float(cfg.slippage_points),
        "discovery_contract_size": float(cfg.contract_size),
        "discovery_batch_size": int(cfg.batch_size),
        "discovery_target_survivors": int(cfg.target_survivors),
        "discovery_max_candidates": int(cfg.max_candidates),
        "discovery_genetic": bool(cfg.genetic),
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
        "discovery_use_custom": bool(cfg.use_custom),
        "discovery_custom_criteria": dict(cfg.custom_criteria),
        "discovery_custom_montecarlo": bool(cfg.custom_montecarlo),
        "discovery_custom_mc_runs": int(cfg.custom_mc_runs),
        "discovery_base_seed": int(cfg.base_seed),
        "recipient_email": str(cfg.recipient_email).strip(),
        "alert_min_score": float(cfg.alert_min_score),
        "progress_email_hours": float(cfg.progress_email_hours),
    }


def build_discovery_payload(
    cfg: DiscoverySettings,
    *,
    symbol: str,
    timeframe: str,
    seed: int | None = None,
) -> dict:
    """Build a full discovery job payload for one symbol/timeframe sweep."""
    cfg.sync_wfo_from_duration()
    start_dt, end_dt = history_start_end(cfg.months)

    payload: dict = {
        "symbol": symbol.strip().upper(),
        "timeframe": timeframe,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "test_duration_months": int(cfg.months),
        "engine": cfg.engine,
        "deposit": float(cfg.deposit),
        "leverage": int(cfg.leverage),
        "spread_points": float(cfg.spread_points),
        "slippage_points": float(cfg.slippage_points),
        "contract_size": float(cfg.contract_size),
        "batch_size": int(cfg.batch_size),
        "target_survivors": int(cfg.target_survivors),
        "max_candidates": int(cfg.max_candidates),
        "genetic": bool(cfg.genetic),
        "mechanics": list(cfg.mechanics),
        "tm_features": list(cfg.tm_features),
        "wfo_train_months": int(cfg.wfo_train_months),
        "wfo_test_months": int(cfg.wfo_test_months),
        "wfo_windows": int(cfg.wfo_windows),
        "advanced_mode": bool(cfg.advanced_mode),
        "complexity_cap": int(cfg.complexity_cap),
        "enable_regime_switching": bool(cfg.enable_regime_switching),
        "enable_mtf_context": bool(cfg.enable_mtf_context),
        "feature_toggles": list(cfg.feature_toggles),
    }
    if seed is not None:
        payload["seed"] = int(seed)

    if cfg.use_custom:
        payload["montecarlo"] = bool(cfg.custom_montecarlo)
        payload["mc_runs"] = int(cfg.custom_mc_runs)
        payload["criteria"] = dict(cfg.custom_criteria)
    else:
        payload["validation_level"] = int(cfg.validation_level)

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
