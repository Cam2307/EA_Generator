"""Logic Matrix strategy generator: random sampling + genetic combination."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from factory.models import (
    EntryFilter, EntryFilterType, ExecutionMechanic, ExecutionMechanicType,
    Lineage, LotMode, ParamRange, RiskBlock, StopLossMode, StrategyDefinition,
    StrategyProfile, TakeProfitMode, TradeManagement, TrailMode,
)

# ---------------------------------------------------------------------------
# Parameter templates per building block
# ---------------------------------------------------------------------------

FILTER_PARAM_SPECS: Dict[EntryFilterType, Dict[str, ParamRange]] = {
    EntryFilterType.PRICE_ACTION_BREAKOUT: {
        "lookback": ParamRange(min=10, max=60, step=2),
        "buffer_points": ParamRange(min=0, max=50, step=2),
    },
    EntryFilterType.MTF_VOLATILITY: {
        "atr_period": ParamRange(min=7, max=28, step=1),
        "atr_mult_min": ParamRange(min=0.5, max=1.5, step=0.1),
    },
    EntryFilterType.LIQUIDITY_ZONE: {
        "zone_lookback": ParamRange(min=20, max=100, step=5),
        "zone_points": ParamRange(min=10, max=100, step=5),
    },
    EntryFilterType.RSI_REVERSION: {
        "rsi_period": ParamRange(min=7, max=21, step=1),
        "oversold": ParamRange(min=20, max=35, step=1),
        "overbought": ParamRange(min=65, max=80, step=1),
    },
    EntryFilterType.MA_CROSS: {
        "fast_period": ParamRange(min=5, max=20, step=1),
        "slow_period": ParamRange(min=30, max=100, step=2),
    },
    EntryFilterType.BOLLINGER_FADE: {
        "bb_period": ParamRange(min=14, max=28, step=1),
        "bb_dev": ParamRange(min=1.5, max=3.0, step=0.25),
    },
    EntryFilterType.MACD_CROSS: {
        "fast_ema": ParamRange(min=8, max=16, step=1),
        "slow_ema": ParamRange(min=20, max=40, step=2),
        "signal_period": ParamRange(min=6, max=12, step=1),
    },
    EntryFilterType.STOCHASTIC: {
        "k_period": ParamRange(min=5, max=21, step=1),
        "oversold": ParamRange(min=15, max=30, step=1),
        "overbought": ParamRange(min=70, max=85, step=1),
    },
    EntryFilterType.ADX_TREND: {
        "adx_period": ParamRange(min=10, max=28, step=1),
        "adx_min": ParamRange(min=18, max=35, step=1),
    },
    EntryFilterType.CCI_REVERSION: {
        "cci_period": ParamRange(min=10, max=30, step=1),
        "cci_level": ParamRange(min=80, max=200, step=10),
    },
    EntryFilterType.MOMENTUM: {
        "mom_period": ParamRange(min=8, max=24, step=1),
        "mom_threshold": ParamRange(min=0.05, max=0.5, step=0.025),
    },
    EntryFilterType.WILLIAMS_R: {
        "wpr_period": ParamRange(min=7, max=28, step=1),
        "wpr_oversold": ParamRange(min=-90, max=-70, step=2),
        "wpr_overbought": ParamRange(min=-30, max=-10, step=2),
    },
    EntryFilterType.VOLUME_SURGE: {
        "vol_period": ParamRange(min=10, max=50, step=2),
        "vol_mult": ParamRange(min=1.2, max=3.0, step=0.1),
    },
    EntryFilterType.PARABOLIC_SAR: {
        "sar_step": ParamRange(min=0.01, max=0.05, step=0.005),
        "sar_max": ParamRange(min=0.1, max=0.4, step=0.05),
    },
    EntryFilterType.ICHIMOKU: {
        "tenkan": ParamRange(min=6, max=12, step=1),
        "kijun": ParamRange(min=22, max=34, step=2),
        "senkou": ParamRange(min=44, max=68, step=4),
    },
    EntryFilterType.DEMARKER: {
        "dem_period": ParamRange(min=7, max=28, step=1),
        "dem_oversold": ParamRange(min=0.2, max=0.35, step=0.025),
        "dem_overbought": ParamRange(min=0.65, max=0.8, step=0.025),
    },
    EntryFilterType.AWESOME: {
        "ao_threshold": ParamRange(min=0.0, max=0.0, step=0.0),
    },
    EntryFilterType.FORCE_INDEX: {
        "force_period": ParamRange(min=8, max=26, step=1),
    },
    EntryFilterType.STDDEV_REGIME: {
        "std_period": ParamRange(min=10, max=40, step=2),
        "std_mult": ParamRange(min=1.0, max=2.5, step=0.1),
    },
    EntryFilterType.ENVELOPES: {
        "env_period": ParamRange(min=14, max=40, step=1),
        "env_deviation": ParamRange(min=0.05, max=0.5, step=0.025),
    },
    EntryFilterType.MFI: {
        "mfi_period": ParamRange(min=7, max=28, step=1),
        "mfi_oversold": ParamRange(min=15, max=30, step=1),
        "mfi_overbought": ParamRange(min=70, max=85, step=1),
    },
    EntryFilterType.RVI: {
        "rvi_period": ParamRange(min=6, max=20, step=1),
    },
    EntryFilterType.DEMA_CROSS: {
        "dema_fast": ParamRange(min=8, max=20, step=1),
        "dema_slow": ParamRange(min=30, max=80, step=2),
    },
}

MECHANIC_PARAM_SPECS: Dict[ExecutionMechanicType, Dict[str, ParamRange]] = {
    ExecutionMechanicType.STANDARD_SLTP: {
        "sl_points": ParamRange(min=80, max=600, step=20),
        "tp_points": ParamRange(min=80, max=900, step=20),
    },
    ExecutionMechanicType.DCA_GRID: {
        "grid_step_points": ParamRange(min=80, max=500, step=20),
        "lot_multiplier": ParamRange(min=1.0, max=2.0, step=0.1),
        "max_levels": ParamRange(min=2, max=6, step=1),
        "basket_tp_points": ParamRange(min=50, max=300, step=10),
    },
    ExecutionMechanicType.HEDGE_LAYER: {
        "sl_points": ParamRange(min=150, max=800, step=25),
        "tp_points": ParamRange(min=150, max=800, step=25),
        "hedge_trigger_points": ParamRange(min=80, max=400, step=20),
        "hedge_ratio": ParamRange(min=0.5, max=1.5, step=0.1),
    },
    ExecutionMechanicType.PARTIAL_CLOSE: {
        "sl_points": ParamRange(min=80, max=600, step=20),
        "tp_points": ParamRange(min=150, max=900, step=20),
        "partial_tp_points": ParamRange(min=40, max=400, step=20),
        "partial_fraction": ParamRange(min=0.25, max=0.75, step=0.05),
    },
}

# ---------------------------------------------------------------------------
# Trade-management (exit / risk overlay) parameter templates
# ---------------------------------------------------------------------------

TM_PARAM_SPECS: Dict[str, ParamRange] = {
    "atr_period": ParamRange(min=10, max=20, step=2),
    "atr_sl_mult": ParamRange(min=1.0, max=4.0, step=0.5),
    "tp_rr": ParamRange(min=1.0, max=4.0, step=0.5),
    "trail_start_points": ParamRange(min=100, max=600, step=50),
    "trail_distance_points": ParamRange(min=100, max=600, step=50),
    "trail_atr_mult": ParamRange(min=1.5, max=4.0, step=0.5),
    "chandelier_lookback": ParamRange(min=10, max=40, step=5),
    "trail_step_points": ParamRange(min=10, max=60, step=10),
    "be_trigger_points": ParamRange(min=100, max=500, step=50),
    "be_offset_points": ParamRange(min=0, max=60, step=10),
    "risk_percent": ParamRange(min=0.5, max=2.0, step=0.25),
    "start_hour": ParamRange(min=0, max=10, step=1),
    "end_hour": ParamRange(min=14, max=23, step=1),
    "max_trades_per_day": ParamRange(min=1, max=10, step=1),
    "daily_loss_pct": ParamRange(min=2, max=10, step=1),
    "cooldown_bars": ParamRange(min=1, max=20, step=1),
}

# User-selectable trade-management feature categories (UI multiselect keys).
TM_FEATURES: Tuple[str, ...] = (
    "adaptive_sl", "risk_reward_tp", "trailing", "breakeven",
    "risk_sizing", "time_filter", "safeguards", "cooldown",
)

# Mechanics whose per-position exits accept the SL/TP/trailing/breakeven
# overlay. Grid + hedge keep their bespoke basket recovery logic; they only
# receive the account-level filters (session / daily-loss / max-trades / cooldown).
_DIRECTIONAL_MECHANICS = (
    ExecutionMechanicType.STANDARD_SLTP, ExecutionMechanicType.PARTIAL_CLOSE,
)


def _add_tm_param(tm: TradeManagement, name: str, rng: random.Random) -> None:
    r = TM_PARAM_SPECS[name]
    n_steps = int(round((r.max - r.min) / r.step)) if r.step > 0 else 0
    tm.params[name] = r.min + rng.randint(0, n_steps) * r.step if n_steps else r.min
    tm.ranges[name] = r


def random_trade_mgmt(mech_type: ExecutionMechanicType, rng: random.Random,
                      allowed: Optional[Sequence[str]] = None) -> TradeManagement:
    """Randomly configure the exit/risk overlay for a new strategy.

    ``allowed`` restricts which feature categories may be switched on (UI
    control); ``None`` allows all of :data:`TM_FEATURES`.
    """
    allow = set(allowed) if allowed is not None else set(TM_FEATURES)
    tm = TradeManagement()
    directional = mech_type in _DIRECTIONAL_MECHANICS

    if directional:
        if "adaptive_sl" in allow and rng.random() < 0.5:
            tm.sl_mode = StopLossMode.ATR
        if "risk_reward_tp" in allow and rng.random() < 0.4:
            tm.tp_mode = TakeProfitMode.RR
        if "trailing" in allow and rng.random() < 0.6:
            tm.trail_mode = rng.choice(
                [TrailMode.FIXED, TrailMode.ATR, TrailMode.CHANDELIER])
        if "breakeven" in allow and rng.random() < 0.5:
            tm.breakeven = True
        if "risk_sizing" in allow and rng.random() < 0.4:
            tm.lot_mode = LotMode.RISK_PERCENT

    # account-level filters apply to every mechanic
    if "time_filter" in allow and rng.random() < 0.3:
        tm.time_filter = True
    if "safeguards" in allow and rng.random() < 0.3:
        tm.limit_trades_per_day = True
    if "safeguards" in allow and rng.random() < 0.3:
        tm.daily_loss_enabled = True
    if "cooldown" in allow and rng.random() < 0.3:
        tm.cooldown_enabled = True

    # populate optimizable sub-parameters only for the enabled features
    if tm.uses_atr():
        _add_tm_param(tm, "atr_period", rng)
    if tm.sl_mode == StopLossMode.ATR:
        _add_tm_param(tm, "atr_sl_mult", rng)
    if tm.tp_mode == TakeProfitMode.RR:
        _add_tm_param(tm, "tp_rr", rng)
    if tm.trail_mode != TrailMode.OFF:
        _add_tm_param(tm, "trail_start_points", rng)
        _add_tm_param(tm, "trail_step_points", rng)
        if tm.trail_mode == TrailMode.FIXED:
            _add_tm_param(tm, "trail_distance_points", rng)
        else:
            _add_tm_param(tm, "trail_atr_mult", rng)
        if tm.trail_mode == TrailMode.CHANDELIER:
            _add_tm_param(tm, "chandelier_lookback", rng)
    if tm.breakeven:
        _add_tm_param(tm, "be_trigger_points", rng)
        _add_tm_param(tm, "be_offset_points", rng)
    if tm.lot_mode == LotMode.RISK_PERCENT:
        _add_tm_param(tm, "risk_percent", rng)
    if tm.time_filter:
        _add_tm_param(tm, "start_hour", rng)
        _add_tm_param(tm, "end_hour", rng)
    if tm.limit_trades_per_day:
        _add_tm_param(tm, "max_trades_per_day", rng)
    if tm.daily_loss_enabled:
        _add_tm_param(tm, "daily_loss_pct", rng)
    if tm.cooldown_enabled:
        _add_tm_param(tm, "cooldown_bars", rng)
    return tm


def describe_trade_mgmt(tm: TradeManagement) -> List[str]:
    """Human-readable summary lines for the exit/risk overlay."""
    p = tm.params
    lines: List[str] = []
    if tm.sl_mode == StopLossMode.ATR:
        lines.append(
            f"Adaptive stop: SL = ATR({p.get('atr_period', 14):.0f}) x "
            f"{p.get('atr_sl_mult', 2):.1f}.")
    if tm.tp_mode == TakeProfitMode.RR:
        lines.append(f"Take profit at {p.get('tp_rr', 2):.1f}x the stop distance (R:R).")
    if tm.trail_mode == TrailMode.FIXED:
        lines.append(
            f"Trailing stop: fixed {p.get('trail_distance_points', 0):.0f} pts, "
            f"arms after +{p.get('trail_start_points', 0):.0f} pts "
            f"(step {p.get('trail_step_points', 0):.0f}).")
    elif tm.trail_mode == TrailMode.ATR:
        lines.append(
            f"Trailing stop: ATR({p.get('atr_period', 14):.0f}) x "
            f"{p.get('trail_atr_mult', 2):.1f}, arms after "
            f"+{p.get('trail_start_points', 0):.0f} pts.")
    elif tm.trail_mode == TrailMode.CHANDELIER:
        lines.append(
            f"Chandelier exit: {p.get('chandelier_lookback', 22):.0f}-bar extreme "
            f"-/+ ATR({p.get('atr_period', 14):.0f}) x {p.get('trail_atr_mult', 3):.1f}.")
    if tm.breakeven:
        lines.append(
            f"Breakeven: after +{p.get('be_trigger_points', 0):.0f} pts, move SL to "
            f"entry +{p.get('be_offset_points', 0):.0f} pts.")
    if tm.lot_mode == LotMode.RISK_PERCENT:
        lines.append(f"Position size: risk {p.get('risk_percent', 1):.2f}% of equity per trade.")
    if tm.time_filter:
        lines.append(
            f"Session filter: only trade {p.get('start_hour', 0):.0f}:00–"
            f"{p.get('end_hour', 23):.0f}:00 (server time).")
    if tm.limit_trades_per_day:
        lines.append(f"Cap {p.get('max_trades_per_day', 0):.0f} trades per day.")
    if tm.daily_loss_enabled:
        lines.append(f"Daily loss limit: halt trading after -{p.get('daily_loss_pct', 0):.0f}% on the day.")
    if tm.cooldown_enabled:
        lines.append(f"Cooldown: wait {p.get('cooldown_bars', 0):.0f} bars after a loss.")
    return lines


# Logic Matrix: which entry filters are considered compatible with which
# execution mechanic. Mean-reversion filters pair with DCA/grid;
# breakout/trend filters pair with standard or partial-close management.
LOGIC_MATRIX: Dict[ExecutionMechanicType, List[EntryFilterType]] = {
    ExecutionMechanicType.STANDARD_SLTP: list(EntryFilterType),
    ExecutionMechanicType.PARTIAL_CLOSE: [
        EntryFilterType.PRICE_ACTION_BREAKOUT, EntryFilterType.MA_CROSS,
        EntryFilterType.MTF_VOLATILITY, EntryFilterType.LIQUIDITY_ZONE,
        EntryFilterType.MACD_CROSS, EntryFilterType.ADX_TREND,
        EntryFilterType.MOMENTUM, EntryFilterType.VOLUME_SURGE,
        EntryFilterType.PARABOLIC_SAR, EntryFilterType.ICHIMOKU,
        EntryFilterType.AWESOME, EntryFilterType.FORCE_INDEX,
        EntryFilterType.DEMA_CROSS, EntryFilterType.STDDEV_REGIME,
        EntryFilterType.RVI,
    ],
    ExecutionMechanicType.DCA_GRID: [
        EntryFilterType.RSI_REVERSION, EntryFilterType.BOLLINGER_FADE,
        EntryFilterType.LIQUIDITY_ZONE, EntryFilterType.STOCHASTIC,
        EntryFilterType.CCI_REVERSION, EntryFilterType.WILLIAMS_R,
        EntryFilterType.DEMARKER, EntryFilterType.MFI,
        EntryFilterType.ENVELOPES,
    ],
    ExecutionMechanicType.HEDGE_LAYER: [
        EntryFilterType.PRICE_ACTION_BREAKOUT, EntryFilterType.MA_CROSS,
        EntryFilterType.MTF_VOLATILITY, EntryFilterType.MACD_CROSS,
        EntryFilterType.ADX_TREND, EntryFilterType.MOMENTUM,
        EntryFilterType.PARABOLIC_SAR, EntryFilterType.ICHIMOKU,
        EntryFilterType.AWESOME, EntryFilterType.DEMA_CROSS,
    ],
}

_ADJECTIVES = ["Quantum", "Iron", "Velvet", "Falcon", "Obsidian", "Aurora",
               "Titan", "Zephyr", "Cobalt", "Ember"]
_NOUNS = ["Scalper", "Sentinel", "Harvester", "Navigator", "Breaker",
          "Weaver", "Guardian", "Pulse", "Drift", "Forge"]

_MAGIC_BASE = 770000

_TREND_FILTERS = {
    EntryFilterType.MA_CROSS, EntryFilterType.MACD_CROSS,
    EntryFilterType.ADX_TREND, EntryFilterType.MOMENTUM,
    EntryFilterType.PARABOLIC_SAR, EntryFilterType.ICHIMOKU,
    EntryFilterType.AWESOME, EntryFilterType.FORCE_INDEX,
    EntryFilterType.DEMA_CROSS, EntryFilterType.RVI,
}
_MEAN_REVERSION_FILTERS = {
    EntryFilterType.RSI_REVERSION, EntryFilterType.BOLLINGER_FADE,
    EntryFilterType.STOCHASTIC, EntryFilterType.CCI_REVERSION,
    EntryFilterType.WILLIAMS_R, EntryFilterType.DEMARKER,
    EntryFilterType.MFI, EntryFilterType.ENVELOPES,
}
_VOLATILITY_FILTERS = {
    EntryFilterType.MTF_VOLATILITY, EntryFilterType.STDDEV_REGIME,
    EntryFilterType.VOLUME_SURGE,
}
_STRUCTURE_FILTERS = {
    EntryFilterType.PRICE_ACTION_BREAKOUT, EntryFilterType.LIQUIDITY_ZONE,
}
_REGIME_FILTERS = {
    EntryFilterType.MTF_VOLATILITY, EntryFilterType.STDDEV_REGIME,
    EntryFilterType.ADX_TREND,
}
_COMPLEXITY_COST = {
    EntryFilterType.PRICE_ACTION_BREAKOUT: 2,
    EntryFilterType.MTF_VOLATILITY: 3,
    EntryFilterType.LIQUIDITY_ZONE: 2,
    EntryFilterType.RSI_REVERSION: 1,
    EntryFilterType.MA_CROSS: 1,
    EntryFilterType.BOLLINGER_FADE: 2,
    EntryFilterType.MACD_CROSS: 2,
    EntryFilterType.STOCHASTIC: 1,
    EntryFilterType.ADX_TREND: 2,
    EntryFilterType.CCI_REVERSION: 2,
    EntryFilterType.MOMENTUM: 1,
    EntryFilterType.WILLIAMS_R: 1,
    EntryFilterType.VOLUME_SURGE: 2,
    EntryFilterType.PARABOLIC_SAR: 2,
    EntryFilterType.ICHIMOKU: 3,
    EntryFilterType.DEMARKER: 1,
    EntryFilterType.AWESOME: 1,
    EntryFilterType.FORCE_INDEX: 1,
    EntryFilterType.STDDEV_REGIME: 2,
    EntryFilterType.ENVELOPES: 1,
    EntryFilterType.MFI: 1,
    EntryFilterType.RVI: 2,
    EntryFilterType.DEMA_CROSS: 2,
}


@dataclass(frozen=True)
class GenerationSettings:
    advanced_mode: bool = False
    complexity_cap: int = 4
    enable_regime_switching: bool = False
    enable_mtf_context: bool = False
    feature_toggles: Optional[Sequence[str]] = None


def _sample_params(specs: Dict[str, ParamRange], rng: random.Random) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, r in specs.items():
        n_steps = int(round((r.max - r.min) / r.step)) if r.step > 0 else 0
        out[name] = r.min + rng.randint(0, n_steps) * r.step if n_steps else r.min
    return out


def _is_allowed_by_toggle(ft: EntryFilterType, toggles: Optional[Sequence[str]]) -> bool:
    if not toggles:
        return True
    enabled = set(toggles)
    if ft in _TREND_FILTERS and "momentum" in enabled:
        return True
    if ft in _MEAN_REVERSION_FILTERS and "mean_reversion" in enabled:
        return True
    if ft in _VOLATILITY_FILTERS and "volatility" in enabled:
        return True
    if ft in _STRUCTURE_FILTERS and "market_structure" in enabled:
        return True
    return False


def _pick_filter_pack(
    compatible: Sequence[EntryFilterType],
    rng: random.Random,
    settings: GenerationSettings,
) -> List[EntryFilterType]:
    allowed = [ft for ft in compatible if _is_allowed_by_toggle(ft, settings.feature_toggles)]
    if not allowed:
        allowed = list(compatible)

    if not settings.advanced_mode:
        return rng.sample(allowed, rng.randint(1, min(3, len(allowed))))

    cap = max(2, min(10, int(settings.complexity_cap)))
    chosen: List[EntryFilterType] = []
    budget = cap
    pool = list(allowed)
    rng.shuffle(pool)

    if settings.enable_regime_switching:
        regime = [f for f in pool if f in _REGIME_FILTERS]
        if regime:
            f = rng.choice(regime)
            chosen.append(f)
            budget -= _COMPLEXITY_COST.get(f, 1)

    if settings.enable_mtf_context and EntryFilterType.MTF_VOLATILITY in pool \
            and EntryFilterType.MTF_VOLATILITY not in chosen:
        cost = _COMPLEXITY_COST[EntryFilterType.MTF_VOLATILITY]
        if budget - cost >= 0:
            chosen.append(EntryFilterType.MTF_VOLATILITY)
            budget -= cost

    for ft in pool:
        if ft in chosen:
            continue
        cost = _COMPLEXITY_COST.get(ft, 1)
        if budget - cost < 0:
            continue
        chosen.append(ft)
        budget -= cost
        # anti-bloat: stop once we have enough expressive signal blocks
        if len(chosen) >= 4 and budget <= 1:
            break

    return chosen[: max(1, len(chosen))] or [rng.choice(allowed)]


def _apply_advanced_risk_profile(strat: StrategyDefinition, rng: random.Random,
                                 settings: GenerationSettings) -> None:
    if not settings.advanced_mode:
        return
    tm = strat.trade_mgmt
    # Volatility targeting approximation using existing risk_% sizing controls.
    if tm.lot_mode == LotMode.RISK_PERCENT:
        tm.params["risk_percent"] = min(
            max(tm.params.get("risk_percent", 1.0), 0.35),
            1.5 if settings.enable_regime_switching else 1.8,
        )
    # Dynamic sizing guardrail for aggressive mechanics.
    if strat.mechanic.type in (ExecutionMechanicType.DCA_GRID, ExecutionMechanicType.HEDGE_LAYER):
        strat.risk.max_open_lots = min(strat.risk.max_open_lots, 2.5)
    # Optional time-based exit proxy: force session windows for advanced mode.
    if rng.random() < 0.35:
        tm.time_filter = True
        tm.params.setdefault("start_hour", 6.0)
        tm.params.setdefault("end_hour", 20.0)


def describe_rules(strategy: StrategyDefinition) -> str:
    """Human-readable entry/exit rule math for the curation UI and .md export."""
    lines: List[str] = []
    for f in strategy.entry_filters:
        p = f.params
        if f.type == EntryFilterType.PRICE_ACTION_BREAKOUT:
            lines.append(
                f"BUY when Close > Highest(High, {p['lookback']:.0f}) + {p['buffer_points']:.0f} pts; "
                f"SELL when Close < Lowest(Low, {p['lookback']:.0f}) - {p['buffer_points']:.0f} pts."
            )
        elif f.type == EntryFilterType.MTF_VOLATILITY:
            lines.append(
                f"Only trade when ATR({p['atr_period']:.0f}) > {p['atr_mult_min']:.2f} x "
                f"median ATR (volatility regime filter)."
            )
        elif f.type == EntryFilterType.LIQUIDITY_ZONE:
            lines.append(
                f"BUY within {p['zone_points']:.0f} pts of Lowest(Low, {p['zone_lookback']:.0f}) "
                f"(demand zone); SELL within {p['zone_points']:.0f} pts of "
                f"Highest(High, {p['zone_lookback']:.0f}) (supply zone)."
            )
        elif f.type == EntryFilterType.RSI_REVERSION:
            lines.append(
                f"BUY when RSI({p['rsi_period']:.0f}) < {p['oversold']:.0f}; "
                f"SELL when RSI({p['rsi_period']:.0f}) > {p['overbought']:.0f}."
            )
        elif f.type == EntryFilterType.MA_CROSS:
            lines.append(
                f"BUY when SMA({p['fast_period']:.0f}) crosses above SMA({p['slow_period']:.0f}); "
                f"SELL on the opposite cross."
            )
        elif f.type == EntryFilterType.BOLLINGER_FADE:
            lines.append(
                f"BUY when Close < BB({p['bb_period']:.0f}, {p['bb_dev']:.1f}) lower band; "
                f"SELL when Close > upper band."
            )
        elif f.type == EntryFilterType.MACD_CROSS:
            lines.append(
                f"BUY when MACD({p['fast_ema']:.0f}, {p['slow_ema']:.0f}, "
                f"{p['signal_period']:.0f}) main crosses above signal; "
                f"SELL on the opposite cross."
            )
        elif f.type == EntryFilterType.STOCHASTIC:
            lines.append(
                f"BUY when Stochastic %K({p['k_period']:.0f}) < {p['oversold']:.0f}; "
                f"SELL when %K > {p['overbought']:.0f}."
            )
        elif f.type == EntryFilterType.ADX_TREND:
            lines.append(
                f"Only trade when ADX({p['adx_period']:.0f}) > {p['adx_min']:.0f}; "
                f"BUY when +DI > -DI, SELL when -DI > +DI (trend-strength filter)."
            )
        elif f.type == EntryFilterType.CCI_REVERSION:
            lines.append(
                f"BUY when CCI({p['cci_period']:.0f}) < -{p['cci_level']:.0f}; "
                f"SELL when CCI > +{p['cci_level']:.0f}."
            )
        elif f.type == EntryFilterType.MOMENTUM:
            lines.append(
                f"BUY when Momentum({p['mom_period']:.0f}) > "
                f"{100 + p['mom_threshold']:.2f}; "
                f"SELL when < {100 - p['mom_threshold']:.2f}."
            )
        elif f.type == EntryFilterType.WILLIAMS_R:
            lines.append(
                f"BUY when Williams %R({p['wpr_period']:.0f}) < "
                f"{p['wpr_oversold']:.0f} (oversold); "
                f"SELL when > {p['wpr_overbought']:.0f} (overbought)."
            )
        elif f.type == EntryFilterType.VOLUME_SURGE:
            lines.append(
                f"Only trade when tick volume > {p['vol_mult']:.2f}x its "
                f"{p['vol_period']:.0f}-bar average (liquidity/participation filter)."
            )
        elif f.type == EntryFilterType.PARABOLIC_SAR:
            lines.append(
                f"BUY when price is above Parabolic SAR (step {p['sar_step']:.2f}, "
                f"max {p['sar_max']:.2f}); SELL when below."
            )
        elif f.type == EntryFilterType.ICHIMOKU:
            lines.append(
                f"BUY when Ichimoku Tenkan({p['tenkan']:.0f}) crosses above "
                f"Kijun({p['kijun']:.0f}); SELL on the opposite cross."
            )
        elif f.type == EntryFilterType.DEMARKER:
            lines.append(
                f"BUY when DeMarker({p['dem_period']:.0f}) < {p['dem_oversold']:.2f}; "
                f"SELL when > {p['dem_overbought']:.2f}."
            )
        elif f.type == EntryFilterType.AWESOME:
            lines.append(
                "BUY when the Awesome Oscillator crosses above zero; "
                "SELL when it crosses below."
            )
        elif f.type == EntryFilterType.FORCE_INDEX:
            lines.append(
                f"BUY when Force Index({p['force_period']:.0f}) crosses above zero "
                f"(buying pressure); SELL when it crosses below."
            )
        elif f.type == EntryFilterType.STDDEV_REGIME:
            lines.append(
                f"Only trade when StdDev({p['std_period']:.0f}) > {p['std_mult']:.2f}x "
                f"its average (volatility-expansion filter)."
            )
        elif f.type == EntryFilterType.ENVELOPES:
            lines.append(
                f"BUY when Close < lower Envelope(SMA {p['env_period']:.0f}, "
                f"{p['env_deviation']:.1f}%); SELL when Close > upper envelope."
            )
        elif f.type == EntryFilterType.MFI:
            lines.append(
                f"BUY when Money Flow Index({p['mfi_period']:.0f}) < "
                f"{p['mfi_oversold']:.0f}; SELL when > {p['mfi_overbought']:.0f}."
            )
        elif f.type == EntryFilterType.RVI:
            lines.append(
                f"BUY when RVI({p['rvi_period']:.0f}) main crosses above its "
                f"signal line; SELL on the opposite cross."
            )
        elif f.type == EntryFilterType.DEMA_CROSS:
            lines.append(
                f"BUY when DEMA({p['dema_fast']:.0f}) crosses above "
                f"DEMA({p['dema_slow']:.0f}); SELL on the opposite cross."
            )
    m, mp = strategy.mechanic, strategy.mechanic.params
    if m.type == ExecutionMechanicType.STANDARD_SLTP:
        lines.append(f"Exit: fixed SL {mp['sl_points']:.0f} pts / TP {mp['tp_points']:.0f} pts.")
    elif m.type == ExecutionMechanicType.DCA_GRID:
        lines.append(
            f"Exit: DCA grid — add every {mp['grid_step_points']:.0f} pts against entry, "
            f"lot x{mp['lot_multiplier']:.2f}, max {mp['max_levels']:.0f} levels; basket TP "
            f"{mp['basket_tp_points']:.0f} pts from average price."
        )
    elif m.type == ExecutionMechanicType.HEDGE_LAYER:
        lines.append(
            f"Exit: SL {mp['sl_points']:.0f} / TP {mp['tp_points']:.0f} pts; open opposite hedge "
            f"({mp['hedge_ratio']:.2f}x lots) when {mp['hedge_trigger_points']:.0f} pts underwater."
        )
    elif m.type == ExecutionMechanicType.PARTIAL_CLOSE:
        lines.append(
            f"Exit: close {mp['partial_fraction'] * 100:.0f}% at {mp['partial_tp_points']:.0f} pts "
            f"and move SL to breakeven; remainder to TP {mp['tp_points']:.0f} pts "
            f"(SL {mp['sl_points']:.0f} pts)."
        )
    lines.extend(describe_trade_mgmt(strategy.trade_mgmt))
    if strategy.profile.advanced_mode:
        lines.append(
            f"Advanced profile: complexity {strategy.profile.complexity_score}/"
            f"{strategy.profile.complexity_cap}; regime switching "
            f"{'on' if strategy.profile.regime_switching else 'off'}; "
            f"MTF context {'on' if strategy.profile.mtf_context else 'off'}."
        )
    return "\n".join(lines)


def random_strategy(symbol: str, timeframe: str,
                    rng: Optional[random.Random] = None,
                    generation: int = 0,
                    allowed_mechanics: Optional[Sequence[ExecutionMechanicType]] = None,
                    allowed_tm_features: Optional[Sequence[str]] = None,
                    generation_settings: Optional[GenerationSettings] = None,
                    ) -> StrategyDefinition:
    """Build a random strategy.

    ``allowed_mechanics`` restricts which execution mechanics may be generated
    (e.g. only DCA/grid + hedging). ``None`` or empty allows every mechanic.
    ``allowed_tm_features`` restricts which trade-management overlays (trailing,
    adaptive SL, etc.) may be switched on; ``None`` allows all.
    """
    rng = rng or random.Random()
    generation_settings = generation_settings or GenerationSettings()
    choices = [m for m in (allowed_mechanics or list(ExecutionMechanicType))
               if m in MECHANIC_PARAM_SPECS] or list(ExecutionMechanicType)
    mech_type = rng.choice(choices)
    compatible = LOGIC_MATRIX[mech_type]
    filter_types = _pick_filter_pack(compatible, rng, generation_settings)

    filters = [
        EntryFilter(type=ft, params=_sample_params(FILTER_PARAM_SPECS[ft], rng),
                    ranges=dict(FILTER_PARAM_SPECS[ft]))
        for ft in filter_types
    ]
    mechanic = ExecutionMechanic(
        type=mech_type, params=_sample_params(MECHANIC_PARAM_SPECS[mech_type], rng),
        ranges=dict(MECHANIC_PARAM_SPECS[mech_type]),
    )
    strat = StrategyDefinition(
        symbol=symbol, timeframe=timeframe, entry_filters=filters,
        mechanic=mechanic, risk=RiskBlock(),
        trade_mgmt=random_trade_mgmt(mech_type, rng, allowed_tm_features),
        lineage=Lineage(generation=generation),
        profile=StrategyProfile(
            advanced_mode=generation_settings.advanced_mode,
            complexity_cap=(
                max(2, int(generation_settings.complexity_cap))
                if generation_settings.advanced_mode else 2
            ),
            regime_switching=bool(generation_settings.enable_regime_switching),
            mtf_context=bool(generation_settings.enable_mtf_context),
            feature_toggles=list(generation_settings.feature_toggles or []),
        ),
    )
    _apply_advanced_risk_profile(strat, rng, generation_settings)
    strat.profile.complexity_score = int(
        sum(_COMPLEXITY_COST.get(f.type, 1) for f in strat.entry_filters)
    )
    strat.profile.portfolio_signature = "|".join(
        [
            strat.symbol,
            strat.timeframe,
            strat.mechanic.type.value,
            ",".join(sorted(f.type.value for f in strat.entry_filters)),
        ]
    )
    strat.name = f"{rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)} {strat.id[:6].upper()}"
    strat.magic_number = _MAGIC_BASE + int(strat.id[:6], 16) % 100000
    strat.rule_description = describe_rules(strat)
    return strat


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def mutate(strategy: StrategyDefinition, rng: Optional[random.Random] = None,
           rate: float = 0.3) -> StrategyDefinition:
    """Perturb a random subset of parameters within their ranges."""
    rng = rng or random.Random()
    clone = strategy.model_copy(deep=True)
    clone.id = StrategyDefinition(mechanic=clone.mechanic).id  # new uuid
    mutations: List[str] = []
    blocks = list(clone.entry_filters) + [clone.mechanic, clone.trade_mgmt]
    for block in blocks:
        for name, r in block.ranges.items():
            if rng.random() < rate:
                n_steps = int(round((r.max - r.min) / r.step)) if r.step > 0 else 0
                new_val = r.min + rng.randint(0, n_steps) * r.step if n_steps else r.min
                if new_val != block.params.get(name):
                    block.params[name] = new_val
                    mutations.append(f"{name}={new_val}")
    clone.lineage = Lineage(parents=[strategy.id], mutations=mutations,
                            generation=strategy.lineage.generation + 1)
    clone.profile.complexity_score = int(
        sum(_COMPLEXITY_COST.get(f.type, 1) for f in clone.entry_filters)
    )
    clone.profile.portfolio_signature = "|".join(
        [
            clone.symbol,
            clone.timeframe,
            clone.mechanic.type.value,
            ",".join(sorted(f.type.value for f in clone.entry_filters)),
        ]
    )
    clone.name = strategy.name.rsplit(" ", 1)[0] + f" {clone.id[:6].upper()}"
    clone.magic_number = _MAGIC_BASE + int(clone.id[:6], 16) % 100000
    clone.rule_description = describe_rules(clone)
    return clone


def crossover(a: StrategyDefinition, b: StrategyDefinition,
              rng: Optional[random.Random] = None) -> StrategyDefinition:
    """Combine filter blocks of two parents; mechanic inherited from parent A.

    Only filters compatible with A's mechanic (per the Logic Matrix) survive.
    """
    rng = rng or random.Random()
    compatible = LOGIC_MATRIX[a.mechanic.type]
    pool = [f for f in (a.entry_filters + b.entry_filters) if f.type in compatible]
    # de-duplicate by type, preferring random parent order
    rng.shuffle(pool)
    seen: Dict[EntryFilterType, EntryFilter] = {}
    for f in pool:
        seen.setdefault(f.type, f)
    chosen = list(seen.values())[: max(1, min(3, len(seen)))]

    child = StrategyDefinition(
        symbol=a.symbol, timeframe=a.timeframe,
        entry_filters=[f.model_copy(deep=True) for f in chosen],
        mechanic=a.mechanic.model_copy(deep=True), risk=a.risk.model_copy(deep=True),
        trade_mgmt=a.trade_mgmt.model_copy(deep=True),
        lineage=Lineage(parents=[a.id, b.id],
                        generation=max(a.lineage.generation, b.lineage.generation) + 1),
        profile=a.profile.model_copy(deep=True),
    )
    child.profile.complexity_score = int(
        sum(_COMPLEXITY_COST.get(f.type, 1) for f in child.entry_filters)
    )
    child.profile.portfolio_signature = "|".join(
        [
            child.symbol,
            child.timeframe,
            child.mechanic.type.value,
            ",".join(sorted(f.type.value for f in child.entry_filters)),
        ]
    )
    child.name = f"{rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)} {child.id[:6].upper()}"
    child.magic_number = _MAGIC_BASE + int(child.id[:6], 16) % 100000
    child.rule_description = describe_rules(child)
    return child


def evolve(population: Sequence[Tuple[StrategyDefinition, float]],
           n_offspring: int, rng: Optional[random.Random] = None,
           tournament_k: int = 3) -> List[StrategyDefinition]:
    """Tournament selection on (strategy, fitness) pairs -> offspring batch."""
    rng = rng or random.Random()
    if not population:
        return []

    def pick() -> StrategyDefinition:
        contenders = [rng.choice(population) for _ in range(min(tournament_k, len(population)))]
        return max(contenders, key=lambda t: t[1])[0]

    offspring: List[StrategyDefinition] = []
    for _ in range(n_offspring):
        if len(population) >= 2 and rng.random() < 0.5:
            child = crossover(pick(), pick(), rng)
        else:
            child = mutate(pick(), rng)
        offspring.append(child)
    return offspring
