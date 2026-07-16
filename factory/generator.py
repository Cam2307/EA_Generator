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
from factory.param_scale import (
    POINT_DISTANCE_PARAM_NAMES, SCALE_RANGE, is_scale_key, sample_log_uniform,
    scale_key_for,
)
from factory.symbol_class import (
    HYPOTHESIS_FAMILIES, PERCENT_SL_SPECS, PERCENT_TP_SPECS, SymbolClass,
    classify_symbol, infer_hypothesis_family as _infer_family_from_values,
    prefers_relative_exits, requires_percent_exits,
    scaled_mechanic_specs, scaled_tm_distance_specs,
)

# ---------------------------------------------------------------------------
# Parameter templates per building block
#
# Point-distance knobs are searched as base × *_scale (SCALE_RANGE 1–20);
# apply_flat_params collapses onto the original key so sim/MQL5 stay stable.
# Multiplier/ratio knobs are widened in place — no second scale dim.
# ---------------------------------------------------------------------------

FILTER_PARAM_SPECS: Dict[EntryFilterType, Dict[str, ParamRange]] = {
    EntryFilterType.PRICE_ACTION_BREAKOUT: {
        "lookback": ParamRange(min=10, max=60, step=2),
        "buffer_points": ParamRange(min=0, max=80, step=2),
        "buffer_scale": SCALE_RANGE,
    },
    EntryFilterType.MTF_VOLATILITY: {
        "atr_period": ParamRange(min=7, max=28, step=1),
        "atr_mult_min": ParamRange(min=0.5, max=2.5, step=0.1),
    },
    EntryFilterType.LIQUIDITY_ZONE: {
        "zone_lookback": ParamRange(min=20, max=100, step=5),
        "zone_points": ParamRange(min=10, max=150, step=5),
        "zone_scale": SCALE_RANGE,
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
        "bb_dev": ParamRange(min=1.5, max=3.5, step=0.25),
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
        "vol_mult": ParamRange(min=1.2, max=4.0, step=0.1),
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
        "std_mult": ParamRange(min=1.0, max=3.5, step=0.1),
    },
    EntryFilterType.ENVELOPES: {
        "env_period": ParamRange(min=14, max=40, step=1),
        "env_deviation": ParamRange(min=0.05, max=0.75, step=0.025),
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

# FX base grids (currency pairs). Wider than legacy 80–800 so Optuna can
# reach swing-style stops; non-FX ignores these and uses percent SL/TP.
MECHANIC_PARAM_SPECS: Dict[ExecutionMechanicType, Dict[str, ParamRange]] = {
    ExecutionMechanicType.STANDARD_SLTP: {
        "sl_points": ParamRange(min=100, max=3000, step=50),
        "sl_scale": SCALE_RANGE,
        "tp_points": ParamRange(min=100, max=4000, step=50),
        "tp_scale": SCALE_RANGE,
    },
    ExecutionMechanicType.DCA_GRID: {
        "grid_step_points": ParamRange(min=100, max=2000, step=50),
        "grid_step_scale": SCALE_RANGE,
        # Mild martingale allowed for search: 1.0 (flat) … 2.0 (2× per level).
        "lot_multiplier": ParamRange(min=1.0, max=2.0, step=0.1),
        "max_levels": ParamRange(min=2, max=4, step=1),
        "basket_tp_points": ParamRange(min=80, max=2500, step=50),
        "basket_tp_scale": SCALE_RANGE,
        # One shared stop for the whole basket, measured from VWAP (same
        # reference as basket TP). When hit, every open leg closes together.
        "basket_sl_points": ParamRange(min=200, max=3500, step=50),
        "basket_sl_scale": SCALE_RANGE,
    },
    ExecutionMechanicType.HEDGE_LAYER: {
        "sl_points": ParamRange(min=150, max=3000, step=50),
        "sl_scale": SCALE_RANGE,
        "tp_points": ParamRange(min=150, max=3500, step=50),
        "tp_scale": SCALE_RANGE,
        "hedge_trigger_points": ParamRange(min=100, max=2000, step=50),
        "hedge_trigger_scale": SCALE_RANGE,
        "hedge_ratio": ParamRange(min=0.5, max=2.0, step=0.1),
    },
    ExecutionMechanicType.PARTIAL_CLOSE: {
        "sl_points": ParamRange(min=100, max=3000, step=50),
        "sl_scale": SCALE_RANGE,
        "tp_points": ParamRange(min=150, max=4000, step=50),
        "tp_scale": SCALE_RANGE,
        "partial_tp_points": ParamRange(min=80, max=2000, step=50),
        "partial_tp_scale": SCALE_RANGE,
        "partial_fraction": ParamRange(min=0.20, max=0.80, step=0.05),
    },
}

# ---------------------------------------------------------------------------
# Trade-management (exit / risk overlay) parameter templates
# ---------------------------------------------------------------------------

TM_PARAM_SPECS: Dict[str, ParamRange] = {
    "atr_period": ParamRange(min=10, max=20, step=2),
    "atr_sl_mult": ParamRange(min=1.0, max=8.0, step=0.5),
    "tp_rr": ParamRange(min=1.0, max=6.0, step=0.5),
    "sl_pct": ParamRange(min=0.1, max=2.5, step=0.1),
    "tp_pct": ParamRange(min=0.1, max=4.0, step=0.1),
    "trail_start_points": ParamRange(min=100, max=2500, step=50),
    "trail_start_scale": SCALE_RANGE,
    "trail_distance_points": ParamRange(min=100, max=2500, step=50),
    "trail_distance_scale": SCALE_RANGE,
    "trail_atr_mult": ParamRange(min=1.5, max=8.0, step=0.5),
    "chandelier_lookback": ParamRange(min=10, max=40, step=5),
    "trail_step_points": ParamRange(min=10, max=200, step=10),
    "trail_step_scale": SCALE_RANGE,
    "be_trigger_points": ParamRange(min=100, max=2000, step=50),
    "be_trigger_scale": SCALE_RANGE,
    "be_offset_points": ParamRange(min=0, max=200, step=10),
    "be_offset_scale": SCALE_RANGE,
    "risk_percent": ParamRange(min=0.5, max=3.0, step=0.25),
    "start_hour": ParamRange(min=0, max=10, step=1),
    "end_hour": ParamRange(min=14, max=23, step=1),
    "max_trades_per_day": ParamRange(min=1, max=10, step=1),
    "daily_loss_pct": ParamRange(min=2, max=10, step=1),
    "cooldown_bars": ParamRange(min=1, max=20, step=1),
    # adaptive regime gate (see factory/regime.py): which regimes to trade
    # (bitmask over codes 0..3) and the classification thresholds — all
    # optimizable, so the search learns which regimes suit the strategy.
    "regime_allow_mask": ParamRange(min=1, max=14, step=1),
    "regime_adx_period": ParamRange(min=10, max=24, step=2),
    "regime_adx_min": ParamRange(min=18, max=32, step=2),
    "regime_atr_period": ParamRange(min=10, max=20, step=2),
    "regime_atr_mult": ParamRange(min=1.05, max=2.0, step=0.05),
    # per-regime entry-lot multipliers (adaptive sizing)
    "regime_size_quiet_range": ParamRange(min=0.25, max=3.0, step=0.25),
    "regime_size_quiet_trend": ParamRange(min=0.25, max=3.0, step=0.25),
    "regime_size_vol_range": ParamRange(min=0.25, max=3.0, step=0.25),
    "regime_size_vol_trend": ParamRange(min=0.25, max=3.0, step=0.25),
    # 2-state Gaussian HMM on log returns (factory/hmm_regime.py)
    "hmm_mu0": ParamRange(min=-0.0005, max=0.0005, step=0.0001),
    "hmm_mu1": ParamRange(min=-0.0005, max=0.0005, step=0.0001),
    "hmm_sigma0": ParamRange(min=0.0002, max=0.0010, step=0.0001),
    "hmm_sigma1": ParamRange(min=0.0010, max=0.0040, step=0.00025),
    "hmm_p00": ParamRange(min=0.80, max=0.98, step=0.02),
    "hmm_p11": ParamRange(min=0.70, max=0.95, step=0.05),
    "hmm_pi0": ParamRange(min=0.3, max=0.7, step=0.1),
    "hmm_min_prob": ParamRange(min=0.50, max=0.80, step=0.05),
    "hmm_allow_mask": ParamRange(min=1, max=3, step=1),
    "hmm_size_state0": ParamRange(min=0.25, max=3.0, step=0.25),
    "hmm_size_state1": ParamRange(min=0.25, max=3.0, step=0.25),
}

# User-selectable trade-management feature categories (UI multiselect keys).
TM_FEATURES: Tuple[str, ...] = (
    "adaptive_sl", "risk_reward_tp", "percent_exits", "trailing", "breakeven",
    "risk_sizing", "time_filter", "safeguards", "cooldown", "regime_filter",
    "regime_sizing", "hmm_regime_filter", "hmm_regime_sizing",
)

# Extra complexity contributed by trade-management overlays (added on top of
# entry-filter costs so advanced EAs actually score higher when HMM/regime
# features are enabled).
_TM_COMPLEXITY_COST: Dict[str, int] = {
    "regime_filter": 2,
    "regime_sizing": 2,
    "hmm_regime_filter": 3,
    "hmm_regime_sizing": 2,
}

# Mechanics whose per-position exits accept the SL/TP/trailing/breakeven
# overlay. Grid + hedge keep their bespoke basket recovery logic; they only
# receive the account-level filters (session / daily-loss / max-trades / cooldown).
_DIRECTIONAL_MECHANICS = (
    ExecutionMechanicType.STANDARD_SLTP, ExecutionMechanicType.PARTIAL_CLOSE,
)


def _tm_specs_for_class(symbol_class: SymbolClass) -> Dict[str, ParamRange]:
    """TM param ranges with point distances + percent bands for ``symbol_class``."""
    specs = scaled_tm_distance_specs(TM_PARAM_SPECS, symbol_class)
    specs["sl_pct"] = PERCENT_SL_SPECS[symbol_class]
    specs["tp_pct"] = PERCENT_TP_SPECS[symbol_class]
    return specs


def _add_tm_param(tm: TradeManagement, name: str, rng: random.Random,
                  specs: Optional[Dict[str, ParamRange]] = None) -> None:
    table = specs or TM_PARAM_SPECS
    r = table[name]
    if is_scale_key(name):
        tm.params[name] = sample_log_uniform(r, rng)
    else:
        n_steps = int(round((r.max - r.min) / r.step)) if r.step > 0 else 0
        tm.params[name] = (
            r.min + rng.randint(0, n_steps) * r.step if n_steps else r.min
        )
    tm.ranges[name] = r
    if name in POINT_DISTANCE_PARAM_NAMES:
        sk = scale_key_for(name)
        if sk in table and sk not in tm.params:
            _add_tm_param(tm, sk, rng, specs=table)


def random_trade_mgmt(mech_type: ExecutionMechanicType, rng: random.Random,
                      allowed: Optional[Sequence[str]] = None,
                      symbol: Optional[str] = None) -> TradeManagement:
    """Randomly configure the exit/risk overlay for a new strategy.

    ``allowed`` restricts which feature categories may be switched on (UI
    control); ``None`` allows all of :data:`TM_FEATURES`. Non-currency-pair
    symbols always use percent SL/TP (0.1%–2.5% / 0.1%–4%). FX stays on
    point / ATR / R:R exits with wider point search bands.
    """
    allow = set(allowed) if allowed is not None else set(TM_FEATURES)
    tm = TradeManagement()
    directional = mech_type in _DIRECTIONAL_MECHANICS
    sym_class = classify_symbol(symbol)
    specs = _tm_specs_for_class(sym_class)
    use_percent = requires_percent_exits(sym_class)
    relative = prefers_relative_exits(sym_class)

    if directional:
        if use_percent:
            # Crypto / metals / indices / oil: percent exits only.
            tm.sl_mode = StopLossMode.PERCENT
            tm.tp_mode = TakeProfitMode.PERCENT
        else:
            if "adaptive_sl" in allow and rng.random() < 0.55:
                tm.sl_mode = StopLossMode.ATR
            if "risk_reward_tp" in allow and rng.random() < 0.45:
                tm.tp_mode = TakeProfitMode.RR
        if "trailing" in allow and rng.random() < 0.6:
            if relative:
                tm.trail_mode = rng.choice(
                    [TrailMode.ATR, TrailMode.ATR, TrailMode.CHANDELIER,
                     TrailMode.FIXED])
            else:
                tm.trail_mode = rng.choice(
                    [TrailMode.FIXED, TrailMode.ATR, TrailMode.CHANDELIER])
        if "breakeven" in allow and rng.random() < 0.5:
            tm.breakeven = True
        if "risk_sizing" in allow and rng.random() < (0.55 if relative else 0.4):
            tm.lot_mode = LotMode.RISK_PERCENT

    if "time_filter" in allow and rng.random() < 0.3:
        tm.time_filter = True
    if "safeguards" in allow and rng.random() < 0.3:
        tm.limit_trades_per_day = True
    if "safeguards" in allow and rng.random() < 0.3:
        tm.daily_loss_enabled = True
    if "cooldown" in allow and rng.random() < 0.3:
        tm.cooldown_enabled = True
    if "regime_filter" in allow and rng.random() < 0.35:
        tm.regime_filter = True
    if "regime_sizing" in allow and rng.random() < 0.3:
        tm.regime_sizing = True
    if "hmm_regime_filter" in allow and rng.random() < 0.28:
        tm.hmm_regime_filter = True
    if "hmm_regime_sizing" in allow and rng.random() < 0.25:
        tm.hmm_regime_sizing = True

    if tm.uses_atr():
        _add_tm_param(tm, "atr_period", rng, specs=specs)
    if tm.sl_mode == StopLossMode.ATR:
        _add_tm_param(tm, "atr_sl_mult", rng, specs=specs)
    if tm.sl_mode == StopLossMode.PERCENT:
        _add_tm_param(tm, "sl_pct", rng, specs=specs)
    if tm.tp_mode == TakeProfitMode.RR:
        _add_tm_param(tm, "tp_rr", rng, specs=specs)
    if tm.tp_mode == TakeProfitMode.PERCENT:
        _add_tm_param(tm, "tp_pct", rng, specs=specs)
    if tm.trail_mode != TrailMode.OFF:
        _add_tm_param(tm, "trail_start_points", rng, specs=specs)
        _add_tm_param(tm, "trail_step_points", rng, specs=specs)
        if tm.trail_mode == TrailMode.FIXED:
            _add_tm_param(tm, "trail_distance_points", rng, specs=specs)
        else:
            _add_tm_param(tm, "trail_atr_mult", rng, specs=specs)
        if tm.trail_mode == TrailMode.CHANDELIER:
            _add_tm_param(tm, "chandelier_lookback", rng, specs=specs)
    if tm.breakeven:
        _add_tm_param(tm, "be_trigger_points", rng, specs=specs)
        _add_tm_param(tm, "be_offset_points", rng, specs=specs)
    if tm.lot_mode == LotMode.RISK_PERCENT:
        _add_tm_param(tm, "risk_percent", rng, specs=specs)
    if tm.time_filter:
        _add_tm_param(tm, "start_hour", rng, specs=specs)
        _add_tm_param(tm, "end_hour", rng, specs=specs)
    if tm.limit_trades_per_day:
        _add_tm_param(tm, "max_trades_per_day", rng, specs=specs)
    if tm.daily_loss_enabled:
        _add_tm_param(tm, "daily_loss_pct", rng, specs=specs)
    if tm.cooldown_enabled:
        _add_tm_param(tm, "cooldown_bars", rng, specs=specs)
    if tm.regime_filter or tm.regime_sizing:
        _add_tm_param(tm, "regime_adx_period", rng, specs=specs)
        _add_tm_param(tm, "regime_adx_min", rng, specs=specs)
        _add_tm_param(tm, "regime_atr_period", rng, specs=specs)
        _add_tm_param(tm, "regime_atr_mult", rng, specs=specs)
    if tm.regime_filter:
        _add_tm_param(tm, "regime_allow_mask", rng, specs=specs)
    if tm.regime_sizing:
        _add_tm_param(tm, "regime_size_quiet_range", rng, specs=specs)
        _add_tm_param(tm, "regime_size_quiet_trend", rng, specs=specs)
        _add_tm_param(tm, "regime_size_vol_range", rng, specs=specs)
        _add_tm_param(tm, "regime_size_vol_trend", rng, specs=specs)
    if tm.hmm_regime_filter or tm.hmm_regime_sizing:
        _add_tm_param(tm, "hmm_mu0", rng, specs=specs)
        _add_tm_param(tm, "hmm_mu1", rng, specs=specs)
        _add_tm_param(tm, "hmm_sigma0", rng, specs=specs)
        _add_tm_param(tm, "hmm_sigma1", rng, specs=specs)
        _add_tm_param(tm, "hmm_p00", rng, specs=specs)
        _add_tm_param(tm, "hmm_p11", rng, specs=specs)
        _add_tm_param(tm, "hmm_pi0", rng, specs=specs)
    if tm.hmm_regime_filter:
        _add_tm_param(tm, "hmm_allow_mask", rng, specs=specs)
        _add_tm_param(tm, "hmm_min_prob", rng, specs=specs)
    if tm.hmm_regime_sizing:
        _add_tm_param(tm, "hmm_size_state0", rng, specs=specs)
        _add_tm_param(tm, "hmm_size_state1", rng, specs=specs)
    return tm


def describe_trade_mgmt(tm: TradeManagement) -> List[str]:
    """Human-readable summary lines for the exit/risk overlay."""
    p = tm.params
    lines: List[str] = []
    if tm.sl_mode == StopLossMode.ATR:
        lines.append(
            f"Adaptive stop: SL = ATR({p.get('atr_period', 14):.0f}) x "
            f"{p.get('atr_sl_mult', 2):.1f}.")
    elif tm.sl_mode == StopLossMode.PERCENT:
        lines.append(f"Percent stop: SL = {p.get('sl_pct', 1):.2f}% of entry price.")
    if tm.tp_mode == TakeProfitMode.RR:
        lines.append(f"Take profit at {p.get('tp_rr', 2):.1f}x the stop distance (R:R).")
    elif tm.tp_mode == TakeProfitMode.PERCENT:
        lines.append(f"Percent target: TP = {p.get('tp_pct', 1.5):.2f}% of entry price.")
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
    if tm.regime_filter:
        from factory.regime import REGIME_NAMES
        mask = int(p.get("regime_allow_mask", 15))
        allowed = [name for code, name in REGIME_NAMES.items()
                   if (mask >> code) & 1]
        lines.append(
            "Regime filter: trade only in "
            + (", ".join(allowed) if allowed else "no")
            + f" conditions (ADX({p.get('regime_adx_period', 14):.0f})"
            f" > {p.get('regime_adx_min', 25):.0f} = trend;"
            f" ATR({p.get('regime_atr_period', 14):.0f}) >"
            f" {p.get('regime_atr_mult', 1.25):.2f}x baseline = volatile).")
    if tm.regime_sizing:
        lines.append(
            "Regime sizing: entry lots x"
            f" {p.get('regime_size_quiet_range', 1):.2f} (quiet range),"
            f" x{p.get('regime_size_quiet_trend', 1):.2f} (quiet trend),"
            f" x{p.get('regime_size_vol_range', 1):.2f} (volatile range),"
            f" x{p.get('regime_size_vol_trend', 1):.2f} (volatile trend).")
    if tm.hmm_regime_filter:
        from factory.hmm_regime import HMM_STATE_NAMES
        mask = int(p.get("hmm_allow_mask", 3))
        allowed = [HMM_STATE_NAMES[c] for c in (0, 1) if (mask >> c) & 1]
        lines.append(
            "HMM regime filter: trade only in "
            + (", ".join(allowed) if allowed else "no")
            + f" (σ₀={p.get('hmm_sigma0', 0.0005):.4f},"
            f" σ₁={p.get('hmm_sigma1', 0.002):.4f},"
            f" min P={p.get('hmm_min_prob', 0.55):.2f}).")
    if tm.hmm_regime_sizing:
        lines.append(
            "HMM regime sizing: entry lots x"
            f"{p.get('hmm_size_state0', 1):.2f} (state 0),"
            f" x{p.get('hmm_size_state1', 1):.2f} (state 1).")
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
    # When True, pick filters from a single hypothesis family (Phase-4 narrowing)
    # instead of freely mixing all compatible indicators.
    hypothesis_families: bool = True
    # Empirical L4+ clear weights (family name / filter type value → weight).
    # Empty/None keeps uniform priors (cold start).
    family_weights: Optional[Dict[str, float]] = None
    filter_weights: Optional[Dict[str, float]] = None


def _sample_params(specs: Dict[str, ParamRange], rng: random.Random) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, r in specs.items():
        if is_scale_key(name):
            out[name] = sample_log_uniform(r, rng)
            continue
        n_steps = int(round((r.max - r.min) / r.step)) if r.step > 0 else 0
        out[name] = r.min + rng.randint(0, n_steps) * r.step if n_steps else r.min
    return out


def infer_hypothesis_family(strategy: StrategyDefinition) -> Optional[str]:
    """Best-matching hypothesis family for ``strategy`` entry filters."""
    return _infer_family_from_values(f.type.value for f in strategy.entry_filters)


def blend_family_weights(
    clear_counts: Optional[Dict[str, int]] = None,
    *,
    prior_strength: float = 8.0,
) -> Dict[str, float]:
    """Blend uniform family priors with empirical L4+ clear counts."""
    weights = {name: 1.0 for name in HYPOTHESIS_FAMILIES}
    if not clear_counts:
        return weights
    for key, n in clear_counts.items():
        if key not in weights:
            continue
        weights[key] = weights[key] + prior_strength * max(0, int(n))
    return weights


def blend_filter_weights(
    clear_counts: Optional[Dict[str, int]] = None,
    *,
    prior_strength: float = 8.0,
) -> Dict[str, float]:
    """Blend uniform filter priors with empirical L4+ clear counts.

    Keys are ``EntryFilterType.value`` strings. Unknown keys are ignored;
    filters with no clears keep weight 1.0.
    """
    weights: Dict[str, float] = {ft.value: 1.0 for ft in EntryFilterType}
    if not clear_counts:
        return weights
    for key, n in clear_counts.items():
        if key not in weights:
            continue
        weights[key] = weights[key] + prior_strength * max(0, int(n))
    return weights


def _weighted_sample(
    items: Sequence[EntryFilterType],
    weights: Dict[str, float],
    k: int,
    rng: random.Random,
) -> List[EntryFilterType]:
    """Sample ``k`` distinct filters with replacement-free weighted draws."""
    pool = list(items)
    if not pool or k <= 0:
        return []
    k = min(k, len(pool))
    chosen: List[EntryFilterType] = []
    for _ in range(k):
        w = [max(1e-9, float(weights.get(ft.value, 1.0))) for ft in pool]
        pick = rng.choices(pool, weights=w, k=1)[0]
        chosen.append(pick)
        pool.remove(pick)
    return chosen


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


def _hypothesis_family_pool(
    compatible: Sequence[EntryFilterType],
    rng: random.Random,
    family_weights: Optional[Dict[str, float]] = None,
) -> List[EntryFilterType]:
    """Restrict to one named hypothesis family intersected with ``compatible``.

    Family choice is weighted by ``family_weights`` (empirical L4+ clears);
    cold start (missing/empty weights) is uniform.
    """
    compat = set(compatible)
    viable_names: List[str] = []
    viable_pools: List[List[EntryFilterType]] = []
    for name, members in HYPOTHESIS_FAMILIES.items():
        family: List[EntryFilterType] = []
        for raw in members:
            try:
                ft = EntryFilterType(raw)
            except ValueError:
                continue
            if ft in compat:
                family.append(ft)
        if family:
            viable_names.append(name)
            viable_pools.append(family)
    if not viable_names:
        return list(compatible)
    weights = [
        max(1e-9, float((family_weights or {}).get(n, 1.0)))
        for n in viable_names
    ]
    idx = rng.choices(range(len(viable_names)), weights=weights, k=1)[0]
    return viable_pools[idx]


def _pick_filter_pack(
    compatible: Sequence[EntryFilterType],
    rng: random.Random,
    settings: GenerationSettings,
) -> List[EntryFilterType]:
    pool_src = compatible
    if settings.hypothesis_families:
        pool_src = _hypothesis_family_pool(
            compatible, rng, family_weights=settings.family_weights)
    allowed = [ft for ft in pool_src if _is_allowed_by_toggle(ft, settings.feature_toggles)]
    if not allowed:
        allowed = list(pool_src) or list(compatible)

    fweights = settings.filter_weights or {}

    if not settings.advanced_mode:
        k = rng.randint(1, min(3, len(allowed)))
        return _weighted_sample(allowed, fweights, k, rng)

    cap = max(2, min(10, int(settings.complexity_cap)))
    chosen: List[EntryFilterType] = []
    budget = cap
    # Stochastic weighted order: higher-weight filters tend to appear first.
    pool = sorted(
        allowed,
        key=lambda ft: rng.random() / max(1e-9, float(fweights.get(ft.value, 1.0))),
    )

    if settings.enable_regime_switching:
        regime = [f for f in pool if f in _REGIME_FILTERS]
        if regime:
            f = _weighted_sample(regime, fweights, 1, rng)[0]
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
    if len(strategy.entry_filters) >= 2:
        logic = getattr(strategy, "signal_logic", "all")
        if logic == "any":
            lines.append("Entry signal: ANY single filter below may trigger.")
        elif logic == "majority":
            need = len(strategy.entry_filters) // 2 + 1
            lines.append(
                f"Entry signal: at least {need} of "
                f"{len(strategy.entry_filters)} filters below must agree.")
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
            f"lot x{mp['lot_multiplier']:.2f}, max {mp['max_levels']:.0f} levels; "
            f"shared basket SL {mp.get('basket_sl_points', 0):.0f} pts / "
            f"TP {mp['basket_tp_points']:.0f} pts from average price."
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


def default_mechanic_weights() -> Dict[ExecutionMechanicType, float]:
    """Priors favoring defined-risk mechanics over recovery grids."""
    return {
        ExecutionMechanicType.STANDARD_SLTP: 4.0,
        ExecutionMechanicType.PARTIAL_CLOSE: 3.0,
        ExecutionMechanicType.DCA_GRID: 1.0,
        ExecutionMechanicType.HEDGE_LAYER: 1.0,
    }


def blend_mechanic_weights(
    clear_counts: Optional[Dict[str, int]] = None,
    *,
    prior_strength: float = 8.0,
) -> Dict[ExecutionMechanicType, float]:
    """Blend fixed priors with empirical L4+ clear counts.

    ``clear_counts`` keys are mechanic ``.value`` strings. Empty/None keeps
    the default priors unchanged.
    """
    weights = default_mechanic_weights()
    if not clear_counts:
        return weights
    for key, n in clear_counts.items():
        try:
            mech = ExecutionMechanicType(key)
        except ValueError:
            continue
        weights[mech] = weights.get(mech, 1.0) + prior_strength * max(0, int(n))
    return weights


def random_strategy(symbol: str, timeframe: str,
                    rng: Optional[random.Random] = None,
                    generation: int = 0,
                    allowed_mechanics: Optional[Sequence[ExecutionMechanicType]] = None,
                    allowed_tm_features: Optional[Sequence[str]] = None,
                    generation_settings: Optional[GenerationSettings] = None,
                    mechanic_weights: Optional[Dict[ExecutionMechanicType, float]] = None,
                    search_phase: Optional[str] = None,
                    ) -> StrategyDefinition:
    """Build a random strategy.

    ``allowed_mechanics`` restricts which execution mechanics may be generated
    (e.g. only DCA/grid + hedging). ``None`` or empty allows every mechanic.
    ``allowed_tm_features`` restricts which trade-management overlays (trailing,
    adaptive SL, etc.) may be switched on; ``None`` allows all.
    ``mechanic_weights`` overrides the default style priors (survivor bias).
    ``search_phase="edge"`` forces a STANDARD_SLTP R:R probe so discovery
    searches entry edges before enumerating execution variants.
    """
    rng = rng or random.Random()
    generation_settings = generation_settings or GenerationSettings()
    edge_phase = (search_phase or "").lower() == "edge"
    sym_class = classify_symbol(symbol)
    mech_specs = scaled_mechanic_specs(MECHANIC_PARAM_SPECS, sym_class)
    # Scale filter point distances (buffer / zone) the same way.
    from factory.symbol_class import point_distance_mult, scale_param_range
    mult = point_distance_mult(sym_class)

    if edge_phase:
        choices = [ExecutionMechanicType.STANDARD_SLTP]
        mech_type = ExecutionMechanicType.STANDARD_SLTP
    else:
        choices = [m for m in (allowed_mechanics or list(ExecutionMechanicType))
                   if m in mech_specs] or list(ExecutionMechanicType)
        # Prefer defined-risk mechanics when several styles are allowed so
        # recovery grids do not crowd the survivor pool under loose gates.
        base_weights = mechanic_weights or default_mechanic_weights()
        weights = [float(base_weights.get(m, 1.0)) for m in choices]
        mech_type = rng.choices(choices, weights=weights, k=1)[0]
    compatible = LOGIC_MATRIX[mech_type]
    filter_types = _pick_filter_pack(compatible, rng, generation_settings)

    filters = []
    for ft in filter_types:
        fspecs = dict(FILTER_PARAM_SPECS[ft])
        if mult != 1.0:
            for pname, pr in list(fspecs.items()):
                if pname.endswith("_points"):
                    fspecs[pname] = scale_param_range(pr, mult)
        filters.append(EntryFilter(
            type=ft, params=_sample_params(fspecs, rng), ranges=dict(fspecs)))
    mspecs = mech_specs[mech_type]
    mechanic = ExecutionMechanic(
        type=mech_type, params=_sample_params(mspecs, rng),
        ranges=dict(mspecs),
    )
    # Signal-composition logic: with 2+ filters, occasionally combine them
    # disjunctively (any) or by vote (majority) instead of the classic AND —
    # a cheap but genuine expansion of the searchable strategy space.
    signal_logic = "all"
    if len(filters) >= 2:
        roll = rng.random()
        if roll > 0.85:
            signal_logic = "majority"
        elif roll > 0.60:
            signal_logic = "any"

    if edge_phase:
        from factory.edge import EDGE_TM_FEATURES, edge_probe_trade_mgmt
        tm = edge_probe_trade_mgmt(symbol, rng)
        tm_allow = allowed_tm_features if allowed_tm_features is not None \
            else EDGE_TM_FEATURES
        # Keep probes lean even if the UI enabled trailing etc.
        _ = tm_allow
        role = "edge"
        phase = "edge"
    else:
        tm = random_trade_mgmt(
            mech_type, rng, allowed_tm_features, symbol=symbol)
        role = ""
        phase = ""

    strat = StrategyDefinition(
        symbol=symbol, timeframe=timeframe, entry_filters=filters,
        signal_logic=signal_logic,
        mechanic=mechanic, risk=RiskBlock(),
        trade_mgmt=tm,
        lineage=Lineage(generation=generation, role=role),
        profile=StrategyProfile(
            advanced_mode=generation_settings.advanced_mode,
            complexity_cap=(
                max(2, int(generation_settings.complexity_cap))
                if generation_settings.advanced_mode else 2
            ),
            regime_switching=bool(generation_settings.enable_regime_switching),
            mtf_context=bool(generation_settings.enable_mtf_context),
            feature_toggles=list(generation_settings.feature_toggles or []),
            search_phase=phase,
        ),
    )
    _apply_advanced_risk_profile(strat, rng, generation_settings)
    strat.profile.complexity_score = _complexity_score(strat)
    if generation_settings.advanced_mode:
        _enforce_complexity_cap(strat, int(generation_settings.complexity_cap))
        strat.profile.complexity_score = _complexity_score(strat)
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


def _complexity_score(strat: StrategyDefinition) -> int:
    filter_cost = sum(_COMPLEXITY_COST.get(f.type, 1) for f in strat.entry_filters)
    tm = strat.trade_mgmt
    tm_cost = 0
    if tm.regime_filter:
        tm_cost += _TM_COMPLEXITY_COST["regime_filter"]
    if tm.regime_sizing:
        tm_cost += _TM_COMPLEXITY_COST["regime_sizing"]
    if tm.hmm_regime_filter:
        tm_cost += _TM_COMPLEXITY_COST["hmm_regime_filter"]
    if tm.hmm_regime_sizing:
        tm_cost += _TM_COMPLEXITY_COST["hmm_regime_sizing"]
    return int(filter_cost + tm_cost)


def _enforce_complexity_cap(strat: StrategyDefinition, cap: int) -> None:
    """Disable costly TM overlays until score fits ``cap`` (filters kept)."""
    cap = max(1, int(cap))
    # Drop highest-cost TM features first.
    drops = (
        ("hmm_regime_sizing", "hmm_regime_sizing"),
        ("hmm_regime_filter", "hmm_regime_filter"),
        ("regime_sizing", "regime_sizing"),
        ("regime_filter", "regime_filter"),
    )
    tm = strat.trade_mgmt
    while _complexity_score(strat) > cap:
        trimmed = False
        for attr, _key in drops:
            if getattr(tm, attr, False):
                setattr(tm, attr, False)
                trimmed = True
                break
        if not trimmed:
            break


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
    if len(clone.entry_filters) >= 2 and rng.random() < 0.10:
        options = [x for x in ("all", "any", "majority")
                   if x != clone.signal_logic]
        clone.signal_logic = rng.choice(options)
        mutations.append(f"signal_logic={clone.signal_logic}")
    blocks = list(clone.entry_filters) + [clone.mechanic, clone.trade_mgmt]
    for block in blocks:
        for name, r in block.ranges.items():
            if rng.random() < rate:
                n_steps = int(round((r.max - r.min) / r.step)) if r.step > 0 else 0
                new_val = r.min + rng.randint(0, n_steps) * r.step if n_steps else r.min
                if new_val != block.params.get(name):
                    block.params[name] = new_val
                    mutations.append(f"{name}={new_val}")
    clone.lineage = Lineage(
        parents=[strategy.id], mutations=mutations,
        generation=strategy.lineage.generation + 1,
        role=strategy.lineage.role,
        edge_id=strategy.lineage.edge_id,
    )
    clone.profile.search_phase = strategy.profile.search_phase
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
        signal_logic=(a.signal_logic if len(chosen) >= 2 else "all"),
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


def evolve_pareto(population: Sequence[Tuple[StrategyDefinition, Sequence[float]]],
                  n_offspring: int, rng: Optional[random.Random] = None
                  ) -> List[StrategyDefinition]:
    """NSGA-II selection on (strategy, objectives) pairs -> offspring batch.

    Parents are chosen by binary tournament on (non-domination rank,
    crowding distance): a lower front wins; within the same front the less
    crowded candidate wins, keeping the search spread along the whole
    profit/risk/stability frontier instead of collapsing onto one scalar
    compromise. Offspring are produced by the same crossover/mutate
    operators as the scalar path.
    """
    rng = rng or random.Random()
    if not population:
        return []
    from factory.pareto import nsga2_rank
    ranks = nsga2_rank([tuple(obj) for _, obj in population])

    def _better(i: int, j: int) -> int:
        (ri, di), (rj, dj) = ranks[i], ranks[j]
        if ri != rj:
            return i if ri < rj else j
        return i if di >= dj else j

    def pick() -> StrategyDefinition:
        i = rng.randrange(len(population))
        j = rng.randrange(len(population))
        return population[_better(i, j)][0]

    offspring: List[StrategyDefinition] = []
    for _ in range(n_offspring):
        if len(population) >= 2 and rng.random() < 0.5:
            child = crossover(pick(), pick(), rng)
        else:
            child = mutate(pick(), rng)
        offspring.append(child)
    return offspring


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
