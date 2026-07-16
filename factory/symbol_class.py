"""Symbol-class economics for exits, ranges, and discovery defaults.

FX point grids are meaningless on BTC/gold/indices. Generation and screening
use this module so SL/TP/trail distances match the instrument's volatility
scale (percent / ATR-first for non-FX; scaled points when FIXED is used).
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

from factory.models import ExecutionMechanicType, ParamRange


class SymbolClass(str, Enum):
    FX = "fx"
    CRYPTO = "crypto"
    METAL = "metal"
    INDEX = "index"
    OIL = "oil"
    OTHER = "other"


# Multiplier applied to the FX-oriented MECHANIC / TM point grids so FIXED
# distances remain economically plausible when percent/ATR is not used.
_POINT_MULT: Dict[SymbolClass, float] = {
    SymbolClass.FX: 1.0,
    SymbolClass.CRYPTO: 250.0,   # ~0.3–2% of BTC at point=0.01
    SymbolClass.METAL: 4.0,      # gold ~0.3–1.5% in typical FIXED bands
    SymbolClass.INDEX: 8.0,
    SymbolClass.OIL: 5.0,
    SymbolClass.OTHER: 2.0,
}

# Percent SL/TP search bands (percent of entry price).
# Non-FX instruments always use these; FX keeps point-based exits.
_NON_FX_SL = ParamRange(min=0.1, max=2.5, step=0.1)
_NON_FX_TP = ParamRange(min=0.1, max=4.0, step=0.1)
PERCENT_SL_SPECS: Dict[SymbolClass, ParamRange] = {
    SymbolClass.CRYPTO: _NON_FX_SL,
    SymbolClass.METAL: _NON_FX_SL,
    SymbolClass.INDEX: _NON_FX_SL,
    SymbolClass.OIL: _NON_FX_SL,
    SymbolClass.OTHER: _NON_FX_SL,
    SymbolClass.FX: ParamRange(min=0.05, max=0.5, step=0.025),
}
PERCENT_TP_SPECS: Dict[SymbolClass, ParamRange] = {
    SymbolClass.CRYPTO: _NON_FX_TP,
    SymbolClass.METAL: _NON_FX_TP,
    SymbolClass.INDEX: _NON_FX_TP,
    SymbolClass.OIL: _NON_FX_TP,
    SymbolClass.OTHER: _NON_FX_TP,
    SymbolClass.FX: ParamRange(min=0.08, max=0.8, step=0.025),
}

# Hypothesis families for Phase-4 search narrowing (filter type value sets).
HYPOTHESIS_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "breakout_atr": (
        "PRICE_ACTION_BREAKOUT", "MTF_VOLATILITY", "ADX_TREND",
        "VOLUME_SURGE", "STDDEV_REGIME",
    ),
    "mean_reversion_pct": (
        "RSI_REVERSION", "BOLLINGER_FADE", "STOCHASTIC", "CCI_REVERSION",
        "WILLIAMS_R", "ENVELOPES", "MFI", "DEMARKER",
    ),
    "trend_follow": (
        "MA_CROSS", "MACD_CROSS", "MOMENTUM", "PARABOLIC_SAR", "ICHIMOKU",
        "DEMA_CROSS", "AWESOME", "FORCE_INDEX", "RVI",
    ),
    "session_structure": (
        "LIQUIDITY_ZONE", "PRICE_ACTION_BREAKOUT", "MTF_VOLATILITY",
        "ADX_TREND",
    ),
}


def infer_hypothesis_family(filter_type_values: Sequence[str]) -> Optional[str]:
    """Best-matching hypothesis family for a set of entry-filter type values."""
    filter_vals = {str(v) for v in filter_type_values if v}
    if not filter_vals:
        return None
    best_name: Optional[str] = None
    best_overlap = 0
    for name, members in HYPOTHESIS_FAMILIES.items():
        overlap = len(filter_vals & set(members))
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name
    return best_name if best_overlap > 0 else None


def normalize_symbol(symbol: Optional[str]) -> str:
    return (symbol or "").upper().replace(".", "").replace(
        " ", "").replace("_", "")


def classify_symbol(symbol: Optional[str]) -> SymbolClass:
    """Map a broker symbol name to an economics class."""
    sym = normalize_symbol(symbol)
    if not sym:
        return SymbolClass.OTHER
    if sym.startswith("BTC") or sym.startswith("ETH") or sym in ("XBTUSD",):
        return SymbolClass.CRYPTO
    if sym.startswith("XAU") or "GOLD" in sym or sym.startswith("XAG") or "SILVER" in sym:
        return SymbolClass.METAL
    if sym in (
        "US30", "DJ30", "US500", "SPX500", "USTEC", "NAS100", "NASDAQ",
        "GER40", "DE40", "UK100", "FTSE", "JP225", "JPN225", "NI225",
    ):
        return SymbolClass.INDEX
    if (sym in ("USOIL", "UKOIL", "WTI", "BRENT", "XTIUSD", "XBRUSD")
            or "OIL" in sym):
        return SymbolClass.OIL
    fx_root = sym[:6] if len(sym) >= 6 else sym
    if len(fx_root) == 6 and fx_root.isalpha():
        return SymbolClass.FX
    return SymbolClass.OTHER


def prefers_relative_exits(symbol_class: SymbolClass) -> bool:
    """Non-FX instruments always use percent SL/TP (not FIXED points)."""
    return symbol_class != SymbolClass.FX


def requires_percent_exits(symbol_class: SymbolClass) -> bool:
    """True for every non-currency-pair instrument."""
    return symbol_class != SymbolClass.FX


def point_distance_mult(symbol_class: SymbolClass) -> float:
    return float(_POINT_MULT.get(symbol_class, 2.0))


def scale_param_range(base: ParamRange, mult: float) -> ParamRange:
    """Scale a point-distance ParamRange by ``mult`` (step floored at 1)."""
    if mult == 1.0:
        return base
    step = max(1.0, round(base.step * mult))
    return ParamRange(
        min=max(step, round(base.min * mult)),
        max=max(step, round(base.max * mult)),
        step=step,
    )


def scaled_mechanic_specs(
    base_specs: Dict[ExecutionMechanicType, Dict[str, ParamRange]],
    symbol_class: SymbolClass,
) -> Dict[ExecutionMechanicType, Dict[str, ParamRange]]:
    """Copy mechanic param specs with point distances scaled for the class."""
    mult = point_distance_mult(symbol_class)
    if mult == 1.0:
        return {m: dict(params) for m, params in base_specs.items()}
    out: Dict[ExecutionMechanicType, Dict[str, ParamRange]] = {}
    for mech, params in base_specs.items():
        scaled: Dict[str, ParamRange] = {}
        for name, rng in params.items():
            if name.endswith("_points"):
                scaled[name] = scale_param_range(rng, mult)
            else:
                scaled[name] = rng
        out[mech] = scaled
    return out


def scaled_tm_distance_specs(
    base_specs: Dict[str, ParamRange],
    symbol_class: SymbolClass,
) -> Dict[str, ParamRange]:
    """Scale trail / breakeven point knobs for the symbol class."""
    mult = point_distance_mult(symbol_class)
    if mult == 1.0:
        return dict(base_specs)
    out: Dict[str, ParamRange] = {}
    for name, rng in base_specs.items():
        if name.endswith("_points"):
            out[name] = scale_param_range(rng, mult)
        else:
            out[name] = rng
    return out


def recommended_history_months(symbol_class: SymbolClass, default: int = 12) -> int:
    """Longer history for crypto so trade-count / WFO gates are reachable."""
    if symbol_class == SymbolClass.CRYPTO:
        return max(int(default), 24)
    if symbol_class in (SymbolClass.METAL, SymbolClass.INDEX):
        return max(int(default), 18)
    return int(default)


def percent_to_points(price: float, pct: float, point: float) -> float:
    """Convert a percent-of-price distance into broker points."""
    if price <= 0.0 or point <= 0.0 or pct <= 0.0:
        return 0.0
    return (price * (pct / 100.0)) / point
