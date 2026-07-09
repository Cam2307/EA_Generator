"""Assemble a standalone, validator-proof .mq5 Expert Advisor from templates.

Templates live in ``templates/`` and are split into named sections with
``//@SECTION <NAME>`` markers. The renderer substitutes:

- ``{I}``        -> filter index
- ``{IN_param}`` -> MQL5 input variable name for that parameter
- ``{P_param}``  -> the parameter's current value

Output is deterministic for a given StrategyDefinition.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

from factory.models import (
    EntryFilter, EntryFilterType, ExecutionMechanic, ExecutionMechanicType,
    LotMode, StopLossMode, StrategyDefinition, TakeProfitMode, TradeManagement,
    TrailMode,
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Neutral defaults for trade-management inputs whose feature is switched off,
# so the generated EA still declares every input with a sane (no-op) value.
_TM_NEUTRAL: Dict[str, float] = {
    "atr_period": 14, "atr_sl_mult": 2.0, "tp_rr": 2.0,
    "trail_start_points": 300, "trail_distance_points": 300,
    "trail_atr_mult": 3.0, "chandelier_lookback": 22, "trail_step_points": 10,
    "be_trigger_points": 300, "be_offset_points": 0, "risk_percent": 1.0,
    "start_hour": 0, "end_hour": 24, "max_trades_per_day": 5,
    "daily_loss_pct": 5, "cooldown_bars": 5,
}
_SL_MODE_INT = {StopLossMode.OFF: 0, StopLossMode.FIXED: 1, StopLossMode.ATR: 2}
_TP_MODE_INT = {TakeProfitMode.OFF: 0, TakeProfitMode.FIXED: 1, TakeProfitMode.RR: 2}
_TRAIL_MODE_INT = {TrailMode.OFF: 0, TrailMode.FIXED: 1, TrailMode.ATR: 2,
                   TrailMode.CHANDELIER: 3}
_LOT_MODE_INT = {LotMode.FIXED: 0, LotMode.RISK_PERCENT: 1}

_FILTER_TEMPLATE_FILES: Dict[EntryFilterType, str] = {
    EntryFilterType.PRICE_ACTION_BREAKOUT: "filter_price_action_breakout.mq5",
    EntryFilterType.MTF_VOLATILITY: "filter_mtf_volatility.mq5",
    EntryFilterType.LIQUIDITY_ZONE: "filter_liquidity_zone.mq5",
    EntryFilterType.RSI_REVERSION: "filter_rsi_reversion.mq5",
    EntryFilterType.MA_CROSS: "filter_ma_cross.mq5",
    EntryFilterType.BOLLINGER_FADE: "filter_bollinger_fade.mq5",
    EntryFilterType.MACD_CROSS: "filter_macd_cross.mq5",
    EntryFilterType.STOCHASTIC: "filter_stochastic.mq5",
    EntryFilterType.ADX_TREND: "filter_adx_trend.mq5",
    EntryFilterType.CCI_REVERSION: "filter_cci_reversion.mq5",
    EntryFilterType.MOMENTUM: "filter_momentum.mq5",
    EntryFilterType.WILLIAMS_R: "filter_williams_r.mq5",
    EntryFilterType.VOLUME_SURGE: "filter_volume_surge.mq5",
    EntryFilterType.PARABOLIC_SAR: "filter_parabolic_sar.mq5",
    EntryFilterType.ICHIMOKU: "filter_ichimoku.mq5",
    EntryFilterType.DEMARKER: "filter_demarker.mq5",
    EntryFilterType.AWESOME: "filter_awesome.mq5",
    EntryFilterType.FORCE_INDEX: "filter_force_index.mq5",
    EntryFilterType.STDDEV_REGIME: "filter_stddev_regime.mq5",
    EntryFilterType.ENVELOPES: "filter_envelopes.mq5",
    EntryFilterType.MFI: "filter_mfi.mq5",
    EntryFilterType.RVI: "filter_rvi.mq5",
    EntryFilterType.DEMA_CROSS: "filter_dema_cross.mq5",
}

_MECHANIC_TEMPLATE_FILES: Dict[ExecutionMechanicType, str] = {
    ExecutionMechanicType.STANDARD_SLTP: "mechanic_standard_sltp.mq5",
    ExecutionMechanicType.DCA_GRID: "mechanic_dca_grid.mq5",
    ExecutionMechanicType.HEDGE_LAYER: "mechanic_hedge_layer.mq5",
    ExecutionMechanicType.PARTIAL_CLOSE: "mechanic_partial_close.mq5",
}

_SECTION_RE = re.compile(r"^//@SECTION\s+(\w+)\s*$", re.MULTILINE)


def _load_sections(filename: str) -> Dict[str, str]:
    text = (TEMPLATES_DIR / filename).read_text(encoding="utf-8")
    sections: Dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(text))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[m.group(1)] = text[start:end].strip("\n")
    return sections


def format_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def mql5_input_name(flat_key: str) -> str:
    """Map a flat parameter key to the generated EA's input variable name.

    ``F0_PRICE_ACTION_BREAKOUT_lookback`` -> ``Inp_F0_lookback``
    ``M_DCA_GRID_grid_step_points``       -> ``Inp_M_grid_step_points``
    """
    m = re.match(r"^F(\d+)_[A-Z_]+?_([a-z][a-z0-9_]*)$", flat_key)
    if m:
        return f"Inp_F{m.group(1)}_{m.group(2)}"
    m = re.match(r"^M_[A-Z_]+?_([a-z][a-z0-9_]*)$", flat_key)
    if m:
        return f"Inp_M_{m.group(1)}"
    m = re.match(r"^X_([a-z][a-z0-9_]*)$", flat_key)
    if m:
        return f"Inp_X_{m.group(1)}"
    raise ValueError(f"Unrecognized flat parameter key: {flat_key}")


def _tm_replacements(tm: TradeManagement) -> Dict[str, str]:
    """Concrete default values for the static trade-management inputs."""
    def val(key: str) -> str:
        return format_value(tm.params.get(key, _TM_NEUTRAL[key]))
    return {
        "__TM_SL_MODE__": str(_SL_MODE_INT[tm.sl_mode]),
        "__TM_ATR_PERIOD__": val("atr_period"),
        "__TM_ATR_SL_MULT__": val("atr_sl_mult"),
        "__TM_TP_MODE__": str(_TP_MODE_INT[tm.tp_mode]),
        "__TM_TP_RR__": val("tp_rr"),
        "__TM_TRAIL_MODE__": str(_TRAIL_MODE_INT[tm.trail_mode]),
        "__TM_TRAIL_START__": val("trail_start_points"),
        "__TM_TRAIL_DIST__": val("trail_distance_points"),
        "__TM_TRAIL_ATR_MULT__": val("trail_atr_mult"),
        "__TM_CHAND_LB__": val("chandelier_lookback"),
        "__TM_TRAIL_STEP__": val("trail_step_points"),
        "__TM_BREAKEVEN__": "1" if tm.breakeven else "0",
        "__TM_BE_TRIGGER__": val("be_trigger_points"),
        "__TM_BE_OFFSET__": val("be_offset_points"),
        "__TM_LOT_MODE__": str(_LOT_MODE_INT[tm.lot_mode]),
        "__TM_RISK_PCT__": val("risk_percent"),
        "__TM_TIME_FILTER__": "1" if tm.time_filter else "0",
        "__TM_START_HOUR__": val("start_hour"),
        "__TM_END_HOUR__": val("end_hour"),
        "__TM_LIMIT_TRADES__": "1" if tm.limit_trades_per_day else "0",
        "__TM_MAX_TRADES__": val("max_trades_per_day"),
        "__TM_DAILY_LOSS__": "1" if tm.daily_loss_enabled else "0",
        "__TM_DAILY_LOSS_PCT__": val("daily_loss_pct"),
        "__TM_COOLDOWN__": "1" if tm.cooldown_enabled else "0",
        "__TM_COOLDOWN_BARS__": val("cooldown_bars"),
    }


def _substitute(text: str, index: int, params: Dict[str, float],
                prefix: str) -> str:
    out = text.replace("{I}", str(index))
    for name, value in params.items():
        out = out.replace(f"{{IN_{name}}}", f"Inp_{prefix}_{name}")
        out = out.replace(f"{{P_{name}}}", format_value(value))
    return out


def _min_bars(strategy: StrategyDefinition) -> int:
    """Bars required before the EA is allowed to trade (history guard)."""
    needed = 100  # MTF volatility baseline window
    for f in strategy.entry_filters:
        for key in ("lookback", "zone_lookback", "slow_period", "bb_period",
                    "rsi_period", "atr_period", "slow_ema", "signal_period",
                    "k_period", "adx_period", "cci_period", "mom_period",
                    "wpr_period", "vol_period", "kijun", "senkou",
                    "dem_period", "force_period", "std_period", "env_period",
                    "mfi_period", "rvi_period", "dema_slow"):
            if key in f.params:
                needed = max(needed, int(f.params[key]) * 3)
    tm = strategy.trade_mgmt
    if "atr_period" in tm.params:
        needed = max(needed, int(tm.params["atr_period"]) * 3)
    if "chandelier_lookback" in tm.params:
        needed = max(needed, int(tm.params["chandelier_lookback"]) + 5)
    return needed + 10


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or "EAFactoryExpert"


def render_ea(strategy: StrategyDefinition) -> str:
    """Render the complete .mq5 source for a strategy."""
    base = (TEMPLATES_DIR / "base.mq5").read_text(encoding="utf-8")

    input_blocks: List[str] = []
    global_decls: List[str] = []
    init_blocks: List[str] = []
    release_blocks: List[str] = []
    long_terms: List[str] = []
    short_terms: List[str] = []
    functions: List[str] = []

    for i, f in enumerate(strategy.entry_filters):
        sections = _load_sections(_FILTER_TEMPLATE_FILES[f.type])
        prefix = f"F{i}"
        sub = lambda key: _substitute(sections.get(key, ""), i, f.params, prefix)
        if sections.get("INPUTS"):
            input_blocks.append(sub("INPUTS"))
        if sections.get("GLOBALS"):
            global_decls.append(sub("GLOBALS"))
        if sections.get("INIT"):
            init_blocks.append(sub("INIT"))
        if sections.get("RELEASE"):
            release_blocks.append(sub("RELEASE"))
        long_terms.append(f"   ok = ok && {sub('LONG_EXPR').strip()};")
        short_terms.append(f"   ok = ok && {sub('SHORT_EXPR').strip()};")
        functions.append(sub("FUNCTIONS"))

    mech = strategy.mechanic
    msections = _load_sections(_MECHANIC_TEMPLATE_FILES[mech.type])
    msub = lambda key: _substitute(msections.get(key, ""), 0, mech.params, "M")
    if msections.get("INPUTS"):
        input_blocks.append(msub("INPUTS"))
    if msections.get("GLOBALS"):
        global_decls.append(msub("GLOBALS"))
    mechanic_functions = msub("FUNCTIONS")

    description = strategy.rule_description.replace("\n", " ")[:250]

    ea = base
    replacements = {
        "__EA_NAME__": _sanitize_name(strategy.name),
        "__STRATEGY_ID__": strategy.id,
        "__EA_DESCRIPTION__": description.replace('"', "'"),
        "__MAGIC__": str(strategy.magic_number),
        "__LOTS__": format_value(strategy.risk.fixed_lots),
        "__MAX_SPREAD__": format_value(strategy.risk.max_spread_points),
        "__MAX_OPEN_LOTS__": format_value(strategy.risk.max_open_lots),
        "__MIN_BARS__": str(_min_bars(strategy)),
        "__INPUT_BLOCKS__": "\n\n".join(input_blocks),
        "__GLOBAL_DECLS__": "\n".join(global_decls),
        "__INIT_INDICATORS__": "\n".join(init_blocks),
        "__RELEASE_INDICATORS__": "\n".join(release_blocks),
        "__SIGNAL_LONG__": "\n".join(long_terms),
        "__SIGNAL_SHORT__": "\n".join(short_terms),
        "__FILTER_FUNCTIONS__": "\n\n".join(functions),
        "__MECHANIC_FUNCTIONS__": mechanic_functions,
    }
    replacements.update(_tm_replacements(strategy.trade_mgmt))
    for token, value in replacements.items():
        ea = ea.replace(token, value)
    return ea


def mql5_inputs_for(strategy: StrategyDefinition):
    """Map a strategy's flat params/ranges to the generated EA's input names.

    Returns ``(inputs, ranges)`` where keys are MQL5 input variable names,
    ready for [TesterInputs] blocks and .set export — including the execution
    mechanic parameters, which are optimizable exactly like filter parameters.
    """
    inputs = {mql5_input_name(k): v for k, v in strategy.all_params().items()}
    ranges = {mql5_input_name(k): r for k, r in strategy.all_ranges().items()}
    return inputs, ranges


def write_ea(strategy: StrategyDefinition, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_sanitize_name(strategy.name)}.mq5"
    path.write_text(render_ea(strategy), encoding="utf-8")
    return path
