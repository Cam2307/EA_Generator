"""Edge-first discovery: signal edge search, then execution-style expansion.

Discovery prefers finding *entry* edges (win rate vs R:R / expectancy) under a
simple defined-risk probe. Only after an edge clears validation do we enumerate
execution/mechanic variants (partial close, trailing overlays, DCA, …) as
candidate MQL5 EA strategies that *use* that edge.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from factory.models import (
    BacktestMetrics, ExecutionMechanic, ExecutionMechanicType, Lineage,
    StopLossMode, StrategyDefinition, TakeProfitMode, TradeManagement,
    TrailMode, ValidationReport,
)
from factory.symbol_class import (
    classify_symbol, requires_percent_exits, scaled_mechanic_specs,
)

# Probe mechanic for edge search — isolates signal quality from recovery grids.
EDGE_PROBE_MECHANIC = ExecutionMechanicType.STANDARD_SLTP

# Default execution styles tried *after* an edge qualifies.
DEFAULT_EXECUTION_MECHANICS: List[str] = [
    ExecutionMechanicType.STANDARD_SLTP.value,
    ExecutionMechanicType.PARTIAL_CLOSE.value,
]

# Minimal TM for edge probes: stop + R:R (or percent on non-FX). No trailing /
# DCA overlays during the search so win-rate vs reward is readable.
EDGE_TM_FEATURES: Tuple[str, ...] = (
    "adaptive_sl",
    "risk_reward_tp",
    "percent_exits",
)

# Soft edge gates used for ranking / expansion eligibility (validation floor
# remains the hard survivor gate).
MIN_EDGE_TRADES = 6
MIN_EDGE_EXPECTANCY = 0.0
MIN_EDGE_PROFIT_FACTOR = 1.05
# Win rate must beat break-even at the realized payoff ratio with a cushion.
EDGE_WR_CUSHION = 0.03


def enrich_trade_stats(m: BacktestMetrics) -> BacktestMetrics:
    """Fill win_rate / expectancy / avg_win / avg_loss when missing.

    Prefer values already set by the simulator; otherwise derive from
    gross_profit / gross_loss / trade_count (works for archived metrics too).
    """
    if m.trade_count <= 0:
        return m
    if m.win_rate > 0.0 or m.expectancy != 0.0:
        return m
    # Reconstruct approximate win count from averages when only aggregates
    # exist — not exact without per-trade list, but usable for ranking.
    if m.avg_win <= 0.0 and m.gross_profit > 0.0 and m.trade_count > 0:
        # Without win count we cannot split; leave win_rate 0 and set expectancy.
        m.expectancy = round(m.net_profit / m.trade_count, 4)
        return m
    return m


def payoff_ratio(m: BacktestMetrics) -> float:
    """Average win / average loss (absolute). 0 when undefined."""
    if m.avg_loss <= 0.0:
        return 0.0 if m.avg_win <= 0.0 else 99.0
    return float(m.avg_win) / float(m.avg_loss)


def break_even_win_rate(rr: float) -> float:
    """Win rate needed for zero expectancy at payoff ratio ``rr``."""
    if rr <= 0.0:
        return 1.0
    return 1.0 / (1.0 + rr)


def edge_score(m: BacktestMetrics) -> float:
    """Scalar ranking for edge probes: expectancy × sample × WR cushion.

    Positive only when expectancy and profit factor look tradeable.
    """
    if m.trade_count < 2 or m.net_profit <= 0.0:
        return -1.0
    exp = m.expectancy if m.expectancy != 0.0 else (
        m.net_profit / max(m.trade_count, 1)
    )
    if exp <= 0.0 or m.profit_factor < 1.0:
        return -1.0
    rr = payoff_ratio(m)
    be = break_even_win_rate(rr) if rr > 0 else 0.5
    wr = m.win_rate if m.win_rate > 0 else 0.5
    wr_edge = wr - be
    sample = min(m.trade_count, 80) / 80.0
    smoothness = 0.5 + 0.5 * max(0.0, min(1.0, m.r_squared))
    return float(exp) * (1.0 + max(wr_edge, 0.0)) * (0.5 + sample) * smoothness


def has_signal_edge(
    m: BacktestMetrics,
    *,
    min_trades: int = MIN_EDGE_TRADES,
    min_expectancy: float = MIN_EDGE_EXPECTANCY,
    min_pf: float = MIN_EDGE_PROFIT_FACTOR,
    wr_cushion: float = EDGE_WR_CUSHION,
) -> bool:
    """True when metrics show a usable entry edge (WR vs realized R:R)."""
    if m.trade_count < min_trades:
        return False
    if m.net_profit <= 0.0 or m.profit_factor < min_pf:
        return False
    exp = m.expectancy if m.expectancy != 0.0 else (
        m.net_profit / max(m.trade_count, 1)
    )
    if exp < min_expectancy:
        return False
    rr = payoff_ratio(m)
    if rr <= 0.0:
        # No average-loss info — fall back to PF + expectancy only.
        return exp > 0.0
    if m.win_rate <= 0.0:
        return exp > 0.0 and m.profit_factor >= min_pf
    return m.win_rate >= break_even_win_rate(rr) + wr_cushion


def report_has_edge(report: ValidationReport) -> bool:
    """Edge check on OOS metrics of a finished validation report."""
    return bool(report.passed) and has_signal_edge(report.oos_metrics)


def is_edge_probe(strategy: StrategyDefinition) -> bool:
    role = (strategy.lineage.role or strategy.profile.search_phase or "").lower()
    if role in ("edge", "edge_probe"):
        return True
    if role in ("execution", "mechanic_variant"):
        return False
    # Legacy / unmarked: treat STANDARD_SLTP without expansion tag as probe.
    return strategy.mechanic.type == EDGE_PROBE_MECHANIC


def is_execution_variant(strategy: StrategyDefinition) -> bool:
    role = (strategy.lineage.role or strategy.profile.search_phase or "").lower()
    return role in ("execution", "mechanic_variant")


def _sample_edge_tp_rr(rng: random.Random, specs: Dict) -> float:
    """Bias R:R toward the break-even-friendly 1.5–3.0 band (~75% of draws)."""
    from factory.models import ParamRange

    r = specs["tp_rr"]
    if not isinstance(r, ParamRange):
        r = ParamRange(min=1.0, max=6.0, step=0.5)
    # Preferred band for edge probes; fall back to full range otherwise.
    if rng.random() < 0.75:
        lo = max(float(r.min), 1.5)
        hi = min(float(r.max), 3.0)
        if hi >= lo and r.step > 0:
            n_steps = int(round((hi - lo) / r.step))
            return lo + rng.randint(0, max(0, n_steps)) * r.step
    n_steps = int(round((r.max - r.min) / r.step)) if r.step > 0 else 0
    return float(r.min + rng.randint(0, n_steps) * r.step if n_steps else r.min)


def edge_probe_trade_mgmt(
    symbol: str,
    rng: random.Random,
) -> TradeManagement:
    """Fixed, readable exit overlay for edge search (RR or percent)."""
    from factory.generator import _add_tm_param, _tm_specs_for_class

    tm = TradeManagement()
    sym_class = classify_symbol(symbol)
    specs = _tm_specs_for_class(sym_class)
    if requires_percent_exits(sym_class):
        tm.sl_mode = StopLossMode.PERCENT
        tm.tp_mode = TakeProfitMode.PERCENT
        _add_tm_param(tm, "sl_pct", rng, specs=specs)
        _add_tm_param(tm, "tp_pct", rng, specs=specs)
    else:
        # Prefer ATR stop + R:R target so win-rate vs reward is explicit.
        tm.sl_mode = StopLossMode.ATR if rng.random() < 0.80 else StopLossMode.FIXED
        tm.tp_mode = TakeProfitMode.RR
        if tm.sl_mode == StopLossMode.ATR:
            _add_tm_param(tm, "atr_period", rng, specs=specs)
            _add_tm_param(tm, "atr_sl_mult", rng, specs=specs)
        tm.params["tp_rr"] = _sample_edge_tp_rr(rng, specs)
        tm.ranges["tp_rr"] = specs["tp_rr"]
    return tm


def _sample_mechanic(
    mech_type: ExecutionMechanicType,
    symbol: str,
    rng: random.Random,
) -> ExecutionMechanic:
    from factory.generator import MECHANIC_PARAM_SPECS, _sample_params

    sym_class = classify_symbol(symbol)
    mech_specs = scaled_mechanic_specs(MECHANIC_PARAM_SPECS, sym_class)
    mspecs = mech_specs[mech_type]
    return ExecutionMechanic(
        type=mech_type,
        params=_sample_params(mspecs, rng),
        ranges=dict(mspecs),
    )


def expand_execution_variants(
    edge: StrategyDefinition,
    *,
    mechanics: Optional[Sequence[ExecutionMechanicType]] = None,
    tm_features: Optional[Sequence[str]] = None,
    rng: Optional[random.Random] = None,
    max_variants: int = 6,
    include_probe_style: bool = False,
) -> List[StrategyDefinition]:
    """Clone a validated edge into execution/mechanic EA variants.

    Entry filters and signal_logic are preserved; mechanic + trade-management
    overlays are re-sampled so each variant is a different *use* of the edge.
    """
    from factory.generator import (
        LOGIC_MATRIX, _ADJECTIVES, _COMPLEXITY_COST, _MAGIC_BASE, _NOUNS,
        _TM_COMPLEXITY_COST, describe_rules, random_trade_mgmt,
    )

    rng = rng or random.Random()
    allowed = list(mechanics) if mechanics else [
        ExecutionMechanicType(m) for m in DEFAULT_EXECUTION_MECHANICS
    ]
    # Only keep mechanics whose Logic Matrix accepts this edge's filters.
    filter_types = {f.type for f in edge.entry_filters}
    compatible = []
    for m in allowed:
        ok_filters = set(LOGIC_MATRIX.get(m, []))
        if filter_types and not filter_types.issubset(ok_filters):
            # Allow if at least one filter is compatible (partial reuse).
            if not (filter_types & ok_filters):
                continue
        compatible.append(m)
    if not compatible:
        compatible = [EDGE_PROBE_MECHANIC]

    variants: List[StrategyDefinition] = []
    seen: set = set()
    # Always try each mechanic once, then extra TM draws until max_variants.
    plans: List[ExecutionMechanicType] = []
    for m in compatible:
        if not include_probe_style and m == EDGE_PROBE_MECHANIC:
            # Still allow STANDARD_SLTP variants with richer TM (trailing…).
            plans.append(m)
        else:
            plans.append(m)
    rng.shuffle(plans)

    attempts = 0
    while len(variants) < max_variants and attempts < max_variants * 4:
        attempts += 1
        mech_type = plans[attempts % len(plans)]
        mechanic = _sample_mechanic(mech_type, edge.symbol, rng)
        # Richer overlays for execution phase.
        tm = random_trade_mgmt(
            mech_type, rng, allowed=tm_features, symbol=edge.symbol)
        # Ensure STANDARD_SLTP expansions differ from the bare probe: prefer
        # trailing / breakeven when those features are allowed.
        if mech_type == EDGE_PROBE_MECHANIC and tm_features:
            allow = set(tm_features)
            if "trailing" in allow and tm.trail_mode == TrailMode.OFF:
                if rng.random() < 0.7:
                    tm = random_trade_mgmt(
                        mech_type, rng, allowed=tm_features, symbol=edge.symbol)
            sig = (
                mech_type.value,
                tm.sl_mode.value,
                tm.tp_mode.value,
                tm.trail_mode.value,
                tm.breakeven,
            )
        else:
            sig = (mech_type.value, tm.sl_mode.value, tm.tp_mode.value,
                   tm.trail_mode.value)
        if sig in seen and attempts < max_variants * 2:
            continue
        seen.add(sig)

        child = edge.model_copy(deep=True)
        child.id = StrategyDefinition(mechanic=mechanic).id
        child.mechanic = mechanic
        child.trade_mgmt = tm
        # Drop filters incompatible with the new mechanic.
        ok = set(LOGIC_MATRIX.get(mech_type, []))
        kept = [f for f in child.entry_filters if f.type in ok]
        if not kept:
            continue
        child.entry_filters = kept
        child.lineage = Lineage(
            parents=[edge.id],
            mutations=[f"edge_expand:{edge.id}", f"mechanic={mech_type.value}"],
            generation=edge.lineage.generation + 1,
            role="execution",
            edge_id=edge.id,
        )
        child.profile = edge.profile.model_copy(deep=True)
        child.profile.search_phase = "execution"
        filter_cost = sum(_COMPLEXITY_COST.get(f.type, 1) for f in child.entry_filters)
        tm_cost = 0
        if tm.regime_filter:
            tm_cost += _TM_COMPLEXITY_COST["regime_filter"]
        if tm.regime_sizing:
            tm_cost += _TM_COMPLEXITY_COST["regime_sizing"]
        if tm.hmm_regime_filter:
            tm_cost += _TM_COMPLEXITY_COST["hmm_regime_filter"]
        if tm.hmm_regime_sizing:
            tm_cost += _TM_COMPLEXITY_COST["hmm_regime_sizing"]
        child.profile.complexity_score = int(filter_cost + tm_cost)
        child.profile.portfolio_signature = "|".join(
            [
                child.symbol,
                child.timeframe,
                child.mechanic.type.value,
                ",".join(sorted(f.type.value for f in child.entry_filters)),
            ]
        )
        child.name = (
            f"{rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)} "
            f"{child.id[:6].upper()}"
        )
        child.magic_number = _MAGIC_BASE + int(child.id[:6], 16) % 100000
        child.rule_description = describe_rules(child)
        variants.append(child)

    return variants


def apply_edge_best_params(
    variant: StrategyDefinition,
    edge: StrategyDefinition,
    best_params: Optional[Dict[str, float]],
) -> StrategyDefinition:
    """Copy optimized *filter* params from the edge onto an execution variant."""
    if not best_params:
        return variant
    # Only transfer F* keys (entry filters); mechanic/TM stay variant-specific.
    filter_flat = {
        k: v for k, v in best_params.items() if k.startswith("F")
    }
    if not filter_flat:
        return variant
    # Align filter prefixes by type order on the variant.
    tuned = variant.apply_flat_params(filter_flat)
    return tuned
