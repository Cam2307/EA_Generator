"""Progressive validation levels.

The raw acceptance-criteria numbers (WFE, R-squared, Sharpe, 95%-worst-case
Monte Carlo drawdown, ...) are hard to reason about unless you already know
what they mean. Validation *levels* wrap them in a single easy dial: a higher
level applies every gate of the levels below it, only stricter, and turns on
progressively heavier robustness testing (Monte Carlo from level 7 up).

Sixteen fine levels nest inside six named bands (Screener → Elite). Pick a low
level to cast a wide net; pick a high level to keep only strategies that survive
punishing, curve-fit-resistant checks. Level names/thresholds are the single
source of truth for both the UI and the discovery worker.

Honesty gates (WFO fold loss rate, DSR, neighborhood stability, MC path
bootstrap) unlock from Standard·A (L7) upward and are applied whenever the
signals are supplied to :func:`level_clears`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

from factory.backtest.montecarlo import MonteCarloConfig
from factory.models import AcceptanceCriteria, BacktestMetrics

if TYPE_CHECKING:
    from factory.models import MonteCarloResult


# Schema version for persisted ``highest_level_passed`` / discovery ceilings.
# v1 = legacy 6 coarse levels; v2 = 16 fine levels (this module).
LEVEL_SCHEMA_VERSION = 2

# Map legacy coarse L1–L6 → fine-level band anchors (A substages / Elite peak).
LEGACY_LEVEL_MAP = {1: 1, 2: 4, 3: 7, 4: 10, 5: 13, 6: 16}

# Monte Carlo unlocks at Standard·A (fine L7). Stage-1 screen stays fine L1.
MC_UNLOCK_LEVEL = 7

# Last fine level that does not require Monte Carlo (Basic·C).
MC_PRE_LEVEL = MC_UNLOCK_LEVEL - 1

# First level that applies honesty gates (WFO / DSR / stability / path MC).
HONESTY_UNLOCK_LEVEL = MC_UNLOCK_LEVEL

# Dashboard KPI band anchors — Basic / Standard / Robust (not soft Screener).
KPI_ANCHORS: Tuple[int, ...] = (4, 7, 10)

DEFAULT_PROGRESSIVE_STEP = 2


@dataclass(frozen=True)
class HonestySignals:
    """Optional selection-bias / fold-consistency signals for level scoring.

    Missing fields (``None``) skip the corresponding honesty gate so unit
    tests and legacy callers that only pass OOS+WFE+MC keep working. The
    Stage-2 pipeline always supplies concrete values.
    """
    p_oos_loss: Optional[float] = None
    dsr: Optional[float] = None
    stability_ratio: Optional[float] = None


@dataclass(frozen=True)
class ValidationLevel:
    level: int
    name: str
    summary: str
    criteria: AcceptanceCriteria
    montecarlo: bool = False
    mc_runs: int = 20
    mc_min_profitable: float = 0.70   # fraction of MC runs that must profit
    mc_max_dd_p95: float = 30.0       # 95%-worst-case MC drawdown limit (%)
    # Path-bootstrap gate (applied only when MC result has path_runs > 0).
    mc_min_path_profitable: float = 0.0
    # Honesty gates (0 / 1.0 = disabled for that metric).
    max_p_oos_loss: float = 1.0
    min_dsr: float = 0.0
    min_stability: float = 0.0
    band: str = ""
    substage: str = ""

    def human_gates(self) -> List[str]:
        """Plain-language bullet list of what this level enforces."""
        c = self.criteria
        bullets: List[str] = []
        if c.min_wfe <= 0:
            bullets.append(
                "Walk-forward efficiency: not required "
                "(OOS screener — DD / trades / PF / net profit only)")
        elif self.band == "Screener":
            bullets.append(
                f"Walk-forward efficiency \u2265 {c.min_wfe:.2f} "
                "(soft: waived when out-of-sample is profitable)")
        else:
            bullets.append(
                f"Walk-forward efficiency \u2265 {c.min_wfe:.2f} "
                "(out-of-sample keeps this share of the in-sample edge)")
        bullets += [
            f"Out-of-sample drawdown < {c.max_dd_pct:.0f}%",
            f"\u2265 {c.min_trades} out-of-sample trades",
        ]
        if c.min_profit_factor > 0:
            bullets.append(f"Profit factor \u2265 {c.min_profit_factor:.2f}")
        if c.min_sharpe > 0:
            bullets.append(f"Sharpe ratio \u2265 {c.min_sharpe:.2f}")
        if c.min_r_squared > 0:
            bullets.append(
                f"Equity-curve smoothness (R\u00b2) \u2265 {c.min_r_squared:.2f}")
        if c.max_consecutive_losses > 0:
            bullets.append(
                f"\u2264 {c.max_consecutive_losses} consecutive losing trades")
        if self.montecarlo:
            bullets.append(
                f"Monte Carlo: \u2265 {self.mc_min_profitable:.0%} of "
                f"{self.mc_runs} randomized runs profitable, "
                f"95%-worst-case drawdown < {self.mc_max_dd_p95:.0f}%")
            if self.mc_min_path_profitable > 0:
                bullets.append(
                    f"Path bootstrap: \u2265 {self.mc_min_path_profitable:.0%} "
                    f"of block-bootstrap paths profitable")
        else:
            bullets.append("Monte Carlo robustness test: not required")
        if self.max_p_oos_loss < 1.0:
            bullets.append(
                f"Walk-forward: \u2264 {self.max_p_oos_loss:.0%} of OOS "
                f"windows unprofitable")
        if self.min_dsr > 0:
            bullets.append(f"Deflated Sharpe ratio \u2265 {self.min_dsr:.2f}")
        if self.min_stability > 0:
            bullets.append(
                f"Neighborhood stability \u2265 {self.min_stability:.2f}")
        return bullets


def _lvl(
    level: int,
    band: str,
    sub: str,
    summary: str,
    *,
    min_wfe: float,
    max_dd_pct: float,
    min_trades: int,
    min_profit_factor: float,
    min_sharpe: float = 0.0,
    min_r_squared: float = 0.0,
    max_consecutive_losses: int = 0,
    montecarlo: bool = False,
    mc_runs: int = 20,
    mc_min_profitable: float = 0.70,
    mc_max_dd_p95: float = 30.0,
    mc_min_path_profitable: float = 0.0,
    max_p_oos_loss: float = 1.0,
    min_dsr: float = 0.0,
    min_stability: float = 0.0,
) -> ValidationLevel:
    return ValidationLevel(
        level=level,
        name=f"{band}\u00b7{sub}",
        summary=summary,
        band=band,
        substage=sub,
        criteria=AcceptanceCriteria(
            min_wfe=min_wfe,
            max_dd_pct=max_dd_pct,
            min_trades=min_trades,
            min_profit_factor=min_profit_factor,
            min_sharpe=min_sharpe,
            min_r_squared=min_r_squared,
            max_consecutive_losses=max_consecutive_losses,
        ),
        montecarlo=montecarlo,
        mc_runs=mc_runs,
        mc_min_profitable=mc_min_profitable,
        mc_max_dd_p95=mc_max_dd_p95,
        mc_min_path_profitable=mc_min_path_profitable,
        max_p_oos_loss=max_p_oos_loss,
        min_dsr=min_dsr,
        min_stability=min_stability,
    )


# Ordered loosest -> strictest. Each level's gates dominate the previous one.
# Screener (L1–L3) is intentionally a wide net; later bands climb smoothly.
# Drawdown caps are calibrated above observed median OOS DD (~40%+) so low
# tiers are not DD-starved; higher tiers still tighten progressively.
VALIDATION_LEVELS: List[ValidationLevel] = [
    _lvl(1, "Screener", "A",
         "Loosest net — barely tradeable and not blown up. Great for a first look.",
         min_wfe=0.00, max_dd_pct=70.0, min_trades=3, min_profit_factor=0.95),
    _lvl(2, "Screener", "B",
         "Slightly tighter screener — thin edge and a few more trades.",
         min_wfe=0.10, max_dd_pct=65.0, min_trades=4, min_profit_factor=1.00),
    _lvl(3, "Screener", "C",
         "Upper screener band before Basic thresholds.",
         min_wfe=0.20, max_dd_pct=60.0, min_trades=5, min_profit_factor=1.02),
    # Basic trade floors are slightly softer than early drafts so genuine
    # edges on shorter OOS windows are not killed solely by sample size;
    # WFE / PF / DD stay the quality bar. Standard+ (L7+) unchanged.
    _lvl(4, "Basic", "A",
         "A modest edge that holds up out of sample.",
         min_wfe=0.30, max_dd_pct=55.0, min_trades=6, min_profit_factor=1.05),
    _lvl(5, "Basic", "B",
         "Stronger basic edge with tighter drawdown and more trades.",
         min_wfe=0.38, max_dd_pct=50.0, min_trades=8, min_profit_factor=1.08),
    _lvl(6, "Basic", "C",
         "Top of Basic — introduces light Sharpe / R\u00b2 before Monte Carlo.",
         min_wfe=0.45, max_dd_pct=45.0, min_trades=10, min_profit_factor=1.12,
         min_sharpe=0.10, min_r_squared=0.30),
    # Honesty gates unlock at Standard·A (L7): WFO fold loss, path MC, then
    # stability / DSR tighten through Robust → Elite.
    _lvl(7, "Standard", "A",
         "Dependable edge with a smooth equity curve, confirmed by Monte Carlo.",
         min_wfe=0.50, max_dd_pct=48.0, min_trades=14, min_profit_factor=1.15,
         min_sharpe=0.15, min_r_squared=0.35,
         montecarlo=True, mc_runs=20, mc_min_profitable=0.60, mc_max_dd_p95=52.0,
         mc_min_path_profitable=0.50, max_p_oos_loss=0.65),
    _lvl(8, "Standard", "B",
         "Standard band mid-stage — tighter risk and MC profitability.",
         min_wfe=0.55, max_dd_pct=40.0, min_trades=16, min_profit_factor=1.18,
         min_sharpe=0.35, min_r_squared=0.50,
         montecarlo=True, mc_runs=20, mc_min_profitable=0.68, mc_max_dd_p95=45.0,
         mc_min_path_profitable=0.52, max_p_oos_loss=0.55),
    _lvl(9, "Standard", "C",
         "Upper Standard — heavier MC sample and tighter tails.",
         min_wfe=0.58, max_dd_pct=38.0, min_trades=18, min_profit_factor=1.22,
         min_sharpe=0.45, min_r_squared=0.55,
         montecarlo=True, mc_runs=25, mc_min_profitable=0.70, mc_max_dd_p95=42.0,
         mc_min_path_profitable=0.55, max_p_oos_loss=0.50),
    _lvl(10, "Robust", "A",
         "Strong risk-adjusted returns that resist parameter and execution noise.",
         min_wfe=0.62, max_dd_pct=35.0, min_trades=22, min_profit_factor=1.28,
         min_sharpe=0.60, min_r_squared=0.60, max_consecutive_losses=12,
         montecarlo=True, mc_runs=30, mc_min_profitable=0.72, mc_max_dd_p95=40.0,
         mc_min_path_profitable=0.55, max_p_oos_loss=0.45, min_stability=0.50),
    _lvl(11, "Robust", "B",
         "Robust mid-stage — more trades and stricter consecutive-loss control.",
         min_wfe=0.66, max_dd_pct=32.0, min_trades=25, min_profit_factor=1.32,
         min_sharpe=0.70, min_r_squared=0.65, max_consecutive_losses=10,
         montecarlo=True, mc_runs=35, mc_min_profitable=0.75, mc_max_dd_p95=38.0,
         mc_min_path_profitable=0.58, max_p_oos_loss=0.40, min_stability=0.55),
    _lvl(12, "Robust", "C",
         "Upper Robust — deep MC stress before Strict.",
         min_wfe=0.70, max_dd_pct=30.0, min_trades=28, min_profit_factor=1.38,
         min_sharpe=0.80, min_r_squared=0.70, max_consecutive_losses=9,
         montecarlo=True, mc_runs=40, mc_min_profitable=0.78, mc_max_dd_p95=35.0,
         mc_min_path_profitable=0.60, max_p_oos_loss=0.35, min_stability=0.60),
    _lvl(13, "Strict", "A",
         "Tight drawdowns and high consistency under heavy Monte Carlo stress.",
         min_wfe=0.74, max_dd_pct=28.0, min_trades=32, min_profit_factor=1.45,
         min_sharpe=0.95, min_r_squared=0.75, max_consecutive_losses=8,
         montecarlo=True, mc_runs=50, mc_min_profitable=0.80, mc_max_dd_p95=32.0,
         mc_min_path_profitable=0.62, max_p_oos_loss=0.35,
         min_dsr=0.45, min_stability=0.60),
    _lvl(14, "Strict", "B",
         "Upper Strict — deeper MC plus neighborhood stability honesty gate.",
         min_wfe=0.78, max_dd_pct=25.0, min_trades=36, min_profit_factor=1.52,
         min_sharpe=1.05, min_r_squared=0.80, max_consecutive_losses=7,
         montecarlo=True, mc_runs=60, mc_min_profitable=0.82, mc_max_dd_p95=30.0,
         mc_min_path_profitable=0.65, max_p_oos_loss=0.30,
         min_dsr=0.55, min_stability=0.70),
    _lvl(15, "Elite", "A",
         "Near-elite consistency with Deflated Sharpe honesty gate.",
         min_wfe=0.82, max_dd_pct=22.0, min_trades=40, min_profit_factor=1.60,
         min_sharpe=1.15, min_r_squared=0.84, max_consecutive_losses=6,
         montecarlo=True, mc_runs=80, mc_min_profitable=0.85, mc_max_dd_p95=28.0,
         mc_min_path_profitable=0.68, max_p_oos_loss=0.28,
         min_dsr=0.70, min_stability=0.75),
    _lvl(16, "Elite", "B",
         "The strictest gate — institutional-grade consistency and drawdown control.",
         min_wfe=0.85, max_dd_pct=20.0, min_trades=45, min_profit_factor=1.70,
         min_sharpe=1.30, min_r_squared=0.88, max_consecutive_losses=6,
         montecarlo=True, mc_runs=100, mc_min_profitable=0.88, mc_max_dd_p95=25.0,
         mc_min_path_profitable=0.70, max_p_oos_loss=0.25,
         min_dsr=0.85, min_stability=0.80),
]

MIN_LEVEL = VALIDATION_LEVELS[0].level
MAX_LEVEL = VALIDATION_LEVELS[-1].level
# Default survivor floor: Basic·A — Screener soft-WFE clears stay triage-only.
DEFAULT_LEVEL = 4
# Gallery / run-dialog filter default — Standard·A and above.
GALLERY_DEFAULT_MIN_LEVEL = MC_UNLOCK_LEVEL

# Band → Streamlit badge color (UI Phase 2).
_BAND_BADGE_COLOR = {
    "Screener": "blue",
    "Basic": "blue",
    "Standard": "green",
    "Robust": "green",
    "Strict": "orange",
    "Elite": "red",
}


def get_level(level: int) -> ValidationLevel:
    """Return the level definition, clamped to the available range."""
    clamped = max(MIN_LEVEL, min(MAX_LEVEL, int(level)))
    return VALIDATION_LEVELS[clamped - MIN_LEVEL]


def band_for_level(level: int) -> str:
    """Return the band name for a fine level (empty when uncleared)."""
    lvl = int(level or 0)
    if lvl <= 0:
        return ""
    return get_level(lvl).band


def display_label(level: int, *, include_number: bool = True) -> str:
    """UI label like ``7 · Standard·A`` (or just ``Standard·A``)."""
    lvl = int(level or 0)
    if lvl <= 0:
        return "L0 none"
    name = get_level(lvl).name
    return f"{lvl} \u00b7 {name}" if include_number else name


def badge_color_for_level(level: int) -> str:
    """Streamlit badge color keyed by band."""
    band = band_for_level(level)
    return _BAND_BADGE_COLOR.get(band, "gray")


def remap_legacy_level(
    level: int,
    schema_version: int | None = None,
) -> int:
    """Map a persisted level into the current fine-level schema.

    Legacy schema (missing / < ``LEVEL_SCHEMA_VERSION``) uses
    ``LEGACY_LEVEL_MAP``. Values already in the fine schema are clamped only.
    Uncleared (0) stays 0.
    """
    lvl = int(level or 0)
    if lvl <= 0:
        return 0
    ver = 1 if schema_version is None else int(schema_version)
    if ver < LEVEL_SCHEMA_VERSION:
        if lvl in LEGACY_LEVEL_MAP:
            return int(LEGACY_LEVEL_MAP[lvl])
        # Unknown legacy value — clamp into fine range as a best effort.
        return max(MIN_LEVEL, min(MAX_LEVEL, lvl))
    return max(MIN_LEVEL, min(MAX_LEVEL, lvl))


def mc_config_for(level: ValidationLevel,
                  seed: Optional[int] = None) -> Optional[MonteCarloConfig]:
    """Build the Monte Carlo config for a level, or None if MC is off."""
    if not level.montecarlo:
        return None
    return MonteCarloConfig(
        n_runs=level.mc_runs,
        min_profitable=level.mc_min_profitable,
        max_dd_p95=level.mc_max_dd_p95,
        min_path_profitable=float(level.mc_min_path_profitable),
        seed=seed,
    )


def soft_wfe_waived(
    level: ValidationLevel,
    oos: BacktestMetrics,
    wfe: float,
) -> bool:
    """True when Screener soft-WFE would waive a hard WFE failure."""
    reasons = level.criteria.evaluate(oos, wfe)
    return bool(
        reasons
        and level.band == "Screener"
        and float(oos.net_profit) > 0.0
        and all("WFE" in r for r in reasons)
    )


def mc_clears_level(mc: Optional["MonteCarloResult"],
                    level: ValidationLevel) -> bool:
    """Whether stored Monte Carlo stats satisfy a level's MC gates.

    One MC run (at ceiling depth) is re-scored against each level's thresholds
    — no re-simulation per level. Path-bootstrap profitability is gated when
    the MC result actually ran path stress (``path_runs > 0``).
    """
    if not level.montecarlo:
        return True
    if mc is None or int(getattr(mc, "n_runs", 0) or 0) <= 0:
        return False
    if float(mc.pct_profitable) < float(level.mc_min_profitable):
        return False
    worst_dd = max(float(mc.dd_p95), float(getattr(mc, "resample_dd_p95", 0.0) or 0.0))
    if worst_dd > float(level.mc_max_dd_p95):
        return False
    path_runs = int(getattr(mc, "path_runs", 0) or 0)
    if path_runs > 0 and float(level.mc_min_path_profitable) > 0:
        if float(getattr(mc, "path_pct_profitable", 0.0) or 0.0) < float(
                level.mc_min_path_profitable):
            return False
    return True


def honesty_clears_level(
    level: ValidationLevel,
    honesty: Optional[HonestySignals] = None,
) -> bool:
    """Whether WFO / DSR / stability honesty gates clear ``level``.

    Gates with thresholds disabled (``max_p_oos_loss >= 1``, ``min_dsr <= 0``,
    ``min_stability <= 0``) always pass. When a threshold is active but the
    corresponding signal is ``None``, the gate is skipped (legacy callers).
    """
    if honesty is None or level.level < HONESTY_UNLOCK_LEVEL:
        return True
    if float(level.max_p_oos_loss) < 1.0 and honesty.p_oos_loss is not None:
        if float(honesty.p_oos_loss) > float(level.max_p_oos_loss):
            return False
    if float(level.min_dsr) > 0.0 and honesty.dsr is not None:
        if float(honesty.dsr) < float(level.min_dsr):
            return False
    if float(level.min_stability) > 0.0 and honesty.stability_ratio is not None:
        if float(honesty.stability_ratio) < float(level.min_stability):
            return False
    return True


def level_clears(
    level: ValidationLevel,
    oos: BacktestMetrics,
    wfe: float,
    montecarlo: Optional["MonteCarloResult"] = None,
    honesty: Optional[HonestySignals] = None,
) -> bool:
    """True when OOS metrics (+ MC + honesty when required) clear ``level``.

    Screener band (L1–L3) uses soft WFE: if the only failing gate is WFE but
    OOS net profit is positive, the level still clears. L4+ keep hard WFE.
    Soft-WFE clears are triage-only — gallery defaults start at Standard.
    """
    reasons = level.criteria.evaluate(oos, wfe)
    if reasons:
        if not soft_wfe_waived(level, oos, wfe):
            return False
    if not mc_clears_level(montecarlo, level):
        return False
    return honesty_clears_level(level, honesty)


def highest_level_cleared(
    oos: BacktestMetrics,
    wfe: float,
    montecarlo: Optional["MonteCarloResult"] = None,
    *,
    ceiling: int = MAX_LEVEL,
    floor: int = MIN_LEVEL,
    honesty: Optional[HonestySignals] = None,
) -> int:
    """Highest validation level cleared, or 0 if none.

    Levels are checked from ``floor`` through ``ceiling`` inclusive. Scoring is
    strictly nested: the first failing level stops the climb — L2 is never
    awarded when L1 failed, and so on.
    """
    lo = max(MIN_LEVEL, int(floor))
    hi = min(MAX_LEVEL, max(lo, int(ceiling)))
    best = 0
    for lvl in range(lo, hi + 1):
        if level_clears(get_level(lvl), oos, wfe, montecarlo, honesty=honesty):
            best = lvl
        else:
            break  # nested ladder — do not evaluate higher levels
    return best


def levels_cleared_map(
    oos: BacktestMetrics,
    wfe: float,
    montecarlo: Optional["MonteCarloResult"] = None,
    *,
    ceiling: int = MAX_LEVEL,
    highest: Optional[int] = None,
    honesty: Optional[HonestySignals] = None,
) -> dict:
    """Per-level pass/fail map for one scored backtest (``{\"1\": true, ...}``).

    Nested levels: every level at or below ``highest`` is True; the rest False.
    When ``highest`` is omitted it is computed via :func:`highest_level_cleared`.
    """
    hi = (
        int(highest)
        if highest is not None
        else highest_level_cleared(
            oos, wfe, montecarlo, ceiling=ceiling, floor=1, honesty=honesty)
    )
    cap = min(MAX_LEVEL, max(MIN_LEVEL, int(ceiling)))
    return {str(lvl): (lvl <= hi) for lvl in range(MIN_LEVEL, cap + 1)}


def levels_snapshot() -> dict:
    """Serializable snapshot of the level table (for results archives)."""
    rows = []
    for lvl in VALIDATION_LEVELS:
        c = lvl.criteria
        rows.append({
            "level": lvl.level,
            "name": lvl.name,
            "band": lvl.band,
            "substage": lvl.substage,
            "summary": lvl.summary,
            "min_wfe": c.min_wfe,
            "max_dd_pct": c.max_dd_pct,
            "min_trades": c.min_trades,
            "min_profit_factor": c.min_profit_factor,
            "min_sharpe": c.min_sharpe,
            "min_r_squared": c.min_r_squared,
            "max_consecutive_losses": c.max_consecutive_losses,
            "montecarlo": lvl.montecarlo,
            "mc_runs": lvl.mc_runs,
            "mc_min_profitable": lvl.mc_min_profitable,
            "mc_max_dd_p95": lvl.mc_max_dd_p95,
            "mc_min_path_profitable": lvl.mc_min_path_profitable,
            "max_p_oos_loss": lvl.max_p_oos_loss,
            "min_dsr": lvl.min_dsr,
            "min_stability": lvl.min_stability,
        })
    return {
        "schema_version": LEVEL_SCHEMA_VERSION,
        "min_level": MIN_LEVEL,
        "max_level": MAX_LEVEL,
        "mc_unlock_level": MC_UNLOCK_LEVEL,
        "honesty_unlock_level": HONESTY_UNLOCK_LEVEL,
        "levels": rows,
    }


def reasons_for_next_level(
    oos: BacktestMetrics,
    wfe: float,
    montecarlo: Optional["MonteCarloResult"] = None,
    *,
    highest: int,
    ceiling: int = MAX_LEVEL,
    honesty: Optional[HonestySignals] = None,
) -> List[str]:
    """Diagnostic reasons for failing the next level above ``highest``."""
    if highest >= min(MAX_LEVEL, int(ceiling)):
        return []
    nxt = get_level(highest + 1)
    reasons = list(nxt.criteria.evaluate(oos, wfe))
    # Align with soft-WFE Screener clearing in level_clears.
    if soft_wfe_waived(nxt, oos, wfe):
        reasons = []
    if not reasons and nxt.montecarlo and not mc_clears_level(montecarlo, nxt):
        if montecarlo is None or int(getattr(montecarlo, "n_runs", 0) or 0) <= 0:
            reasons.append(f"Monte Carlo required for level {nxt.level} ({nxt.name})")
        else:
            if float(montecarlo.pct_profitable) < float(nxt.mc_min_profitable):
                reasons.append(
                    f"Monte Carlo: only {montecarlo.pct_profitable:.0%} profitable "
                    f"(need ≥ {nxt.mc_min_profitable:.0%} for {nxt.name})")
            worst_dd = max(
                float(montecarlo.dd_p95),
                float(getattr(montecarlo, "resample_dd_p95", 0.0) or 0.0),
            )
            if worst_dd > float(nxt.mc_max_dd_p95):
                reasons.append(
                    f"Monte Carlo: worst-case DD {worst_dd:.1f}% "
                    f"(limit {nxt.mc_max_dd_p95:.0f}% for {nxt.name})")
            path_runs = int(getattr(montecarlo, "path_runs", 0) or 0)
            if (
                path_runs > 0
                and float(nxt.mc_min_path_profitable) > 0
                and float(getattr(montecarlo, "path_pct_profitable", 0.0) or 0.0)
                < float(nxt.mc_min_path_profitable)
            ):
                reasons.append(
                    f"Monte Carlo path bootstrap: "
                    f"{float(montecarlo.path_pct_profitable):.0%} profitable "
                    f"(need ≥ {nxt.mc_min_path_profitable:.0%} for {nxt.name})")
    if not reasons and not honesty_clears_level(nxt, honesty):
        if (
            honesty is not None
            and float(nxt.max_p_oos_loss) < 1.0
            and honesty.p_oos_loss is not None
            and float(honesty.p_oos_loss) > float(nxt.max_p_oos_loss)
        ):
            reasons.append(
                f"WFO OOS loss rate {float(honesty.p_oos_loss):.0%} "
                f"(limit {nxt.max_p_oos_loss:.0%} for {nxt.name})")
        if (
            honesty is not None
            and float(nxt.min_dsr) > 0.0
            and honesty.dsr is not None
            and float(honesty.dsr) < float(nxt.min_dsr)
        ):
            reasons.append(
                f"Deflated Sharpe {float(honesty.dsr):.2f} "
                f"(need ≥ {nxt.min_dsr:.2f} for {nxt.name})")
        if (
            honesty is not None
            and float(nxt.min_stability) > 0.0
            and honesty.stability_ratio is not None
            and float(honesty.stability_ratio) < float(nxt.min_stability)
        ):
            reasons.append(
                f"Neighborhood stability {float(honesty.stability_ratio):.2f} "
                f"(need ≥ {nxt.min_stability:.2f} for {nxt.name})")
    if not reasons:
        reasons.append(f"Did not clear level {nxt.level} ({nxt.name})")
    return [f"L{nxt.level} {nxt.name}: {r}" for r in reasons]
