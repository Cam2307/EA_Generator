"""Progressive validation levels.

The raw acceptance-criteria numbers (WFE, R-squared, Sharpe, 95%-worst-case
Monte Carlo drawdown, ...) are hard to reason about unless you already know
what they mean. Validation *levels* wrap them in a single easy dial: a higher
level applies every gate of the levels below it, only stricter, and turns on
progressively heavier robustness testing (Monte Carlo from level 3 up).

Pick a low level to cast a wide net and eyeball many candidates; pick a high
level to keep only strategies that survive punishing, curve-fit-resistant
checks. Level names/thresholds are the single source of truth for both the UI
and the discovery worker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from factory.backtest.montecarlo import MonteCarloConfig
from factory.models import AcceptanceCriteria


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

    def human_gates(self) -> List[str]:
        """Plain-language bullet list of what this level enforces."""
        c = self.criteria
        bullets = [
            f"Walk-forward efficiency \u2265 {c.min_wfe:.2f} "
            "(out-of-sample keeps this share of the in-sample edge)",
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
        else:
            bullets.append("Monte Carlo robustness test: not required")
        return bullets


# Ordered loosest -> strictest. Each level's gates dominate the previous one.
VALIDATION_LEVELS: List[ValidationLevel] = [
    ValidationLevel(
        level=1, name="Screener",
        summary="Loosest net — just profitable and tradeable. Great for a "
                "first look at what the factory can find.",
        criteria=AcceptanceCriteria(
            min_wfe=0.30, max_dd_pct=35.0, min_trades=5,
            min_profit_factor=1.0),
        montecarlo=False),
    ValidationLevel(
        level=2, name="Basic",
        summary="A modest edge that holds up out of sample.",
        criteria=AcceptanceCriteria(
            min_wfe=0.45, max_dd_pct=30.0, min_trades=10,
            min_profit_factor=1.1),
        montecarlo=False),
    ValidationLevel(
        level=3, name="Standard",
        summary="A dependable edge with a reasonably smooth equity curve, "
                "confirmed by Monte Carlo stress testing.",
        criteria=AcceptanceCriteria(
            min_wfe=0.55, max_dd_pct=25.0, min_trades=15,
            min_profit_factor=1.2, min_sharpe=0.3, min_r_squared=0.50),
        montecarlo=True, mc_runs=20, mc_min_profitable=0.70, mc_max_dd_p95=30.0),
    ValidationLevel(
        level=4, name="Robust",
        summary="Strong risk-adjusted returns that resist parameter and "
                "execution noise.",
        criteria=AcceptanceCriteria(
            min_wfe=0.65, max_dd_pct=20.0, min_trades=25,
            min_profit_factor=1.3, min_sharpe=0.7, min_r_squared=0.65,
            max_consecutive_losses=10),
        montecarlo=True, mc_runs=30, mc_min_profitable=0.75, mc_max_dd_p95=25.0),
    ValidationLevel(
        level=5, name="Strict",
        summary="Tight drawdowns and high consistency across heavy Monte "
                "Carlo stress. Few strategies get here.",
        criteria=AcceptanceCriteria(
            min_wfe=0.75, max_dd_pct=15.0, min_trades=35,
            min_profit_factor=1.5, min_sharpe=1.0, min_r_squared=0.80,
            max_consecutive_losses=8),
        montecarlo=True, mc_runs=50, mc_min_profitable=0.82, mc_max_dd_p95=20.0),
    ValidationLevel(
        level=6, name="Elite",
        summary="The strictest gate — institutional-grade consistency and "
                "drawdown control under the most demanding Monte Carlo runs.",
        criteria=AcceptanceCriteria(
            min_wfe=0.85, max_dd_pct=10.0, min_trades=45,
            min_profit_factor=1.7, min_sharpe=1.3, min_r_squared=0.88,
            max_consecutive_losses=6),
        montecarlo=True, mc_runs=100, mc_min_profitable=0.88, mc_max_dd_p95=15.0),
]

MIN_LEVEL = VALIDATION_LEVELS[0].level
MAX_LEVEL = VALIDATION_LEVELS[-1].level
# Default to a practical broad-net profile: keep gates meaningful but avoid
# early over-pruning so users can rank more candidates in the gallery.
DEFAULT_LEVEL = 1


def get_level(level: int) -> ValidationLevel:
    """Return the level definition, clamped to the available range."""
    clamped = max(MIN_LEVEL, min(MAX_LEVEL, int(level)))
    return VALIDATION_LEVELS[clamped - MIN_LEVEL]


def mc_config_for(level: ValidationLevel,
                  seed: Optional[int] = None) -> Optional[MonteCarloConfig]:
    """Build the Monte Carlo config for a level, or None if MC is off."""
    if not level.montecarlo:
        return None
    return MonteCarloConfig(
        n_runs=level.mc_runs,
        min_profitable=level.mc_min_profitable,
        max_dd_p95=level.mc_max_dd_p95,
        seed=seed,
    )
