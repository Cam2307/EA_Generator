"""Monte Carlo robustness testing on the event-driven simulator.

A validated strategy is re-run ``n_runs`` times, each run randomized along
the axes EA Studio (and FSB Pro / Express Generator) stress:

- **spread / slippage randomization** — execution costs drawn per run
  between the base value and a configured maximum (slippage is always
  adverse, matching real-broker behavior);
- **indicator/mechanic parameter perturbation** — each optimizable
  parameter is nudged by up to ``param_max_steps`` grid steps within its
  own :class:`~factory.models.ParamRange` with probability
  ``param_change_prob`` (a curve-fit strategy collapses under tiny nudges);
- **entry skipping + first-bar jitter** — entries are randomly dropped and
  the usable history start is randomly offset, so results cannot hinge on
  one lucky fill or one lucky anchoring of the data;
- **trade-order resampling** — the base run's trade PnL sequence is
  permuted many times to estimate the drawdown distribution independent of
  the particular order in which trades happened to occur.

Output: per-run stats, profit percentiles, 5/50/95% equity confidence
bands, a 0-100 robustness score, and a pass/fail gate.

Honesty note (also in the docs): a strategy passing Monte Carlo is *less
likely to be curve-fit*. Nothing here guarantees future profitability.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from config import settings
from factory.backtest.simulator import SymbolSpec, run_simulation
from factory.models import (
    JobCancelled, MonteCarloResult, MonteCarloRun, ParamRange,
    StrategyDefinition,
)

# Cooperative cancellation probe (see factory.backtest.validation).
CancelCheck = Optional[Callable[[], bool]]


@dataclass
class MonteCarloConfig:
    n_runs: int = settings.MC_RUNS
    spread_max_points: float = settings.MC_SPREAD_MAX_POINTS
    slippage_max_points: float = settings.MC_SLIPPAGE_MAX_POINTS
    param_change_prob: float = settings.MC_PARAM_CHANGE_PROB
    param_max_steps: int = settings.MC_PARAM_MAX_STEPS
    skip_entry_prob: float = settings.MC_SKIP_ENTRY_PROB
    start_jitter_bars: int = settings.MC_START_JITTER_BARS
    n_resamples: int = settings.MC_RESAMPLES
    min_profitable: float = settings.MC_MIN_PROFITABLE
    max_dd_p95: float = settings.MC_MAX_DD_P95
    seed: Optional[int] = None


def perturb_flat_params(strategy: StrategyDefinition, rng: random.Random,
                        change_prob: float, max_steps: int) -> Dict[str, float]:
    """Randomly nudge parameters within their ParamRange grid.

    Each parameter moves by +/- 1..max_steps grid steps with probability
    ``change_prob``; the result is clamped and snapped to the range grid.
    Parameters without a declared range are never touched.
    """
    flat = strategy.all_params()
    ranges = strategy.all_ranges()
    out: Dict[str, float] = {}
    for key, value in flat.items():
        r: Optional[ParamRange] = ranges.get(key)
        if r is None or r.step <= 0 or rng.random() >= change_prob:
            out[key] = value
            continue
        steps = rng.randint(1, max(1, max_steps)) * rng.choice((-1, 1))
        out[key] = r.clamp(value + steps * r.step)
    return out


def _entry_mask(n_bars: int, rng: random.Random, skip_prob: float,
                jitter_bars: int) -> np.ndarray:
    mask = np.ones(n_bars, dtype=bool)
    if jitter_bars > 0 and n_bars > jitter_bars * 2:
        mask[: rng.randint(0, jitter_bars)] = False
    if skip_prob > 0:
        drops = np.asarray(
            [rng.random() < skip_prob for _ in range(n_bars)], dtype=bool)
        mask &= ~drops
    return mask


def resample_drawdowns(trade_profits: List[float], deposit: float,
                       n_resamples: int, rng: random.Random) -> List[float]:
    """Max drawdown %% distribution over random permutations of trade order."""
    if not trade_profits or n_resamples <= 0:
        return []
    profits = np.asarray(trade_profits, dtype=float)
    out: List[float] = []
    for _ in range(n_resamples):
        order = rng.sample(range(len(profits)), len(profits))
        equity = deposit + np.cumsum(profits[order])
        peak = np.maximum.accumulate(np.maximum(equity, deposit))
        dd = (peak - equity) / np.maximum(peak, 1e-9) * 100.0
        out.append(float(dd.max()))
    return out


def robustness_score(pct_profitable: float, profit_p05: float,
                     profit_p50: float, dd_p95: float,
                     dd_limit: float) -> float:
    """0-100 composite: profitable fraction, dispersion, DD containment.

    - 40%: fraction of MC runs that end profitable;
    - 30%: dispersion — how close the 5th-percentile outcome sits to the
      median (1.0 when even the bad draws match the median, 0 when the bad
      draws wipe out more than the median gain);
    - 30%: worst-case drawdown containment relative to twice the gate.
    """
    profitable_part = max(0.0, min(1.0, pct_profitable))
    denom = abs(profit_p50) + 1e-9
    dispersion_part = max(0.0, min(1.0, 0.5 + 0.5 * profit_p05 / denom))
    dd_part = max(0.0, min(1.0, 1.0 - dd_p95 / max(2.0 * dd_limit, 1e-9)))
    return round(100.0 * (0.4 * profitable_part + 0.3 * dispersion_part
                          + 0.3 * dd_part), 1)


def run_montecarlo(strategy: StrategyDefinition, df: pd.DataFrame,
                   spec: SymbolSpec, deposit: float,
                   config: Optional[MonteCarloConfig] = None,
                   cancel_check: CancelCheck = None) -> MonteCarloResult:
    """Run the full Monte Carlo battery for one strategy on one dataset."""
    cfg = config or MonteCarloConfig()
    rng = random.Random(cfg.seed)
    n_bars = len(df)

    if cancel_check is not None and cancel_check():
        raise JobCancelled()

    # base (unperturbed) run: source of the trade sequence for resampling
    base_metrics, base_book = run_simulation(df, strategy, spec, deposit,
                                             cancel_check=cancel_check)
    base_profits = [tr.profit for tr in
                    sorted(base_book.closed, key=lambda t: t.close_time)]

    runs: List[MonteCarloRun] = []
    curves: List[List[float]] = []
    band_ts: List[float] = []

    for _ in range(cfg.n_runs):
        if cancel_check is not None and cancel_check():
            raise JobCancelled()
        run_spec = replace(
            spec,
            spread_points=rng.uniform(spec.spread_points,
                                      max(spec.spread_points,
                                          cfg.spread_max_points)),
            slippage_points=rng.uniform(spec.slippage_points,
                                        max(spec.slippage_points,
                                            cfg.slippage_max_points)),
        )
        perturbed = strategy.apply_flat_params(perturb_flat_params(
            strategy, rng, cfg.param_change_prob, cfg.param_max_steps))
        mask = _entry_mask(n_bars, rng, cfg.skip_entry_prob,
                           cfg.start_jitter_bars)
        try:
            m, _ = run_simulation(df, perturbed, run_spec, deposit,
                                  entry_mask=mask, cancel_check=cancel_check)
        except Exception:
            continue
        runs.append(MonteCarloRun(
            net_profit=m.net_profit, max_dd_pct=m.max_dd_pct,
            profit_factor=m.profit_factor, trade_count=m.trade_count))
        curves.append(m.equity)
        if not band_ts:
            band_ts = m.equity_ts

    reasons: List[str] = []
    if not runs:
        return MonteCarloResult(
            n_runs=0, passed=False,
            reasons=["Monte Carlo produced no completed runs"])

    profits = np.asarray([r.net_profit for r in runs], dtype=float)
    dds = np.asarray([r.max_dd_pct for r in runs], dtype=float)
    pct_profitable = float((profits > 0).mean())
    profit_p05 = float(np.percentile(profits, 5))
    profit_p50 = float(np.percentile(profits, 50))
    profit_p95 = float(np.percentile(profits, 95))
    dd_p95 = float(np.percentile(dds, 95))

    resampled = resample_drawdowns(base_profits, deposit,
                                   cfg.n_resamples, rng)
    resample_dd_p95 = float(np.percentile(resampled, 95)) if resampled else 0.0

    # confidence bands: percentile across runs at each (thinned) bar index
    band_p05: List[float] = []
    band_p50: List[float] = []
    band_p95: List[float] = []
    min_len = min(len(c) for c in curves)
    if min_len > 1:
        stack = np.asarray([c[:min_len] for c in curves], dtype=float)
        band_p05 = [float(x) for x in np.percentile(stack, 5, axis=0)]
        band_p50 = [float(x) for x in np.percentile(stack, 50, axis=0)]
        band_p95 = [float(x) for x in np.percentile(stack, 95, axis=0)]
        band_ts = band_ts[:min_len]
    else:
        band_ts = []

    if base_metrics.trade_count == 0:
        reasons.append("base run produced no trades")
    if pct_profitable < cfg.min_profitable:
        reasons.append(
            f"only {pct_profitable:.0%} of MC runs profitable"
            f" (gate {cfg.min_profitable:.0%})")
    if dd_p95 > cfg.max_dd_p95:
        reasons.append(
            f"MC 95%-worst-case drawdown {dd_p95:.1f}%"
            f" > limit {cfg.max_dd_p95}%")
    if resample_dd_p95 > cfg.max_dd_p95:
        reasons.append(
            f"trade-order-resampled 95% drawdown {resample_dd_p95:.1f}%"
            f" > limit {cfg.max_dd_p95}%")

    score = robustness_score(pct_profitable, profit_p05, profit_p50,
                             max(dd_p95, resample_dd_p95), cfg.max_dd_p95)

    return MonteCarloResult(
        n_runs=len(runs), runs=runs,
        pct_profitable=round(pct_profitable, 4),
        profit_p05=round(profit_p05, 2),
        profit_p50=round(profit_p50, 2),
        profit_p95=round(profit_p95, 2),
        dd_p95=round(dd_p95, 3),
        resample_dd_p95=round(resample_dd_p95, 3),
        robustness_score=score,
        band_ts=band_ts, band_p05=band_p05, band_p50=band_p50,
        band_p95=band_p95,
        passed=not reasons, reasons=reasons,
    )


def montecarlo_for_strategy(strategy: StrategyDefinition, start, end,
                            deposit: float,
                            config: Optional[MonteCarloConfig] = None,
                            params_override: Optional[Dict[str, float]] = None,
                            cancel_check: CancelCheck = None,
                            spec_overrides: Optional[Mapping[str, float]] = None
                            ) -> MonteCarloResult:
    """Convenience wrapper: load the strategy's own data, then run MC.

    ``spec_overrides`` carries the user-chosen account/execution economics
    (leverage, spread, slippage, contract size) so the Monte Carlo battery
    uses the same symbol economics as the IS/OOS and walk-forward runs. The
    user's spread/slippage act as the lower bound the MC randomization draws
    from, exactly as before.
    """
    from factory import data as data_mod

    if params_override:
        strategy = strategy.apply_flat_params(params_override)
    df = data_mod.load_ohlc(strategy.symbol, strategy.timeframe, start, end)
    spec = SymbolSpec.infer(float(df["close"].iloc[0]), spec_overrides)
    return run_montecarlo(strategy, df, spec, deposit, config,
                          cancel_check=cancel_check)
