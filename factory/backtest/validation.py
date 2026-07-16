"""IS/OOS split, walk-forward optimization, WFE scoring, acceptance gates.

Over-fitting protections (see docs/ea_studio_reference.md):

- the IS optimizer can score a candidate parameter set by the **average
  fitness of its +/-1-step neighbors** instead of its own peak, so a sharp
  isolated peak in parameter space loses to a stable plateau;
- IS-vs-OOS degradation of the annualized profit rate is computed and
  reported prominently;
- the pass/fail decision is driven by a user-configurable
  :class:`~factory.models.AcceptanceCriteria` (trade count, drawdown,
  profit factor, Sharpe, equity R-squared, consecutive losses, WFE);
- survivors of those gates are stress-tested by the Monte Carlo module and
  must additionally pass its robustness gate.
"""
from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from config import settings
from factory.backtest.base import BacktestEngine
from factory.models import (
    AcceptanceCriteria, BacktestMetrics, JobCancelled, ParamRange,
    StrategyDefinition, ValidationReport, WFOWindowResult,
)

# A cooperative cancellation probe: returns True when the caller should abort.
CancelCheck = Optional[Callable[[], bool]]


def _raise_if_cancelled(cancel_check: CancelCheck) -> None:
    """Raise :class:`JobCancelled` when the cancel probe reports a request.

    Called at safe points throughout the expensive validation pipeline so a
    cancel takes effect within a fraction of a second instead of only being
    seen by the worker loop after the whole candidate finishes.
    """
    if cancel_check is not None and cancel_check():
        raise JobCancelled()


def default_criteria() -> AcceptanceCriteria:
    """Acceptance criteria built from the settings defaults."""
    return AcceptanceCriteria(
        min_wfe=settings.WFE_THRESHOLD,
        max_dd_pct=settings.OOS_MAX_DD_PCT,
        min_trades=settings.MIN_OOS_TRADES,
        min_profit_factor=settings.MIN_PROFIT_FACTOR,
        min_sharpe=settings.MIN_SHARPE,
        min_r_squared=settings.MIN_R_SQUARED,
        max_consecutive_losses=settings.MAX_CONSECUTIVE_LOSSES,
    )


def estimate_stop_points(strategy: StrategyDefinition, *,
                         price: float, atr_price: float,
                         point: float) -> float:
    """Estimate initial SL distance in points from structural exit modes."""
    from factory.models import StopLossMode
    from factory.param_scale import collapse_scaled_point_params

    strat = collapse_scaled_point_params(strategy)
    tm = strat.trade_mgmt
    tp = tm.params
    mp = strat.mechanic.params
    if tm.sl_mode == StopLossMode.OFF:
        return 0.0
    if tm.sl_mode == StopLossMode.ATR and atr_price > 0.0 and point > 0.0:
        return float(atr_price / point * tp.get("atr_sl_mult", 2.0))
    if tm.sl_mode == StopLossMode.PERCENT and price > 0.0 and point > 0.0:
        return float(price * (float(tp.get("sl_pct", 1.0)) / 100.0) / point)
    return float(mp.get("sl_points", 0.0) or 0.0)


def stop_economics_sane(strategy: StrategyDefinition, *,
                        price: float, atr_price: float, point: float,
                        spread_points: float, slippage_points: float
                        ) -> Tuple[bool, str]:
    """Reject FX-tiny / sub-cost stops before Optuna burns trials.

    Returns ``(ok, reason)``. ATR/percent/FIXED stops must clear a floor of
    ``SCREEN_MIN_STOP_ATR_MULT × ATR`` and ``SCREEN_MIN_STOP_COST_MULT ×``
    one-way spread+slip (proxy for round-trip friction).
    """
    from factory.models import ExecutionMechanicType, StopLossMode

    if strategy.mechanic.type == ExecutionMechanicType.DCA_GRID:
        return True, ""
    if strategy.trade_mgmt.sl_mode == StopLossMode.OFF:
        return True, ""
    if point <= 0.0 or price <= 0.0:
        return True, ""

    sl_pts = estimate_stop_points(
        strategy, price=price, atr_price=atr_price, point=point)
    if sl_pts <= 0.0:
        return False, "stop distance is zero"

    atr_pts = (atr_price / point) if atr_price > 0.0 else 0.0
    min_atr = float(getattr(settings, "SCREEN_MIN_STOP_ATR_MULT", 0.25))
    if atr_pts > 0.0 and sl_pts < min_atr * atr_pts:
        return False, (
            f"stop {sl_pts:.0f} pts < {min_atr:.2f}×ATR ({atr_pts:.0f} pts)"
        )

    cost = max(0.0, float(spread_points) + float(slippage_points))
    min_cost = float(getattr(settings, "SCREEN_MIN_STOP_COST_MULT", 2.0))
    if cost > 0.0 and sl_pts < min_cost * cost:
        return False, (
            f"stop {sl_pts:.0f} pts < {min_cost:.1f}× round-trip cost "
            f"({cost:.0f} pts)"
        )
    return True, ""


def quick_screen(engine: BacktestEngine, strategy: StrategyDefinition,
                 start: datetime, end: datetime, deposit: float,
                 criteria: AcceptanceCriteria,
                 *, ohlc=None, spec=None
                 ) -> Tuple[bool, BacktestMetrics]:
    """Cheap single-pass pre-filter used to triage thousands of candidates.

    One full-range simulation (no optimization, no walk-forward, no Monte
    Carlo) is ~50x cheaper than :func:`validate_strategy`. Candidates that
    are not even profitable on a single straight run over the whole period
    are discarded before spending the expensive full pipeline on them.
    Returns ``(promising, metrics)``.

    When ``ohlc`` / ``spec`` are supplied, economically absurd stops are
    rejected before the bar loop runs.
    """
    if ohlc is not None and spec is not None and len(ohlc) > 20:
        try:
            import numpy as np
            closes = ohlc["close"].to_numpy(dtype=float)
            highs = ohlc["high"].to_numpy(dtype=float)
            lows = ohlc["low"].to_numpy(dtype=float)
            price = float(closes[-1])
            # Simple ATR(14) from the last bars for the economics gate.
            tr = np.maximum(
                highs[-15:] - lows[-15:],
                np.maximum(
                    np.abs(highs[-15:] - np.roll(closes[-15:], 1)),
                    np.abs(lows[-15:] - np.roll(closes[-15:], 1)),
                ),
            )[1:]
            atr_price = float(np.mean(tr)) if len(tr) else 0.0
            ok, reason = stop_economics_sane(
                strategy, price=price, atr_price=atr_price,
                point=float(spec.point),
                spread_points=float(getattr(spec, "spread_points", 0.0)),
                slippage_points=float(getattr(spec, "slippage_points", 0.0)),
            )
            if not ok:
                return False, BacktestMetrics()
        except Exception:
            pass
    try:
        m = engine.run(strategy, start, end, deposit=deposit)
    except Exception:
        return False, BacktestMetrics()
    # loose gates: the strict acceptance criteria are applied later on the
    # OOS zone by the full pipeline. Here we only weed out obvious junk.
    # Keep the funnel aligned with easy Screener·A (L1) so low tiers fill.
    dd_ceiling = max(criteria.max_dd_pct * 1.5, criteria.max_dd_pct + 10.0)
    promising = (
        m.trade_count >= 2
        and m.net_profit > 0.0
        and m.profit_factor >= 1.0
        and m.max_dd_pct < dd_ceiling
    )
    return promising, m


def _equity_curve_fitness(m: BacktestMetrics) -> float:
    """Fitness that targets a smooth, steadily-rising equity curve.

    Profit shaded by drawdown is the base score; *winners* are then scaled by
    the equity curve's linearity (``r_squared`` in [0, 1], where 1.0 is a dead-
    straight line) so both the genetic search and the in-sample optimizer steer
    toward straight, steadily-climbing equity instead of profitable-but-choppy
    curves. A floor keeps some weight on profit so a very strong earner isn't
    fully discarded for a little chop.

    Losing/empty runs keep the plain profit-based score (no smoothness scaling)
    so they stay correctly ranked *below* every winner — a smoothly falling
    curve must never outrank a modest winner.
    """
    if m.trade_count < 3:
        return -1e9
    # /4.0 weights drawdown ~2.5× harder than the legacy /10.0 divisor so
    # genetic search prefers lower-DD equity curves among profitable runs.
    base = m.net_profit / (1.0 + m.max_dd_pct / 4.0)
    if base <= 0.0:
        return base
    smoothness = 0.25 + 0.75 * max(0.0, min(1.0, m.r_squared))
    return base * smoothness


def screen_fitness(m: BacktestMetrics) -> float:
    """Genetic-search fitness — smooth rising curve, boosted by edge expectancy."""
    base = _equity_curve_fitness(m)
    if base <= 0.0 or m.trade_count < 2:
        return base
    exp = m.expectancy if m.expectancy != 0.0 else (
        m.net_profit / max(m.trade_count, 1)
    )
    if exp <= 0.0:
        return base
    # Mild boost so positive-expectancy edges outrank equal-profit chop.
    from factory.edge import edge_score
    boost = 1.0 + 0.15 * max(0.0, min(2.0, edge_score(m) / max(abs(exp), 1e-9)))
    return base * boost


def compute_wfe(is_metrics: BacktestMetrics, oos_metrics: BacktestMetrics) -> float:
    """Walk-Forward Efficiency = OOS annualized profit rate / IS annualized rate.

    A meaningless (non-positive) IS rate yields WFE 0 — such strategies never
    pass the gate anyway, and dividing by a negative rate would flip signs.
    """
    is_rate = is_metrics.annualized_profit_rate()
    oos_rate = oos_metrics.annualized_profit_rate()
    if is_rate <= 0:
        return 0.0
    return oos_rate / is_rate


def compute_degradation_pct(is_metrics: BacktestMetrics,
                            oos_metrics: BacktestMetrics) -> float:
    """IS -> OOS degradation of the annualized profit rate, in percent.

    0 = no degradation, 100 = the whole IS edge vanished out of sample,
    negative = OOS actually outperformed IS. 100 when IS was not profitable.
    """
    is_rate = is_metrics.annualized_profit_rate()
    if is_rate <= 0:
        return 100.0
    oos_rate = oos_metrics.annualized_profit_rate()
    return round((1.0 - oos_rate / is_rate) * 100.0, 2)


def _sample_param_set(ranges: Dict[str, ParamRange], rng: random.Random) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, r in ranges.items():
        n_steps = int(round((r.max - r.min) / r.step)) if r.step > 0 else 0
        out[name] = r.min + rng.randint(0, n_steps) * r.step if n_steps else r.min
    return out


def _neighbor_sets(params: Dict[str, float], ranges: Dict[str, ParamRange],
                   rng: random.Random, n: int) -> List[Dict[str, float]]:
    """Random +/-1-step neighbors of a parameter set (snapped to the grid)."""
    out: List[Dict[str, float]] = []
    movable = [name for name, r in ranges.items() if r.step > 0]
    if not movable:
        return out
    for _ in range(n):
        nb = dict(params)
        changed = False
        for name in movable:
            if rng.random() < 0.5:
                r = ranges[name]
                nb[name] = r.clamp(params.get(name, r.min)
                                   + rng.choice((-1, 1)) * r.step)
                changed = changed or nb[name] != params.get(name)
        if not changed:               # force at least one step
            name = rng.choice(movable)
            r = ranges[name]
            nb[name] = r.clamp(params.get(name, r.min)
                               + rng.choice((-1, 1)) * r.step)
        out.append(nb)
    return out


def _is_continuous_heavy(ranges: Dict[str, ParamRange]) -> bool:
    """True when most params have fine-grained steps (favor CMA-ES)."""
    if not ranges:
        return False
    fine = 0
    for r in ranges.values():
        if r.step <= 0:
            continue
        n_steps = int(round((r.max - r.min) / r.step))
        if n_steps >= 20:
            fine += 1
    return fine >= max(1, len(ranges) // 2)


def _suggest_params(trial, ranges: Dict[str, ParamRange], *,
                    continuous: bool = False) -> Dict[str, float]:
    """Map ParamRange grids onto Optuna suggestions, then snap to the grid."""
    out: Dict[str, float] = {}
    for name, r in ranges.items():
        if r.step <= 0 or r.max <= r.min:
            out[name] = float(r.min)
            continue
        if continuous:
            raw = trial.suggest_float(name, float(r.min), float(r.max))
            out[name] = r.clamp(raw)
            continue
        n_steps = int(round((r.max - r.min) / r.step))
        if n_steps <= 0:
            out[name] = float(r.min)
            continue
        idx = trial.suggest_int(f"{name}__idx", 0, n_steps)
        out[name] = r.clamp(r.min + idx * r.step)
    return out


def _to_dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _segment_bounds(start: datetime, end: datetime, n_segments: int
                    ) -> List[Tuple[datetime, datetime]]:
    """Chronological IS chunks for MedianPruner intermediate reporting."""
    n_segments = max(1, int(n_segments))
    t0, t1 = start.timestamp(), end.timestamp()
    if t1 <= t0 or n_segments == 1:
        return [(start, end)]
    span = (t1 - t0) / n_segments
    out: List[Tuple[datetime, datetime]] = []
    for i in range(n_segments):
        a = t0 + i * span
        b = t1 if i == n_segments - 1 else t0 + (i + 1) * span
        out.append((_to_dt(a), _to_dt(b)))
    return out


def optimize_is(engine: BacktestEngine, strategy: StrategyDefinition,
                start: datetime, end: datetime, deposit: float,
                n_samples: int, rng: random.Random,
                stability: bool = False,
                cancel_check: CancelCheck = None
                ) -> Tuple[Dict[str, float], BacktestMetrics, float, int, List[Dict]]:
    """Optuna (TPE / CMA-ES) optimization over the strategy's parameter ranges.

    Trial 0 is always the strategy's own current parameters (enqueued), so the
    optimizer can never return something worse than the incumbent.

    With ``stability=True`` the top candidates are re-scored by the average
    fitness of their random +/-1-step neighbors (plateau beats peak).

    Returns ``(best_params, best_metrics, stability_ratio, is_trials,
    param_search_trace)``.
    """
    import optuna
    from optuna.exceptions import TrialPruned
    from optuna.pruners import MedianPruner
    from optuna.samplers import CmaEsSampler, TPESampler

    # Optuna is chatty by default; keep discovery logs clean.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ranges = strategy.all_ranges()
    incumbent = strategy.all_params()
    n_samples = max(1, int(n_samples))
    n_segments = max(1, int(getattr(settings, "OPTUNA_PRUNE_SEGMENTS", 4)))
    segments = _segment_bounds(start, end, n_segments)
    seed = rng.randint(0, 2**31 - 1)

    sampler_name = str(getattr(settings, "OPTUNA_SAMPLER", "auto")).lower()
    if sampler_name == "auto":
        sampler_name = "cmaes" if _is_continuous_heavy(ranges) else "tpe"
    n_startup = int(getattr(settings, "OPTUNA_N_STARTUP_TRIALS", 3))
    use_cmaes = sampler_name == "cmaes" and bool(ranges)
    if use_cmaes:
        try:
            import cmaes  # noqa: F401  — Optuna's CmaEsSampler needs this package
        except ImportError:
            use_cmaes = False
    if use_cmaes:
        # CMA-ES requires float space; suggest floats then clamp to the grid.
        sampler = CmaEsSampler(seed=seed, n_startup_trials=min(n_startup, n_samples))
    else:
        sampler = TPESampler(seed=seed, n_startup_trials=min(n_startup, n_samples))
        use_cmaes = False

    pruner = MedianPruner(n_startup_trials=min(n_startup, n_samples),
                          n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                pruner=pruner)

    # Enqueue incumbent so trial 0 evaluates the current parameters.
    if ranges:
        enqueue = {}
        for name, r in ranges.items():
            if r.step <= 0 or r.max <= r.min:
                continue
            val = float(incumbent.get(name, r.min))
            if use_cmaes:
                enqueue[name] = max(float(r.min), min(float(r.max), val))
            else:
                n_steps = int(round((r.max - r.min) / r.step))
                if n_steps <= 0:
                    continue
                idx = int(round((val - r.min) / r.step))
                enqueue[f"{name}__idx"] = max(0, min(n_steps, idx))
        if enqueue:
            study.enqueue_trial(enqueue)

    scored: List[Tuple[Dict[str, float], BacktestMetrics, float]] = []

    def objective(trial: "optuna.Trial") -> float:
        _raise_if_cancelled(cancel_check)
        if not ranges:
            params = dict(incumbent)
        else:
            params = _suggest_params(trial, ranges, continuous=use_cmaes)
            for k, v in incumbent.items():
                params.setdefault(k, v)

        # Chronological segments: report intermediate fitness for MedianPruner.
        last_fitness = float("-inf")
        last_metrics: Optional[BacktestMetrics] = None
        for step, (_seg_start, seg_end) in enumerate(segments):
            _raise_if_cancelled(cancel_check)
            try:
                m = engine.run(strategy, start, seg_end,
                               params_override=params, deposit=deposit)
            except JobCancelled:
                raise
            except Exception:
                raise TrialPruned()
            last_metrics = m
            last_fitness = _fitness(m)
            trial.report(last_fitness, step)
            if trial.should_prune():
                raise TrialPruned()

        assert last_metrics is not None
        scored.append((params, last_metrics, last_fitness))
        return last_fitness

    try:
        study.optimize(objective, n_trials=n_samples, catch=(TrialPruned,))
    except JobCancelled:
        raise

    is_trials = len(study.trials)
    trace_n = int(getattr(settings, "OPTUNA_TRACE_TOP_N", 10))
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    completed.sort(key=lambda t: t.value, reverse=True)
    param_trace: List[Dict] = []
    for t in completed[:trace_n]:
        if ranges:
            params = {}
            for name, r in ranges.items():
                if use_cmaes and name in t.params:
                    params[name] = r.clamp(float(t.params[name]))
                else:
                    key = f"{name}__idx"
                    if key in t.params:
                        params[name] = r.clamp(
                            r.min + int(t.params[key]) * r.step)
                    else:
                        params[name] = incumbent.get(name, r.min)
            for k, v in incumbent.items():
                params.setdefault(k, v)
        else:
            params = dict(incumbent)
        param_trace.append({"params": params, "value": float(t.value)})

    if not scored:
        m = engine.run(strategy, start, end, deposit=deposit)
        return strategy.all_params(), m, 1.0, is_trials, param_trace

    scored.sort(key=lambda t: t[2], reverse=True)

    if not (stability and settings.NEIGHBORHOOD_STABILITY and ranges):
        best_params, best_metrics, _ = scored[0]
        return best_params, best_metrics, 1.0, is_trials, param_trace

    # neighborhood re-scoring of the top candidates
    best_entry = None
    best_nb_score = float("-inf")
    for params, metrics, own_fitness in scored[:settings.NEIGHBOR_TOP_K]:
        _raise_if_cancelled(cancel_check)
        fitnesses = [own_fitness]
        for nb in _neighbor_sets(params, ranges, rng, settings.NEIGHBOR_SAMPLES):
            try:
                nb_m = engine.run(strategy, start, end,
                                  params_override=nb, deposit=deposit)
            except Exception:
                continue
            fitnesses.append(_fitness(nb_m))
        nb_score = sum(fitnesses) / len(fitnesses)
        if nb_score > best_nb_score:
            best_nb_score = nb_score
            ratio = nb_score / own_fitness if own_fitness > 0 else 1.0
            best_entry = (params, metrics, max(0.0, round(ratio, 4)))
    assert best_entry is not None
    return (*best_entry, is_trials, param_trace)


def _fitness(m: BacktestMetrics) -> float:
    """IS-optimization fitness — same smooth-rising-equity objective as the
    genetic search, so each optimized window prefers straight, steady equity."""
    return _equity_curve_fitness(m)


def _month_secs(months: int) -> float:
    """Average calendar month length in seconds (matches DAYS_PER_MONTH)."""
    return months * settings.DAYS_PER_MONTH * 86400.0


def walk_forward(engine: BacktestEngine, strategy: StrategyDefinition,
                 start: datetime, end: datetime, deposit: float,
                 n_windows: int, mode: str, rng: random.Random,
                 train_months: Optional[int] = None,
                 test_months: Optional[int] = None,
                 cancel_check: CancelCheck = None) -> List[WFOWindowResult]:
    """Anchored or rolling walk-forward: optimize IS window, test on the next slice.

  When ``train_months`` and ``test_months`` are set, window boundaries use
  fixed calendar-month lengths instead of splitting the total range into
  equal fractions (e.g. 2 months train + 1 month test).
    """
    t0, t1 = start.timestamp(), end.timestamp()
    results: List[WFOWindowResult] = []

    if train_months and test_months:
        train_len = _month_secs(train_months)
        test_len = _month_secs(test_months)
        if mode == "anchored":
            max_fit = int((t1 - t0 - train_len) / test_len) if t1 - t0 > train_len else 0
        else:
            max_fit = int((t1 - t0 - train_len) / test_len) if t1 - t0 > train_len + test_len else 0
        n = min(n_windows, max(0, max_fit))
        for w in range(n):
            oos_end = t1 - w * test_len
            oos_start = oos_end - test_len
            if oos_start < t0:
                break
            is_end = oos_start
            is_start = t0 if mode == "anchored" else max(t0, is_end - train_len)
            if is_end - is_start < train_len * 0.5:
                continue
            _raise_if_cancelled(cancel_check)
            try:
                params, is_m, *_rest = optimize_is(
                    engine, strategy, _to_dt(is_start), _to_dt(is_end), deposit,
                    settings.WFO_OPT_SAMPLES, rng, cancel_check=cancel_check,
                )
                oos_m = engine.run(strategy, _to_dt(oos_start), _to_dt(oos_end),
                                   params_override=params, deposit=deposit)
            except Exception:
                continue
            results.append(WFOWindowResult(
                mode=mode, index=w,
                is_start_ts=is_start, is_end_ts=is_end,
                oos_start_ts=oos_start, oos_end_ts=oos_end,
                is_metrics=is_m, oos_metrics=oos_m,
                wfe=round(compute_wfe(is_m, oos_m), 4),
            ))
        return results

    total = t1 - t0
    oos_len = total / (n_windows + 2)          # each OOS slice

    for w in range(n_windows):
        oos_start = t0 + total - (n_windows - w) * oos_len
        oos_end = oos_start + oos_len
        if mode == "anchored":
            is_start = t0
        else:                                   # rolling: fixed-length IS window
            is_start = max(t0, oos_start - 2 * oos_len)
        is_end = oos_start

        _raise_if_cancelled(cancel_check)
        try:
            params, is_m, *_rest = optimize_is(
                engine, strategy, _to_dt(is_start), _to_dt(is_end), deposit,
                settings.WFO_OPT_SAMPLES, rng, cancel_check=cancel_check,
            )
            oos_m = engine.run(strategy, _to_dt(oos_start), _to_dt(oos_end),
                               params_override=params, deposit=deposit)
        except Exception:
            continue
        results.append(WFOWindowResult(
            mode=mode, index=w,
            is_start_ts=is_start, is_end_ts=is_end,
            oos_start_ts=oos_start, oos_end_ts=oos_end,
            is_metrics=is_m, oos_metrics=oos_m,
            wfe=round(compute_wfe(is_m, oos_m), 4),
        ))
    return results


def validate_strategy(engine: BacktestEngine, strategy: StrategyDefinition,
                      start: datetime, end: datetime,
                      deposit: float = settings.DEFAULT_DEPOSIT,
                      seed: Optional[int] = None,
                      criteria: Optional[AcceptanceCriteria] = None,
                      run_montecarlo: Optional[bool] = None,
                      mc_config=None,
                      wfo_train_months: Optional[int] = None,
                      wfo_test_months: Optional[int] = None,
                      wfo_windows: Optional[int] = None,
                      data_source: Optional[str] = None,
                      cancel_check: CancelCheck = None,
                      spec_overrides: Optional[Mapping[str, float]] = None,
                      n_trials: int = 1,
                      floor_level: Optional[int] = None,
                      ceiling_level: Optional[int] = None,
                      ) -> ValidationReport:
    """Full pipeline: 70/30 chronological split -> IS optimization (with
    neighborhood-stability scoring) -> OOS test -> anchored + rolling WFO
    -> multi-level population scoring -> Monte Carlo when mid-tier clears.

    When ``floor_level`` / ``ceiling_level`` are set, the report is scored
    against every named validation level L1–L16 (one backtest, nested gates).
    ``passed`` means the candidate cleared at least ``floor_level``;
    ``highest_level_passed`` is the best tier cleared (0 if none). Monte Carlo
    runs once the OOS metrics clear the last pre-MC level (``MC_PRE_LEVEL``),
    using the L16 MC config so L7–L16 can be scored from one stress test.

    Legacy single-criteria mode (no floor/ceiling) keeps the previous boolean
    gate behavior, with ``highest_level_passed`` set to 1 on pass / 0 on fail.
    """
    from factory import validation_levels
    import time as _time

    t_start = _time.perf_counter()
    rng = random.Random(seed)
    criteria = criteria or default_criteria()
    t0, t1 = start.timestamp(), end.timestamp()
    split_ts = t0 + (t1 - t0) * settings.IS_OOS_SPLIT

    tier_mode = floor_level is not None or ceiling_level is not None
    floor = validation_levels.MIN_LEVEL if floor_level is None else int(floor_level)
    # Always score the full ladder in tier mode; ceiling_level is accepted for
    # API compat but scoring/MC depth use MAX_LEVEL.
    ceiling = validation_levels.MAX_LEVEL
    floor = max(validation_levels.MIN_LEVEL,
                min(validation_levels.MAX_LEVEL, floor))
    ceiling = max(floor, min(validation_levels.MAX_LEVEL, ceiling))

    _raise_if_cancelled(cancel_check)
    best_params, is_metrics, stability_ratio, is_trials, param_trace = optimize_is(
        engine, strategy, start, _to_dt(split_ts), deposit,
        settings.OPT_SAMPLES, rng,
        stability=settings.NEIGHBORHOOD_STABILITY, cancel_check=cancel_check,
    )
    oos_metrics = engine.run(strategy, _to_dt(split_ts), end,
                             params_override=best_params, deposit=deposit)

    wfe = compute_wfe(is_metrics, oos_metrics)

    # Selection-bias statistics (reported, not gated): how likely is this
    # OOS Sharpe to be real given how many candidates the run has tried?
    from factory.backtest.statistics import dsr_from_metrics, p_oos_loss

    # Per-regime OOS breakdown (simulator only — needs the trade book).
    # Reported always; gated only when criteria.max_regime_loss_pct is set.
    regime_stats: List = []
    regime_reasons: List[str] = []
    if hasattr(engine, "run_with_trades"):
        from factory.regime import regime_stats_from_book, worst_regime_net
        try:
            _raise_if_cancelled(cancel_check)
            _, oos_book, oos_df = engine.run_with_trades(
                strategy, _to_dt(split_ts), end,
                params_override=best_params, deposit=deposit)
            regime_stats = regime_stats_from_book(oos_df, oos_book)
        except JobCancelled:
            raise
        except Exception:
            regime_stats = []
        if regime_stats and criteria.max_regime_loss_pct > 0:
            worst = worst_regime_net(regime_stats)
            limit = -deposit * criteria.max_regime_loss_pct / 100.0
            if worst < limit:
                worst_name = min(regime_stats,
                                 key=lambda s: s.net_profit).name
                regime_reasons.append(
                    f"worst regime '{worst_name}' loses {worst:.2f}"
                    f" (> {criteria.max_regime_loss_pct}% of deposit)")

    # Nested ladder: if L1 fails, do not spend budget on WFO / MC / higher
    # levels. Higher tiers are only evaluated when every lower gate clears.
    l1_clears = True
    if tier_mode and not regime_reasons:
        l1_clears = validation_levels.level_clears(
            validation_levels.get_level(validation_levels.MIN_LEVEL),
            oos_metrics, wfe, montecarlo=None)
    elif tier_mode and regime_reasons:
        l1_clears = False

    windows: List[WFOWindowResult] = []
    wfo_n = wfo_windows if wfo_windows is not None else settings.WFO_WINDOWS
    wfo_train = wfo_train_months if wfo_train_months is not None else settings.WFO_TRAIN_MONTHS
    wfo_test = wfo_test_months if wfo_test_months is not None else settings.WFO_TEST_MONTHS
    wfo_kwargs = dict(train_months=wfo_train, test_months=wfo_test)
    if l1_clears:
        for mode in settings.WFO_MODES:
            windows += walk_forward(engine, strategy, start, end, deposit,
                                    wfo_n, mode, rng, cancel_check=cancel_check,
                                    **wfo_kwargs)

    dsr = dsr_from_metrics(oos_metrics, n_trials)
    oos_loss_frac = p_oos_loss(windows) if windows else 0.0
    # Empty WFO after L1 clear is a hard honesty miss (cannot prove folds).
    if windows:
        honesty_p_oos = float(oos_loss_frac)
    elif l1_clears and tier_mode:
        honesty_p_oos = 1.0
    else:
        honesty_p_oos = None
    honesty = validation_levels.HonestySignals(
        p_oos_loss=honesty_p_oos,
        dsr=float(dsr),
        stability_ratio=float(stability_ratio),
    )

    # Monte Carlo: only after nested clear through the last pre-MC level
    # (never run MC when L1–L6 already failed).
    mc_result = None
    if run_montecarlo is None:
        run_montecarlo = settings.MC_ENABLED and \
            getattr(engine, "name", "") == "simulator"

    should_run_mc = bool(run_montecarlo) and l1_clears
    if tier_mode and should_run_mc:
        pre_hi = validation_levels.highest_level_cleared(
            oos_metrics, wfe, None,
            ceiling=validation_levels.MC_PRE_LEVEL, floor=1,
            honesty=honesty)
        should_run_mc = pre_hi >= validation_levels.MC_PRE_LEVEL
        if should_run_mc and mc_config is None:
            mc_config = validation_levels.mc_config_for(
                validation_levels.get_level(validation_levels.MAX_LEVEL))
            if mc_config is None:
                should_run_mc = False
    elif should_run_mc:
        base_reasons = criteria.evaluate(oos_metrics, wfe) + regime_reasons
        should_run_mc = not base_reasons

    mc_fail_reasons: List[str] = []
    if should_run_mc:
        from factory.backtest.montecarlo import (
            MonteCarloConfig, montecarlo_for_strategy,
        )
        _raise_if_cancelled(cancel_check)
        cfg = mc_config or MonteCarloConfig(seed=rng.randint(0, 2**31))
        try:
            mc_result = montecarlo_for_strategy(
                strategy, start, end, deposit, cfg,
                params_override=best_params, cancel_check=cancel_check,
                spec_overrides=spec_overrides)
            if not tier_mode and not mc_result.passed:
                mc_fail_reasons += [
                    f"Monte Carlo: {r}" for r in mc_result.reasons]
        except JobCancelled:
            raise
        except Exception as exc:
            mc_fail_reasons.append(f"Monte Carlo failed to run: {exc}")

    if tier_mode:
        highest = validation_levels.highest_level_cleared(
            oos_metrics, wfe, mc_result, ceiling=ceiling, floor=1,
            honesty=honesty)
        if regime_reasons:
            # Regime gate (rare) demotes below floor when configured.
            highest = 0
        passed = highest >= floor and not regime_reasons
        reasons = list(regime_reasons)
        if not passed:
            reasons += validation_levels.reasons_for_next_level(
                oos_metrics, wfe, mc_result,
                highest=highest, ceiling=ceiling, honesty=honesty)
            if not reasons and highest < floor:
                fl = validation_levels.get_level(floor)
                reasons = [
                    f"Cleared L{highest} only; floor is L{floor} ({fl.name})"
                ]
        elif highest < ceiling:
            # Still surface what blocks the next tier for population triage.
            reasons = validation_levels.reasons_for_next_level(
                oos_metrics, wfe, mc_result,
                highest=highest, ceiling=ceiling, honesty=honesty)
        gate_criteria = validation_levels.get_level(floor).criteria
        levels_map = validation_levels.levels_cleared_map(
            oos_metrics, wfe, mc_result, ceiling=ceiling, highest=highest,
            honesty=honesty)
    else:
        reasons = criteria.evaluate(oos_metrics, wfe) + regime_reasons + mc_fail_reasons
        passed = not reasons
        highest = 1 if passed else 0
        gate_criteria = criteria
        levels_map = {"1": bool(passed)} if passed else {}

    duration_ms = round((_time.perf_counter() - t_start) * 1000.0, 3)
    soft_wfe_clear = False
    if tier_mode and int(highest) > 0:
        hi_lvl = validation_levels.get_level(int(highest))
        soft_wfe_clear = validation_levels.soft_wfe_waived(hi_lvl, oos_metrics, wfe)

    return ValidationReport(
        strategy_id=strategy.id,
        is_metrics=is_metrics,
        oos_metrics=oos_metrics,
        wfo_windows=windows,
        wfe=round(wfe, 4),
        passed=passed,
        highest_level_passed=int(highest),
        soft_wfe_clear=bool(soft_wfe_clear),
        levels_cleared=levels_map,
        duration_ms=duration_ms,
        reasons=reasons,
        best_params=best_params,
        engine=getattr(engine, "name", "unknown"),
        is_range=(t0, split_ts),
        oos_range=(split_ts, t1),
        criteria=gate_criteria,
        montecarlo=mc_result,
        degradation_pct=compute_degradation_pct(is_metrics, oos_metrics),
        stability_ratio=stability_ratio,
        data_source=data_source or "unknown",
        wfo_train_months=wfo_train,
        wfo_test_months=wfo_test,
        dsr=dsr,
        n_trials=max(1, int(n_trials)),
        is_trials=int(is_trials),
        param_search_trace=param_trace,
        p_oos_loss=oos_loss_frac,
        regime_stats=regime_stats,
    )
