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


def quick_screen(engine: BacktestEngine, strategy: StrategyDefinition,
                 start: datetime, end: datetime, deposit: float,
                 criteria: AcceptanceCriteria
                 ) -> Tuple[bool, BacktestMetrics]:
    """Cheap single-pass pre-filter used to triage thousands of candidates.

    One full-range simulation (no optimization, no walk-forward, no Monte
    Carlo) is ~50x cheaper than :func:`validate_strategy`. Candidates that
    are not even profitable on a single straight run over the whole period
    are discarded before spending the expensive full pipeline on them.
    Returns ``(promising, metrics)``.
    """
    try:
        m = engine.run(strategy, start, end, deposit=deposit)
    except Exception:
        return False, BacktestMetrics()
    # loose gates: the strict acceptance criteria are applied later on the
    # OOS zone by the full pipeline. Here we only weed out obvious junk.
    dd_ceiling = max(criteria.max_dd_pct * 1.5, criteria.max_dd_pct + 10.0)
    promising = (
        m.trade_count >= max(3, criteria.min_trades)
        and m.net_profit > 0.0
        and m.profit_factor >= 1.05
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
    base = m.net_profit / (1.0 + m.max_dd_pct / 10.0)
    if base <= 0.0:
        return base
    smoothness = 0.25 + 0.75 * max(0.0, min(1.0, m.r_squared))
    return base * smoothness


def screen_fitness(m: BacktestMetrics) -> float:
    """Genetic-search fitness — biases discovery toward a smooth rising curve."""
    return _equity_curve_fitness(m)


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


def optimize_is(engine: BacktestEngine, strategy: StrategyDefinition,
                start: datetime, end: datetime, deposit: float,
                n_samples: int, rng: random.Random,
                stability: bool = False,
                cancel_check: CancelCheck = None
                ) -> Tuple[Dict[str, float], BacktestMetrics, float]:
    """Random-search optimization over the strategy's parameter ranges.

    Candidate 0 is always the strategy's own current parameters, so the
    optimizer can never return something worse than the incumbent.

    With ``stability=True`` the top candidates are re-scored by the average
    fitness of their random +/-1-step neighbors (plateau beats peak).
    Returns ``(best_params, best_metrics, stability_ratio)`` where the
    ratio is neighborhood-average / own fitness for the winner (1.0 when
    the stability pass is disabled or not applicable).
    """
    candidates: List[Dict[str, float]] = [strategy.all_params()]
    ranges = strategy.all_ranges()
    candidates += [_sample_param_set(ranges, rng) for _ in range(max(0, n_samples - 1))]

    scored: List[Tuple[Dict[str, float], BacktestMetrics, float]] = []
    for params in candidates:
        _raise_if_cancelled(cancel_check)
        try:
            m = engine.run(strategy, start, end, params_override=params, deposit=deposit)
        except Exception:
            continue
        scored.append((params, m, _fitness(m)))

    if not scored:
        m = engine.run(strategy, start, end, deposit=deposit)
        return strategy.all_params(), m, 1.0

    scored.sort(key=lambda t: t[2], reverse=True)

    if not (stability and settings.NEIGHBORHOOD_STABILITY and ranges):
        best_params, best_metrics, _ = scored[0]
        return best_params, best_metrics, 1.0

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
    return best_entry


def _fitness(m: BacktestMetrics) -> float:
    """IS-optimization fitness — same smooth-rising-equity objective as the
    genetic search, so each optimized window prefers straight, steady equity."""
    return _equity_curve_fitness(m)


def _to_dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _month_secs(months: int) -> float:
    """Average calendar month length in seconds."""
    return months * 30.4375 * 86400.0


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
                params, is_m, _ = optimize_is(
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
            params, is_m, _ = optimize_is(
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
                      spec_overrides: Optional[Mapping[str, float]] = None
                      ) -> ValidationReport:
    """Full pipeline: 70/30 chronological split -> IS optimization (with
    neighborhood-stability scoring) -> OOS test -> anchored + rolling WFO
    -> acceptance-criteria gates -> Monte Carlo robustness gate.

    ``run_montecarlo=None`` (default) runs Monte Carlo only when it is
    enabled in settings AND the engine is the simulator (MC is always
    simulator-based; a stub/MT5 engine skips it unless forced).

    ``spec_overrides`` carries the user-chosen account/execution economics
    (leverage, spread, slippage, contract size). IS optimization, OOS and
    walk-forward pick these up through the engine (which was constructed with
    them); Monte Carlo builds its own spec, so they are forwarded to it
    explicitly here to keep all stages consistent.
    """
    rng = random.Random(seed)
    criteria = criteria or default_criteria()
    t0, t1 = start.timestamp(), end.timestamp()
    split_ts = t0 + (t1 - t0) * settings.IS_OOS_SPLIT

    _raise_if_cancelled(cancel_check)
    best_params, is_metrics, stability_ratio = optimize_is(
        engine, strategy, start, _to_dt(split_ts), deposit,
        settings.OPT_SAMPLES, rng, stability=True, cancel_check=cancel_check,
    )
    oos_metrics = engine.run(strategy, _to_dt(split_ts), end,
                             params_override=best_params, deposit=deposit)

    windows: List[WFOWindowResult] = []
    wfo_n = wfo_windows if wfo_windows is not None else settings.WFO_WINDOWS
    wfo_train = wfo_train_months if wfo_train_months is not None else settings.WFO_TRAIN_MONTHS
    wfo_test = wfo_test_months if wfo_test_months is not None else settings.WFO_TEST_MONTHS
    wfo_kwargs = dict(train_months=wfo_train, test_months=wfo_test)
    for mode in ("anchored", "rolling"):
        windows += walk_forward(engine, strategy, start, end, deposit,
                                wfo_n, mode, rng, cancel_check=cancel_check,
                                **wfo_kwargs)

    wfe = compute_wfe(is_metrics, oos_metrics)
    reasons: List[str] = criteria.evaluate(oos_metrics, wfe)

    # Monte Carlo robustness gate — only for strategies that pass the base
    # acceptance criteria (no point stress-testing rejects).
    mc_result = None
    if run_montecarlo is None:
        run_montecarlo = settings.MC_ENABLED and \
            getattr(engine, "name", "") == "simulator"
    if run_montecarlo and not reasons:
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
            if not mc_result.passed:
                reasons += [f"Monte Carlo: {r}" for r in mc_result.reasons]
        except JobCancelled:
            raise
        except Exception as exc:
            reasons.append(f"Monte Carlo failed to run: {exc}")

    return ValidationReport(
        strategy_id=strategy.id,
        is_metrics=is_metrics,
        oos_metrics=oos_metrics,
        wfo_windows=windows,
        wfe=round(wfe, 4),
        passed=not reasons,
        reasons=reasons,
        best_params=best_params,
        engine=getattr(engine, "name", "unknown"),
        is_range=(t0, split_ts),
        oos_range=(split_ts, t1),
        criteria=criteria,
        montecarlo=mc_result,
        degradation_pct=compute_degradation_pct(is_metrics, oos_metrics),
        stability_ratio=stability_ratio,
        data_source=data_source or "unknown",
        wfo_train_months=wfo_train,
        wfo_test_months=wfo_test,
    )
