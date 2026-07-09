# EA_Generator research addendum: MT5 optimization and robustness validation (2026-07)

This addendum complements the existing architecture and phased roadmap with concise, implementation-focused guidance for MetaTrader 5 optimization, profitability validation, and production gating in a 24/7 discovery pipeline.

## 1) MT5 optimization modes and when to use each

- **`Slow complete algorithm` for final parameter certainty.** MT5 defines complete optimization as testing all combinations, which is the only mode that guarantees full parameter-space coverage. Use it for narrowed ranges or final confirmation runs. [Strategy Optimization (MT5 Help)](https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization), [Optimization Types (MT5 Help)](https://www.metatrader5.com/en/terminal/help/algotrading/optimization_types)
- **`Fast genetic algorithm` for broad search.** MT5 documents it as much faster and often near the quality of complete search, but it intentionally skips combinations; treat it as a discovery accelerator, not the final verdict. [Optimization Types (MT5 Help)](https://www.metatrader5.com/en/terminal/help/algotrading/optimization_types)
- **Forward optimization should be default in optimization runs.** MT5 splits your date range into IS and forward/OOS segments (1/2, 1/3, 1/4, or custom), then retests top optimization runs on forward data to reduce curve-fit acceptance. [Strategy Testing (MT5 Help)](https://www.metatrader5.com/en/terminal/help/algotrading/testing), [Strategy Optimization (MT5 Help)](https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization)
- **Know the retest fraction behavior.** MT5 states forward retests select top 10% (complete search) or 25% (genetic) from optimization results. [Strategy Optimization (MT5 Help)](https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization)
- **Cloud network policy matters.** MT5 cloud docs note complete optimization can use full network power, while genetic optimization is constrained to one access point due to generation synchronization needs; this affects latency and scaling assumptions. [MQL5 Cloud use (MT5 Help)](https://www.metatrader5.com/en/terminal/help/mql5cloud/mql5cloud_use), [MQL5 Cloud FAQ](https://cloud.mql5.com/en/faq)
- **Tick modeling caveat is non-negotiable.** MT5/MQL5 docs explicitly rank speed vs realism: `Open prices only` fastest/least accurate, `1 minute OHLC` faster but rougher, `Every tick` more realistic, and `Every tick based on real ticks` highest fidelity but slowest. [Test preparation (MT5 Help)](https://www.metatrader5.com/en/terminal/help/algotrading/test_preparation), [Tick generation (MT5 Help)](https://www.metatrader5.com/en/terminal/help/algotrading/tick_generation), [Real ticks article (MQL5)](https://www.mql5.com/en/articles/2612)
- **Practical rule:** allow rough modes in discovery, but require final promotion runs in `Every tick` or `Every tick based on real ticks`. [Fundamentals of testing in MT5 (MQL5)](https://www.mql5.com/en/articles/239)

## 2) Advanced profitability and robustness validation methods

- **Walk-forward optimization (WFO):** optimize on IS, freeze parameters, test on adjacent OOS, roll forward, and evaluate stitched OOS only. This is materially stronger than one static split. [Walk forward optimization](https://en.wikipedia.org/wiki/Walk_forward_optimization), [Custom WFO in MT5 (MQL5)](https://www.mql5.com/en/articles/3279)
- **Use both rolling and anchored WFO.** Rolling windows stress regime adaptation; anchored windows stress long-horizon consistency. Using both catches regime-specific overfit that a single scheme can miss. [WFO controlled study](https://wfo.marketmaker.cc/)
- **Treat fixed WFE folklore cautiously.** Common pass/fail heuristics (for example 0.5 or 0.8) are widely used but not universally calibrated; use them as soft priors, not hard truth. [WFO controlled study](https://wfo.marketmaker.cc/)
- **Run multi-axis Monte Carlo stress, not just trade-order shuffle.** MQL5 references support perturbing sequence, execution frictions, and data/parameter conditions to surface fragility before capital allocation. [MC optimization (MQL5)](https://www.mql5.com/en/articles/4347), [MC permutation tests (MQL5)](https://www.mql5.com/en/articles/13162), [MC stress article (MQL5)](https://www.mql5.com/en/articles/22291)
- **Correct for multiple testing and data snooping.** White’s Reality Check asks whether the best discovered model truly beats benchmark after search effects; Hansen SPA improves power and reduces sensitivity to poor alternatives. [White Reality Check](https://doi.org/10.1111/1468-0262.00152), [Hansen SPA](https://doi.org/10.1198/073500105000000063)
- **Quantify overfitting likelihood with PBO/CSCV.** Bailey et al. propose estimating the probability that an IS winner underperforms OOS, directly targeting backtest-overfit risk in model selection pipelines. [Probability of Backtest Overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)
- **Deflate Sharpe-based confidence.** DSR adjusts for multiple trials and non-normal returns, making it more suitable than raw Sharpe for generator workflows. [Deflated Sharpe Ratio (JPM)](https://doi.org/10.3905/jpm.2014.40.5.094), [Deflated Sharpe Ratio (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)

## 3) Recommended thresholds/ranges (with strategy-frequency caveats)

These are practical defaults for EA_Generator policy, not universal constants. The goal is minimizing false-positive promotions while keeping enough throughput.

- **Sample sufficiency floors (per OOS aggregate and per WFO fold):**
  - Low-frequency (H4+): target `>= 80` trades, hard floor `50`.
  - Medium-frequency (M30-H1): target `>= 120` trades.
  - High-frequency/scalping (M1-M15): target `>= 200` trades.
  - Rationale: sparse samples inflate estimation error and selection bias effects. [PBO](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253), [DSR](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- **OOS degradation bounds (WFE-like):**
  - Minimum median OOS/IS ratio: `>= 0.50`.
  - Stronger band for promotion priority: `0.60-0.80+`.
  - Caveat: never gate on this metric alone. [WFO controlled study](https://wfo.marketmaker.cc/)
- **Profitability quality (OOS after costs):**
  - Profit Factor: `>= 1.15` low-frequency, `>= 1.20` medium/high-frequency.
  - Expectancy must remain positive after spread/slippage assumptions.
  - Sharpe acceptance should be DSR-significant (for example alpha 5-10%), not only raw SR threshold. [DSR (JPM)](https://doi.org/10.3905/jpm.2014.40.5.094)
- **Stress risk controls:**
  - MC 95th percentile max DD cap: <= `1.5x-2.0x` historical DD (low/medium frequency), <= `2.5x` (high frequency, higher execution noise).
  - Probability of ruin under target leverage: `< 1%` policy default.
  - Rationale: MC stress routinely reveals materially worse tail DD than single historical path. [MC stress (MQL5)](https://www.mql5.com/en/articles/22291), [MC optimization (MQL5)](https://www.mql5.com/en/articles/4347)
- **Parameter stability control:**
  - Require local-neighborhood pass rate (plateau behavior) `>= 70%` in perturbation checks.
  - Reject isolated sharp peaks even with good point metrics.
  - Rationale: consistent with overfit-control literature and search-bias corrections. [PBO](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253), [White Reality Check](https://doi.org/10.1111/1468-0262.00152)

## 4) Integration guidance for EA_Generator 24/7 discovery pipeline

- **Stage A (`discover_fast`)**
  - Genetic optimization + rough tick mode (`Open prices` or `1m OHLC`).
  - Loose filters: trade count, basic PF, return/DD.
  - Purpose: maximize candidate throughput. [Optimization types](https://www.metatrader5.com/en/terminal/help/algotrading/optimization_types), [Test preparation](https://www.metatrader5.com/en/terminal/help/algotrading/test_preparation)
- **Stage B (`validate_realism`)**
  - Re-run survivors in `Every tick` or real ticks.
  - Apply conservative spread/slippage assumptions.
  - Purpose: remove simulator-mode artifacts. [Tick generation](https://www.metatrader5.com/en/terminal/help/algotrading/tick_generation), [Real ticks article](https://www.mql5.com/en/articles/2612)
- **Stage C (`validate_wfo`)**
  - Execute rolling + anchored WFO; store fold-level OOS metrics.
  - Gate on fold consistency and degradation bounds, not pooled averages only.
  - Purpose: verify cross-regime persistence. [Custom WFO (MQL5)](https://www.mql5.com/en/articles/3279), [WFO study](https://wfo.marketmaker.cc/)
- **Stage D (`stress_mc`)**
  - Run MC battery: trade-order bootstrap, cost perturbations, execution perturbations, start-bar/parameter jitter.
  - Gate on `stress_dd_95`, `ruin_prob`, and positive-tail robustness.
  - Purpose: enforce tail-risk realism. [MC stress (MQL5)](https://www.mql5.com/en/articles/22291), [MC permutation (MQL5)](https://www.mql5.com/en/articles/13162)
- **Stage E (`correct_selection_bias`)**
  - Compute DSR for finalists.
  - Periodic Reality Check/SPA batch against benchmark family.
  - Purpose: ensure “best found” is not just data snooping. [DSR](https://doi.org/10.3905/jpm.2014.40.5.094), [White RC](https://doi.org/10.1111/1468-0262.00152), [SPA](https://doi.org/10.1198/073500105000000063)
- **Stage F (`promote_edge_positive`)**
  - Promote only if all prior gates pass.
  - Save machine-readable pass/fail diagnostics and live demotion triggers.
  - If any output is externally shared as hypothetical performance, enforce required disclaimer workflow. [17 CFR 4.41](https://www.law.cornell.edu/cfr/text/17/4.41)

## What to implement next

- [ ] Add explicit `discover_fast` vs `validate_realism` MT5 mode policy in config.
- [ ] Implement dual WFO engine (rolling + anchored) with fold-level persistence.
- [ ] Add WFO degradation gate (median OOS/IS) and fold-consistency gate.
- [ ] Add DSR calculator and promotion gate on DSR significance.
- [ ] Add PBO/CSCV estimator for shortlisted candidates.
- [ ] Add periodic White RC or Hansen SPA batch validation job.
- [ ] Expand Monte Carlo runner to include execution and parameter perturbation profiles.
- [ ] Add `stress_dd_95` and `ruin_prob` hard gates to promotion logic.
- [ ] Add local parameter-neighborhood stability score to optimizer output.
- [ ] Add promotion ledger with per-gate evidence and demotion triggers.

## Explicit source list by major claim

- **MT5 optimization mode behavior (complete vs genetic, forward retest percentages):**
  - https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization
  - https://www.metatrader5.com/en/terminal/help/algotrading/optimization_types
  - https://www.metatrader5.com/en/terminal/help/algotrading/testing
- **Cloud network scaling and genetic constraints:**
  - https://www.metatrader5.com/en/terminal/help/mql5cloud/mql5cloud_use
  - https://cloud.mql5.com/en/faq
- **Tick modeling quality caveats and realism hierarchy:**
  - https://www.metatrader5.com/en/terminal/help/algotrading/test_preparation
  - https://www.metatrader5.com/en/terminal/help/algotrading/tick_generation
  - https://www.mql5.com/en/articles/2612
  - https://www.mql5.com/en/articles/239
- **WFO method and window-design caveats:**
  - https://www.mql5.com/en/articles/3279
  - https://wfo.marketmaker.cc/
  - https://en.wikipedia.org/wiki/Walk_forward_optimization
- **Monte Carlo and permutation stress rationale in MQL5 context:**
  - https://www.mql5.com/en/articles/4347
  - https://www.mql5.com/en/articles/13162
  - https://www.mql5.com/en/articles/22291
- **Multiple-testing/data-snooping controls (RC/SPA):**
  - https://doi.org/10.1111/1468-0262.00152
  - https://doi.org/10.1198/073500105000000063
- **Backtest overfitting and Sharpe deflation (PBO/DSR):**
  - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
  - https://doi.org/10.3905/jpm.2014.40.5.094
  - https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- **Hypothetical-performance disclosure requirements (if externally published):**
  - https://www.law.cornell.edu/cfr/text/17/4.41
  - https://www.cftc.gov/LawRegulation/FederalRegister/FinalRules/e7-3122.html

## Sources

- [Strategy Optimization - MetaTrader 5 Help](https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization)
- [Optimization Types - MetaTrader 5 Help](https://www.metatrader5.com/en/terminal/help/algotrading/optimization_types)
- [Strategy Testing - MetaTrader 5 Help](https://www.metatrader5.com/en/terminal/help/algotrading/testing)
- [How the Tester Downloads Historical Data - MetaTrader 5 Help](https://www.metatrader5.com/en/terminal/help/algotrading/test_preparation)
- [Real and Generated Ticks - MetaTrader 5 Help](https://www.metatrader5.com/en/terminal/help/algotrading/tick_generation)
- [How to Use - MQL5 Cloud Network - MetaTrader 5 Help](https://www.metatrader5.com/en/terminal/help/mql5cloud/mql5cloud_use)
- [MQL5 Cloud Network FAQ](https://cloud.mql5.com/en/faq)
- [Custom Walk Forward optimization in MetaTrader 5 - MQL5 Articles](https://www.mql5.com/en/articles/3279)
- [Testing trading strategies on real ticks - MQL5 Articles](https://www.mql5.com/en/articles/2612)
- [The Fundamentals of Testing in MetaTrader 5 - MQL5 Articles](https://www.mql5.com/en/articles/239)
- [Applying the Monte Carlo method for optimizing trading strategies - MQL5 Articles](https://www.mql5.com/en/articles/4347)
- [Monte Carlo Permutation Tests in MetaTrader 5 - MQL5 Articles](https://www.mql5.com/en/articles/13162)
- [Stress Testing Trade Sequences with Monte Carlo in MQL5 - MQL5 Articles](https://www.mql5.com/en/articles/22291)
- [The Probability of Backtest Overfitting (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)
- [The Deflated Sharpe Ratio (Journal of Portfolio Management)](https://doi.org/10.3905/jpm.2014.40.5.094)
- [The Deflated Sharpe Ratio (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- [A Reality Check for Data Snooping (Econometrica)](https://doi.org/10.1111/1468-0262.00152)
- [A Test for Superior Predictive Ability (Journal of Business & Economic Statistics)](https://doi.org/10.1198/073500105000000063)
- [Does Walk-Forward Validation Predict Out-of-Sample Performance?](https://wfo.marketmaker.cc/)
- [17 CFR § 4.41](https://www.law.cornell.edu/cfr/text/17/4.41)
- [CFTC Regulation 4.41 Amendments (Federal Register)](https://www.cftc.gov/LawRegulation/FederalRegister/FinalRules/e7-3122.html)
