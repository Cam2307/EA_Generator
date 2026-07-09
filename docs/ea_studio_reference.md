# EA Studio feature reference (research notes)

Research target: **EA Studio** (expert-advisor-studio.com) by Forex Software Ltd.
(forexsb.com), plus its scriptable sibling **Express Generator**. Sources:
forexsb.com wiki (`eas-guide`, `ea-studio-academy`, `express-generator`
sections), the product page, and the forexsb forum. Collected 2026-07.

Purpose of this document: define the benchmark this project must match or
beat, feature by feature. Honest caveat that EA Studio itself states in its
docs: *nothing here guarantees profitability* — every tool below only raises
the odds that a strategy is not curve-fit garbage.

---

## 1. Workflow overview

EA Studio's pipeline: **Generator → Acceptance Criteria → Collection →
(Optimizer / Validator / Monte Carlo / Multi Market) → Portfolio → EA export**.
The "Reactor" automates the whole chain: generate, optimize, robustness-test,
multi-market-test, and push survivors into the Collection without manual
clicks. Our equivalent chain is: generator → simulator pre-filter →
IS/OOS + walk-forward validation → acceptance criteria → Monte Carlo →
multi-market → gallery/portfolio → package export.

## 2. Acceptance criteria

- Act as a *mandatory filter* everywhere (Generator, Reactor, Validator,
  Optimizer): a strategy enters the Collection only if it passes **all**
  criteria in the list.
- Applied to any of three backtest zones independently: **complete backtest**,
  **in-sample (training) part**, **out-of-sample (trading) part**.
- Metrics available (union of EA Studio + Express Generator):
  - min/max **count of trades** (default min is 100 — statistical significance
    is treated as the first-class citizen),
  - max **drawdown %** and max equity drawdown (money),
  - min **profit**, min **profit per day**, min **profit factor**,
  - min **return-to-drawdown ratio**, min **win/loss ratio**,
  - min **R-squared** (stability/linearity of the equity curve),
  - max **consecutive losses**,
  - max **stagnation** (days and %),
  - min **Sharpe** (in EA Studio's stats panel; Express Generator gates via
    R-squared / return-to-drawdown instead),
  - "backtest quality" %.
- Docs recommend deriving thresholds from a strategy you like and loosening
  them slightly ("liked 8% DD → gate at 10%").

**Gap this project closes:** a user-configurable `AcceptanceCriteria` model
(min trades, min profit factor, min Sharpe, min R², max DD, max consecutive
losses, min net profit) evaluated on the OOS zone during validation, with
per-criterion pass/fail reasons persisted and shown in the UI.

## 3. Monte Carlo robustness testing

- Purpose stated bluntly in the docs: the enemy is **over-optimization /
  curve fitting**; Monte Carlo verifies a strategy survives random changes to
  market data, execution, and its own parameters.
- Default: **20 tests**; a strategy is robust if **≥ 80% of tests pass** the
  MC validation criteria (`valid_tests_percent = 80`).
- Variation classes (EA Studio + FSB Pro + Express Generator):
  - **Randomize spread** — spread varies per position up to a configured max.
  - **Randomize slippage** — always applied *against* the trade (brokers'
    slippage is adversarial in practice); applied to entry and exit orders,
    not to SL/TP fills.
  - **Randomize history data** (FSB Pro) — perturb bar high/low.
  - **Randomize indicator parameters** — each numeric parameter has a
    "change probability" (default 20%) and a max change %; an over-fit
    strategy collapses when its parameters are nudged.
  - **Execution problems** (Express Generator) — `skip_entries_percent`,
    `skip_exits_percent`, `rand_close_percent` (random skipped/forced fills).
  - **Randomize first bar** (`rand_first_bar_percent`) — start-bar jitter,
    i.e. the strategy must not depend on one lucky anchoring of history.
- Results shown as a fan of simulated equity curves; validation criteria
  (same metric family as acceptance criteria: `mc_min_profit`,
  `mc_max_drawdown_percent`, `mc_min_profit_factor`, `mc_min_r_squared`,
  `mc_min_count_of_trades`, ...) are evaluated per test, and the strategy
  passes when enough tests pass.

**Gap this project closes:** `factory/backtest/montecarlo.py` re-runs a
validated strategy N times on the event-driven simulator with randomized
spread/slippage, ParamRange-constrained parameter perturbation, random entry
skipping/first-bar jitter, plus **trade-order (equity-curve) resampling**;
computes 5/50/95% confidence bands, a 0–100 robustness score, and a gate
(default: ≥ 80% of runs profitable AND 95%-worst-case DD within limit).

## 4. Multi-market validation

- One click re-tests a strategy against several symbols/timeframes; the
  theory: an edge that exists on more than one market is much less likely to
  be curve-fit to one market's noise.
- Configurable "**validated markets**" count: how many of the extra market
  tests must pass for the strategy to survive.
- Express Generator's `mm.js` does the same for whole collections
  ("keep only those passing your required count of tests").

**Scope note:** multi-market validation is deliberately **out of scope for
now** — the factory currently targets single-symbol EAs only. This section
is kept as the reference for when that changes.

## 5. Collection, correlation analysis, portfolio

- The **Collection** stores the best strategies, sortable/filterable by any
  stat.
- **Correlation analysis**: strategies whose balance lines correlate above a
  threshold (default **0.98**, recommended 0.90–0.99) are flagged as
  duplicates; "Resolve correlations" keeps only the best of each correlated
  pair (by the current sort metric). A second algorithm detects similar
  *trading rules* (same indicators/SL/TP shape, different numeric values).
- **Portfolio**: selected strategies are combined and can be exported as one
  portfolio expert; the vendor's own workflow is explicitly
  quantity-over-attachment: trade many uncorrelated strategies, cull losers
  weekly, replace with fresh survivors.

**Scope note:** correlation analysis and the portfolio view are **out of
scope for now** (single-symbol, single-strategy focus). This section is the
reference for a future Collection upgrade.

## 6. Optimizer over-fitting protections

- The Optimizer sweeps numeric indicator parameters + SL/TP; docs repeatedly
  warn that a wider search range = better fit = *more fragile* strategy, and
  recommend always pairing optimization with Monte Carlo.
- OOS options inside the optimizer (optimize on IS, judge on OOS).
- Acceptance criteria are enforced *during* optimization so losing strategies
  are never "optimized into" the collection.
- Walk-forward is available in FSB Pro; EA Studio leans on OOS + MC + multi
  market instead.

**Gap this project closes (and goes beyond EA Studio):**
- **Parameter-neighborhood stability**: a candidate parameter set is scored
  by the *average fitness of its ±1-step neighbors*, not its own peak — a
  sharp isolated peak in parameter space loses to a stable plateau.
- **IS-vs-OOS degradation** reported prominently (annualized OOS rate vs IS
  rate, i.e. 1 − WFE, per validation and per WFO window).

## 7. Exported EA quality

- EA Studio exports self-contained MQL4/MQL5 source with an embedded
  indicator engine, broker-digit auto-detection, spread/slippage protections,
  optional "long only/short only" mode, and settable magic number. Exports
  are deliberately dependency-free (no DLLs, no external includes).
- Their EAs are known to pass the MQL5 Market validator; that quality bar —
  normalized volumes, checked stops, no unchecked buffer access — is the
  baseline for our renderer (see `mql5_validator_checklist.md`).

## 8. What "better than EA Studio" means here

1. Event-driven simulator that models **path-dependent mechanics EA Studio
   does not generate at all** (DCA/grid baskets, hedge layers, partial
   closes) — with those mechanics' parameters optimizable.
2. Real **walk-forward** (anchored + rolling) with WFE gating, not just a
   single OOS split.
3. Monte Carlo including **equity-curve resampling** (trade-order shuffle)
   on top of EA Studio's spread/slippage/parameter variations.
4. Neighborhood-stability optimization (EA Studio optimizes to the peak).
5. Full pipeline scripted + versionable (SQLite, pytest, headless MT5 lane),
   not a browser session.
6. Honest reporting: every gate produces machine-readable pass/fail reasons;
   no strategy is presented without its failure modes.

None of this guarantees profitability — the same disclaimer EA Studio's own
documentation makes. The goal is maximizing robustness and eliminating
mechanical failure classes.
