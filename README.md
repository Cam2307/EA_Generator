# MQL5 EA Factory & Curation Dashboard

A Streamlit-based factory for **generating, backtesting, validating, curating,
and exporting MetaTrader 5 Expert Advisors**. Strategies are assembled from a
Logic Matrix of entry filters and execution mechanics, validated with an
IS/OOS + walk-forward pipeline, and exported as Marketplace Packages
(`.mq5` + `.set` + `.md`).

## Quick start

```bat
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
run_dashboard.bat
```

or manually: `.venv\Scripts\streamlit run app\dashboard.py`.

## Discovery runs

The **Discovery** tab is a single panel: live status, configuration, and controls
all live in one bordered container. Every sweep goes through the same pipeline
(`build_discovery_payload` → `JobQueue.submit_discovery` → `_run_discovery`), so
engine, mechanics, account economics, validation level, walk-forward windows,
and advanced generation options are identical across runs.

| Control | What it does |
|---------|----------------|
| **Start discovery** | Saves settings and starts the orchestrator. It cycles through every selected **symbol × timeframe** pair continuously until you stop. Sends an hourly progress digest and one-time alerts for exceptional strategies (SMTP via `EA_SMTP_*` env vars). |
| **Stop discovery** | Appears while a run is active. Stops the orchestrator after the current sweep; use **Stop** on the in-flight job card to cancel the active backtest immediately. |

### How to use

1. Configure **symbols**, **timeframes**, **validation**, **engine**, **mechanics**,
   and the rest of the options in the discovery panel.
2. Press **Start discovery** — the orchestrator cycles through every
   symbol×timeframe pair with identical settings and sends hourly progress emails.
3. Press **Stop discovery** when you want the cycle to end after the current sweep.

Sweep scope is the cartesian product of the selected symbols and timeframes
(e.g. 3 symbols × 2 timeframes = 6 sweeps per cycle). Only one discovery job
runs at a time; the orchestrator queues the next combination when the current
job completes.

## Architecture

```
config/settings.py          MT5 paths (auto-detected by default), validation gates, dirs
factory/models.py           Pydantic models (StrategyDefinition, BacktestMetrics, ...)
factory/storage.py          SQLite (WAL + busy_timeout, per-thread context-managed connections)
factory/generator.py        Logic Matrix: random sampling + genetic crossover/mutation
factory/backtest/base.py    BacktestEngine ABC
factory/backtest/mt5_runner.py   Headless MT5: compile via metaeditor64, tester .ini, XML reports
factory/backtest/simulator.py    Event-driven bar-by-bar fallback engine (stateful PositionBook)
factory/backtest/costs.py        Session-aware dynamic spread/slippage model
factory/backtest/validation.py   70/30 IS/OOS, WFO, acceptance gates, MC gate
factory/backtest/statistics.py   Deflated Sharpe Ratio + selection-bias stats
factory/backtest/montecarlo.py   Monte Carlo robustness + price-path block bootstrap
factory/backtest/reconcile.py    Simulator vs MT5 bias quantification
factory/regime.py            Market-regime classification + per-regime breakdown
factory/pareto.py            NSGA-II multi-objective primitives for the GA
factory/correlation.py       Return-stream correlation (duplicate-edge curation)
factory/manifest.py          Reproducibility manifests (seed, data hash, versions)
factory/reoptimize.py        Online re-optimization of promoted strategies
factory/portfolio.py         HRP portfolio weights + combined-portfolio metrics
factory/holdout.py           Untouched holdout: reserved window, one-shot scoring
factory/publication.py       Publication-tier gates + publish records
factory/metalabel.py         Meta-labeling diagnostics (logistic, chrono split)
app/components/theme.py      Shared design system (hero, KPI strip, chips)
app/components/portfolio_panel.py  Gallery "Portfolio" view (HRP, heatmap)
factory/discovery_config.py   Shared discovery settings + job payload builder
factory/agent_alerts.py       Hourly progress digests + quality alerts for the agent
docs/ea_studio_reference.md      EA Studio benchmark notes
docs/mql5_validator_checklist.md MQL5 Market validator checklist
factory/mql5/renderer.py    Validator-proof .mq5 assembly from templates/
factory/assets/             .set writer, marketplace .md writer, package exporter
jobs/worker.py              Singleton JobQueue, single-slot MT5 lane, SQLite-persisted progress
jobs/mt5_pool.py            Multi-instance MT5 pool (portable installs, parallel testers)
jobs/orchestrator.py        Detached discovery agent (batch + continuous sweep modes)
jobs/sweep.py               Symbol × timeframe sweep planner
scripts/discovery_agent_service.py  Orchestrator entry point (spawned by the dashboard)
app/dashboard.py            Streamlit UI (discovery, gallery, export)
app/components/discovery_panel.py   Unified discovery form, live progress, agent controls
data/ reports/ output/      Runtime artifacts (git-ignored)
```

## The two backtest engines

- **Simulator (pre-filter).** An *event-driven*, bar-by-bar loop over a
  stateful `PositionBook` that models sequential DCA/grid fills, hedge
  layers, partial closes, floating drawdown, margin usage, and spread +
  slippage per fill. Vectorization is used only to precompute indicator and
  signal arrays. Execution realism (see `config/settings.py`):
  - **Session-aware dynamic costs** (`SIMULATOR_DYNAMIC_COSTS`): spread
    widens by UTC hour/weekday (rollover spike, Asian session, weekend
    gaps) and slippage scales with realized volatility; the configured
    spread/slippage are the *typical London-session* base costs. Entries
    are skipped when the session-widened spread exceeds the strategy's
    `max_spread_points`.
  - **Intrabar exit resolution** (`SIMULATOR_INTRABAR_MODE`): same-bar
    SL/TP ambiguity is resolved along the bar's OHLC path (bullish bars
    walk open→low→high→close) or, in `"m1"` mode, by replaying real M1
    bars inside each strategy bar (auto-falls back to the path heuristic
    when M1 data is unavailable or synthetic). Gap-throughs still exit.

  **It is a pre-filter, not the truth**: every surviving
  strategy must still pass a real MT5 Strategy Tester run before you ship it.
- **MT5 runner (source of truth).** Auto-detects the terminal via the
  `MetaTrader5` package, compiles rendered EAs with `metaeditor64.exe`,
  writes a tester `.ini` with an explicit static `Report=` path under
  `reports/`, runs `terminal64.exe /config:...` headlessly with a timeout,
  and parses the XML report. Runs are **strictly sequential** (single-slot
  lane) to avoid data-directory corruption.
  **The MT5 terminal must be closed** while headless runs execute — a
  running interactive terminal owns the data directory, and a second
  instance exits silently without running the tester. The runner detects
  this and reports it as a clear job error.
- **MT5 pool (optional, parallel).** Configure `MT5_INSTANCE_PATHS` in
  `config/settings.py` with the `terminal64.exe` paths of N *portable-mode*
  installs (each owns its own data directory) and MT5 discovery validates
  candidates **in parallel across the pool** — each candidate leases one
  instance exclusively for its whole validation. Provisioning steps are in
  `jobs/mt5_pool.py`. With no instances configured, the legacy single lane
  applies.
- **Reconciliation harness.** `python scripts/reconcile_engines.py` runs the
  same strategies through both engines and reports per-metric deltas plus
  the simulator's aggregate optimism/pessimism bias — the number that tells
  you how much to trust (and how to calibrate) the pre-filter. Requires MT5.
- If neither MT5 nor cached data is available, the simulator falls back to a
  deterministic **synthetic random-walk series** so the pipeline can be
  developed and demoed offline (clearly not market data — treat results as
  plumbing checks only).

## How strategies are generated, optimized, and scored

Each strategy is a recipe built from three layers:

### 1. How strategies are generated

**Entry rules (when to trade).** The factory picks 1–2 indicators from the
Logic Matrix (RSI, MA cross, breakout, Bollinger, etc.) and random values for
their settings — e.g. RSI period 14, oversold level 25.

**Exit style (how trades are managed).** One of four execution mechanics:

| Mechanic | What it does |
|----------|--------------|
| **Standard SL/TP** | Fixed stop and take-profit in points |
| **Partial close** | Take partial profit, move stop to breakeven |
| **DCA / Grid** | Add positions as price moves against you |
| **Hedge layer** | Open opposite position when underwater |

**Trade-management overlay (optional).** On top of the mechanic, each strategy
may also get adaptive or fixed stop loss, risk-reward take profit, trailing
stop (fixed / ATR / chandelier), breakeven triggers, session filters, daily
loss limits, and related controls — plus an **adaptive regime filter**: the
strategy classifies every bar into quiet/volatile × range/trend (ADX +
ATR-vs-baseline proxies) and only enters in the regimes enabled by an
optimizable bitmask. The mask *and* the classification thresholds are tuned
by the optimizer, exported in the `.set`, and the generated EA computes the
identical classification on-chart (`TM_RegimeAllowed()` — same formula as
`factory/regime.py::classify_regimes_filter`), so a validated regime edge
survives export unchanged. A companion **regime sizing** feature scales the
entry lot size by an optimizable per-regime multiplier
(`Inp_X_regime_size_*`), applied identically in the simulator and the EA.

New strategies are built randomly, then the genetic loop keeps promising
candidates and breeds better ones — mutating parameter values and combining
entry filters from two parents across generations. Parent selection is
**multi-objective (NSGA-II)** by default (`PARETO_EVOLUTION`): candidates are
ranked by Pareto dominance over *(net profit, max drawdown, equity R²,
trade count)* with crowding-distance diversity, so discovery explores the
whole profit/risk/stability frontier instead of collapsing onto one scalar
compromise. Set `PARETO_EVOLUTION = False` for the legacy scalar tournament.

### 2. What gets optimized (including SL distance)

**Yes — stop and exit distances are both generated randomly and optimized.**

For a standard SL/TP strategy, typical tunable parameters include:

| Parameter | Typical range | Meaning |
|-----------|---------------|---------|
| `sl_points` | 100–600 pts | Fixed stop distance |
| `tp_points` | 100–900 pts | Fixed take-profit distance |
| `atr_sl_mult` | 1.0–4.0 | Adaptive SL: stop = ATR × multiplier |
| `trail_distance_points` | 100–600 | Fixed trailing distance |
| `trail_atr_mult` | 1.5–4.0 | ATR / chandelier trailing |
| `be_trigger_points` | 100–500 | When to move stop to breakeven |
| `tp_rr` | 1.0–4.0 | Take-profit as multiple of stop (R:R) |

Entry-filter settings are tuned too (e.g. RSI period, MA fast/slow, lookback).

**In-sample optimization** runs a random search over all of these ranges on
the first ~70% of history. The winning combination is saved as `best_params`
and exported in the `.set` file. If a strategy shows SL at 250 pts, that value
was chosen by the optimizer from the allowed range for that run — not hard-coded.

Entry-filter *and* execution-mechanic parameters (grid step spacing, grid
level count, lot multiplier, hedge trigger distance in points, hedge ratio,
partial-close level and fraction, SL/TP, trade-management `X_*` params) all
carry `min/max/step` ranges:

- the IS optimizer and walk-forward windows sweep them,
- the genetic loop mutates and crosses them over,
- rendered EAs expose them as `input` variables (`Inp_M_grid_step_points`,
  `Inp_M_hedge_trigger_points`, ...),
- tester `.ini` files and exported `.set` files carry them in the
  `Value||Start||Step||Stop||Y` optimizable format.

### 3. How optimization scores strategies

Both the genetic search and the in-sample optimizer share the same fitness
objective — biased toward **stable, steadily rising equity curves**:

```
fitness = (net profit / (1 + drawdown%)) × smoothness
```

**Smoothness** is the equity-curve **R²** (how straight and steadily rising the
curve is). A choppy but profitable curve scores lower than a smooth riser with
similar profit. Neighborhood-stability scoring (see Validation gates below)
further prefers parameter *plateaus* over isolated peaks when re-ranking top
candidates.

## Validation gates

Chronological 70/30 IS/OOS split; random-search IS optimization with
**neighborhood-stability scoring** (plateau beats isolated peak); anchored and
rolling walk-forward windows. Pass/fail is driven by user-configurable
**acceptance criteria** (WFE, OOS drawdown, min trades, profit factor,
Sharpe, equity R-squared, max consecutive losses). Survivors are then
stress-tested by a **Monte Carlo** module (randomized spread/slippage,
parameter perturbation, entry jitter, trade-order resampling, and
**price-path block bootstrap** — the strategy is re-run on counterfactual
histories rebuilt from resampled blocks of real bar returns) and must pass
its robustness gates (default: ≥ 80% of MC runs profitable, 95%-worst-case DD
within limit, ≥ 60% of bootstrap paths profitable). IS→OOS **degradation %**
and optimizer **stability ratio** are reported on every strategy card.

Additional honesty statistics on every validation:

- **Deflated Sharpe Ratio (DSR)** — the probability the OOS Sharpe beats the
  *expected best-of-N zero-skill* Sharpe, where N is how many candidates the
  run had tried. Near 1.0 = likely a real edge; ≤ ~0.5 = plausibly pure
  selection luck. Shown on strategy cards and available as a sort order.
- **Per-regime breakdown** — every OOS trade is attributed to one of four
  market regimes (quiet/volatile × range/trend, from ADX + ATR-percentile
  proxies) so "profitable overall" decomposes into per-regime PF/net/win
  rate. The optional `max_regime_loss_pct` acceptance gate rejects
  strategies whose worst regime loses more than a set % of the deposit.
- **Return-stream correlation** — promotion scoring penalizes candidates
  whose daily OOS returns are > 0.6 correlated with an already-promoted
  strategy (the same edge under a different name).
- **Run manifests** — every discovery run persists its concrete seed, full
  payload, a SHA-256 fingerprint of the exact bar data, the realism settings
  in force, and library versions, so any gallery strategy is re-derivable.

## Untouched holdout & publication tier

- **Untouched holdout** (`HOLDOUT_MONTHS`, default 12): the most recent
  months of history are reserved — the discovery worker clamps every run's
  end date to the boundary, so no candidate is ever generated or optimized
  on that window. Each strategy may be scored on the holdout exactly
  **once** (re-runs require an explicit `force`); the aggregate hit rate of
  those one-shot evaluations is the factory's master KPI, shown on the
  Export page.
- **Publication tier** (`factory/publication.py`, Export page checklist):
  a far higher bar than the discovery gates — ≥200 OOS trades, DSR ≥ 0.95,
  WFE ≥ 0.70, MC robustness ≥ 85, edge positive in ≥2 regimes, return
  stream <0.5 correlated with anything already published, real
  (non-synthetic) data, and a passed holdout. `publish()` exports the
  package and records the decision; forced publications are marked.
- **Risk-style labels**: martingale DCA grids (lot multiplier > 1.0), flat
  DCA grids, and hedge-recovery strategies carry a warning badge on every
  result card and a disclosure warning in the publication checklist —
  flagged, never silently blocked.

## Search-space expansion

- **Composite signal logic** (`StrategyDefinition.signal_logic`): entry
  filters can combine as ALL (classic AND), ANY (disjunctive), or MAJORITY
  (vote). Sampled and mutated by the genetic search, simulated exactly, and
  rendered as a hits-counting `SignalLong()/SignalShort()` in the EA.
- **Behavioral novelty search** (`NOVELTY_ENABLED`): each candidate's
  daily-return fingerprint is compared against a reservoir of recent
  candidates; `1 - max|corr|` joins the NSGA-II objectives so discovery
  explores new behaviors instead of rediscovering one edge.
- **Meta-labeling diagnostics** (`factory/metalabel.py`): a chronologically
  split logistic model tests whether a strategy's winners are predictable
  from regime/session/direction context (test AUC + expectancy uplift) —
  the honest precursor to premium filtered variants.

## Maintenance & portfolio

- **Online re-optimization** — `python scripts/reoptimize_promoted.py`
  (schedule it weekly/monthly) re-runs the IS optimizer for every promoted
  strategy on the trailing window. When the fitness plateau has genuinely
  shifted (params changed AND the incumbent trails the fresh optimum by
  >10%), it writes an updated `.set` into `output/reoptimized/` and flags
  the strategy — the factory maintains its fleet instead of only growing it.
- **HRP portfolio** — the Gallery's **Portfolio** view computes Hierarchical
  Risk Parity weights over the selected strategies' OOS daily-return streams
  (with gap-aware bisection so near-duplicate edges share one risk bucket),
  plus a correlation heatmap, combined equity curve, and diversification
  ratio. HRP balances risk; it cannot create edge.

See `config/settings.py` and `docs/ea_studio_reference.md`.

## Validator-proof MQL5 output

Every exported `.mq5` is assembled from hardened templates and checked against
`docs/mql5_validator_checklist.md`. All generated EAs are **single-symbol**
(they trade only the chart symbol `_Symbol`). Highlights:

- `#property copyright`, `#property link`, `#property version "1.00"`,
  `#property description` (Market-required metadata)
- `CheckVolumeValue()` and `CheckMoneyForTrade()` (official Market validation
  patterns) called before every order via `OrderPreflight()`
- `SYMBOL_TRADE_MODE_FULL` + session checks; `TradingAllowed()` respects
  `TERMINAL_TRADE_ALLOWED` / `MQL_TRADE_ALLOWED` without failing `OnInit`
- `AdjustStops()` / `FreezeOK()` for `SYMBOL_TRADE_STOPS_LEVEL` and freeze level
- Checked `CopyBuffer`/`CopyRates`, `SafeDiv`, bounded transient retcode retries
- Lazy indicator handle creation (`OnInit` always succeeds on any symbol)
- Netting vs hedging account branching for DCA/grid and hedge mechanics
- Tester-only fallback trade so the validator always sees at least one operation
- Only `<Trade\Trade.mqh>` — no DLLs, no file/network I/O

Compile verification: `python scripts/compile_verify.py` (requires MetaEditor).

## Tests & smoke run

```bat
.venv\Scripts\python -m pytest tests -q          & rem unit tests
.venv\Scripts\python scripts\smoke_run.py        & rem end-to-end (simulator)
.venv\Scripts\python scripts\compile_verify.py   & rem DCA + hedge compile check
.venv\Scripts\python scripts\mt5_verify.py       & rem render -> compile -> real tester
```

The unit tests cover DCA/grid accounting, WFE and acceptance gates, Monte Carlo
math, renderer validator snippets (`CheckMoneyForTrade`, `#property version`,
…), and `.ini`/`.set` writers.

## Scope & honesty

- **Single-symbol only.** Every EA trades the chart symbol (`_Symbol`) with no
  multi-market or portfolio export. Correlation analysis and portfolio
  combination are deliberately out of scope (see `docs/ea_studio_reference.md`).
- **Robustness ≠ profitability.** Passing acceptance criteria, walk-forward,
  and Monte Carlo gates means a strategy is *less likely to be curve-fit* and
  has survived several stress tests — it does **not** guarantee future profits.
  The simulator is a pre-filter; real MT5 Strategy Tester runs are the source of
  truth before you ship anything live.
- **Market validation is mechanical, not economic.** Exported EAs are hardened
  against MQL5.com automatic validator failure classes (volume, margin, stops,
  netting/hedging, no-trade fallback, …). Passing compile and validator checks
  does not mean the strategy edge is real.
