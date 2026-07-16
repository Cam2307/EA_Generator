"""Central configuration for the EA Factory.

Paths are resolved relative to the project root so the app can be launched
from any working directory. MT5 paths default to filesystem auto-detection
(AppData origin.txt / common install dirs); set them explicitly here to override.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = PROJECT_ROOT / "results"
DB_PATH = DATA_DIR / "factory.db"

# ---------------------------------------------------------------------------
# MetaTrader 5 integration
# ---------------------------------------------------------------------------
# None -> auto-detect via AppData origin.txt / common install paths
# (never via MetaTrader5.initialize() — that launches an interactive terminal).
MT5_TERMINAL_PATH: Optional[str] = None
MT5_METAEDITOR_PATH: Optional[str] = None
MT5_RUN_TIMEOUT_SECONDS = 1800       # single headless backtest hard timeout
MT5_COMPILE_TIMEOUT_SECONDS = 180    # metaeditor64 compile hard timeout

# Multi-instance MT5 pool (jobs/mt5_pool.py): terminal64.exe paths of
# PORTABLE-MODE installs, one per concurrent tester lane. Each install owns
# its own data directory (launched with /portable), so N instances can run
# testers in parallel. Empty = legacy single shared install, one run at a
# time, interactive terminal must be closed. See jobs/mt5_pool.py for the
# provisioning steps.
MT5_INSTANCE_PATHS: tuple = ()
# e.g. (r"C:\\MT5-farm\\a\\terminal64.exe", r"C:\\MT5-farm\\b\\terminal64.exe")

# ---------------------------------------------------------------------------
# Validation gates (defaults for the AcceptanceCriteria model — every gate is
# user-configurable per discovery batch from the dashboard)
# ---------------------------------------------------------------------------
IS_OOS_SPLIT = 0.70          # chronological in-sample fraction
WFE_THRESHOLD = 0.55         # pass gate: walk-forward efficiency
OOS_MAX_DD_PCT = 15.0        # pass gate: max OOS drawdown percent
MIN_OOS_TRADES = 5           # sanity gate: refuse statistically empty OOS runs
MIN_PROFIT_FACTOR = 0.0      # 0 disables the gate
MIN_SHARPE = 0.0             # 0 disables the gate
MIN_R_SQUARED = 0.0          # 0 disables the gate (equity-curve stability)
MAX_CONSECUTIVE_LOSSES = 0   # 0 disables the gate
DAYS_PER_MONTH = 30.4375     # shared month length for history + WFO folds
WFO_WINDOWS = 2              # default walk-forward windows (auto-derived from duration)
WFO_TRAIN_MONTHS = 2         # default rolling WFO IS window (auto-derived from duration)
WFO_TEST_MONTHS = 1          # default rolling WFO OOS window (auto-derived from duration)
WFO_MODES = ("rolling",)     # "anchored" + "rolling" doubles WFO cost
OPT_SAMPLES = 48             # Optuna trials for main IS optimization (percent/ATR settle)
WFO_OPT_SAMPLES = 16         # lighter Optuna budget inside each WFO window
OPTUNA_SAMPLER = "auto"      # "tpe" | "cmaes" | "auto" (CMA-ES when continuous-heavy)
OPTUNA_N_STARTUP_TRIALS = 3  # random trials before TPE/CMA-ES kicks in
OPTUNA_PRUNE_SEGMENTS = 4    # chronological IS chunks for MedianPruner
OPTUNA_TRACE_TOP_N = 10      # top trials persisted for explainability

# Over-fitting protection in the optimizer: score a candidate parameter set
# by the average fitness of its +/-1-step neighbors instead of its own peak.
# On by default so Robust+ stability honesty gates see real ratios (not 1.0).
NEIGHBORHOOD_STABILITY = True
NEIGHBOR_SAMPLES = 4         # neighbors evaluated per top candidate
NEIGHBOR_TOP_K = 3           # top raw candidates re-scored by neighborhood

# ---------------------------------------------------------------------------
# Monte Carlo robustness (simulator-based; see factory/backtest/montecarlo.py)
# ---------------------------------------------------------------------------
MC_ENABLED = True
MC_RUNS = 20                 # randomized re-runs (EA Studio default: 20)
MC_SPREAD_MAX_POINTS = 30.0  # spread randomized between base and this max
MC_SLIPPAGE_MAX_POINTS = 10.0
MC_PARAM_CHANGE_PROB = 0.2   # chance each parameter is perturbed per run
MC_PARAM_MAX_STEPS = 2       # max ParamRange steps a parameter may move
MC_SKIP_ENTRY_PROB = 0.05    # chance an entry signal is randomly skipped
MC_START_JITTER_BARS = 50    # random warm-up offset (first-bar jitter)
MC_RESAMPLES = 200           # trade-order resampling draws (equity bootstrap)
MC_MIN_PROFITABLE = 0.80     # gate: fraction of MC runs that must be profitable
MC_MAX_DD_P95 = 25.0         # gate: 95th-percentile max drawdown limit (%)
# Price-path block bootstrap: re-run the strategy on counterfactual histories
# rebuilt from resampled blocks of real bar returns (autocorrelation and vol
# clustering preserved inside blocks, the realized path destroyed). A milder
# gate than the perturbation battery because genuine long-horizon trends are
# legitimately absent from bootstrapped paths. 0 runs disables.
MC_PATH_RUNS = 10
MC_PATH_BLOCK_BARS = 96      # block length in bars (~1 day of M15)
MC_MIN_PATH_PROFITABLE = 0.60

# ---------------------------------------------------------------------------
# Simulator execution realism
# ---------------------------------------------------------------------------
# Session-aware dynamic costs: spread widens by hour-of-day/weekday (rollover,
# Asian session, weekend gaps) and slippage scales with realized volatility.
# The user's configured spread/slippage remain the typical (London) base cost.
# Applies to specs inferred from data; explicitly constructed SymbolSpec
# objects (tests) keep flat static costs unless dynamic_costs=True is set.
SIMULATOR_DYNAMIC_COSTS = True

# Intrabar SL/TP ambiguity resolution ("which was hit first?"):
#   "conservative" — legacy: SL always assumed first (pessimistic)
#   "path"         — OHLC path heuristic (bullish bar: open->low->high->close)
#   "m1"           — replay real M1 bars inside each strategy bar; falls back
#                    to "path" when M1 data is unavailable or synthetic
# "path" keeps discovery throughput high (Numba-eligible) while still
# resolving intrabar SL/TP better than conservative. Use "m1" when you want
# tick-path realism on a short survivor-confirm pass.
SIMULATOR_INTRABAR_MODE = "path"

# Stage-1 quick_screen intrabar mode. "path" keeps Numba eligible for the
# high-volume triage pass; full validation still uses SIMULATOR_INTRABAR_MODE.
SIMULATOR_SCREEN_INTRABAR_MODE = "path"

# After Stage-1 path triage, re-screen promising candidates with Stage-2
# realism (SIMULATOR_INTRABAR_MODE) before Optuna/WFO to cut false promotes.
SIMULATOR_SCREEN_CONFIRM = True

# When the user picks MT5 but an interactive terminal owns the data directory,
# fall back to the simulator for Stage-2 instead of aborting the whole sweep
# (which previously burned promising screens as empty INFRA fails).
DISCOVERY_MT5_FALLBACK_TO_SIMULATOR = True

# After a simulator survivor clears MT5_CONFIRM_MIN_LEVEL, re-run a cheap
# MT5 confirmation pass and record ``mt5_confirmed`` on the report so
# promotion can require real-tester agreement.
DISCOVERY_MT5_CONFIRM_SURVIVORS = True
MT5_CONFIRM_MIN_LEVEL = 7

# Stage-1 reject: stop distance must be >= this multiple of recent ATR
# (and >= SCREEN_MIN_STOP_COST_MULT × round-trip spread+slip).
SCREEN_MIN_STOP_ATR_MULT = 0.25
SCREEN_MIN_STOP_COST_MULT = 2.0

# Inject mutants of prior L4+ survivors (same symbol/TF, else same class) into
# each generation.
DISCOVERY_ELITE_SEED_COUNT = 8
# After a signal edge clears the validation floor, how many execution/mechanic
# EA variants to enqueue for screening (edge-first mode).
DISCOVERY_MAX_EDGE_VARIANTS = 8

# Numba JIT of the Standard SL/TP + Partial-close bar loop (see
# factory/backtest/sim_numba_core.py). Falls back to the Python PositionBook
# path for DCA/grid/hedge, M1 intrabar, ATR overlays, or when numba is absent.
SIMULATOR_NUMBA = True

# ---------------------------------------------------------------------------
# Untouched holdout (factory/holdout.py)
# ---------------------------------------------------------------------------
# The most recent HOLDOUT_MONTHS of history are reserved: discovery end dates
# are clamped to the boundary, and each strategy may be scored on the holdout
# exactly once. The aggregate hit rate of those one-shot evaluations is the
# factory's master KPI.
HOLDOUT_ENABLED = True
HOLDOUT_MONTHS = 12
HOLDOUT_MAX_DD_PCT = 25.0    # holdout pass also requires DD under this

# Promotion to edge_positive / promoted_live_watchlist requires a one-shot
# holdout pass. Validated (lower) states remain available without it.
PROMOTION_REQUIRE_HOLDOUT = True
# Highest promotion state also requires MT5 confirmation when available.
PROMOTION_REQUIRE_MT5_FOR_WATCHLIST = True

# ---------------------------------------------------------------------------
# Publication tier (factory/publication.py) — a far higher bar than the
# discovery gates. These decide what carries your marketplace reputation.
# ---------------------------------------------------------------------------
PUB_MIN_OOS_TRADES = 200
PUB_MIN_DSR = 0.95
PUB_MIN_WFE = 0.70
PUB_MIN_MC_SCORE = 85.0
PUB_MAX_CORR = 0.5           # vs anything already published
PUB_MIN_POSITIVE_REGIMES = 2
PUB_ALLOWED_DATA_SOURCES = ("mt5", "cache")   # synthetic can never publish
PUB_REQUIRE_HOLDOUT = True

# ---------------------------------------------------------------------------
# Genetic search
# ---------------------------------------------------------------------------
# NSGA-II multi-objective evolution: parents are selected by Pareto rank +
# crowding over (net profit, -max DD, equity R^2, trade count) instead of the
# single scalar fitness, so discovery explores the whole profit/risk/
# stability frontier. False falls back to scalar tournament selection.
PARETO_EVOLUTION = True

# Behavioral novelty search: append 1 - max|corr| against a reservoir of
# recent candidates' daily-return fingerprints as an extra NSGA-II objective,
# so discovery explores new behaviors instead of rediscovering one edge.
NOVELTY_ENABLED = True
NOVELTY_RESERVOIR = 200      # fingerprints kept for the novelty comparison

# ---------------------------------------------------------------------------
# Engine / account defaults
# ---------------------------------------------------------------------------
DEFAULT_ENGINE = "simulator"     # "simulator" | "mt5"
DEFAULT_DEPOSIT = 10_000.0
DEFAULT_LEVERAGE = 100
DEFAULT_SYMBOL = "EURUSD"
DEFAULT_TIMEFRAME = "M15"

# Symbols offered in the discovery dropdown. Exact tradable names vary by
# broker; the simulator falls back to synthetic data for any symbol, and MT5
# uses whatever the connected terminal exposes. Extend freely.
SYMBOLS = [
    # FX majors
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    # FX crosses
    "EURGBP", "EURJPY", "GBPJPY", "EURCHF", "EURAUD", "AUDJPY", "GBPCHF",
    "CADJPY", "NZDJPY", "AUDNZD", "AUDCAD", "GBPAUD",
    # Metals
    "XAUUSD", "XAGUSD",
    # Indices
    "US30", "US500", "USTEC", "GER40", "UK100", "JP225",
    # Energy
    "USOIL", "UKOIL",
    # Crypto
    "BTCUSD", "ETHUSD",
]

# Discovery-agent defaults (persisted overrides live in SQLite app_settings).
# Leave empty — set via dashboard or EA_ALERT_RECIPIENT env; never hardcode.
DEFAULT_ALERT_RECIPIENT = ""
DEFAULT_ALERT_MIN_SCORE = 80.0
DEFAULT_ALERT_COOLDOWN_MINUTES = 60
DEFAULT_PROGRESS_EMAIL_HOURS = 1.0


def ensure_dirs() -> None:
    for d in (DATA_DIR, REPORTS_DIR, OUTPUT_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
