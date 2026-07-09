"""Central configuration for the EA Factory.

Paths are resolved relative to the project root so the app can be launched
from any working directory. MT5 paths default to auto-detection via the
MetaTrader5 python package; set them explicitly here to override.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "output"
DB_PATH = DATA_DIR / "factory.db"

# ---------------------------------------------------------------------------
# MetaTrader 5 integration
# ---------------------------------------------------------------------------
# None -> auto-detect via MetaTrader5.initialize() / terminal_info().path
MT5_TERMINAL_PATH: Optional[str] = None
MT5_METAEDITOR_PATH: Optional[str] = None
MT5_RUN_TIMEOUT_SECONDS = 1800       # single headless backtest hard timeout
MT5_COMPILE_TIMEOUT_SECONDS = 180    # metaeditor64 compile hard timeout

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
WFO_WINDOWS = 3              # walk-forward windows per mode (anchored + rolling)
WFO_TRAIN_MONTHS = 2         # rolling WFO in-sample window (calendar months)
WFO_TEST_MONTHS = 1          # rolling WFO out-of-sample window (calendar months)
OPT_SAMPLES = 8              # random parameter samples for IS optimization
WFO_OPT_SAMPLES = 3          # lighter sampling inside each WFO window

# Over-fitting protection in the optimizer: score a candidate parameter set
# by the average fitness of its +/-1-step neighbors instead of its own peak.
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


def ensure_dirs() -> None:
    for d in (DATA_DIR, REPORTS_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
