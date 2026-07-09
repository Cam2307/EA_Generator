"""Pydantic data models shared across the factory."""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Strategy building blocks
# ---------------------------------------------------------------------------

class EntryFilterType(str, Enum):
    PRICE_ACTION_BREAKOUT = "PRICE_ACTION_BREAKOUT"
    MTF_VOLATILITY = "MTF_VOLATILITY"
    LIQUIDITY_ZONE = "LIQUIDITY_ZONE"
    RSI_REVERSION = "RSI_REVERSION"
    MA_CROSS = "MA_CROSS"
    BOLLINGER_FADE = "BOLLINGER_FADE"
    MACD_CROSS = "MACD_CROSS"
    STOCHASTIC = "STOCHASTIC"
    ADX_TREND = "ADX_TREND"
    CCI_REVERSION = "CCI_REVERSION"
    MOMENTUM = "MOMENTUM"
    WILLIAMS_R = "WILLIAMS_R"
    VOLUME_SURGE = "VOLUME_SURGE"
    PARABOLIC_SAR = "PARABOLIC_SAR"
    ICHIMOKU = "ICHIMOKU"
    DEMARKER = "DEMARKER"
    AWESOME = "AWESOME"
    FORCE_INDEX = "FORCE_INDEX"
    STDDEV_REGIME = "STDDEV_REGIME"
    ENVELOPES = "ENVELOPES"
    MFI = "MFI"
    RVI = "RVI"
    DEMA_CROSS = "DEMA_CROSS"


class ExecutionMechanicType(str, Enum):
    STANDARD_SLTP = "STANDARD_SLTP"
    DCA_GRID = "DCA_GRID"
    HEDGE_LAYER = "HEDGE_LAYER"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"


# ---------------------------------------------------------------------------
# Trade-management (exit / risk) option modes
# ---------------------------------------------------------------------------

class StopLossMode(str, Enum):
    OFF = "OFF"          # no protective stop
    FIXED = "FIXED"      # fixed distance in points (mechanic sl_points)
    ATR = "ATR"          # adaptive: distance = ATR(period) * multiplier


class TakeProfitMode(str, Enum):
    OFF = "OFF"          # no fixed target (rely on trailing / mechanic exits)
    FIXED = "FIXED"      # fixed distance in points (mechanic tp_points)
    RR = "RR"            # target = risk_reward * initial stop distance


class TrailMode(str, Enum):
    OFF = "OFF"
    FIXED = "FIXED"          # trail at a fixed point distance behind price
    ATR = "ATR"             # trail at ATR(period) * multiplier behind price
    CHANDELIER = "CHANDELIER"  # highest-high/lowest-low over lookback -/+ ATR*mult


class LotMode(str, Enum):
    FIXED = "FIXED"              # fixed lots (risk.fixed_lots)
    RISK_PERCENT = "RISK_PERCENT"  # size lots from equity risk % over the stop


class ParamRange(BaseModel):
    """Numeric parameter with an optimization range."""
    min: float
    max: float
    step: float

    def clamp(self, value: float) -> float:
        v = max(self.min, min(self.max, value))
        # snap to grid
        steps = round((v - self.min) / self.step) if self.step > 0 else 0
        return self.min + steps * self.step


class EntryFilter(BaseModel):
    type: EntryFilterType
    params: Dict[str, float] = Field(default_factory=dict)
    ranges: Dict[str, ParamRange] = Field(default_factory=dict)


class ExecutionMechanic(BaseModel):
    type: ExecutionMechanicType
    params: Dict[str, float] = Field(default_factory=dict)
    ranges: Dict[str, ParamRange] = Field(default_factory=dict)


class RiskBlock(BaseModel):
    fixed_lots: float = 0.10
    risk_percent: float = 1.0          # informational; lot sizing uses fixed lots
    max_spread_points: float = 30.0
    max_open_lots: float = 5.0


class TradeManagement(BaseModel):
    """Optional exit / risk overlay applied on top of the execution mechanic.

    Modes/flags are structural choices fixed at generation (like which
    indicator a filter uses); the numeric sub-parameters in ``params`` carry
    ``ranges`` and are tuned by the genetic search + walk-forward optimizer and
    exported into the .set file. Only the sub-parameters for the *enabled*
    features are populated, so a simple strategy stays cheap to optimize.

    Parameter keys used in ``params`` (all in points unless noted):
      atr_period, atr_sl_mult, tp_rr, trail_start_points, trail_distance_points,
      trail_atr_mult, chandelier_lookback, trail_step_points, be_trigger_points,
      be_offset_points, risk_percent (%), start_hour, end_hour,
      max_trades_per_day, daily_loss_pct (%), cooldown_bars.
    """
    sl_mode: StopLossMode = StopLossMode.FIXED
    tp_mode: TakeProfitMode = TakeProfitMode.FIXED
    trail_mode: TrailMode = TrailMode.OFF
    lot_mode: LotMode = LotMode.FIXED
    breakeven: bool = False
    time_filter: bool = False
    limit_trades_per_day: bool = False
    daily_loss_enabled: bool = False
    cooldown_enabled: bool = False
    params: Dict[str, float] = Field(default_factory=dict)
    ranges: Dict[str, "ParamRange"] = Field(default_factory=dict)

    def uses_atr(self) -> bool:
        return (self.sl_mode == StopLossMode.ATR
                or self.trail_mode in (TrailMode.ATR, TrailMode.CHANDELIER))


class Lineage(BaseModel):
    parents: List[str] = Field(default_factory=list)
    mutations: List[str] = Field(default_factory=list)
    generation: int = 0


class StrategyProfile(BaseModel):
    """Describes how sophisticated a generated strategy is."""
    advanced_mode: bool = False
    complexity_score: int = 0
    complexity_cap: int = 2
    regime_switching: bool = False
    mtf_context: bool = False
    feature_toggles: List[str] = Field(default_factory=list)
    portfolio_signature: str = ""


class StrategyDefinition(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    magic_number: int = 0
    symbol: str = "EURUSD"
    timeframe: str = "M15"
    entry_filters: List[EntryFilter] = Field(default_factory=list)
    mechanic: ExecutionMechanic
    risk: RiskBlock = Field(default_factory=RiskBlock)
    trade_mgmt: TradeManagement = Field(default_factory=TradeManagement)
    rule_description: str = ""
    lineage: Lineage = Field(default_factory=Lineage)
    profile: StrategyProfile = Field(default_factory=StrategyProfile)
    created_at: float = Field(default_factory=time.time)

    def all_params(self) -> Dict[str, float]:
        """Flat parameter dict (prefixed) used for optimization and .set export."""
        out: Dict[str, float] = {}
        for i, f in enumerate(self.entry_filters):
            for k, v in f.params.items():
                out[f"F{i}_{f.type.value}_{k}"] = v
        for k, v in self.mechanic.params.items():
            out[f"M_{self.mechanic.type.value}_{k}"] = v
        for k, v in self.trade_mgmt.params.items():
            out[f"X_{k}"] = v
        return out

    def all_ranges(self) -> Dict[str, ParamRange]:
        out: Dict[str, ParamRange] = {}
        for i, f in enumerate(self.entry_filters):
            for k, r in f.ranges.items():
                out[f"F{i}_{f.type.value}_{k}"] = r
        for k, r in self.mechanic.ranges.items():
            out[f"M_{self.mechanic.type.value}_{k}"] = r
        for k, r in self.trade_mgmt.ranges.items():
            out[f"X_{k}"] = r
        return out

    def apply_flat_params(self, flat: Dict[str, float]) -> "StrategyDefinition":
        """Return a copy with the flat (prefixed) parameter overrides applied."""
        clone = self.model_copy(deep=True)
        for i, f in enumerate(clone.entry_filters):
            prefix = f"F{i}_{f.type.value}_"
            for key, val in flat.items():
                if key.startswith(prefix):
                    f.params[key[len(prefix):]] = val
        mprefix = f"M_{clone.mechanic.type.value}_"
        for key, val in flat.items():
            if key.startswith(mprefix):
                clone.mechanic.params[key[len(mprefix):]] = val
        for key, val in flat.items():
            if key.startswith("X_"):
                clone.trade_mgmt.params[key[2:]] = val
        return clone


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class BacktestMetrics(BaseModel):
    net_profit: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    recovery_factor: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_dd_pct: float = 0.0
    max_dd_money: float = 0.0
    trade_count: int = 0
    r_squared: float = 0.0             # linearity/stability of the equity curve
    max_consecutive_losses: int = 0
    initial_deposit: float = 0.0
    start_ts: float = 0.0              # unix seconds
    end_ts: float = 0.0
    # equity curve as parallel lists (json-friendly): timestamps + equity
    equity_ts: List[float] = Field(default_factory=list)
    equity: List[float] = Field(default_factory=list)

    @property
    def years(self) -> float:
        return max((self.end_ts - self.start_ts) / (365.25 * 86400), 1e-9)

    def annualized_profit_rate(self) -> float:
        """Net profit per unit deposit per year."""
        if self.initial_deposit <= 0:
            return 0.0
        return (self.net_profit / self.initial_deposit) / self.years


class AcceptanceCriteria(BaseModel):
    """User-configurable pass/fail gates applied to the OOS backtest zone.

    A value of 0 (or 0.0) disables the corresponding gate, except
    ``min_wfe`` / ``max_dd_pct`` / ``min_trades`` which are always active.
    Mirrors EA Studio's Acceptance Criteria metric family.
    """
    min_wfe: float = 0.55
    max_dd_pct: float = 15.0
    min_trades: int = 5
    min_net_profit: float = 0.0        # OOS net profit must exceed this
    min_profit_factor: float = 0.0     # 0 disables
    min_sharpe: float = 0.0            # 0 disables
    min_r_squared: float = 0.0         # 0 disables
    max_consecutive_losses: int = 0    # 0 disables

    def evaluate(self, oos: "BacktestMetrics", wfe: float) -> List[str]:
        """Return the list of failed-gate reasons (empty == all gates pass)."""
        reasons: List[str] = []
        if wfe <= self.min_wfe:
            reasons.append(f"WFE {wfe:.3f} <= threshold {self.min_wfe}")
        if oos.max_dd_pct >= self.max_dd_pct:
            reasons.append(
                f"OOS max drawdown {oos.max_dd_pct:.2f}% >= {self.max_dd_pct}%")
        if oos.trade_count < self.min_trades:
            reasons.append(
                f"OOS trade count {oos.trade_count} < {self.min_trades}")
        if oos.net_profit <= self.min_net_profit:
            reasons.append(
                f"OOS net profit {oos.net_profit:.2f} <= {self.min_net_profit:.2f}")
        if self.min_profit_factor > 0 and oos.profit_factor < self.min_profit_factor:
            reasons.append(
                f"OOS profit factor {oos.profit_factor:.2f} < {self.min_profit_factor}")
        if self.min_sharpe > 0 and oos.sharpe < self.min_sharpe:
            reasons.append(f"OOS Sharpe {oos.sharpe:.2f} < {self.min_sharpe}")
        if self.min_r_squared > 0 and oos.r_squared < self.min_r_squared:
            reasons.append(
                f"OOS equity R-squared {oos.r_squared:.3f} < {self.min_r_squared}")
        if (self.max_consecutive_losses > 0
                and oos.max_consecutive_losses > self.max_consecutive_losses):
            reasons.append(
                f"OOS max consecutive losses {oos.max_consecutive_losses}"
                f" > {self.max_consecutive_losses}")
        return reasons


class MonteCarloRun(BaseModel):
    """Summary of a single randomized Monte Carlo re-run."""
    net_profit: float = 0.0
    max_dd_pct: float = 0.0
    profit_factor: float = 0.0
    trade_count: int = 0


class MonteCarloResult(BaseModel):
    """Aggregate of N randomized simulator re-runs + trade-order resampling.

    Robustness maximization only — a high score never guarantees future
    profitability.
    """
    n_runs: int = 0
    runs: List[MonteCarloRun] = Field(default_factory=list)
    pct_profitable: float = 0.0        # 0..1 fraction of profitable MC runs
    profit_p05: float = 0.0            # 5th percentile of final net profit
    profit_p50: float = 0.0
    profit_p95: float = 0.0
    dd_p95: float = 0.0                # 95th percentile (worst-case) max DD %
    resample_dd_p95: float = 0.0       # 95th pct DD from trade-order resampling
    robustness_score: float = 0.0      # 0..100
    # confidence bands over the common bar timeline (thinned)
    band_ts: List[float] = Field(default_factory=list)
    band_p05: List[float] = Field(default_factory=list)
    band_p50: List[float] = Field(default_factory=list)
    band_p95: List[float] = Field(default_factory=list)
    passed: bool = False
    reasons: List[str] = Field(default_factory=list)


class WFOWindowResult(BaseModel):
    mode: str                          # "anchored" | "rolling"
    index: int
    is_start_ts: float
    is_end_ts: float
    oos_start_ts: float
    oos_end_ts: float
    is_metrics: BacktestMetrics
    oos_metrics: BacktestMetrics
    wfe: float = 0.0


class ValidationReport(BaseModel):
    strategy_id: str
    # Discovery run that produced this result (the job id, e.g. "disc_ab12cd34ef").
    # Not stored inside the body — it is populated on load from the validations
    # table's job_id column so the gallery can show which run a result came from.
    run_id: Optional[str] = None
    is_metrics: BacktestMetrics
    oos_metrics: BacktestMetrics
    wfo_windows: List[WFOWindowResult] = Field(default_factory=list)
    wfe: float = 0.0
    passed: bool = False
    reasons: List[str] = Field(default_factory=list)
    best_params: Dict[str, float] = Field(default_factory=dict)
    engine: str = "simulator"
    is_range: Tuple[float, float] = (0.0, 0.0)   # unix ts of IS region
    oos_range: Tuple[float, float] = (0.0, 0.0)
    criteria: Optional[AcceptanceCriteria] = None
    montecarlo: Optional[MonteCarloResult] = None
    # IS -> OOS degradation of the annualized profit rate, in percent
    # (0 = no degradation, 100 = all IS edge lost out of sample)
    degradation_pct: float = 0.0
    # neighborhood-average fitness / peak fitness of the chosen parameter set
    # (1.0 = perfectly stable plateau, near 0 = fragile isolated peak)
    stability_ratio: float = 1.0
    # OHLC provenance for the backtest period (mt5 | cache | synthetic)
    data_source: str = "unknown"
    # Rolling walk-forward window sizing (months); None = legacy fraction-based
    wfo_train_months: Optional[int] = None
    wfo_test_months: Optional[int] = None
    # Composite discovery quality score (0..100), used for promotion/alerting.
    quality_score: float = 0.0
    # Human-readable score components for transparent ranking/triage.
    quality_breakdown: Dict[str, float] = Field(default_factory=dict)
    # Promotion lifecycle: candidate -> validated -> edge_positive -> promoted_live_watchlist.
    promotion_state: str = "candidate"
    # Hard-gate flag for alert eligibility (strict minimum safety gates).
    hard_gates_passed: bool = False


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class JobCancelled(BaseException):
    """Raised inside the validation pipeline when a job's cancel flag is set.

    Lets long-running per-candidate work (IS optimization, walk-forward, Monte
    Carlo) abort promptly at a safe point instead of only being noticed by the
    worker loop between candidates.

    Subclasses :class:`BaseException` (not :class:`Exception`) on purpose: the
    optimizer, walk-forward and Monte Carlo hot loops wrap each backtest in a
    broad ``except Exception: continue`` to skip individual failed runs. If a
    cancel were an ordinary ``Exception`` those guards would silently swallow
    it and the run would keep going. As a ``BaseException`` it sails straight
    past them (like ``KeyboardInterrupt``) and is only caught by the explicit
    ``except JobCancelled`` handlers that stop the job.
    """


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Job(BaseModel):
    id: str
    kind: str = "discovery"
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0              # 0..1
    message: str = ""
    error: Optional[str] = None
    payload: Dict = Field(default_factory=dict)
    cancel_requested: bool = False
    # Live discovery counters, persisted incrementally so the UI can render
    # determinate progress WHILE the run is in flight (not just at the end).
    tested: int = 0                    # candidates that ran the fast screen
    promising: int = 0                 # candidates promoted to full validation
    survivors: int = 0                 # strategies passing every gate
    generation: int = 0                # current evolution round
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
