"""Causal 2-state HMM regime overlay: math, generator, simulator, MQL5."""
import random

import numpy as np
import pandas as pd
import pytest

from factory.backtest import simulator as sim
from factory.backtest.simulator import SymbolSpec, run_simulation
from factory.generator import (
    GenerationSettings, _TM_COMPLEXITY_COST, describe_trade_mgmt,
    random_strategy, random_trade_mgmt,
)
from factory.hmm_regime import (
    allowed_by_hmm, classify_hmm_filter, forward_filter, forward_step,
    log_returns,
)
from factory.models import (
    EntryFilter, EntryFilterType, ExecutionMechanic, ExecutionMechanicType,
    RiskBlock, StrategyDefinition, TradeManagement,
)
from factory.mql5.renderer import mql5_inputs_for, render_ea

SPEC = SymbolSpec(point=0.01, contract_size=100.0, leverage=100.0,
                  spread_points=0.0, slippage_points=0.0)

_HMM_PARAMS = {
    "hmm_mu0": 0.0, "hmm_mu1": 0.0,
    "hmm_sigma0": 0.0003, "hmm_sigma1": 0.0025,
    "hmm_p00": 0.95, "hmm_p11": 0.90, "hmm_pi0": 0.5,
    "hmm_min_prob": 0.55, "hmm_allow_mask": 3.0,
    "hmm_size_state0": 1.0, "hmm_size_state1": 1.0,
}


def _closes_two_regime(n0=300, n1=300, seed=3):
    """Low-vol then high-vol Gaussian returns → close path."""
    rng = np.random.default_rng(seed)
    r0 = rng.normal(0.0, 0.0003, n0)
    r1 = rng.normal(0.0, 0.0025, n1)
    rets = np.concatenate([r0, r1])
    close = 100.0 * np.exp(np.cumsum(rets))
    return close, rets


def _df_from_close(close):
    n = len(close)
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="1h", tz="utc"),
        "open": close, "high": close + 0.02, "low": close - 0.02,
        "close": close, "volume": np.ones(n),
    })


def test_forward_filter_recovers_vol_regimes():
    close, _ = _closes_two_regime()
    rets = log_returns(close)
    post = forward_filter(
        rets, mu0=0.0, mu1=0.0, sigma0=0.0003, sigma1=0.0025,
        p00=0.95, p11=0.90, pi0=0.5)
    codes = np.argmax(post, axis=1)
    # After warmup, first half mostly state 0; second half mostly state 1
    assert (codes[50:280] == 0).mean() > 0.7
    assert (codes[350:580] == 1).mean() > 0.7


def test_forward_filter_is_causal():
    close, _ = _closes_two_regime(200, 200)
    rets = log_returns(close)
    full = forward_filter(rets, 0.0, 0.0, 0.0003, 0.0025, 0.95, 0.90)
    truncated = forward_filter(rets[:250], 0.0, 0.0, 0.0003, 0.0025, 0.95, 0.90)
    assert truncated.shape == (250, 2)
    np.testing.assert_allclose(full[:250], truncated, rtol=1e-10, atol=1e-12)


def test_recursive_step_matches_batch():
    """Parity check for the MQL5 one-step recurrence."""
    close, _ = _closes_two_regime(80, 80, seed=11)
    rets = log_returns(close)
    batch = forward_filter(rets, 0.0, 0.0, 0.0004, 0.002, 0.92, 0.88, 0.5)
    a0, a1 = 0.5, 0.5
    for t in range(1, len(rets)):
        a0, a1 = forward_step(a0, a1, float(rets[t]),
                              0.0, 0.0, 0.0004, 0.002, 0.92, 0.88)
        assert a0 == pytest.approx(batch[t, 0], rel=1e-9, abs=1e-12)
        assert a1 == pytest.approx(batch[t, 1], rel=1e-9, abs=1e-12)


def test_allowed_by_hmm_mask_and_confidence():
    codes = np.array([0, 0, 1, 1], dtype=np.int8)
    post = np.array([
        [0.9, 0.1],
        [0.52, 0.48],
        [0.2, 0.8],
        [0.45, 0.55],
    ])
    ok = allowed_by_hmm(codes, post, mask=0b01, min_prob=0.55)
    assert ok.tolist() == [True, False, False, False]
    ok_both = allowed_by_hmm(codes, post, mask=0b11, min_prob=0.50)
    assert ok_both.tolist() == [True, True, True, True]


def _strategy(tm):
    return StrategyDefinition(
        symbol="TEST", timeframe="H1",
        entry_filters=[EntryFilter(
            type=EntryFilterType.RSI_REVERSION,
            params={"rsi_period": 14, "oversold": 30, "overbought": 70})],
        mechanic=ExecutionMechanic(
            type=ExecutionMechanicType.STANDARD_SLTP,
            params={"sl_points": 1000.0, "tp_points": 0.0}),
        risk=RiskBlock(fixed_lots=0.10, max_open_lots=50.0),
        trade_mgmt=tm)


def _hmm_filter_tm(mask=0b01, min_prob=0.55):
    p = dict(_HMM_PARAMS)
    p["hmm_allow_mask"] = float(mask)
    p["hmm_min_prob"] = float(min_prob)
    return TradeManagement(hmm_regime_filter=True, params=p)


def _hmm_sizing_tm(s0=0.5, s1=2.0):
    p = dict(_HMM_PARAMS)
    p["hmm_size_state0"] = float(s0)
    p["hmm_size_state1"] = float(s1)
    return TradeManagement(hmm_regime_sizing=True, params=p)


def test_generator_populates_hmm_filter_params():
    rng = random.Random(0)
    found = False
    for _ in range(80):
        tm = random_trade_mgmt(ExecutionMechanicType.STANDARD_SLTP, rng,
                               allowed=["hmm_regime_filter"])
        if tm.hmm_regime_filter:
            found = True
            for key in ("hmm_mu0", "hmm_sigma0", "hmm_sigma1", "hmm_p00",
                        "hmm_p11", "hmm_allow_mask", "hmm_min_prob"):
                assert key in tm.params and key in tm.ranges
            assert any("HMM regime filter" in ln for ln in describe_trade_mgmt(tm))
    assert found


def test_generator_populates_hmm_sizing_params():
    rng = random.Random(2)
    found = False
    for _ in range(80):
        tm = random_trade_mgmt(ExecutionMechanicType.STANDARD_SLTP, rng,
                               allowed=["hmm_regime_sizing"])
        if tm.hmm_regime_sizing:
            found = True
            for key in ("hmm_size_state0", "hmm_size_state1", "hmm_sigma0"):
                assert key in tm.params and key in tm.ranges
            assert any("HMM regime sizing" in ln for ln in describe_trade_mgmt(tm))
    assert found


def test_hmm_raises_complexity_score():
    assert (
        _TM_COMPLEXITY_COST["hmm_regime_filter"]
        + _TM_COMPLEXITY_COST["hmm_regime_sizing"]
    ) == 5
    strat = random_strategy(
        "EURUSD", "H1", rng=random.Random(9),
        allowed_tm_features=["hmm_regime_filter"],
        generation_settings=GenerationSettings(
            advanced_mode=True, complexity_cap=10))
    if strat.trade_mgmt.hmm_regime_filter:
        assert strat.profile.complexity_score >= _TM_COMPLEXITY_COST[
            "hmm_regime_filter"]


def test_simulator_gates_entries_by_hmm(monkeypatch):
    close, _ = _closes_two_regime(250, 250)
    df = _df_from_close(close)
    n = len(df)
    monkeypatch.setattr(sim, "compute_signals",
                        lambda *a, **k: (np.ones(n, dtype=bool),
                                         np.zeros(n, dtype=bool), 1))

    codes, post = classify_hmm_filter(df, _HMM_PARAMS)
    state0_only = _hmm_filter_tm(mask=0b01, min_prob=0.50)
    state1_only = _hmm_filter_tm(mask=0b10, min_prob=0.50)

    _, book0 = run_simulation(df, _strategy(state0_only), SPEC, 100_000.0)
    _, book1 = run_simulation(df, _strategy(state1_only), SPEC, 100_000.0)

    ok0 = allowed_by_hmm(codes, post, 0b01, 0.50)
    ok1 = allowed_by_hmm(codes, post, 0b10, 0.50)
    bar_of = {float(t): i for i, t in enumerate(
        pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        .to_numpy().astype("datetime64[s]").astype("int64"))}

    assert book0.closed and book1.closed
    for tr in book0.closed:
        assert ok0[bar_of[tr.open_time]]
    for tr in book1.closed:
        assert ok1[bar_of[tr.open_time]]


def test_simulator_scales_lots_by_hmm(monkeypatch):
    close, _ = _closes_two_regime(250, 250)
    df = _df_from_close(close)
    n = len(df)
    monkeypatch.setattr(sim, "compute_signals",
                        lambda *a, **k: (np.ones(n, dtype=bool),
                                         np.zeros(n, dtype=bool), 1))
    codes, _ = classify_hmm_filter(df, _HMM_PARAMS)
    bar_of = {float(t): i for i, t in enumerate(
        pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        .to_numpy().astype("datetime64[s]").astype("int64"))}

    strat = _strategy(_hmm_sizing_tm(s0=0.5, s1=2.0))
    strat.mechanic.params = {"sl_points": 10.0, "tp_points": 10.0}
    _, book = run_simulation(df, strat, SPEC, 100_000.0)
    assert book.closed
    seen0 = seen1 = False
    for tr in book.closed:
        code = codes[bar_of[tr.open_time]]
        if code == 0:
            assert tr.lots == pytest.approx(0.05)
            seen0 = True
        else:
            assert tr.lots == pytest.approx(0.20)
            seen1 = True
    assert seen0 and seen1


def test_rendered_ea_contains_hmm_block():
    tm = _hmm_filter_tm(mask=0b10, min_prob=0.60)
    tm.hmm_regime_sizing = True
    tm.params["hmm_size_state0"] = 0.5
    tm.params["hmm_size_state1"] = 1.5
    strat = _strategy(tm)
    strat.name = "HMM Test"
    src = render_ea(strat)
    assert "input int    Inp_X_hmm_regime_filter  = 1;" in src
    assert "Inp_X_hmm_allow_mask     = 2;" in src
    assert "Inp_X_hmm_min_prob       = 0.6;" in src
    assert "TM_HmmAllowed" in src
    assert "TM_HmmLotMult" in src
    assert "TM_HmmUpdate" in src
    assert "NormalizeLots(lots * TM_RegimeLotMult() * TM_HmmLotMult())" in src
    inputs, _ = mql5_inputs_for(strat)
    assert inputs["Inp_X_hmm_size_state1"] == 1.5

    off = render_ea(_strategy(TradeManagement()))
    assert "input int    Inp_X_hmm_regime_filter  = 0;" in off
