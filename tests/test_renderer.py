"""Renderer output checked against Market validator-ready snippets."""
import re

import pytest

from factory.generator import random_strategy
from factory.models import (
    EntryFilter, EntryFilterType, ExecutionMechanic, ExecutionMechanicType,
    ParamRange, RiskBlock, StrategyDefinition,
)
from factory.mql5.renderer import mql5_input_name, mql5_inputs_for, render_ea


def _strategy():
    return StrategyDefinition(
        name="Iron Weaver F00BA4", symbol="EURUSD", timeframe="M15",
        magic_number=774242,
        entry_filters=[EntryFilter(
            type=EntryFilterType.RSI_REVERSION,
            params={"rsi_period": 14, "oversold": 30, "overbought": 70},
            ranges={"rsi_period": ParamRange(min=7, max=21, step=7)})],
        mechanic=ExecutionMechanic(
            type=ExecutionMechanicType.DCA_GRID,
            params={"grid_step_points": 200.0, "lot_multiplier": 1.5,
                    "max_levels": 4.0, "basket_tp_points": 100.0,
                    "basket_sl_points": 300.0},
            ranges={"grid_step_points": ParamRange(min=100, max=500, step=50)}),
        risk=RiskBlock(fixed_lots=0.1, max_spread_points=30),
    )


@pytest.fixture(scope="module")
def ea_source():
    return render_ea(_strategy())


def test_no_unresolved_placeholders(ea_source):
    assert "{I}" not in ea_source
    assert "{IN_" not in ea_source
    assert "{P_" not in ea_source
    assert not re.search(r"__[A-Z_]+__", ea_source)


def test_property_headers(ea_source):
    assert '#property copyright' in ea_source
    assert '#property link' in ea_source
    assert '#property version   "1.00"' in ea_source
    assert '#property description' in ea_source


def test_validator_proof_guards(ea_source):
    assert "SERIES_SYNCHRONIZED" in ea_source
    assert "BarsCalculated" in ea_source
    assert "ArrayResize" in ea_source
    assert "ArraySetAsSeries" in ea_source
    assert "copied == count" in ea_source
    assert "SafeDiv" in ea_source
    assert "SYMBOL_TRADE_MODE" in ea_source
    assert "InpMaxSpreadPoints" in ea_source
    assert "TRADE_RETCODE_DONE" in ea_source
    assert "RetcodeTransient" in ea_source
    assert "Sleep(InpRetryBaseMs * attempt)" in ea_source
    assert "INVALID_HANDLE" in ea_source
    assert "IndicatorRelease" in ea_source
    assert "OBJ_LABEL" in ea_source
    assert ea_source.count("input group") >= 3
    assert "#include <Trade\\Trade.mqh>" in ea_source
    assert ea_source.count("#include") == 1


def test_market_validation_helpers(ea_source):
    assert "bool CheckMoneyForTrade(" in ea_source
    assert "OrderCalcMargin" in ea_source
    assert "ACCOUNT_MARGIN_FREE" in ea_source
    assert "bool CheckVolumeValue(" in ea_source
    assert "SYMBOL_VOLUME_MIN" in ea_source
    assert "SYMBOL_VOLUME_MAX" in ea_source
    assert "SYMBOL_VOLUME_STEP" in ea_source
    assert "AdjustStops" in ea_source
    assert "SYMBOL_TRADE_STOPS_LEVEL" in ea_source
    assert "SYMBOL_TRADE_FREEZE_LEVEL" in ea_source
    assert "FreezeOK" in ea_source
    assert "TradingAllowed" in ea_source
    assert "MQLInfoInteger(MQL_TESTER)" in ea_source
    assert "TERMINAL_TRADE_ALLOWED" in ea_source
    assert "MQL_TRADE_ALLOWED" in ea_source
    assert "OrderPreflight" in ea_source
    assert "CheckMoneyForTrade(_Symbol" in ea_source
    assert "CheckVolumeValue(lots" in ea_source


def test_netting_hedging_and_tester_fallback(ea_source):
    assert "IsHedgingAccount" in ea_source
    assert "ACCOUNT_MARGIN_MODE" in ea_source
    assert "TesterFallbackTrade" in ea_source
    assert "g_any_trade" in ea_source


def test_mechanic_params_are_inputs_not_literals(ea_source):
    assert "input double Inp_M_grid_step_points = 200;" in ea_source
    assert "input double Inp_M_lot_multiplier   = 1.5;" in ea_source
    assert "input int    Inp_M_max_levels       = 4;" in ea_source
    assert "input double Inp_M_basket_tp_points = 100;" in ea_source
    assert "input double Inp_M_basket_sl_points = 300;" in ea_source
    assert "SyncSharedStopOnAll" in ea_source
    assert "adverse_points >= Inp_M_grid_step_points" in ea_source
    assert "Inp_F0_rsi_period" in ea_source


def test_strategy_metadata_embedded(ea_source):
    assert "InpMagic           = 774242" in ea_source
    assert "Iron_Weaver_F00BA4" in ea_source


def test_hedge_template_exposes_hedge_distance_input():
    strat = _strategy()
    strat.mechanic = ExecutionMechanic(
        type=ExecutionMechanicType.HEDGE_LAYER,
        params={"sl_points": 400.0, "tp_points": 600.0,
                "hedge_trigger_points": 250.0, "hedge_ratio": 1.0},
        ranges={"hedge_trigger_points": ParamRange(min=100, max=400, step=50)})
    src = render_ea(strat)
    assert "input double Inp_M_hedge_trigger_points = 250;" in src
    assert "adverse_points >= Inp_M_hedge_trigger_points" in src
    assert "CheckMoneyForTrade" in src
    assert '#property version   "1.00"' in src


def test_dca_grid_netting_state_globals():
    strat = _strategy()
    src = render_ea(strat)
    assert "g_grid_levels" in src
    assert "IsHedgingAccount()" in src


def test_lazy_indicator_init_no_init_failed():
    strat = _strategy()
    src = render_ea(strat)
    assert "INIT_FAILED" not in src
    assert "Filter0_Ensure" in src


def test_input_name_mapping():
    assert mql5_input_name("F0_PRICE_ACTION_BREAKOUT_lookback") == "Inp_F0_lookback"
    assert mql5_input_name("M_DCA_GRID_grid_step_points") == "Inp_M_grid_step_points"
    with pytest.raises(ValueError):
        mql5_input_name("garbage")


def test_inputs_for_strategy_cover_all_params():
    strat = _strategy()
    inputs, ranges = mql5_inputs_for(strat)
    assert inputs["Inp_M_grid_step_points"] == 200.0
    assert ranges["Inp_M_grid_step_points"].min == 100
    assert inputs["Inp_F0_rsi_period"] == 14


def test_random_strategies_always_render():
    import random
    rng = random.Random(7)
    for _ in range(12):
        strat = random_strategy("EURUSD", "M15", rng)
        src = render_ea(strat)
        assert "OnTick" in src and "OnInit" in src
        assert "CheckMoneyForTrade" in src
        assert "CheckVolumeValue" in src
        assert not re.search(r"\{(IN|P)_[a-z_]+\}", src)
