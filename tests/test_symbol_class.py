"""Symbol-class exit economics and percent/ATR generation bias."""
from __future__ import annotations

import random

from factory.backtest.validation import estimate_stop_points, stop_economics_sane
from factory.generator import random_strategy
from factory.models import StopLossMode, TakeProfitMode
from factory.symbol_class import (
    SymbolClass, classify_symbol, point_distance_mult, recommended_history_months,
)


def test_classify_btc_crypto() -> None:
    assert classify_symbol("BTCUSD") == SymbolClass.CRYPTO
    assert classify_symbol("ETHUSD") == SymbolClass.CRYPTO
    assert classify_symbol("EURUSD") == SymbolClass.FX
    assert classify_symbol("XAUUSD") == SymbolClass.METAL


def test_crypto_point_mult_much_larger_than_fx() -> None:
    assert point_distance_mult(SymbolClass.CRYPTO) >= 100.0
    assert point_distance_mult(SymbolClass.FX) == 1.0


def test_crypto_history_default_longer() -> None:
    assert recommended_history_months(SymbolClass.CRYPTO, default=12) >= 24


def test_non_fx_always_uses_percent_exits() -> None:
    from factory.models import ExecutionMechanicType

    rng = random.Random(42)
    directional = [
        ExecutionMechanicType.STANDARD_SLTP,
        ExecutionMechanicType.PARTIAL_CLOSE,
    ]
    for symbol in ("BTCUSD", "XAUUSD", "US30", "USOIL", "ETHUSD"):
        for _ in range(15):
            s = random_strategy(
                symbol, "M15", rng, allowed_mechanics=directional)
            assert s.trade_mgmt.sl_mode == StopLossMode.PERCENT, symbol
            assert s.trade_mgmt.tp_mode == TakeProfitMode.PERCENT, symbol
            sl_pct = round(float(s.trade_mgmt.params["sl_pct"]), 2)
            tp_pct = round(float(s.trade_mgmt.params["tp_pct"]), 2)
            assert 0.1 <= sl_pct <= 2.5
            assert 0.1 <= tp_pct <= 4.0


def test_fx_point_ranges_are_wide() -> None:
    from factory.generator import MECHANIC_PARAM_SPECS
    from factory.models import ExecutionMechanicType

    sl = MECHANIC_PARAM_SPECS[ExecutionMechanicType.STANDARD_SLTP]["sl_points"]
    tp = MECHANIC_PARAM_SPECS[ExecutionMechanicType.STANDARD_SLTP]["tp_points"]
    assert sl.min <= 100 and sl.max >= 3000
    assert tp.max >= 4000


def test_percent_stop_estimate() -> None:
    rng = random.Random(1)
    s = random_strategy("BTCUSD", "M15", rng)
    assert s.trade_mgmt.sl_mode == StopLossMode.PERCENT
    s.trade_mgmt.params["sl_pct"] = 1.0
    s.trade_mgmt.params["tp_pct"] = 2.0
    pts = estimate_stop_points(s, price=100_000.0, atr_price=1500.0, point=0.01)
    # 1% of 100k = 1000 USD = 100_000 points at point=0.01
    assert abs(pts - 100_000.0) < 1.0


def test_absurd_fx_stop_on_btc_rejected() -> None:
    rng = random.Random(2)
    s = random_strategy("EURUSD", "M15", rng)
    s.symbol = "BTCUSD"
    s.trade_mgmt.sl_mode = StopLossMode.FIXED
    s.mechanic.params["sl_points"] = 100.0  # $1 on BTC
    ok, reason = stop_economics_sane(
        s, price=100_000.0, atr_price=2000.0, point=0.01,
        spread_points=500.0, slippage_points=50.0,
    )
    assert ok is False
    assert "ATR" in reason or "cost" in reason
