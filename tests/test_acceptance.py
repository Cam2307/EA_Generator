"""Acceptance criteria gates."""
from datetime import datetime, timezone

import pytest

from factory.models import AcceptanceCriteria, BacktestMetrics


def _oos(**kw):
    defaults = dict(net_profit=500.0, profit_factor=1.5, sharpe=1.2,
                    max_dd_pct=8.0, trade_count=30, r_squared=0.7,
                    max_consecutive_losses=3, initial_deposit=10_000.0,
                    start_ts=0.0, end_ts=365.25 * 86400)
    defaults.update(kw)
    return BacktestMetrics(**defaults)


def test_all_gates_pass():
    c = AcceptanceCriteria(min_wfe=0.5, max_dd_pct=15.0, min_trades=5,
                           min_profit_factor=1.2, min_sharpe=1.0,
                           min_r_squared=0.5, max_consecutive_losses=5)
    assert c.evaluate(_oos(), wfe=0.8) == []


def test_wfe_gate():
    c = AcceptanceCriteria(min_wfe=0.6)
    reasons = c.evaluate(_oos(), wfe=0.4)
    assert any("WFE" in r for r in reasons)


def test_min_wfe_zero_disables_wfe_gate():
    """min_wfe <= 0 skips WFE entirely (L1 OOS screener)."""
    c = AcceptanceCriteria(min_wfe=0.0, max_dd_pct=70.0, min_trades=3,
                           min_profit_factor=0.95)
    assert c.evaluate(_oos(max_dd_pct=20.0, trade_count=10), wfe=0.0) == []
    assert c.evaluate(_oos(max_dd_pct=20.0, trade_count=10), wfe=-1.0) == []


def test_profit_factor_gate():
    c = AcceptanceCriteria(min_profit_factor=2.0)
    reasons = c.evaluate(_oos(profit_factor=1.1), wfe=1.0)
    assert any("profit factor" in r for r in reasons)


def test_r_squared_gate():
    c = AcceptanceCriteria(min_r_squared=0.8)
    reasons = c.evaluate(_oos(r_squared=0.3), wfe=1.0)
    assert any("R-squared" in r for r in reasons)


def test_consecutive_losses_gate():
    c = AcceptanceCriteria(max_consecutive_losses=5)
    reasons = c.evaluate(_oos(max_consecutive_losses=8), wfe=1.0)
    assert any("consecutive" in r for r in reasons)
