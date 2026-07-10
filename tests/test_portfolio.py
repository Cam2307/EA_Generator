"""HRP portfolio layer (factory.portfolio)."""
import numpy as np

from factory.models import BacktestMetrics, ValidationReport
from factory.portfolio import (
    build_portfolio, build_return_matrix, correlation_matrix, hrp_weights,
    portfolio_metrics,
)


def _metrics(daily_pnl, start_day=19_700, deposit=10_000.0):
    ts, eq = [], []
    equity = deposit
    for d, pnl in enumerate(daily_pnl):
        equity += pnl
        ts.append((start_day + d) * 86400.0 + 43200.0)
        eq.append(equity)
    return BacktestMetrics(equity_ts=ts, equity=eq)


def _report(sid, pnl):
    return ValidationReport(strategy_id=sid, is_metrics=BacktestMetrics(),
                            oos_metrics=_metrics(pnl), passed=True)


def test_return_matrix_alignment():
    rng = np.random.default_rng(1)
    a = rng.normal(10, 40, 60).tolist()
    b = rng.normal(5, 30, 60).tolist()
    ids, days, matrix = build_return_matrix([_report("a", a), _report("b", b)])
    assert ids == ["a", "b"]
    assert matrix.shape == (len(days), 2)
    assert matrix.shape[0] >= 59
    # a report with no usable curve is dropped
    ids2, _, m2 = build_return_matrix([_report("a", a),
                                       ValidationReport(
                                           strategy_id="empty",
                                           is_metrics=BacktestMetrics(),
                                           oos_metrics=BacktestMetrics())])
    assert ids2 == ["a"] and m2.shape[1] == 1


def test_hrp_downweights_the_correlated_pair():
    rng = np.random.default_rng(7)
    base = rng.normal(0.001, 0.01, 500)
    twin = base + rng.normal(0, 0.001, 500)        # ~same stream
    indep = rng.normal(0.001, 0.01, 500)           # independent
    matrix = np.column_stack([base, twin, indep])
    w = hrp_weights(matrix)
    assert w.shape == (3,) and abs(w.sum() - 1.0) < 1e-9
    # the independent strategy must get more weight than either twin
    assert w[2] > w[0] and w[2] > w[1]
    # degenerate cases
    assert hrp_weights(np.empty((0, 0))).size == 0
    assert hrp_weights(matrix[:, :1]).tolist() == [1.0]


def test_correlation_matrix_safe_on_flat_column():
    rng = np.random.default_rng(3)
    m = np.column_stack([rng.normal(0, 0.01, 100), np.zeros(100)])
    corr = correlation_matrix(m)
    assert corr[0, 1] == 0.0 and corr[0, 0] == 1.0


def test_portfolio_metrics_diversification():
    rng = np.random.default_rng(9)
    a = rng.normal(0.001, 0.01, 500)
    b = rng.normal(0.001, 0.01, 500)               # independent, same vol
    m = np.column_stack([a, b])
    w = np.array([0.5, 0.5])
    ann_ret, sharpe, dd, equity, div = portfolio_metrics(m, w)
    assert len(equity) == 500
    assert dd >= 0.0
    # two independent equal-vol streams: portfolio vol ~ 1/sqrt(2) of avg
    assert 0.6 < div < 0.85


def test_build_portfolio_end_to_end():
    rng = np.random.default_rng(11)
    reports = [
        _report("s1", rng.normal(12, 45, 90).tolist()),
        _report("s2", rng.normal(8, 35, 90).tolist()),
        _report("s3", rng.normal(10, 50, 90).tolist()),
    ]
    port = build_portfolio(reports)
    assert set(port.strategy_ids) == {"s1", "s2", "s3"}
    assert abs(sum(port.weights.values()) - 1.0) < 1e-3
    assert port.days >= 89
    assert len(port.equity) == port.days
    assert 0.0 <= port.avg_pairwise_corr <= 1.0
    assert port.max_pairwise_corr >= port.avg_pairwise_corr
    assert build_portfolio([]).strategy_ids == []
