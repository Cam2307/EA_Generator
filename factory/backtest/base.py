"""Common backtest engine interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional

from factory.models import BacktestMetrics, StrategyDefinition


class BacktestEngine(ABC):
    """Runs one strategy over one date range and returns metrics.

    Implementations: the event-driven fallback simulator (pre-filter) and the
    headless MT5 Strategy Tester wrapper (source of truth).
    """

    name: str = "base"

    @abstractmethod
    def run(self, strategy: StrategyDefinition, start: datetime, end: datetime,
            params_override: Optional[Dict[str, float]] = None,
            deposit: float = 10_000.0) -> BacktestMetrics:
        """Backtest ``strategy`` between ``start`` and ``end``.

        ``params_override`` is a flat prefixed parameter dict (see
        ``StrategyDefinition.all_params``) applied before the run.
        """
        raise NotImplementedError
