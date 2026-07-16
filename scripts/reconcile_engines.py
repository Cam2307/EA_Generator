"""Reconcile the simulator against the real MT5 Strategy Tester.

Runs N randomly generated strategies (or the current gallery survivors with
``--survivors``) through BOTH engines on the same window and prints per-metric
deltas plus the aggregate simulator-bias summary. Requires a machine with MT5
installed and the interactive terminal CLOSED (see README).

Usage:
    python scripts/reconcile_engines.py [--n 10] [--symbol EURUSD]
        [--timeframe M15] [--days 365] [--survivors] [--seed 42]

    # After percent/ATR exits ship, also reconcile a crypto sample:
    python scripts/reconcile_engines.py --n 8 --symbol BTCUSD --timeframe H1

Exit code 1 when fewer than half the strategies reconcile within tolerance —
suitable as a CI gate on machines with MT5 provisioned.
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings                                   # noqa: E402
from factory.backtest.mt5_runner import MT5Runner, detect_mt5  # noqa: E402
from factory.backtest.reconcile import (                      # noqa: E402
    format_report, reconcile_strategies,
)
from factory.backtest.simulator import SimulatorEngine        # noqa: E402
from factory.generator import random_strategy                 # noqa: E402
from factory.storage import Storage                           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10, help="random strategies to test")
    ap.add_argument("--symbol", default=settings.DEFAULT_SYMBOL)
    ap.add_argument("--timeframe", default=settings.DEFAULT_TIMEFRAME)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--survivors", action="store_true",
                    help="reconcile gallery survivors instead of random strategies")
    args = ap.parse_args()

    try:
        detect_mt5()
    except Exception as exc:
        print(f"MT5 not available: {exc}")
        print("This harness needs a machine with MetaTrader 5 installed.")
        return 2
    mt5 = MT5Runner()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)

    if args.survivors:
        storage = Storage()
        reports = storage.list_validated(passed_only=True)
        strategies = [s for s in (storage.get_strategy(r.strategy_id)
                                  for r in reports) if s is not None][: args.n]
        if not strategies:
            print("No validated survivors in the gallery to reconcile.")
            return 2
    else:
        rng = random.Random(args.seed)
        strategies = [random_strategy(args.symbol, args.timeframe, rng)
                      for _ in range(args.n)]

    sim = SimulatorEngine()
    print(f"Reconciling {len(strategies)} strategies on "
          f"{args.symbol} {args.timeframe}, {args.days} days...\n")
    results = reconcile_strategies(sim, mt5, strategies, start, end,
                                   deposit=settings.DEFAULT_DEPOSIT)
    print(format_report(results))

    n_ok = sum(1 for r in results if r.ok)
    return 0 if n_ok * 2 >= len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
