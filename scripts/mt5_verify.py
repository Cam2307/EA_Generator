"""Real-terminal verification: render -> MetaEditor compile -> headless tester run.

Degrades gracefully: any missing piece (terminal, editor, market data) is
reported and the script exits 0 with a clear status instead of crashing.

Usage:  python scripts/mt5_verify.py
"""
from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings


def main() -> int:
    settings.ensure_dirs()
    from factory.backtest.mt5_runner import MT5Runner, MT5RunnerError, detect_mt5
    from factory.generator import random_strategy
    from factory.mql5.renderer import write_ea

    # 1) detection
    try:
        paths = detect_mt5()
    except MT5RunnerError as exc:
        print(f"[mt5-verify] SKIPPED — MT5 not detectable: {exc}")
        return 0
    print(f"[mt5-verify] terminal   : {paths.terminal_exe}")
    print(f"[mt5-verify] metaeditor : {paths.metaeditor_exe}")
    print(f"[mt5-verify] data dir   : {paths.data_dir}")

    # 2) render + compile
    rng = random.Random(42)
    strategy = random_strategy("EURUSD", "M15", rng)
    print(f"[mt5-verify] rendering strategy: {strategy.name} "
          f"({strategy.mechanic.type.value})")
    mq5_path = write_ea(strategy, settings.OUTPUT_DIR / "_mt5_verify")
    print(f"[mt5-verify] rendered {mq5_path}")

    runner = MT5Runner(paths)
    try:
        ex5 = runner.compile_ea(mq5_path)
        print(f"[mt5-verify] COMPILE OK -> {ex5}")
    except MT5RunnerError as exc:
        print(f"[mt5-verify] COMPILE FAILED: {exc}")
        return 1

    # 3) one short headless tester pass (kept brief)
    settings.MT5_RUN_TIMEOUT_SECONDS = 420
    end = datetime.now(timezone.utc) - timedelta(days=7)
    start = end - timedelta(days=60)
    try:
        metrics = runner.run_backtest(
            expert=f"EAFactory\\{ex5.stem}.ex5", symbol="EURUSD",
            timeframe="M15", start=start, end=end, deposit=10_000.0,
        )
        print(f"[mt5-verify] TESTER OK — net={metrics.net_profit} "
              f"trades={metrics.trade_count} pf={metrics.profit_factor} "
              f"dd%={metrics.max_dd_pct}")
    except MT5RunnerError as exc:
        print(f"[mt5-verify] TESTER RUN DEGRADED (captured, not fatal): {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
