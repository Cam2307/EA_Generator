"""Compile two hardened EAs (DCA/grid + hedge) via MetaEditor.

Usage:  python scripts/compile_verify.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from factory.backtest.mt5_runner import MT5Runner, MT5RunnerError, detect_mt5
from factory.generator import random_strategy
from factory.models import ExecutionMechanicType
from factory.mql5.renderer import render_ea, write_ea


def _compile_one(runner: MT5Runner, strategy, out_dir: Path) -> None:
    path = write_ea(strategy, out_dir)
    print(f"[compile] rendering {strategy.name} ({strategy.mechanic.type.value})")
    print(f"[compile]   -> {path}")
    ex5 = runner.compile_ea(path)
    print(f"[compile]   OK -> {ex5}")


def main() -> int:
    settings.ensure_dirs()
    try:
        paths = detect_mt5()
    except MT5RunnerError as exc:
        print(f"[compile] SKIPPED — MT5 not detectable: {exc}")
        return 1

    runner = MT5Runner(paths)
    out_dir = settings.OUTPUT_DIR / "_compile_verify"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(20260708)

    dca = hedge = None
    for _ in range(50):
        s = random_strategy("EURUSD", "M15", rng)
        if s.mechanic.type == ExecutionMechanicType.DCA_GRID and dca is None:
            dca = s
        if s.mechanic.type == ExecutionMechanicType.HEDGE_LAYER and hedge is None:
            hedge = s
        if dca and hedge:
            break
    if dca is None or hedge is None:
        print("[compile] could not sample required mechanic types")
        return 1

    errors = 0
    for strat in (dca, hedge):
        try:
            _compile_one(runner, strat, out_dir)
        except MT5RunnerError as exc:
            print(f"[compile] FAILED: {exc}")
            errors += 1

    if errors:
        print(f"[compile] {errors} compilation(s) failed")
        return 1
    print("[compile] BOTH EAs compiled with 0 errors, 0 warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
