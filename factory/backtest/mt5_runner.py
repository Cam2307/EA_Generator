"""Headless MetaTrader 5 Strategy Tester wrapper.

Design rules (see plan):
- Terminal auto-detected via MetaTrader5.initialize() -> terminal_info().path,
  overridable in config/settings.py.
- Every tester .ini sets an explicit static Report= path under reports/.
- XML reports parsed (not HTML) to avoid file-locking issues.
- Strictly sequential execution guarded by a module lock; the worker also
  routes all MT5 jobs through a single-slot lane.
- Every failure degrades gracefully into MT5RunnerError with a clear message
  that the job runner captures into the Job record — never a crash.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from config import settings
from factory.backtest.base import BacktestEngine
from factory.models import BacktestMetrics, JobCancelled, ParamRange, StrategyDefinition

_MT5_LOCK = threading.Lock()   # hard guarantee: one terminal at a time
_CANCEL_POLL_SECONDS = 0.5


def _subprocess_kwargs() -> dict:
    """Suppress flashing console windows for helper commands on Windows."""
    if os.name != "nt":
        return {}
    return {"creationflags": subprocess.CREATE_NO_WINDOW}  # type: ignore[attr-defined]


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort kill of an MT5 tester process (and children on Windows)."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt" and proc.pid:
            subprocess.run(  # noqa: S603,S607
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                **_subprocess_kwargs(),
            )
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

TIMEFRAME_TO_PERIOD = {
    "M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30",
    "H1": "H1", "H4": "H4", "D1": "Daily",
}


class MT5RunnerError(RuntimeError):
    """Raised for any MT5 pipeline failure; message is job-record friendly."""


@dataclass
class MT5Paths:
    terminal_exe: Path
    metaeditor_exe: Path
    data_dir: Path              # terminal data folder containing MQL5/

    @property
    def experts_dir(self) -> Path:
        return self.data_dir / "MQL5" / "Experts" / "EAFactory"


def detect_mt5() -> MT5Paths:
    """Locate terminal64.exe / metaeditor64.exe, honoring settings overrides.

    Raises MT5RunnerError with a clear message when MT5 is not available.
    """
    if settings.MT5_TERMINAL_PATH:
        term = Path(settings.MT5_TERMINAL_PATH)
        if not term.exists():
            raise MT5RunnerError(
                f"Configured MT5_TERMINAL_PATH does not exist: {term}")
        editor = Path(settings.MT5_METAEDITOR_PATH) if settings.MT5_METAEDITOR_PATH \
            else term.parent / "metaeditor64.exe"
        return MT5Paths(terminal_exe=term, metaeditor_exe=editor,
                        data_dir=term.parent)

    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise MT5RunnerError(
            "MetaTrader5 python package is not installed; cannot auto-detect "
            "the terminal.") from exc

    try:
        if not mt5.initialize():
            code, desc = mt5.last_error()
            raise MT5RunnerError(
                f"MetaTrader 5 terminal not detected on this machine "
                f"(initialize failed: {code} {desc}). Install MT5 or set "
                f"MT5_TERMINAL_PATH in config/settings.py.")
        info = mt5.terminal_info()
        term_path = Path(info.path)
        data_path = Path(info.data_path) if getattr(info, "data_path", None) else term_path
        mt5.shutdown()
    except MT5RunnerError:
        raise
    except Exception as exc:
        raise MT5RunnerError(f"MT5 auto-detection failed: {exc}") from exc

    terminal_exe = term_path / "terminal64.exe"
    metaeditor_exe = term_path / "metaeditor64.exe"
    if not terminal_exe.exists():
        raise MT5RunnerError(f"terminal64.exe not found under {term_path}")
    return MT5Paths(terminal_exe=terminal_exe, metaeditor_exe=metaeditor_exe,
                    data_dir=data_path)


def _terminal_running() -> bool:
    """True when a terminal64.exe instance is already running."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq terminal64.exe", "/FO", "CSV"],
            capture_output=True, text=True, timeout=15,
            **_subprocess_kwargs(),
        ).stdout
    except Exception:
        return False
    return "terminal64.exe" in out


# ---------------------------------------------------------------------------
# .ini writer (standalone for unit testing)
# ---------------------------------------------------------------------------

def _fmt_num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def write_tester_ini(path: Path, *, expert: str, symbol: str, timeframe: str,
                     from_date: datetime, to_date: datetime, deposit: float,
                     report_path: Path, leverage: int = 100,
                     optimization: int = 0,
                     inputs: Optional[Dict[str, float]] = None,
                     input_ranges: Optional[Dict[str, ParamRange]] = None) -> Path:
    """Write a headless Strategy Tester configuration file.

    ``report_path`` must be an explicit static file path (extension omitted in
    the ini; MT5 appends .xml). ``optimization=2`` selects the genetic
    optimizer.

    When ``input_ranges`` provides a range for an input (this includes
    execution-mechanic parameters such as grid step and hedge distance),
    the [TesterInputs] line uses the optimizable
    ``Value||Start||Step||Stop||Y`` format; otherwise the parameter is
    emitted fixed (``Value||Value||0||Value||N``).
    """
    period = TIMEFRAME_TO_PERIOD.get(timeframe)
    if period is None:
        raise MT5RunnerError(f"Unsupported timeframe for MT5 tester: {timeframe}")

    lines = [
        "[Tester]",
        f"Expert={expert}",
        f"Symbol={symbol}",
        f"Period={period}",
        "Model=0",                       # every tick
        f"FromDate={from_date:%Y.%m.%d}",
        f"ToDate={to_date:%Y.%m.%d}",
        f"Deposit={deposit:g}",
        f"Leverage={leverage}",
        f"Optimization={optimization}",
        "OptimizationCriterion=0",
        f"Report={report_path.with_suffix('')}",
        "ReplaceReport=1",
        "ShutdownTerminal=1",
        "Visual=0",
    ]
    if inputs:
        lines.append("")
        lines.append("[TesterInputs]")
        ranges = input_ranges or {}
        for name, value in inputs.items():
            v = _fmt_num(value)
            r = ranges.get(name)
            if r is not None:
                lines.append(
                    f"{name}={v}||{_fmt_num(r.min)}||{_fmt_num(r.step)}"
                    f"||{_fmt_num(r.max)}||Y")
            else:
                lines.append(f"{name}={v}||{v}||0||{v}||N")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# XML report parsing
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "total net profit": "net_profit",
    "gross profit": "gross_profit",
    "gross loss": "gross_loss",
    "profit factor": "profit_factor",
    "recovery factor": "recovery_factor",
    "sharpe ratio": "sharpe",
    "balance drawdown maximal": "max_dd_money",
    "equity drawdown maximal": "max_dd_money",
    "total trades": "trade_count",
    "initial deposit": "initial_deposit",
}

_DD_PCT_RE = re.compile(r"\(([\d.,]+)\s*%\)")


def parse_xml_report(path: Path) -> BacktestMetrics:
    """Parse an MT5 tester XML report (SpreadsheetML) into BacktestMetrics.

    The report is a label/value grid; the parser scans rows tolerant of
    locale formatting and MT5 build differences.
    """
    if not path.exists():
        raise MT5RunnerError(f"Tester report not found at {path}")
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise MT5RunnerError(f"Cannot parse tester XML report {path}: {exc}") from exc

    # SpreadsheetML uses a namespace; strip tags generically.
    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1].lower()

    values: Dict[str, float] = {}
    dd_pct: Optional[float] = None

    for row in tree.iter():
        if local(row.tag) != "row":
            continue
        cells = [el for el in row.iter() if local(el.tag) == "data"]
        texts = [(el.text or "").strip() for el in cells]
        for idx, text in enumerate(texts):
            label = text.rstrip(":").lower()
            if label in _METRIC_LABELS and idx + 1 < len(texts):
                raw = texts[idx + 1]
                m = _DD_PCT_RE.search(raw)
                if m and "drawdown" in label:
                    dd_pct = float(m.group(1).replace(",", "."))
                num = re.sub(r"[^\d.\-]", "", raw.split("(")[0].replace(",", "."))
                try:
                    values[_METRIC_LABELS[label]] = float(num)
                except ValueError:
                    continue

    if not values:
        raise MT5RunnerError(
            f"Tester XML report {path} contained no recognizable metrics")

    return BacktestMetrics(
        net_profit=values.get("net_profit", 0.0),
        gross_profit=values.get("gross_profit", 0.0),
        gross_loss=abs(values.get("gross_loss", 0.0)),
        profit_factor=values.get("profit_factor", 0.0),
        recovery_factor=values.get("recovery_factor", 0.0),
        sharpe=values.get("sharpe", 0.0),
        max_dd_money=values.get("max_dd_money", 0.0),
        max_dd_pct=dd_pct if dd_pct is not None else 0.0,
        trade_count=int(values.get("trade_count", 0)),
        initial_deposit=values.get("initial_deposit", 0.0),
    )


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class MT5Runner(BacktestEngine):
    name = "mt5"

    def __init__(self, paths: Optional[MT5Paths] = None,
                 leverage: Optional[int] = None,
                 portable: bool = False, exclusive: bool = True):
        self._paths = paths
        # User-chosen account leverage for the tester .ini (None -> MT5 default).
        self.leverage = leverage
        # Portable mode launches terminal/metaeditor with /portable so the
        # instance uses its own directory as the data dir — the basis of the
        # multi-instance pool (jobs/mt5_pool.py).
        self.portable = portable
        # exclusive=True keeps the legacy safety net for a single shared
        # install: one tester at a time process-wide, and refuse to run while
        # an interactive terminal owns the data directory. Pool-managed
        # portable instances don't share a data dir, so leasing already
        # guarantees safety and these global guards are skipped.
        self.exclusive = exclusive

    @property
    def paths(self) -> MT5Paths:
        if self._paths is None:
            self._paths = detect_mt5()      # raises MT5RunnerError if absent
        return self._paths

    # -- compilation ------------------------------------------------------
    def compile_ea(self, mq5_path: Path) -> Path:
        """Copy the .mq5 into the terminal Experts folder and compile it."""
        paths = self.paths
        if not paths.metaeditor_exe.exists():
            raise MT5RunnerError(f"metaeditor64.exe not found at {paths.metaeditor_exe}")
        paths.experts_dir.mkdir(parents=True, exist_ok=True)
        dest = paths.experts_dir / mq5_path.name
        shutil.copyfile(mq5_path, dest)

        log_path = dest.with_suffix(".log")
        cmd = [str(paths.metaeditor_exe), f"/compile:{dest}", f"/log:{log_path}"]
        if self.portable:
            cmd.append("/portable")
        try:
            subprocess.run(
                cmd,
                timeout=settings.MT5_COMPILE_TIMEOUT_SECONDS,
                capture_output=True,
                **_subprocess_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise MT5RunnerError(f"MetaEditor compile timed out for {dest.name}") from exc

        if not log_path.exists():
            raise MT5RunnerError(f"MetaEditor produced no log for {dest.name}")
        log_text = log_path.read_text(encoding="utf-16", errors="ignore")
        m = re.search(r"(\d+)\s+errors?,\s*(\d+)\s+warnings?", log_text, re.IGNORECASE)
        if not m:
            raise MT5RunnerError(
                f"Could not find compile result in MetaEditor log for {dest.name}")
        errors, warnings = int(m.group(1)), int(m.group(2))
        if errors > 0:
            raise MT5RunnerError(
                f"Compilation failed for {dest.name}: {errors} errors, "
                f"{warnings} warnings.\n{log_text[-2000:]}")
        ex5 = dest.with_suffix(".ex5")
        if not ex5.exists():
            raise MT5RunnerError(f"Compile reported success but {ex5.name} is missing")
        return ex5

    # -- backtest -----------------------------------------------------------
    def run(self, strategy: StrategyDefinition, start: datetime, end: datetime,
            params_override: Optional[Dict[str, float]] = None,
            deposit: float = 10_000.0) -> BacktestMetrics:
        """Render is assumed done by the caller; expert must be compiled already
        as EAFactory/<strategy.id>.ex5. Runs strictly sequentially."""
        from factory.mql5.renderer import mql5_inputs_for

        if params_override:
            strategy = strategy.apply_flat_params(params_override)
        inputs, _ = mql5_inputs_for(strategy)
        expert_rel = f"EAFactory\\{strategy.id}.ex5"
        return self.run_backtest(
            expert=expert_rel, symbol=strategy.symbol, timeframe=strategy.timeframe,
            start=start, end=end, deposit=deposit, inputs=inputs,
        )

    def run_backtest(self, *, expert: str, symbol: str, timeframe: str,
                     start: datetime, end: datetime, deposit: float,
                     inputs: Optional[Dict[str, float]] = None,
                     input_ranges: Optional[Dict[str, ParamRange]] = None,
                     optimization: int = 0,
                     leverage: Optional[int] = None) -> BacktestMetrics:
        paths = self.paths
        settings.ensure_dirs()
        run_id = uuid.uuid4().hex[:12]
        report_path = settings.REPORTS_DIR / f"report_{run_id}.xml"
        ini_path = settings.REPORTS_DIR / f"tester_{run_id}.ini"
        eff_leverage = leverage if leverage is not None else self.leverage
        write_tester_ini(
            ini_path, expert=expert, symbol=symbol, timeframe=timeframe,
            from_date=start, to_date=end, deposit=deposit,
            report_path=report_path, optimization=optimization, inputs=inputs,
            input_ranges=input_ranges,
            leverage=int(eff_leverage) if eff_leverage is not None else 100,
        )

        # A headless tester run shares the terminal's data directory. If the
        # terminal is already open interactively, the new instance exits
        # immediately without running the tester (and without any error), so
        # fail fast with a clear, job-record-friendly explanation instead.
        # Pool-managed portable instances own their data dir, so the check
        # (which sees ANY terminal64.exe, including other pool members) and
        # the process-wide lock are skipped for them.
        if self.exclusive and _terminal_running():
            raise MT5RunnerError(
                "MetaTrader 5 terminal is already running interactively. "
                "Close the MT5 application and retry: headless tester runs "
                "need exclusive use of the terminal data directory.")

        cmd = [str(paths.terminal_exe), f"/config:{ini_path}"]
        if self.portable:
            cmd.append("/portable")
        cancel_check = getattr(self, "_cancel_check", None)
        returncode = 0
        lane = _MT5_LOCK if self.exclusive else nullcontext()
        with lane:                          # sequential per shared install
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **_subprocess_kwargs(),
            )
            deadline = time.monotonic() + float(settings.MT5_RUN_TIMEOUT_SECONDS)
            try:
                while True:
                    ret = proc.poll()
                    if ret is not None:
                        returncode = int(ret)
                        break
                    if cancel_check is not None and cancel_check():
                        _terminate_process_tree(proc)
                        raise JobCancelled()
                    if time.monotonic() >= deadline:
                        _terminate_process_tree(proc)
                        raise MT5RunnerError(
                            f"MT5 tester run timed out after "
                            f"{settings.MT5_RUN_TIMEOUT_SECONDS}s for {expert}")
                    time.sleep(_CANCEL_POLL_SECONDS)
            except JobCancelled:
                raise
            except MT5RunnerError:
                raise
            except Exception:
                _terminate_process_tree(proc)
                raise
        # MT5 may exit non-zero even on success; the report is the truth.
        try:
            metrics = parse_xml_report(report_path)
        except MT5RunnerError as exc:
            raise MT5RunnerError(
                f"{exc} (terminal exit code {returncode})") from exc
        finally:
            ini_path.unlink(missing_ok=True)

        metrics.start_ts = start.timestamp()
        metrics.end_ts = end.timestamp()
        if metrics.initial_deposit <= 0:
            metrics.initial_deposit = deposit
        return metrics

    def run_optimization(self, strategy: StrategyDefinition, *,
                         start: datetime, end: datetime,
                         deposit: float = 10_000.0) -> BacktestMetrics:
        """Genetic optimization pass (Optimization=2) sweeping every
        optimizable parameter — entry filters AND execution mechanics (grid
        step, hedge distance, lot multiplier, partial-close levels, ...).
        Returns the summary metrics of the best pass as reported by the
        tester."""
        from factory.mql5.renderer import mql5_inputs_for

        inputs, ranges = mql5_inputs_for(strategy)
        expert_rel = f"EAFactory\\{strategy.id}.ex5"
        return self.run_backtest(
            expert=expert_rel, symbol=strategy.symbol,
            timeframe=strategy.timeframe, start=start, end=end,
            deposit=deposit, inputs=inputs, input_ranges=ranges,
            optimization=2,
        )
