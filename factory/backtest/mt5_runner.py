"""Headless MetaTrader 5 Strategy Tester wrapper.

Design rules (see plan):
- Terminal auto-detected from the filesystem (settings override, AppData
  origin.txt, common install dirs) — never via MetaTrader5.initialize(),
  which starts an interactive terminal and leaves it running after shutdown.
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


def _read_origin_install(origin: Path) -> Optional[Path]:
    """Parse MetaQuotes ``origin.txt`` (install path of a terminal data dir)."""
    for encoding in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            text = origin.read_text(encoding=encoding).strip()
        except Exception:
            continue
        if not text:
            continue
        line = text.splitlines()[0].strip().strip("\x00")
        if line:
            return Path(line)
    return None


def _appdata_terminal_candidates() -> list[MT5Paths]:
    """Discover installs via ``%APPDATA%\\MetaQuotes\\Terminal\\*\\origin.txt``.

    This never launches MT5 — unlike ``MetaTrader5.initialize()``, which starts
    an interactive ``terminal64.exe`` and leaves it running after shutdown.
    """
    appdata = os.environ.get("APPDATA") or ""
    root = Path(appdata) / "MetaQuotes" / "Terminal"
    if not root.is_dir():
        return []
    skip = {"Common", "Community", "Help"}
    found: list[MT5Paths] = []
    for data_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not data_dir.is_dir() or data_dir.name in skip:
            continue
        origin = data_dir / "origin.txt"
        if not origin.is_file():
            continue
        install = _read_origin_install(origin)
        if install is None:
            continue
        terminal_exe = install / "terminal64.exe"
        if not terminal_exe.is_file():
            continue
        editor = install / "metaeditor64.exe"
        found.append(MT5Paths(
            terminal_exe=terminal_exe,
            metaeditor_exe=editor,
            data_dir=data_dir,
        ))
    return found


def _common_install_candidates() -> list[MT5Paths]:
    """Fallback: well-known install directories (portable-style data dir = install)."""
    bases = [
        Path(r"C:\Program Files\MetaTrader 5"),
        Path(r"C:\Program Files\MetaTrader"),
        Path(r"C:\Program Files (x86)\MetaTrader 5"),
        Path(r"C:\Program Files (x86)\MetaTrader"),
    ]
    found: list[MT5Paths] = []
    for base in bases:
        terminal_exe = base / "terminal64.exe"
        if terminal_exe.is_file():
            found.append(MT5Paths(
                terminal_exe=terminal_exe,
                metaeditor_exe=base / "metaeditor64.exe",
                data_dir=base,
            ))
    return found


def _pick_preferred(paths: list[MT5Paths]) -> MT5Paths:
    """Prefer a vanilla MetaTrader install over prop-firm / renamed copies."""
    if len(paths) == 1:
        return paths[0]

    def score(p: MT5Paths) -> tuple:
        name = str(p.terminal_exe.parent).lower()
        # Lower is better.
        vanilla = 0 if name.rstrip("\\/").endswith(("metatrader 5", "metatrader")) else 1
        return (vanilla, len(name), name)

    return sorted(paths, key=score)[0]


def _paths_for_terminal_exe(term: Path, editor: Optional[Path] = None) -> MT5Paths:
    """Resolve data_dir for an explicit terminal exe (AppData hash folder if any)."""
    term = term.resolve()
    editor = editor if editor is not None else term.parent / "metaeditor64.exe"
    for candidate in _appdata_terminal_candidates():
        try:
            if candidate.terminal_exe.resolve() == term:
                return MT5Paths(
                    terminal_exe=term,
                    metaeditor_exe=editor if editor.exists() else candidate.metaeditor_exe,
                    data_dir=candidate.data_dir,
                )
        except OSError:
            continue
    # Portable / unknown: data lives next to the exe.
    return MT5Paths(terminal_exe=term, metaeditor_exe=editor, data_dir=term.parent)


def detect_mt5() -> MT5Paths:
    """Locate terminal64.exe / metaeditor64.exe, honoring settings overrides.

    Resolves paths from the filesystem (settings, AppData origin.txt, common
    install dirs). Does **not** call ``MetaTrader5.initialize()`` — that API
    starts an interactive terminal and ``shutdown()`` does not close it, which
    then poisons headless Strategy Tester runs.

    Raises MT5RunnerError with a clear message when MT5 is not available.
    """
    if settings.MT5_TERMINAL_PATH:
        term = Path(settings.MT5_TERMINAL_PATH)
        if not term.exists():
            raise MT5RunnerError(
                f"Configured MT5_TERMINAL_PATH does not exist: {term}")
        editor = Path(settings.MT5_METAEDITOR_PATH) if settings.MT5_METAEDITOR_PATH \
            else term.parent / "metaeditor64.exe"
        return _paths_for_terminal_exe(term, editor)

    candidates = _appdata_terminal_candidates() or _common_install_candidates()
    if candidates:
        return _pick_preferred(candidates)

    raise MT5RunnerError(
        "MetaTrader 5 terminal not detected on this machine. Install MT5 or "
        "set MT5_TERMINAL_PATH in config/settings.py.")


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
    return "terminal64.exe" in out.lower()


def kill_stray_terminals() -> None:
    """Best-effort: terminate every ``terminal64.exe`` (Windows).

    Used when *we* started the terminal via the Python API for a data pull and
    must not leave it owning the data directory for headless tester runs.
    """
    if os.name != "nt":
        return
    try:
        subprocess.run(  # noqa: S603,S607
            ["taskkill", "/IM", "terminal64.exe", "/T", "/F"],
            check=False,
            capture_output=True,
            **_subprocess_kwargs(),
        )
    except Exception:
        pass


def interactive_terminal_running() -> bool:
    """Public preflight helper: True when a terminal64.exe instance is running.

    Headless Strategy Tester runs share the terminal data directory with any
    open ``terminal64.exe``, so discovery should fail fast rather than burn
    the candidate budget on empty abort reports.
    """
    return _terminal_running()


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

        # Pool-managed portable instances own their data dir, so the busy
        # check (which sees ANY terminal64.exe, including other pool members)
        # and the process-wide lock are skipped for them.
        cmd = [str(paths.terminal_exe), f"/config:{ini_path}"]
        if self.portable:
            cmd.append("/portable")
        cancel_check = getattr(self, "_cancel_check", None)
        returncode = 0
        lane = _MT5_LOCK if self.exclusive else nullcontext()
        with lane:                          # sequential per shared install
            # Check *inside* the lock so a leftover/orphan terminal from a
            # prior run is visible before we spawn another /config instance.
            if self.exclusive and interactive_terminal_running():
                raise MT5RunnerError(
                    "MetaTrader 5 terminal is already running interactively. "
                    "Close the MT5 application and retry: headless tester runs "
                    "need exclusive use of the terminal data directory.")
            # Do not pass CREATE_NO_WINDOW here: terminal64 is a GUI app and
            # needs a normal process/window lifetime for ShutdownTerminal=1.
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
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
            finally:
                # If our Popen handle is still alive, ShutdownTerminal=1 failed —
                # tear the tree down so the next candidate is not INFRA-blocked.
                if proc.poll() is None:
                    _terminate_process_tree(proc)
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
