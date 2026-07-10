"""Multi-instance MetaTrader 5 pool: parallel headless tester runs.

The single-install MT5 lane is the factory's throughput ceiling: one tester
run at a time, and the interactive terminal must be closed. The way MT5 is
actually scaled in production farms is N *portable-mode* installs — each
terminal launched with ``/portable`` owns its own data directory next to its
exe, so N instances can run testers concurrently without corrupting each
other.

Configure ``MT5_INSTANCE_PATHS`` in ``config/settings.py`` with the
``terminal64.exe`` path of every portable install. Each instance is leased
exclusively to one job at a time via :meth:`MT5InstancePool.lease`; the pool
blocks callers until an instance frees up, so oversubscription is impossible
by construction. With no instances configured the worker keeps the legacy
single-lane behavior.

Provisioning a portable install (once per instance):
1. copy an existing MT5 install directory (or run the installer to a new dir);
2. start ``terminal64.exe /portable`` once, log in to the data source
   account, let history sync, close it;
3. add the exe path to ``MT5_INSTANCE_PATHS``.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

from config import settings
from factory.backtest.mt5_runner import MT5Paths


class MT5InstancePool:
    """Thread-safe exclusive leasing over a fixed set of portable installs."""

    def __init__(self, instances: Sequence[MT5Paths]):
        self._instances: List[MT5Paths] = list(instances)
        self._free: List[MT5Paths] = list(instances)
        self._cond = threading.Condition()

    @property
    def size(self) -> int:
        return len(self._instances)

    @contextmanager
    def lease(self, timeout: Optional[float] = None) -> Iterator[MT5Paths]:
        """Borrow an instance exclusively; blocks until one is available.

        Raises ``TimeoutError`` when ``timeout`` (seconds) elapses first.
        """
        with self._cond:
            if not self._cond.wait_for(lambda: bool(self._free),
                                       timeout=timeout):
                raise TimeoutError(
                    f"No MT5 instance free within {timeout}s "
                    f"(pool size {self.size})")
            instance = self._free.pop()
        try:
            yield instance
        finally:
            with self._cond:
                self._free.append(instance)
                self._cond.notify()

    def runner_for(self, instance: MT5Paths, *,
                   leverage: Optional[int] = None):
        """An :class:`MT5Runner` bound to one portable instance.

        Portable mode + non-exclusive execution: the per-instance data
        directory removes the shared-directory hazard the global lock and
        running-terminal check exist to guard against.
        """
        from factory.backtest.mt5_runner import MT5Runner
        return MT5Runner(paths=instance, leverage=leverage,
                         portable=True, exclusive=False)


def instances_from_settings() -> List[MT5Paths]:
    """Build instance descriptors from ``settings.MT5_INSTANCE_PATHS``.

    Silently skips configured paths that do not exist (an unprovisioned
    machine must not break the worker at import time).
    """
    out: List[MT5Paths] = []
    for raw in getattr(settings, "MT5_INSTANCE_PATHS", ()) or ():
        term = Path(raw)
        if not term.exists():
            continue
        out.append(MT5Paths(
            terminal_exe=term,
            metaeditor_exe=term.parent / "metaeditor64.exe",
            data_dir=term.parent,          # portable mode: data next to exe
        ))
    return out


def pool_from_settings() -> Optional[MT5InstancePool]:
    """The configured pool, or ``None`` for legacy single-lane behavior."""
    instances = instances_from_settings()
    if not instances:
        return None
    return MT5InstancePool(instances)
