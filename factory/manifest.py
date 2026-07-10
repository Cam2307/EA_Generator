"""Reproducibility manifests for discovery runs.

Every discovery job gets a manifest recording *everything needed to re-derive
its results*: the concrete RNG seed, the full job payload, a fingerprint of
the exact bar data the run saw (so a silently-changed cache or re-downloaded
history is detectable), the engine-realism settings in force, and the library
versions. Stored in SQLite next to the job so any strategy in the gallery can
be traced back to a reproducible run.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

from config import settings

# Settings that change simulation results. Captured verbatim so a manifest
# from an old run still documents the realism knobs it ran under.
_RESULT_AFFECTING_SETTINGS = (
    "IS_OOS_SPLIT", "OPT_SAMPLES", "WFO_OPT_SAMPLES", "WFO_WINDOWS",
    "WFO_TRAIN_MONTHS", "WFO_TEST_MONTHS", "WFO_MODES",
    "NEIGHBORHOOD_STABILITY", "NEIGHBOR_SAMPLES", "NEIGHBOR_TOP_K",
    "MC_ENABLED", "MC_RUNS", "MC_SPREAD_MAX_POINTS", "MC_SLIPPAGE_MAX_POINTS",
    "MC_PARAM_CHANGE_PROB", "MC_PARAM_MAX_STEPS", "MC_SKIP_ENTRY_PROB",
    "MC_START_JITTER_BARS", "MC_RESAMPLES", "MC_MIN_PROFITABLE",
    "MC_MAX_DD_P95", "SIMULATOR_DYNAMIC_COSTS", "SIMULATOR_INTRABAR_MODE",
    "DEFAULT_DEPOSIT",
)


def data_fingerprint(df: pd.DataFrame) -> dict:
    """Compact, order-sensitive fingerprint of a bar series.

    Hashes the raw close/time arrays so *any* change to the underlying data
    (revised history, different feed, silently regrown cache) changes the
    fingerprint even when bar counts match.
    """
    if len(df) == 0:
        return {"bars": 0, "sha256": "", "first_ts": None, "last_ts": None}
    times = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    closes = np.ascontiguousarray(df["close"].to_numpy(dtype=np.float64))
    ts = np.ascontiguousarray(
        times.to_numpy().astype("datetime64[s]").astype("int64"))
    h = hashlib.sha256()
    h.update(ts.tobytes())
    h.update(closes.tobytes())
    return {
        "bars": int(len(df)),
        "sha256": h.hexdigest(),
        "first_ts": int(ts[0]),
        "last_ts": int(ts[-1]),
        "source": str(df.attrs.get("source", "unknown")),
    }


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=settings.PROJECT_ROOT,
            capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def build_manifest(job_id: str, payload: dict, seed: int,
                   df: pd.DataFrame) -> dict:
    """Assemble the reproducibility manifest for one discovery run."""
    import pydantic

    return {
        "job_id": job_id,
        "seed": int(seed),
        "payload": dict(payload),
        "data": data_fingerprint(df),
        "settings": {name: getattr(settings, name, None)
                     for name in _RESULT_AFFECTING_SETTINGS},
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pydantic": pydantic.VERSION,
        },
        "git_commit": _git_commit(),
        "created_at": time.time(),
    }
