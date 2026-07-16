"""OHLC miss-cache and lightweight peek_source behaviour."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from factory import data as data_mod


@pytest.fixture(autouse=True)
def _clean_caches():
    data_mod.clear_range_cache()
    yield
    data_mod.clear_range_cache()


def test_mark_unavailable_skips_repeated_mt5_attempts(monkeypatch):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 2, 1, tzinfo=timezone.utc)
    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        return None

    monkeypatch.setattr(data_mod, "_try_mt5", boom)
    data_mod.mark_unavailable("EURGBP", "M1", start, end)

    with pytest.raises(RuntimeError, match="cached miss"):
        data_mod.load_ohlc("EURGBP", "M1", start, end, allow_synthetic=False)
    assert calls["n"] == 0


def test_failed_m1_load_is_cached(monkeypatch, tmp_path):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 2, 1, tzinfo=timezone.utc)
    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        return None

    monkeypatch.setattr(data_mod, "_try_mt5", boom)
    monkeypatch.setattr(data_mod.settings, "DATA_DIR", tmp_path)
    # Ensure no parquet cache exists for this range.
    assert not list(tmp_path.glob("ohlc_*.parquet"))

    with pytest.raises(RuntimeError):
        data_mod.load_ohlc("EURGBP", "M1", start, end, allow_synthetic=False)
    with pytest.raises(RuntimeError, match="cached miss"):
        data_mod.load_ohlc("EURGBP", "M1", start, end, allow_synthetic=False)
    assert calls["n"] == 1


def test_peek_source_uses_parquet_without_load(monkeypatch, tmp_path):
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 2, 1, tzinfo=timezone.utc)
    cache = tmp_path / "ohlc_EURGBP_M15_20240101_20240201.parquet"
    cache.write_bytes(b"x")
    monkeypatch.setattr(data_mod.settings, "DATA_DIR", tmp_path)

    def fail_load(*_a, **_k):
        raise AssertionError("peek_source must not load OHLC when cache exists")

    monkeypatch.setattr(data_mod, "load_ohlc", fail_load)
    assert data_mod.peek_source("EURGBP", "M15", start, end) == "cache"


def test_pack_unpack_ohlc_blob_roundtrip():
    from jobs.worker import _pack_ohlc_blob, _unpack_ohlc_blob
    import pandas as pd

    df = pd.DataFrame({"time": [1], "open": [1.0], "high": [1.0],
                       "low": [1.0], "close": [1.0], "volume": [1.0]})
    blob = _pack_ohlc_blob([("EURGBP", "M15", df), ("EURGBP", "M1", df)])
    frames = _unpack_ohlc_blob(blob)
    assert len(frames) == 2
    assert frames[0][0] == "EURGBP" and frames[0][1] == "M15"
    assert frames[1][1] == "M1"

    # Legacy single-triple blob still unpacks.
    import pickle
    legacy = pickle.dumps(("EURUSD", "H1", df))
    one = _unpack_ohlc_blob(legacy)
    assert len(one) == 1 and one[0][1] == "H1"
