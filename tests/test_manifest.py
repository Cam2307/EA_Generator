"""Reproducibility run manifests (factory.manifest + storage)."""
import numpy as np
import pandas as pd

from factory.manifest import build_manifest, data_fingerprint
from factory.storage import Storage


def _df(n=100, base=1.10):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="utc")
    close = base + np.arange(n) * 0.0001
    df = pd.DataFrame({"time": idx, "open": close, "high": close,
                       "low": close, "close": close, "volume": 1.0})
    df.attrs["source"] = "synthetic"
    return df


def test_data_fingerprint_detects_any_change():
    a = data_fingerprint(_df())
    assert a["bars"] == 100 and a["source"] == "synthetic"

    changed = _df()
    changed.loc[50, "close"] += 0.0001     # one revised bar
    b = data_fingerprint(changed)
    assert b["sha256"] != a["sha256"]

    # identical data -> identical fingerprint (deterministic)
    assert data_fingerprint(_df())["sha256"] == a["sha256"]


def test_build_manifest_contents():
    payload = {"symbol": "EURUSD", "timeframe": "M15", "batch_size": 50}
    m = build_manifest("disc_test01", payload, seed=12345, df=_df())
    assert m["job_id"] == "disc_test01"
    assert m["seed"] == 12345
    assert m["payload"]["symbol"] == "EURUSD"
    assert m["data"]["bars"] == 100 and m["data"]["sha256"]
    assert "SIMULATOR_DYNAMIC_COSTS" in m["settings"]
    assert "SIMULATOR_INTRABAR_MODE" in m["settings"]
    assert m["versions"]["numpy"] and m["versions"]["pandas"]


def test_manifest_roundtrip_through_storage(tmp_path):
    st = Storage(db_path=tmp_path / "t.db")
    m = build_manifest("disc_rt", {"symbol": "GBPUSD"}, seed=7, df=_df())
    st.save_run_manifest(m)
    loaded = st.get_run_manifest("disc_rt")
    assert loaded is not None
    assert loaded["seed"] == 7
    assert loaded["data"]["sha256"] == m["data"]["sha256"]
    assert st.get_run_manifest("missing") is None
