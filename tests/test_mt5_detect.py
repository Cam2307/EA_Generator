"""Filesystem MT5 path detection must not launch an interactive terminal."""
from __future__ import annotations

from pathlib import Path

import pytest

from factory.backtest import mt5_runner as mr


def test_read_origin_install_utf8(tmp_path: Path):
    origin = tmp_path / "origin.txt"
    origin.write_text("C:\\Program Files\\MetaTrader\n", encoding="utf-8")
    assert mr._read_origin_install(origin) == Path(r"C:\Program Files\MetaTrader")


def test_appdata_terminal_candidates(tmp_path: Path, monkeypatch):
    appdata = tmp_path / "AppData"
    term_root = appdata / "MetaQuotes" / "Terminal"
    data_dir = term_root / "ABCDEF0123456789"
    data_dir.mkdir(parents=True)
    install = tmp_path / "MetaTrader"
    install.mkdir()
    (install / "terminal64.exe").write_bytes(b"MZ")
    (install / "metaeditor64.exe").write_bytes(b"MZ")
    (data_dir / "origin.txt").write_text(str(install), encoding="utf-8")
    # Noise folders the scanner must ignore
    (term_root / "Common").mkdir()
    (term_root / "Help").mkdir()

    monkeypatch.setenv("APPDATA", str(appdata))
    found = mr._appdata_terminal_candidates()
    assert len(found) == 1
    assert found[0].terminal_exe == install / "terminal64.exe"
    assert found[0].data_dir == data_dir


def test_pick_preferred_vanilla_over_prop_firm():
    vanilla = mr.MT5Paths(
        terminal_exe=Path(r"C:\Program Files\MetaTrader\terminal64.exe"),
        metaeditor_exe=Path(r"C:\Program Files\MetaTrader\metaeditor64.exe"),
        data_dir=Path(r"C:\data\vanilla"),
    )
    prop = mr.MT5Paths(
        terminal_exe=Path(r"C:\Program Files\Goat Funded MT5 Terminal\terminal64.exe"),
        metaeditor_exe=Path(r"C:\Program Files\Goat Funded MT5 Terminal\metaeditor64.exe"),
        data_dir=Path(r"C:\data\prop"),
    )
    assert mr._pick_preferred([prop, vanilla]).data_dir == vanilla.data_dir


def test_detect_mt5_uses_settings_and_resolves_appdata(
        tmp_path: Path, monkeypatch):
    install = tmp_path / "MetaTrader"
    install.mkdir()
    term = install / "terminal64.exe"
    term.write_bytes(b"MZ")
    (install / "metaeditor64.exe").write_bytes(b"MZ")

    appdata = tmp_path / "AppData"
    data_dir = appdata / "MetaQuotes" / "Terminal" / "HASH1"
    data_dir.mkdir(parents=True)
    (data_dir / "origin.txt").write_text(str(install), encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setattr(mr.settings, "MT5_TERMINAL_PATH", str(term))
    monkeypatch.setattr(mr.settings, "MT5_METAEDITOR_PATH", None)

    paths = mr.detect_mt5()
    assert paths.terminal_exe.resolve() == term.resolve()
    assert paths.data_dir.resolve() == data_dir.resolve()


def test_detect_mt5_never_calls_initialize(monkeypatch):
    """Regression: initialize() opens interactive MT5 and poisons headless runs."""
    def _boom(*_a, **_k):
        raise AssertionError("MetaTrader5.initialize must not be used for detect_mt5")

    import sys
    fake = type(sys)("MetaTrader5")
    fake.initialize = _boom
    monkeypatch.setitem(sys.modules, "MetaTrader5", fake)
    monkeypatch.setattr(mr.settings, "MT5_TERMINAL_PATH", None)
    monkeypatch.setattr(mr, "_appdata_terminal_candidates", list)
    monkeypatch.setattr(mr, "_common_install_candidates", list)

    with pytest.raises(mr.MT5RunnerError, match="not detected"):
        mr.detect_mt5()
