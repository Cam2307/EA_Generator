from __future__ import annotations

from pathlib import Path

from factory.storage import Storage
from jobs import orchestrator as orch_mod


def test_clear_stale_lock_removes_dead_pid(tmp_path: Path, monkeypatch) -> None:
    lock = tmp_path / "discovery_orchestrator.lock"
    lock.write_text("424242", encoding="utf-8")
    monkeypatch.setattr(orch_mod, "LOCK_PATH", lock)
    monkeypatch.setattr(orch_mod, "_pid_alive", lambda _pid: False)

    assert orch_mod.clear_stale_orchestrator_lock() is True
    assert not lock.exists()


def test_clear_stale_lock_keeps_live_pid(tmp_path: Path, monkeypatch) -> None:
    lock = tmp_path / "discovery_orchestrator.lock"
    lock.write_text("4242", encoding="utf-8")
    monkeypatch.setattr(orch_mod, "LOCK_PATH", lock)
    monkeypatch.setattr(orch_mod, "_pid_alive", lambda pid: pid == 4242)

    assert orch_mod.clear_stale_orchestrator_lock() is False
    assert lock.exists()


def test_python_executable_prefers_project_venv(monkeypatch, tmp_path: Path) -> None:
    venv_py = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_py.parent.mkdir(parents=True)
    venv_py.touch()
    system_py = tmp_path / "system" / "python.exe"
    system_py.parent.mkdir(parents=True)
    system_py.touch()
    monkeypatch.setattr(orch_mod, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(orch_mod.sys, "executable", str(system_py))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)

    assert orch_mod._python_executable() == str(venv_py)


def test_python_executable_windowless_prefers_pythonw(monkeypatch, tmp_path: Path) -> None:
    venv_py = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_pyw = tmp_path / ".venv" / "Scripts" / "pythonw.exe"
    venv_py.parent.mkdir(parents=True)
    venv_py.touch()
    venv_pyw.touch()
    monkeypatch.setattr(orch_mod, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(orch_mod.sys, "executable", str(venv_py))
    monkeypatch.setattr(orch_mod.os, "name", "nt")

    assert orch_mod._python_executable(windowless=True) == str(venv_pyw)


def test_sync_agent_with_live_lock(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "sync.db"
    storage = Storage(db)
    lock = tmp_path / "discovery_orchestrator.lock"
    lock.write_text("4242", encoding="utf-8")
    monkeypatch.setattr(orch_mod, "LOCK_PATH", lock)
    monkeypatch.setattr(orch_mod, "_pid_alive", lambda pid: pid == 4242)
    storage.update_agent_state(
        enabled=1,
        status="starting",
        pid=None,
        message="Starting discovery…",
    )

    assert orch_mod.sync_agent_with_orchestrator_lock(storage) is True
    state = storage.get_agent_state()
    assert state["status"] == "running"
    assert int(state["pid"]) == 4242


def test_start_clears_stale_lock_before_spawn(tmp_path: Path, monkeypatch) -> None:
    lock = tmp_path / "discovery_orchestrator.lock"
    lock.write_text("99999", encoding="utf-8")
    monkeypatch.setattr(orch_mod, "LOCK_PATH", lock)
    monkeypatch.setattr(orch_mod, "_pid_alive", lambda _pid: False)
    spawned: list[list[str]] = []

    def _fake_popen(cmd, **kwargs):
        spawned.append(cmd)
        return object()

    monkeypatch.setattr(orch_mod.subprocess, "Popen", _fake_popen)

    assert orch_mod.start_orchestrator_process() is True
    assert not lock.exists()
    assert spawned


def test_recover_stuck_starting_agent_retries_spawn(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "recover.db"
    storage = Storage(db)
    storage.update_agent_state(
        enabled=1,
        status="starting",
        pid=99999,
        updated_at=0.0,
        spawn_attempts=0,
        message="Starting discovery…",
    )
    lock = tmp_path / "discovery_orchestrator.lock"
    lock.write_text("99999", encoding="utf-8")
    monkeypatch.setattr(orch_mod, "LOCK_PATH", lock)
    monkeypatch.setattr(orch_mod, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(orch_mod.subprocess, "Popen", lambda *a, **k: object())

    assert orch_mod.recover_stuck_starting_agent(storage) is True
    assert not lock.exists()
    state = storage.get_agent_state()
    assert int(state.get("spawn_attempts") or 0) == 1
