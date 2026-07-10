from __future__ import annotations

import tempfile
from pathlib import Path

from factory.models import Job, JobStatus
from factory.storage import Storage
from jobs import worker as worker_mod
from jobs.worker import JobQueue


def test_reconcile_keeps_job_with_live_runner_pid(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "reconcile.db"
    storage = Storage(db)
    storage.upsert_job(Job(id="live_job", kind="discovery", status=JobStatus.RUNNING, runner_pid=4242))

    monkeypatch.setattr(worker_mod, "_pid_alive", lambda pid: pid == 4242)

    queue = JobQueue(storage)
    job = storage.get_job("live_job")
    assert job is not None
    assert job.status == JobStatus.RUNNING


def test_reconcile_cancels_job_with_dead_runner_pid(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "reconcile_dead.db"
    storage = Storage(db)
    storage.upsert_job(Job(id="dead_job", kind="discovery", status=JobStatus.RUNNING, runner_pid=99999))

    monkeypatch.setattr(worker_mod, "_pid_alive", lambda _pid: False)

    JobQueue(storage)
    job = storage.get_job("dead_job")
    assert job is not None
    assert job.status == JobStatus.CANCELLED
