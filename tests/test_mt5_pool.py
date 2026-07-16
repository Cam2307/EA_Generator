"""Multi-instance MT5 pool (jobs.mt5_pool) + worker integration."""
import threading
import time
from pathlib import Path

import pytest

from factory.backtest.mt5_runner import MT5Paths
from jobs.mt5_pool import MT5InstancePool, instances_from_settings


def _paths(tag: str) -> MT5Paths:
    base = Path(f"C:/fake/{tag}")
    return MT5Paths(terminal_exe=base / "terminal64.exe",
                    metaeditor_exe=base / "metaeditor64.exe",
                    data_dir=base)


def test_lease_grants_exclusive_instances():
    pool = MT5InstancePool([_paths("a"), _paths("b")])
    assert pool.size == 2
    with pool.lease() as first:
        with pool.lease() as second:
            assert first.data_dir != second.data_dir
        # released instance is reusable
        with pool.lease() as again:
            assert again.data_dir != first.data_dir


def test_lease_blocks_until_release_and_times_out():
    pool = MT5InstancePool([_paths("solo")])
    order = []

    def worker():
        with pool.lease(timeout=5):
            order.append("second")

    with pool.lease():
        order.append("first")
        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.1)
        assert order == ["first"]         # still blocked while we hold it
        with pytest.raises(TimeoutError):
            with pool.lease(timeout=0.05):
                pass
    t.join(timeout=5)
    assert order == ["first", "second"]


def test_concurrent_leases_never_oversubscribe():
    pool = MT5InstancePool([_paths("a"), _paths("b"), _paths("c")])
    active = []
    peak = []
    lock = threading.Lock()

    def worker(i):
        with pool.lease(timeout=10) as inst:
            with lock:
                active.append(inst.data_dir)
                peak.append(len(active))
                assert len(set(active)) == len(active)   # no duplicates
            time.sleep(0.02)
            with lock:
                active.remove(inst.data_dir)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert max(peak) <= pool.size


def test_runner_for_binds_portable_nonexclusive():
    pool = MT5InstancePool([_paths("a")])
    with pool.lease() as inst:
        runner = pool.runner_for(inst, leverage=200)
    assert runner.portable is True
    assert runner.exclusive is False
    assert runner.leverage == 200
    assert runner.paths.data_dir == _paths("a").data_dir


def test_worker_parallel_mt5_generation(tmp_path, monkeypatch):
    """Discovery with engine=mt5 + a pool >1 validates via leased runners."""
    import time
    from datetime import datetime, timezone

    from factory.models import BacktestMetrics, JobStatus
    from factory.storage import Storage
    from jobs.worker import JobQueue

    YEAR = 365.25 * 86400

    class StubRunner:
        name = "mt5"
        run_count = 0

        def run(self, strategy, start, end, params_override=None,
                deposit=10_000.0):
            StubRunner.run_count += 1
            years = max((end - start).total_seconds() / YEAR, 1e-9)
            return BacktestMetrics(
                net_profit=deposit * 0.25 * years, initial_deposit=deposit,
                start_ts=start.timestamp(), end_ts=end.timestamp(),
                max_dd_pct=5.0, trade_count=50, profit_factor=1.5)

    class ScreenStub(StubRunner):
        name = "simulator"

    class StubPool(MT5InstancePool):
        def __init__(self, instances):
            super().__init__(instances)
            self.leased_runners = 0

        def runner_for(self, instance, leverage=None):
            self.leased_runners += 1
            return StubRunner()

    storage = Storage(tmp_path / "pool.db")
    queue = JobQueue(storage)
    pool = StubPool([_paths("a"), _paths("b")])
    queue._mt5_pool = pool
    engines = {"simulator": ScreenStub(), "mt5": StubRunner()}
    monkeypatch.setattr(queue, "_make_engine", lambda name: engines[name])
    # Preflight must not see a real interactive terminal on the developer machine.
    monkeypatch.setattr(
        "jobs.worker.interactive_terminal_running", lambda: False)

    job_id = "mt5_pool_job"
    assert queue.submit_discovery(job_id, {
        "symbol": "EURUSD", "timeframe": "H1", "engine": "mt5",
        "batch_size": 4, "target_survivors": 2, "max_candidates": 8,
        "genetic": False, "seed": 7,
        "start": datetime(2023, 1, 1, tzinfo=timezone.utc).isoformat(),
        "end": datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat(),
        "validation_level": 1, "data_source": "synthetic",
        "wfo_train_months": 2, "wfo_test_months": 1, "wfo_windows": 1,
    })
    deadline = time.time() + 120
    while time.time() < deadline:
        job = storage.get_job(job_id)
        if job.status in (JobStatus.DONE, JobStatus.FAILED,
                          JobStatus.CANCELLED):
            break
        time.sleep(0.05)
    assert job.status == JobStatus.DONE, job.error
    assert pool.leased_runners >= 1          # validations went through the pool
    reports = storage.list_validated(passed_only=False)
    assert reports and any(r.passed for r in reports)
    assert all(r.engine == "mt5" for r in reports)


def test_instances_from_settings_skips_missing(tmp_path, monkeypatch):
    from config import settings
    real = tmp_path / "inst1" / "terminal64.exe"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"")
    monkeypatch.setattr(settings, "MT5_INSTANCE_PATHS",
                        (str(real), r"C:\does\not\exist\terminal64.exe"),
                        raising=False)
    instances = instances_from_settings()
    assert len(instances) == 1
    assert instances[0].terminal_exe == real
    assert instances[0].data_dir == real.parent
