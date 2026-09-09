from __future__ import annotations

import pytest

from langcode_agent.storage.job_queue import JobQueue


def test_job_queue_disabled_status_uses_direct_backend() -> None:
    queue = JobQueue(enabled="0")

    status = queue.status()

    assert status["enabled"] is False
    assert status["available"] is False
    assert status["backend"] == "direct"


def test_job_queue_disabled_cannot_enqueue() -> None:
    queue = JobQueue(enabled="0")

    with pytest.raises(RuntimeError):
        queue.enqueue("chat_stream", {"sessionId": "s1"})


class _RecordingQueue:
    """Minimal JobQueue stand-in that records what a worker publishes."""

    def __init__(self, jobs: list[dict] | None = None, *, failing_done: int = 0) -> None:
        self.queue_name = "test"
        self.jobs = list(jobs or [])
        self.events: list[tuple[str, dict]] = []
        self.done: list[tuple[str, dict | None]] = []
        self.failing_done = failing_done
        self.reserved = 0

    def reserve(self, *, timeout_seconds: int = 1) -> dict | None:
        self.reserved += 1
        return self.jobs.pop(0) if self.jobs else None

    def publish_event(self, job_id: str, event: dict) -> None:
        self.events.append((job_id, event))

    def publish_done(self, job_id: str, event: dict | None = None) -> None:
        if self.failing_done > 0:
            self.failing_done -= 1
            raise RuntimeError("redis went away")
        self.done.append((job_id, event))


def test_worker_always_publishes_a_terminal_event(tmp_path, monkeypatch) -> None:
    """Item 6: a publish_done that itself fails used to hang the client forever."""
    from langcode_agent.interfaces import worker as worker_module

    class _App:
        def chat_events(self, _payload):
            raise RuntimeError("model exploded")

    queue = _RecordingQueue(failing_done=1)

    worker_module._dispatch_job(_App(), queue, {"id": "job-1", "kind": "chat_stream", "payload": {}})

    assert [job_id for job_id, _event in queue.done] == ["job-1"]
    terminal = queue.done[0][1]
    assert terminal["type"] == "error"
    assert terminal["ok"] is False


def test_worker_publishes_the_exception_terminal_event_when_redis_is_healthy() -> None:
    from langcode_agent.interfaces import worker as worker_module

    class _App:
        def chat_events(self, _payload):
            raise RuntimeError("model exploded")

    queue = _RecordingQueue()

    worker_module._dispatch_job(_App(), queue, {"id": "job-2", "kind": "chat_stream", "payload": {}})

    assert len(queue.done) == 1
    job_id, terminal = queue.done[0]
    assert job_id == "job-2"
    assert terminal["type"] == "error"
    assert terminal["ok"] is False
    assert terminal["error"]


def test_worker_unknown_job_kind_still_terminates() -> None:
    from langcode_agent.interfaces import worker as worker_module

    queue = _RecordingQueue()

    worker_module._dispatch_job(object(), queue, {"id": "job-3", "kind": "nonsense", "payload": {}})

    assert queue.done[0][1]["error"].startswith("Unknown job kind")


def test_worker_loop_reserves_only_as_many_jobs_as_it_can_run() -> None:
    """Item 6: unbounded prefetch hid reserved jobs from every other worker."""
    import threading

    from langcode_agent.interfaces import worker as worker_module

    started = threading.Event()
    release = threading.Event()
    handled: list[str] = []

    class _BlockingQueue(_RecordingQueue):
        def reserve(self, *, timeout_seconds: int = 1) -> dict | None:
            job = super().reserve(timeout_seconds=timeout_seconds)
            if job is None and started.is_set():
                # nothing left to hand out; let the loop idle
                stop.set()
            return job

    def fake_handle(_app, _queue, job):
        handled.append(str(job.get("id")))
        started.set()
        release.wait(timeout=5)

    queue = _BlockingQueue([{"id": f"job-{index}", "kind": "chat_stream", "payload": {}} for index in range(4)])
    stop = threading.Event()

    original = worker_module._handle_job
    worker_module._handle_job = fake_handle
    try:
        loop = threading.Thread(
            target=worker_module.run_worker_loop,
            args=(object(), queue),
            kwargs={"concurrency": 1, "poll_seconds": 1, "stop": stop, "install_signal_handlers": False},
        )
        loop.start()
        assert started.wait(timeout=5)
        # concurrency is 1, so exactly one job may be in flight and reserved
        assert queue.reserved == 1
        assert handled == ["job-0"]
        release.set()
        stop.set()
        loop.join(timeout=10)
        assert not loop.is_alive()
    finally:
        worker_module._handle_job = original
        release.set()
        stop.set()


def test_worker_loop_stops_on_the_stop_event_and_drains_the_pool() -> None:
    import threading

    from langcode_agent.interfaces import worker as worker_module

    queue = _RecordingQueue()
    stop = threading.Event()
    stop.set()

    worker_module.run_worker_loop(
        object(), queue, concurrency=2, poll_seconds=1, stop=stop, install_signal_handlers=False
    )

    assert queue.reserved == 0
    assert queue.done == []
