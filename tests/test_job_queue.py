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
