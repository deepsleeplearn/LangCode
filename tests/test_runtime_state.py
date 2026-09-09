from __future__ import annotations

import pytest

from langcode_agent.storage.runtime_state import RuntimeLockTimeout, RuntimeStateStore


def test_runtime_state_memory_cancel_lifecycle() -> None:
    store = RuntimeStateStore(enabled="0")

    assert store.status()["backend"] == "memory"
    assert store.is_run_cancelled("s1", "r1") is False

    store.cancel_run("s1", "r1")
    assert store.is_run_cancelled("s1", "r1") is True

    store.forget_cancelled_run("s1", "r1")
    assert store.is_run_cancelled("s1", "r1") is False


def test_runtime_state_memory_clear_session() -> None:
    store = RuntimeStateStore(enabled="0")

    store.cancel_run("s1", "r1")
    store.mark_run_started("s1", "r1")
    store.clear_session("s1")

    assert store.is_run_cancelled("s1", "r1") is False


def test_runtime_state_clear_session_clears_local_state_with_redis() -> None:
    store = RuntimeStateStore(enabled="0")

    class FakeRedis:
        def scan_iter(self, _pattern):
            return []

        def delete(self, *_keys):
            return None

    store._redis = FakeRedis()
    store._memory_cancelled["s1"] = {"r1"}
    store._memory_active["s1"] = {"runId": "r1"}

    store.clear_session("s1")

    assert store.is_run_cancelled_local("s1", "r1") is False
    assert "s1" not in store._memory_active


def test_runtime_state_lock_is_noop_without_redis() -> None:
    store = RuntimeStateStore(enabled="0")

    with store.acquire_session_lock("s1"):
        store.cancel_run("s1", "r1")

    assert store.is_run_cancelled("s1", "r1") is True


def test_runtime_state_strict_redis_mode_reports_connection_failure() -> None:
    with pytest.raises(RuntimeError):
        RuntimeStateStore(redis_url="redis://127.0.0.1:1/0", prefix="langcode-test", enabled="1")


def test_runtime_state_retries_redis_after_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    store = RuntimeStateStore(enabled="0")
    store.enabled = "auto"
    store._redis_disabled_until = 0.0
    connected = object()

    def reconnect() -> None:
        store._redis = connected

    monkeypatch.setattr(store, "_connect_redis", reconnect)

    assert store.redis_available is True
    assert store._redis is connected
