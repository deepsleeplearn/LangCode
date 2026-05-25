from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import os
import threading
import time
from typing import Any
from uuid import uuid4


DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
DEFAULT_PREFIX = "langcode"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "y"}


def _falsy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"0", "false", "no", "off", "n"}


@dataclass
class RuntimeStateStatus:
    backend: str
    redis_url: str
    available: bool
    error: str = ""


class RuntimeLockTimeout(TimeoutError):
    pass


class RuntimeLease(AbstractContextManager):
    def __init__(self, store: "RuntimeStateStore", key: str, token: str, active: bool) -> None:
        self._store = store
        self._key = key
        self._token = token
        self._active = active

    def __enter__(self) -> "RuntimeLease":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

    def release(self) -> None:
        if not self._active:
            return
        self._active = False
        self._store._release_redis_lock(self._key, self._token)


class RuntimeStateStore:
    """Shared runtime state for web workers.

    SQLite remains the source of truth for durable conversations. This store is
    intentionally limited to short-lived runtime coordination: cancellation
    flags, active run markers, and session locks. Redis is used when available;
    otherwise the class falls back to process-local memory so single-user local
    development keeps working.
    """

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        prefix: str | None = None,
        enabled: str | None = None,
    ) -> None:
        self.redis_url = (redis_url or os.getenv("LANGCODE_REDIS_URL") or DEFAULT_REDIS_URL).strip()
        self.prefix = (prefix or os.getenv("LANGCODE_REDIS_PREFIX") or DEFAULT_PREFIX).strip() or DEFAULT_PREFIX
        self.enabled = (enabled or os.getenv("LANGCODE_REDIS_ENABLED") or "auto").strip().lower()
        self._memory_lock = threading.RLock()
        self._memory_cancelled: dict[str, set[str]] = {}
        self._memory_active: dict[str, dict[str, Any]] = {}
        self._redis: Any | None = None
        self._redis_error = ""
        self._connect_redis()

    @property
    def redis_available(self) -> bool:
        return self._redis is not None

    def status(self) -> dict:
        state = RuntimeStateStatus(
            backend="redis" if self.redis_available else "memory",
            redis_url=_redact_redis_url(self.redis_url),
            available=self.redis_available,
            error=self._redis_error,
        )
        return {
            "backend": state.backend,
            "redisUrl": state.redis_url,
            "available": state.available,
            "error": state.error,
        }

    def cancel_run(self, session_id: str, run_id: str, *, ttl_seconds: int = 24 * 60 * 60) -> None:
        if not session_id or not run_id:
            return
        if self._redis is not None:
            try:
                self._redis.set(self._key("cancelled", session_id, run_id), "1", ex=ttl_seconds)
                return
            except Exception as exc:
                self._disable_redis(exc)
        with self._memory_lock:
            self._memory_cancelled.setdefault(session_id, set()).add(run_id)

    def is_run_cancelled(self, session_id: str, run_id: str | None) -> bool:
        if not session_id or not run_id:
            return False
        if self._redis is not None:
            try:
                return bool(self._redis.exists(self._key("cancelled", session_id, run_id)))
            except Exception as exc:
                self._disable_redis(exc)
        with self._memory_lock:
            return run_id in self._memory_cancelled.get(session_id, set())

    def forget_cancelled_run(self, session_id: str, run_id: str | None) -> None:
        if not session_id or not run_id:
            return
        if self._redis is not None:
            try:
                self._redis.delete(self._key("cancelled", session_id, run_id))
                return
            except Exception as exc:
                self._disable_redis(exc)
        with self._memory_lock:
            cancelled = self._memory_cancelled.get(session_id)
            if not cancelled:
                return
            cancelled.discard(run_id)
            if not cancelled:
                self._memory_cancelled.pop(session_id, None)

    def clear_session(self, session_id: str) -> None:
        if not session_id:
            return
        if self._redis is not None:
            try:
                keys = list(self._redis.scan_iter(self._key("cancelled", session_id, "*")))
                keys.extend(list(self._redis.scan_iter(self._key("active", session_id, "*"))))
                keys.append(self._key("lock", session_id))
                if keys:
                    self._redis.delete(*keys)
                return
            except Exception as exc:
                self._disable_redis(exc)
        with self._memory_lock:
            self._memory_cancelled.pop(session_id, None)
            self._memory_active.pop(session_id, None)

    def mark_run_started(self, session_id: str, run_id: str | None, *, ttl_seconds: int = 2 * 60 * 60) -> None:
        if not session_id or not run_id:
            return
        payload = {"runId": run_id, "startedAt": time.time()}
        if self._redis is not None:
            try:
                self._redis.set(self._key("active", session_id, run_id), _json_dumps(payload), ex=ttl_seconds)
                return
            except Exception as exc:
                self._disable_redis(exc)
        with self._memory_lock:
            self._memory_active[session_id] = payload

    def mark_run_finished(self, session_id: str, run_id: str | None) -> None:
        if not session_id or not run_id:
            return
        if self._redis is not None:
            try:
                self._redis.delete(self._key("active", session_id, run_id))
                return
            except Exception as exc:
                self._disable_redis(exc)
        with self._memory_lock:
            active = self._memory_active.get(session_id)
            if active and active.get("runId") == run_id:
                self._memory_active.pop(session_id, None)

    def acquire_session_lock(
        self,
        session_id: str,
        *,
        wait_timeout_seconds: float = 30.0,
        ttl_seconds: int = 2 * 60 * 60,
    ) -> RuntimeLease:
        if self._redis is None or not session_id:
            return RuntimeLease(self, "", "", False)
        key = self._key("lock", session_id)
        token = uuid4().hex
        deadline = time.monotonic() + max(0.1, wait_timeout_seconds)
        sleep_seconds = 0.05
        while time.monotonic() < deadline:
            try:
                if self._redis.set(key, token, nx=True, ex=ttl_seconds):
                    return RuntimeLease(self, key, token, True)
            except Exception as exc:
                self._disable_redis(exc)
                return RuntimeLease(self, "", "", False)
            time.sleep(sleep_seconds)
            sleep_seconds = min(0.5, sleep_seconds * 1.5)
        raise RuntimeLockTimeout(f"Session {session_id} is busy")

    def _release_redis_lock(self, key: str, token: str) -> None:
        if self._redis is None or not key:
            return
        script = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
  return redis.call("DEL", KEYS[1])
end
return 0
"""
        try:
            self._redis.eval(script, 1, key, token)
        except Exception:
            try:
                value = self._redis.get(key)
                if _decode(value) == token:
                    self._redis.delete(key)
            except Exception as exc:
                self._disable_redis(exc)

    def _connect_redis(self) -> None:
        if _falsy(self.enabled):
            self._redis_error = "disabled"
            return
        try:
            import redis  # type: ignore
        except Exception as exc:
            self._redis_error = f"redis package unavailable: {exc}"
            if _truthy(self.enabled):
                raise RuntimeError(self._redis_error) from exc
            return
        try:
            client = redis.Redis.from_url(
                self.redis_url,
                socket_connect_timeout=0.3,
                socket_timeout=1.0,
                decode_responses=True,
            )
            client.ping()
            self._redis = client
            self._redis_error = ""
        except Exception as exc:
            self._redis = None
            self._redis_error = f"redis unavailable: {exc}"
            if _truthy(self.enabled):
                raise RuntimeError(self._redis_error) from exc

    def _key(self, *parts: str) -> str:
        escaped = [str(part).replace(":", "_") for part in parts]
        return ":".join([self.prefix, *escaped])

    def _disable_redis(self, exc: Exception) -> None:
        self._redis = None
        self._redis_error = f"redis runtime failure: {exc}"


def _redact_redis_url(url: str) -> str:
    if "@" not in url:
        return url
    scheme, rest = url.split("://", 1) if "://" in url else ("", url)
    _credentials, host = rest.rsplit("@", 1)
    return f"{scheme}://***@{host}" if scheme else f"***@{host}"


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
