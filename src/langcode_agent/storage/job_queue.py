from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Iterator
from uuid import uuid4

from .runtime_state import DEFAULT_REDIS_URL, _falsy, _redact_redis_url, _truthy


DEFAULT_QUEUE_NAME = "default"


@dataclass
class QueueStatus:
    backend: str
    enabled: bool
    available: bool
    redis_url: str
    queue: str
    error: str = ""


class JobQueue:
    """Redis-backed job queue for long-running LangCode work.

    The web server enqueues work and streams events from a per-job Redis Stream.
    Worker processes pop jobs from a Redis list, execute model/tool/TTS work, and
    publish structured events back to the job stream.
    """

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        prefix: str | None = None,
        queue_name: str | None = None,
        enabled: str | None = None,
    ) -> None:
        self.redis_url = (redis_url or os.getenv("LANGCODE_REDIS_URL") or DEFAULT_REDIS_URL).strip()
        self.prefix = (prefix or os.getenv("LANGCODE_REDIS_PREFIX") or "langcode").strip() or "langcode"
        self.queue_name = (queue_name or os.getenv("LANGCODE_QUEUE_NAME") or DEFAULT_QUEUE_NAME).strip() or DEFAULT_QUEUE_NAME
        self.enabled = (enabled or os.getenv("LANGCODE_QUEUE_ENABLED") or "0").strip().lower()
        self._redis: Any | None = None
        self._error = ""
        self._connect()

    @property
    def available(self) -> bool:
        return self._redis is not None

    def status(self) -> dict:
        state = QueueStatus(
            backend="redis" if self.available else "direct",
            enabled=not _falsy(self.enabled),
            available=self.available,
            redis_url=_redact_redis_url(self.redis_url),
            queue=self.queue_name,
            error=self._error,
        )
        return {
            "backend": state.backend,
            "enabled": state.enabled,
            "available": state.available,
            "redisUrl": state.redis_url,
            "queue": state.queue,
            "error": state.error,
        }

    def enqueue(self, kind: str, payload: dict) -> str:
        self._require_redis()
        job_id = uuid4().hex
        job = {
            "id": job_id,
            "kind": kind,
            "payload": payload,
            "createdAt": time.time(),
        }
        self._redis.rpush(self._jobs_key(), _json_dumps(job))
        self.publish_event(job_id, {"type": "queued", "ok": True, "jobId": job_id, "kind": kind})
        self._redis.expire(self._events_key(job_id), 24 * 60 * 60)
        return job_id

    def reserve(self, *, timeout_seconds: int = 1) -> dict | None:
        self._require_redis()
        item = self._redis.blpop(self._jobs_key(), timeout=timeout_seconds)
        if not item:
            return None
        _key, raw = item
        return json.loads(_decode(raw))

    def publish_event(self, job_id: str, event: dict) -> None:
        self._require_redis()
        self._redis.xadd(self._events_key(job_id), {"event": _json_dumps(event)}, maxlen=2000, approximate=True)

    def publish_done(self, job_id: str, event: dict | None = None) -> None:
        done_event = event or {"type": "done", "ok": True}
        self.publish_event(job_id, done_event)
        self._redis.expire(self._events_key(job_id), 10 * 60)

    def iter_events(self, job_id: str, *, block_ms: int = 4000) -> Iterator[dict | None]:
        self._require_redis()
        stream_key = self._events_key(job_id)
        last_id = "0-0"
        while True:
            rows = self._redis.xread({stream_key: last_id}, block=block_ms, count=20)
            if not rows:
                yield None
                continue
            for _stream, messages in rows:
                for message_id, fields in messages:
                    last_id = _decode(message_id)
                    raw_event = fields.get("event") if isinstance(fields, dict) else None
                    if raw_event is None:
                        continue
                    yield json.loads(_decode(raw_event))

    def _connect(self) -> None:
        if _falsy(self.enabled):
            self._error = "disabled"
            return
        try:
            import redis  # type: ignore
        except Exception as exc:
            self._error = f"redis package unavailable: {exc}"
            if _truthy(self.enabled):
                raise RuntimeError(self._error) from exc
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
            self._error = ""
        except Exception as exc:
            self._redis = None
            self._error = f"redis unavailable: {exc}"
            if _truthy(self.enabled):
                raise RuntimeError(self._error) from exc

    def _require_redis(self) -> None:
        if self._redis is None:
            raise RuntimeError(self._error or "job queue is unavailable")

    def _jobs_key(self) -> str:
        return f"{self.prefix}:queue:{self.queue_name}:jobs"

    def _events_key(self, job_id: str) -> str:
        return f"{self.prefix}:queue:{self.queue_name}:events:{job_id}"


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _json_dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
