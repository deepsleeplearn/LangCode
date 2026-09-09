"""Shared NDJSON event producer for the TTS streaming endpoints.

``/api/tts/stream`` exists in three places (web.py, worker.py, voice_worker.py)
and each used to inline the same event-building loop. ``iter_tts_events`` is the
single implementation of that wire format so the copies cannot drift, and
``TtsTurnRegistry`` is the single implementation of the barge-in bookkeeping the
web process and the voice worker both need.
"""

from __future__ import annotations

import base64
from collections import OrderedDict
from collections.abc import Callable, Iterator
import inspect
import logging
import threading
import time
from typing import Any

from .tts import FALLBACK_NOTICE


logger = logging.getLogger("langcode.voice")

# How many sessions keep a "newest turn" entry, and how many cancelled turn ids
# are remembered per session. Both caps only bound memory - a forgotten turn
# merely plays one more audio chunk than it had to.
TTS_TRACKED_SESSIONS = 64
TTS_CANCELLED_TURN_MEMORY = 32


class TtsTurnRegistry:
    """Which TTS turn of a session is still worth speaking.

    Full duplex means the user can start the next turn while the previous answer
    is still being spoken. Two things end a turn early: a newer ``turnId`` for
    the same session (``claim``) and an explicit ``cancel`` from the client.
    Producers ask ``is_stale`` between chunks. Deliberately in-memory,
    process-local and bounded: a lost entry only ever costs one extra chunk of
    audio, never correctness.
    """

    def __init__(self) -> None:
        self._latest: OrderedDict[str, str] = OrderedDict()
        self._cancelled: dict[str, OrderedDict[str, bool]] = {}
        self._lock = threading.Lock()

    def claim(self, session_id: str, turn_id: str) -> None:
        """Make ``turn_id`` the newest turn of ``session_id``; older ones go stale."""
        session_id, turn_id = _clean(session_id), _clean(turn_id)
        if not session_id or not turn_id:
            return
        with self._lock:
            self._latest[session_id] = turn_id
            self._latest.move_to_end(session_id)
            while len(self._latest) > TTS_TRACKED_SESSIONS:
                evicted, _ = self._latest.popitem(last=False)
                self._cancelled.pop(evicted, None)

    def cancel(self, session_id: str, turn_id: str) -> bool:
        """Mark one turn cancelled. ``False`` when either id is missing."""
        session_id, turn_id = _clean(session_id), _clean(turn_id)
        if not session_id or not turn_id:
            return False
        with self._lock:
            cancelled = self._cancelled.setdefault(session_id, OrderedDict())
            cancelled[turn_id] = True
            cancelled.move_to_end(turn_id)
            while len(cancelled) > TTS_CANCELLED_TURN_MEMORY:
                cancelled.popitem(last=False)
        return True

    def is_stale(self, session_id: str, turn_id: str) -> bool:
        """True when this turn was cancelled or superseded by a newer one.

        Requests that carry no session/turn keep the old fire-and-forget
        behaviour: only a client disconnect stops them.
        """
        session_id, turn_id = _clean(session_id), _clean(turn_id)
        if not session_id or not turn_id:
            return False
        with self._lock:
            if turn_id in self._cancelled.get(session_id, {}):
                return True
            latest = self._latest.get(session_id)
            return latest is not None and latest != turn_id


def _clean(value: Any) -> str:
    return str(value or "").strip()


def iter_tts_events(
    tts: Any,
    text: str,
    voice_id: str = "",
    *,
    should_stop: Callable[[], bool] | None = None,
    turn_id: str = "",
    started_at: float | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield the NDJSON events for one TTS stream request.

    Event order::

        [notice]? audio+ done        # success
        [notice]? audio* error       # failure part way through
        [notice]? audio* cancelled   # the caller asked to stop (barge-in)

    * ``{"type": "notice", "kind": "tts_fallback", "message": ...}`` — emitted at
      most once, immediately **before** the next audio event, when the
      synthesizer fell back to macOS ``say``. Only the first segment can be
      known about before any audio is sent: when a later segment is the one that
      falls back, the notice necessarily arrives *after* audio the client is
      already playing and carries ``"late": True`` so the UI can word it as
      "the rest of this answer switched voices" instead of retracting.
    * ``{"type": "audio", "index": <1-based int>, "contentType": str,
      "audio": <base64 str>}`` — one per synthesized sentence group. The first
      one also carries ``"firstAudioMs"``: server-side milliseconds from
      ``started_at`` (request start) to the moment that audio was ready, which
      is the number the client's time-to-first-sound is measured against.
    * ``{"type": "done", "ok": True}`` — terminal, success.
    * ``{"type": "error", "ok": False, "error": "<Type>: <message>"}`` —
      terminal, failure; no ``done`` event follows it. Producing zero audio
      events for non-empty text is a failure, not an empty success.
    * ``{"type": "cancelled", "turnId": str}`` — terminal, the caller's
      ``should_stop`` flipped (client gone, turn superseded, explicit cancel).
      Synthesis of a Chinese sentence group takes hundreds of ms, so the check
      runs before every chunk: a barge-in costs at most one chunk of work.

    Every event carries ``"seq"``: a 0-based counter over the events of this
    one request, so a client can tell a dropped event from a slow one.

    The iterator is fully synchronous and blocking (model inference happens
    inside it), so callers should drive it from a worker thread.
    """
    entered_at = time.perf_counter() if started_at is None else started_at
    metas: list[dict[str, Any]] = []
    notified = False
    index = 0
    seq = 0

    def stamped(event: dict[str, Any]) -> dict[str, Any]:
        nonlocal seq
        event["seq"] = seq
        seq += 1
        return event

    def stopped() -> bool:
        return should_stop is not None and bool(should_stop())

    def cancelled() -> dict[str, Any]:
        logger.info(
            "tts cancelled after_audio=%d chars=%d voice=%s turn=%s",
            index,
            len(str(text or "")),
            voice_id or "default",
            turn_id or "-",
        )
        return stamped({"type": "cancelled", "turnId": turn_id})

    try:
        kwargs: dict[str, Any] = {"on_meta": metas.append} if _supports_on_meta(tts.synthesize_chunks) else {}
        if stopped():
            yield cancelled()
            return
        for audio, content_type in tts.synthesize_chunks(text, voice_id=voice_id, **kwargs):
            # The loop body runs after one chunk is synthesized and before the
            # next ``next()`` call, so this single check is both "before the
            # yield" and "before synthesizing the following chunk".
            if stopped():
                yield cancelled()
                return
            if not notified and any(item.get("fallback") for item in metas):
                notified = True
                notice = {"type": "notice", "kind": "tts_fallback", "message": FALLBACK_NOTICE}
                if index:
                    notice["late"] = True
                yield stamped(notice)
            index += 1
            event = {
                "type": "audio",
                "index": index,
                "contentType": content_type,
                "audio": base64.b64encode(audio).decode("ascii"),
            }
            if index == 1:
                first_audio_ms = (time.perf_counter() - entered_at) * 1000.0
                event["firstAudioMs"] = round(first_audio_ms, 1)
                logger.info(
                    "tts first_audio_ms=%.0f chars=%d voice=%s turn=%s",
                    first_audio_ms,
                    len(str(text or "")),
                    voice_id or "default",
                    turn_id or "-",
                )
            yield stamped(event)
        if index == 0 and str(text or "").strip():
            yield stamped({"type": "error", "ok": False, "error": "RuntimeError: TTS 未产生任何音频"})
            return
        if stopped():
            yield cancelled()
            return
        yield stamped({"type": "done", "ok": True})
    except Exception as exc:
        yield stamped({"type": "error", "ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _supports_on_meta(fn: Any) -> bool:
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if "on_meta" in parameters:
        return True
    return any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())
