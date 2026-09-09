from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..interfaces.web import WebApp, _error_event, _exception_error_event, _session_id
from ..voice.stream import iter_tts_events
from ..storage.job_queue import JobQueue


logger = logging.getLogger("langcode.worker")

DEFAULT_WORKER_CONCURRENCY = 4


def worker_concurrency() -> int:
    try:
        value = int(str(os.getenv("LANGCODE_WORKER_CONCURRENCY", str(DEFAULT_WORKER_CONCURRENCY))).strip())
    except (TypeError, ValueError):
        value = DEFAULT_WORKER_CONCURRENCY
    return max(1, value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run LangCode queue worker")
    parser.add_argument("--workspace", default=".", help="Workspace root")
    parser.add_argument("--frontend-dir", default="frontend/dist", help="Built frontend directory")
    parser.add_argument("--queue-name", default=None, help="Redis queue name")
    parser.add_argument("--poll-seconds", type=float, default=1.0, help="Queue polling timeout")
    parser.add_argument(
        "--enable-voice",
        action="store_true",
        help="Load local ASR/TTS models in this worker (off by default; tts_stream jobs then need a voice worker).",
    )
    args = parser.parse_args(argv)

    frontend_dir = Path(args.frontend_dir).expanduser().resolve()
    # Item 34: the queue worker never loads local voice models unless asked to.
    app = WebApp(Path(args.workspace), frontend_dir, enable_voice=bool(args.enable_voice))
    queue = JobQueue(prefix=app.runtime_state.prefix, queue_name=args.queue_name, enabled="1")
    concurrency = worker_concurrency()
    print(
        f"LangCode worker started. queue={queue.queue_name} backend={queue.status()['backend']} "
        f"concurrency={concurrency} voice={'on' if args.enable_voice else 'off'}"
    )
    run_worker_loop(app, queue, concurrency=concurrency, poll_seconds=args.poll_seconds)


def run_worker_loop(
    app: WebApp,
    queue: JobQueue,
    *,
    concurrency: int,
    poll_seconds: float = 1.0,
    stop: threading.Event | None = None,
    install_signal_handlers: bool = True,
) -> None:
    """Reserve and dispatch queue jobs until stopped.

    Item 6: a job is reserved only when the pool has a free slot, so a worker
    never pulls jobs off Redis it cannot start - an unbounded prefetch left them
    invisible to other workers and lost on a crash. SIGTERM stops the loop, and
    any job that was reserved but never started gets an explicit terminal error
    event so its client is not left hanging.
    """
    # Item 25: dispatch jobs concurrently. Per-session serialization still comes
    # from the distributed session lock taken inside _run_event_job, so two jobs
    # for the same session can never execute at the same time.
    concurrency = max(1, int(concurrency))
    stop = stop or threading.Event()
    slots = threading.Semaphore(concurrency)
    pending: dict[Future, str] = {}
    pending_guard = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="queue-job")

    previous_handlers: list[tuple[int, Any]] = []
    if install_signal_handlers:
        for signal_number in (signal.SIGTERM, signal.SIGINT):
            try:
                previous_handlers.append((signal_number, signal.getsignal(signal_number)))
                signal.signal(signal_number, lambda _signum, _frame: stop.set())
            except (ValueError, OSError):  # not the main thread / unsupported
                pass

    try:
        while not stop.is_set():
            if not slots.acquire(timeout=0.5):
                continue
            try:
                job = queue.reserve(timeout_seconds=max(1, int(poll_seconds)))
            except Exception:
                slots.release()
                if stop.is_set():
                    break
                logger.exception("Queue reserve failed; retrying")
                time.sleep(min(5.0, max(0.5, float(poll_seconds))))
                continue
            if job is None:
                slots.release()
                continue
            future = pool.submit(_dispatch_job, app, queue, job, slots, pending, pending_guard)
            with pending_guard:
                pending[future] = str(job.get("id") or "")
    except KeyboardInterrupt:
        stop.set()
    finally:
        for signal_number, handler in previous_handlers:
            try:
                signal.signal(signal_number, handler)
            except (ValueError, OSError):
                pass
        pool.shutdown(wait=True, cancel_futures=True)
        # Jobs that were reserved but cancelled before they ever ran own no
        # terminal event, so publish one rather than let the client hang.
        with pending_guard:
            orphans = [job_id for future, job_id in pending.items() if future.cancelled() and job_id]
            pending.clear()
        for job_id in orphans:
            try:
                queue.publish_done(job_id, _error_event("Worker shut down before this job started.", "internal"))
            except Exception:
                logger.exception("Could not publish the shutdown terminal event for job %s", job_id)


def _dispatch_job(
    app: WebApp,
    queue: JobQueue,
    job: dict[str, Any],
    slots: threading.Semaphore | None = None,
    pending: dict[Future, str] | None = None,
    pending_guard: threading.Lock | None = None,
) -> None:
    try:
        _handle_job(app, queue, job)
    except Exception:
        logger.exception("Queue job crashed: %s", job.get("id"))
    finally:
        if pending is not None and pending_guard is not None:
            with pending_guard:
                for future, job_id in list(pending.items()):
                    if job_id == str(job.get("id") or "") or future.done():
                        pending.pop(future, None)
        if slots is not None:
            slots.release()


def _handle_job(app: WebApp, queue: JobQueue, job: dict[str, Any]) -> None:
    job_id = str(job.get("id") or "")
    kind = str(job.get("kind") or "")
    payload = dict(job.get("payload") or {})
    # Item 6: every path out of here has to leave exactly one terminal event on
    # the job stream. If publish_done itself blew up in the except branch, the
    # client used to wait forever, so the finally block retries once.
    terminal_published = False
    try:
        try:
            if kind == "chat_stream":
                _run_event_job(app, queue, job_id, payload, app.chat_events)
            elif kind == "approval_stream":
                _run_event_job(app, queue, job_id, payload, app.approval_events)
            elif kind == "tts_stream":
                _run_tts_job(app, queue, job_id, payload)
            else:
                queue.publish_done(job_id, _error_event(f"Unknown job kind: {kind}", "internal"))
            terminal_published = True
        except Exception as exc:
            queue.publish_done(job_id, _exception_error_event(exc, context=f"queue job {kind}"))
            terminal_published = True
    finally:
        if not terminal_published and job_id:
            try:
                queue.publish_done(
                    job_id,
                    _error_event(f"Queue job {kind} ended without a terminal event.", "internal"),
                )
            except Exception:
                logger.exception("Could not publish any terminal event for job %s", job_id)


def _run_event_job(app: WebApp, queue: JobQueue, job_id: str, payload: dict, producer) -> None:
    session_id = _session_id(payload)
    run_id = str(payload.get("runId") or "").strip() or None
    with app.runtime_state.acquire_session_lock(session_id):
        app.refresh_session_from_store(session_id)
        app.runtime_state.mark_run_started(session_id, run_id)
        try:
            for event in producer(payload):
                queue.publish_event(job_id, event)
                if event.get("type") == "done":
                    return
        finally:
            app.runtime_state.mark_run_finished(session_id, run_id)
            app._forget_cancelled_run(session_id, run_id)
    queue.publish_done(job_id)


def _run_tts_job(app: WebApp, queue: JobQueue, job_id: str, payload: dict) -> None:
    if app.voice_worker is not None:
        asyncio.run(_run_tts_proxy_job(app, queue, job_id, payload))
        return
    if app.tts is None:
        raise RuntimeError("TTS service is not available.")
    text = str(payload.get("text") or "")
    voice_id = str(payload.get("voiceId") or "")
    terminal: dict | None = None
    # Shared producer with web.py/voice_worker: audio events, an optional
    # tts_fallback notice, then exactly one terminal done/error event.
    for event in iter_tts_events(app.tts, text, voice_id=voice_id):
        if event.get("type") in {"done", "error"}:
            terminal = event
            break
        queue.publish_event(job_id, event)
    if terminal is not None and terminal.get("type") == "error":
        queue.publish_done(job_id, terminal)
        return
    queue.publish_done(job_id, {"type": "done", "ok": True})


async def _run_tts_proxy_job(app: WebApp, queue: JobQueue, job_id: str, payload: dict) -> None:
    if app.voice_worker is None:
        raise RuntimeError("Voice worker is not configured.")
    async for event in app.voice_worker.stream_tts(payload):
        queue.publish_event(job_id, event)
    queue.publish_done(job_id, {"type": "done", "ok": True})


if __name__ == "__main__":
    main()
