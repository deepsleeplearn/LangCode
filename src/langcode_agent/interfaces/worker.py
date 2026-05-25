from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any

from ..interfaces.web import WebApp, _session_id
from ..storage.job_queue import JobQueue


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run LangCode queue worker")
    parser.add_argument("--workspace", default=".", help="Workspace root")
    parser.add_argument("--frontend-dir", default="frontend/dist", help="Built frontend directory")
    parser.add_argument("--queue-name", default=None, help="Redis queue name")
    parser.add_argument("--poll-seconds", type=float, default=1.0, help="Queue polling timeout")
    args = parser.parse_args(argv)

    frontend_dir = Path(args.frontend_dir).expanduser().resolve()
    app = WebApp(Path(args.workspace), frontend_dir)
    queue = JobQueue(prefix=app.runtime_state.prefix, queue_name=args.queue_name, enabled="1")
    print(f"LangCode worker started. queue={queue.queue_name} backend={queue.status()['backend']}")
    while True:
        job = queue.reserve(timeout_seconds=max(1, int(args.poll_seconds)))
        if job is None:
            continue
        _handle_job(app, queue, job)


def _handle_job(app: WebApp, queue: JobQueue, job: dict[str, Any]) -> None:
    job_id = str(job.get("id") or "")
    kind = str(job.get("kind") or "")
    payload = dict(job.get("payload") or {})
    try:
        if kind == "chat_stream":
            _run_event_job(app, queue, job_id, payload, app.chat_events)
        elif kind == "approval_stream":
            _run_event_job(app, queue, job_id, payload, app.approval_events)
        elif kind == "tts_stream":
            _run_tts_job(app, queue, job_id, payload)
        else:
            queue.publish_done(job_id, {"type": "error", "ok": False, "error": f"Unknown job kind: {kind}"})
    except Exception as exc:
        queue.publish_done(job_id, {"type": "error", "ok": False, "error": f"{type(exc).__name__}: {exc}"})


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
    for index, (audio, content_type) in enumerate(app.tts.synthesize_chunks(text, voice_id=voice_id), start=1):
        queue.publish_event(
            job_id,
            {
                "type": "audio",
                "index": index,
                "contentType": content_type,
                "audio": base64.b64encode(audio).decode("ascii"),
            },
        )
    queue.publish_done(job_id, {"type": "done", "ok": True})


async def _run_tts_proxy_job(app: WebApp, queue: JobQueue, job_id: str, payload: dict) -> None:
    if app.voice_worker is None:
        raise RuntimeError("Voice worker is not configured.")
    async for event in app.voice_worker.stream_tts(payload):
        queue.publish_event(job_id, event)
    queue.publish_done(job_id, {"type": "done", "ok": True})


if __name__ == "__main__":
    main()
