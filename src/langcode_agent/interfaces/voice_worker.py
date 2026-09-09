from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import unquote

from sanic import Sanic, response
from sanic.request import Request

from ..core.config import load_env_files
from ..voice.asr import QwenAsrService, websocket_asr_loop
from ..voice.stream import TtsTurnRegistry, iter_tts_events
from ..voice.tts import TtsService, content_type_for_path
from ..voice.turnsense import TurnSenseService


class VoiceWorker:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.expanduser().resolve()
        load_env_files(Path.cwd(), self.workspace_root)
        self.turnsense = TurnSenseService()
        self.asr = QwenAsrService(turnsense=self.turnsense)
        self.tts = TtsService()
        # This process owns the producer when a web process proxies to it, so it
        # also owns the barge-in bookkeeping (same class as web.py uses).
        self.tts_turns = TtsTurnRegistry()

    def start_preload(self) -> None:
        self.asr.start_preload()
        self.tts.start_preload()

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "workspace": str(self.workspace_root),
            "asr": self.asr.status(),
            "turnsense": self.turnsense.status(),
            "tts": self.tts.status(),
        }


def create_voice_worker_app(worker: VoiceWorker) -> Sanic:
    sanic_app = Sanic("langcode-voice-worker")
    sanic_app.config.REQUEST_TIMEOUT = 3600
    sanic_app.config.RESPONSE_TIMEOUT = 3600
    sanic_app.ctx.worker = worker

    @sanic_app.before_server_start
    async def _preload(_app, _loop) -> None:
        worker.start_preload()

    @sanic_app.get("/health")
    async def health(_request: Request):
        return response.json({"ok": True})

    @sanic_app.get("/api/voice/status")
    async def voice_status(_request: Request):
        return await _json_thread(worker.status)

    @sanic_app.get("/api/asr/status")
    async def asr_status(_request: Request):
        return await _json_thread(worker.asr.status)

    @sanic_app.get("/api/tts/status")
    async def tts_status(_request: Request):
        return await _json_thread(worker.tts.status)

    @sanic_app.get("/api/tts/voices")
    async def tts_voices(_request: Request):
        voices = await asyncio.to_thread(worker.tts.list_voices)
        return response.json({"ok": True, "voices": voices})

    @sanic_app.post("/api/tts/voices")
    async def tts_create_voice(request: Request):
        payload = _request_json(request)
        try:
            profile = await asyncio.to_thread(
                worker.tts.create_voice_profile,
                name=str(payload.get("name") or ""),
                prompt_text=str(payload.get("promptText") or ""),
                style=str(payload.get("style") or ""),
                wav_bytes=_decode_data_url_or_base64(str(payload.get("audio") or "")),
            )
            voices = await asyncio.to_thread(worker.tts.list_voices)
            return response.json({"ok": True, "voice": profile, "voices": voices})
        except Exception as exc:
            return response.json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=400)

    @sanic_app.get("/api/tts/voices/<voice_id:path>/preview")
    async def tts_voice_preview(_request: Request, voice_id: str):
        try:
            voice_id = unquote(voice_id)
            preview_path = await asyncio.to_thread(worker.tts.voice_preview_path, voice_id)
            if preview_path is None:
                raise FileNotFoundError(f"未找到音色试听文件：{voice_id}")
            return await response.file(
                preview_path,
                mime_type=content_type_for_path(preview_path),
                headers={"Cache-Control": "no-cache"},
            )
        except Exception as exc:
            return response.json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=404)

    @sanic_app.post("/api/tts/speech")
    async def tts_speech(request: Request):
        payload = _request_json(request)
        try:
            audio, content_type = await asyncio.to_thread(
                worker.tts.synthesize,
                str(payload.get("text") or ""),
                str(payload.get("voiceId") or ""),
            )
            return response.raw(audio, content_type=content_type, headers={"Cache-Control": "no-cache"})
        except Exception as exc:
            return response.json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

    @sanic_app.post("/api/tts/stream")
    async def tts_stream(request: Request):
        payload = _request_json(request)
        text = str(payload.get("text") or "")
        voice_id = str(payload.get("voiceId") or "")
        session_id = str(payload.get("sessionId") or "")
        turn_id = str(payload.get("turnId") or "")
        # A newer turn of the same session supersedes this one; the producer
        # notices between chunks. Same rule as web.py's local endpoint.
        worker.tts_turns.claim(session_id, turn_id)

        async def stream(streaming_response):
            started_at = time.perf_counter()
            event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()
            disconnected = threading.Event()

            def should_stop() -> bool:
                return disconnected.is_set() or worker.tts_turns.is_stale(session_id, turn_id)

            def produce_audio() -> None:
                try:
                    for event in iter_tts_events(
                        worker.tts,
                        text,
                        voice_id=voice_id,
                        should_stop=should_stop,
                        turn_id=turn_id,
                        started_at=started_at,
                    ):
                        loop.call_soon_threadsafe(event_queue.put_nowait, event)
                finally:
                    loop.call_soon_threadsafe(event_queue.put_nowait, None)

            producer = asyncio.create_task(asyncio.to_thread(produce_audio))
            try:
                while True:
                    event = await event_queue.get()
                    if event is None:
                        break
                    await streaming_response.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
                await producer
            except (asyncio.CancelledError, ConnectionError, BrokenPipeError):
                # The web process dropped the upstream connection: stop
                # synthesizing instead of speaking into a dead socket.
                disconnected.set()
                raise

        return response.ResponseStream(
            stream,
            content_type="application/x-ndjson; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    @sanic_app.post("/api/tts/cancel")
    async def tts_cancel(request: Request):
        payload = _request_json(request)
        if not worker.tts_turns.cancel(payload.get("sessionId"), payload.get("turnId")):
            return response.json({"ok": False, "error": "Session id and turn id are required"}, status=400)
        return response.json({"ok": True})

    @sanic_app.websocket("/api/asr/stream")
    async def asr_stream(_request: Request, ws):
        await websocket_asr_loop(ws, worker.asr)

    return sanic_app


async def _json_thread(fn):
    try:
        return response.json(await asyncio.to_thread(fn))
    except Exception as exc:
        return response.json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)


def _request_json(request: Request) -> dict[str, Any]:
    payload = request.json
    return payload if isinstance(payload, dict) else {}


def _decode_data_url_or_base64(value: str) -> bytes:
    if "," in value and value.strip().lower().startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value, validate=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LangCode voice worker")
    parser.add_argument("--workspace", default=".", help="Workspace root")
    parser.add_argument("--host", default="127.0.0.1", help="Host")
    parser.add_argument("--port", type=int, default=8879, help="Port")
    parser.add_argument("--workers", type=int, default=1, help="Sanic worker processes; keep 1 for model memory reuse")
    args = parser.parse_args(argv)

    worker = VoiceWorker(Path(args.workspace))
    sanic_app = create_voice_worker_app(worker)
    print(f"LangCode voice worker: http://{args.host}:{args.port}")
    print(f"Workspace: {Path(args.workspace).expanduser().resolve()}")
    sanic_app.run(
        host=args.host,
        port=args.port,
        workers=args.workers,
        access_log=False,
        single_process=args.workers == 1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
