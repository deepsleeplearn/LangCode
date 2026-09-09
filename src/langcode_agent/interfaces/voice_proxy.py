from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote, urlparse, urlunparse

import httpx
import websockets


class VoiceWorkerClient:
    """Small HTTP/WebSocket client for the out-of-process voice worker."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def status(self) -> dict[str, Any]:
        return self._get_json("/api/voice/status")

    def asr_status(self) -> dict[str, Any]:
        return self._get_json("/api/asr/status")

    def tts_status(self) -> dict[str, Any]:
        return self._get_json("/api/tts/status")

    def list_tts_voices(self) -> dict[str, Any]:
        return self._get_json("/api/tts/voices")

    def create_tts_voice(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json("/api/tts/voices", payload, timeout_seconds=max(30.0, self.timeout_seconds))

    def tts_speech(self, payload: dict[str, Any]) -> tuple[bytes, str]:
        with httpx.Client(timeout=None) as client:
            result = client.post(self._url("/api/tts/speech"), json=payload)
            result.raise_for_status()
            return result.content, result.headers.get("content-type", "audio/wav")

    def tts_voice_preview(self, voice_id: str) -> tuple[bytes, str]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            result = client.get(self._url(f"/api/tts/voices/{quote(voice_id)}/preview"))
            result.raise_for_status()
            return result.content, result.headers.get("content-type", "audio/wav")

    async def cancel_tts(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Forward a barge-in cancel; the worker owns the producer to stop."""
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            result = await client.post(self._url("/api/tts/cancel"), json=payload)
            result.raise_for_status()
            return dict(result.json())

    async def stream_tts(self, payload: dict[str, Any]):
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", self._url("/api/tts/stream"), json=payload) as upstream:
                if upstream.status_code >= 400:
                    body = await upstream.aread()
                    yield {
                        "type": "error",
                        "ok": False,
                        "error": _decode_error_body(body, upstream.status_code),
                    }
                    return
                async for line in upstream.aiter_lines():
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield {"type": "error", "ok": False, "error": f"无效的语音 worker 流事件：{line[:120]}"}

    async def proxy_asr_websocket(self, client_ws: Any) -> None:
        async with websockets.connect(self._ws_url("/api/asr/stream"), max_size=None) as worker_ws:
            async def client_to_worker() -> None:
                while True:
                    message = await client_ws.recv()
                    if message is None:
                        break
                    await worker_ws.send(message)

            async def worker_to_client() -> None:
                async for message in worker_ws:
                    await client_ws.send(message)

            import asyncio

            tasks = {
                asyncio.create_task(client_to_worker()),
                asyncio.create_task(worker_to_client()),
            }
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                task.result()

    def _get_json(self, path: str) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            result = client.get(self._url(path))
            result.raise_for_status()
            return dict(result.json())

    def _post_json(self, path: str, payload: dict[str, Any], *, timeout_seconds: float | None = None) -> dict[str, Any]:
        with httpx.Client(timeout=timeout_seconds or self.timeout_seconds) as client:
            result = client.post(self._url(path), json=payload)
            result.raise_for_status()
            return dict(result.json())

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _ws_url(self, path: str) -> str:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunparse((scheme, parsed.netloc, path, "", "", ""))


def _decode_error_body(body: bytes, status_code: int) -> str:
    text = body.decode("utf-8", errors="replace").strip()
    if not text:
        return f"语音 worker 请求失败：HTTP {status_code}"
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return f"语音 worker 请求失败：HTTP {status_code} {text[:300]}"
    if isinstance(value, dict) and value.get("error"):
        return str(value.get("error"))
    return f"语音 worker 请求失败：HTTP {status_code} {text[:300]}"
