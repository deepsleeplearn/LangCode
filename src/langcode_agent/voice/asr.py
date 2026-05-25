from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from .turnsense import TurnSenseService


SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AsrSettings:
    model: str = "Qwen/Qwen3-ASR-0.6B"
    backend: str = "transformers"
    device: str = "auto"
    dtype: str = "auto"
    language: str = "Chinese"
    partial_interval_sec: float = 1.4
    min_audio_sec: float = 1.0
    final_silence_ms: int = 900
    audio_vad_silero_threshold: float = 0.50
    audio_vad_energy_threshold: float = 0.014
    audio_vad_min_silence_ms: int = 800
    pre_speech_buffer_ms: int = 350
    max_utterance_sec: float = 30.0
    chunk_size_sec: float = 1.0
    unfixed_chunk_num: int = 4
    unfixed_token_num: int = 5

    @classmethod
    def from_env(cls) -> "AsrSettings":
        return cls(
            model=_default_model_path(),
            backend=(os.getenv("LANGCODE_ASR_BACKEND") or "transformers").strip().lower(),
            device=(os.getenv("LANGCODE_ASR_DEVICE") or "auto").strip().lower(),
            dtype=(os.getenv("LANGCODE_ASR_DTYPE") or "auto").strip().lower(),
            language=(os.getenv("LANGCODE_ASR_LANGUAGE") or "Chinese").strip(),
            partial_interval_sec=_float_env("LANGCODE_ASR_PARTIAL_INTERVAL_SEC", 1.4),
            min_audio_sec=_float_env("LANGCODE_ASR_MIN_AUDIO_SEC", 1.0),
            final_silence_ms=_int_env("LANGCODE_ASR_FINAL_SILENCE_MS", 900),
            audio_vad_silero_threshold=_float_env("LANGCODE_AUDIO_VAD_SILERO_THRESHOLD", 0.50),
            audio_vad_energy_threshold=_float_env("LANGCODE_AUDIO_VAD_ENERGY_THRESHOLD", 0.014),
            audio_vad_min_silence_ms=_int_env("LANGCODE_AUDIO_VAD_MIN_SILENCE_MS", 800),
            pre_speech_buffer_ms=_int_env("LANGCODE_ASR_PRE_SPEECH_BUFFER_MS", 350),
            max_utterance_sec=_float_env("LANGCODE_ASR_MAX_UTTERANCE_SEC", 30.0),
            chunk_size_sec=_float_env("LANGCODE_ASR_CHUNK_SIZE_SEC", 1.0),
            unfixed_chunk_num=_int_env("LANGCODE_ASR_UNFIXED_CHUNK_NUM", 4),
            unfixed_token_num=_int_env("LANGCODE_ASR_UNFIXED_TOKEN_NUM", 5),
        )


@dataclass
class AsrStatus:
    state: str = "idle"
    model: str = ""
    backend: str = ""
    vad: str = ""
    error: str = ""
    loaded_at: float | None = None


class QwenAsrService:
    def __init__(self, settings: AsrSettings | None = None, turnsense: TurnSenseService | None = None) -> None:
        self.settings = settings or AsrSettings.from_env()
        self.turnsense = turnsense or TurnSenseService()
        self._status = AsrStatus(
            state="idle",
            model=self.settings.model,
            backend=self.settings.backend,
            vad="silero-vad",
        )
        self._lock = threading.RLock()
        self._model: Any | None = None
        self._load_error: str = ""
        self._preload_thread: threading.Thread | None = None

    def start_preload(self) -> None:
        if os.getenv("PYTEST_CURRENT_TEST"):
            return
        if _truthy(os.getenv("LANGCODE_ASR_PRELOAD", "1")) is False:
            return
        with self._lock:
            if self._model is not None or (self._preload_thread is not None and self._preload_thread.is_alive()):
                return
            self._preload_thread = threading.Thread(target=self._load_safely, name="langcode-asr-preload", daemon=True)
            self._preload_thread.start()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ok": self._status.state != "error",
                "state": self._status.state,
                "model": self._status.model,
                "backend": self._status.backend,
                "device": self.settings.device,
                "dtype": self.settings.dtype,
                "vad": self._status.vad,
                "error": self._status.error,
                "loadedAt": self._status.loaded_at,
                "sampleRate": SAMPLE_RATE,
                "audioVad": {
                    "provider": self._status.vad,
                    "sileroThreshold": self.settings.audio_vad_silero_threshold,
                    "energyThreshold": self.settings.audio_vad_energy_threshold,
                    "minSilenceMs": self.settings.audio_vad_min_silence_ms,
                    "preSpeechBufferMs": self.settings.pre_speech_buffer_ms,
                    "maxUtteranceSec": self.settings.max_utterance_sec,
                },
                "semanticVad": self.turnsense.status(),
            }

    def load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            self._status.state = "loading"
            self._status.error = ""
        try:
            model = self._build_model()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._load_error = message
                self._status.state = "error"
                self._status.error = message
            raise RuntimeError(message) from exc
        with self._lock:
            self._model = model
            self._status.state = "ready"
            self._status.error = ""
            self._status.loaded_at = time.time()
            return model

    def create_session(self) -> "AsrStreamSession":
        model = self.load()
        return AsrStreamSession(self.settings, model, self.turnsense)

    def _load_safely(self) -> None:
        try:
            self.load()
        except Exception:
            pass

    def _build_model(self) -> Any:
        from qwen_asr import Qwen3ASRModel

        if self.settings.backend == "vllm":
            return Qwen3ASRModel.LLM(
                model=self.settings.model,
                max_new_tokens=64,
                gpu_memory_utilization=_float_env("LANGCODE_ASR_GPU_MEMORY_UTILIZATION", 0.8),
            )
        return Qwen3ASRModel.from_pretrained(
            self.settings.model,
            trust_remote_code=True,
            **_transformers_device_kwargs(self.settings.device, self.settings.dtype),
            max_new_tokens=128,
        )


class AsrStreamSession:
    def __init__(self, settings: AsrSettings, model: Any, turnsense: TurnSenseService | None = None) -> None:
        self.settings = settings
        self.model = model
        self.started_at = time.time()
        self.updated_at = self.started_at
        self.text = ""
        self.language = ""
        self.turn_state = "unknown"
        self.turn_confidence = 0.0
        self.turn_source = ""
        self._audio = np.zeros(0, dtype=np.float32)
        self._last_partial_at = 0.0
        self._has_speech = False
        self._lock = threading.RLock()
        self._vad = _SileroPauseDetector(
            silero_silence_ms=settings.final_silence_ms,
            silero_threshold=settings.audio_vad_silero_threshold,
            energy_threshold=settings.audio_vad_energy_threshold,
            energy_silence_ms=settings.audio_vad_min_silence_ms,
        )
        self._turnsense = turnsense or TurnSenseService()
        self._streaming_state: Any | None = None
        if settings.backend == "vllm" and hasattr(model, "init_streaming_state"):
            self._streaming_state = model.init_streaming_state(
                context="",
                language=settings.language or None,
                unfixed_chunk_num=settings.unfixed_chunk_num,
                unfixed_token_num=settings.unfixed_token_num,
                chunk_size_sec=settings.chunk_size_sec,
            )

    def push_float32(self, chunk: np.ndarray) -> dict[str, Any]:
        chunk = _sanitize_audio(chunk)
        if chunk.size == 0:
            return self._event("partial", final=False)
        with self._lock:
            self.updated_at = time.time()
            self._audio = np.concatenate([self._audio, chunk])
            speech_event = self._vad.push(chunk)
            if speech_event == "start":
                self._has_speech = True
            self._trim_audio_buffer()
            should_finalize = (
                self._has_speech
                and speech_event == "end"
                and self._audio.size >= int(SAMPLE_RATE * self.settings.min_audio_sec)
            )
            now = time.time()
            should_partial = self._has_speech and now - self._last_partial_at >= self.settings.partial_interval_sec
        if should_finalize:
            event = self.transcribe(final=False)
            if event.get("type") == "error":
                return event
            return self._event("final", final=True) if self.turn_state in {"complete", "unknown"} else event
        if should_partial:
            return self.transcribe(final=False)
        return self._event("partial", final=False)

    def transcribe(self, *, final: bool) -> dict[str, Any]:
        with self._lock:
            audio = self._audio.copy()
            if audio.size < int(SAMPLE_RATE * self.settings.min_audio_sec):
                return self._event("partial", final=False)
            self._last_partial_at = time.time()
        try:
            if self._streaming_state is not None:
                self.model.streaming_transcribe(audio, self._streaming_state)
                text = getattr(self._streaming_state, "text", "") or ""
                language = getattr(self._streaming_state, "language", "") or ""
            else:
                result = self.model.transcribe((audio, SAMPLE_RATE), language=self.settings.language or None)[0]
                text = getattr(result, "text", "") or ""
                language = getattr(result, "language", "") or ""
        except Exception as exc:
            return {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
        with self._lock:
            self.text = text.strip()
            self.language = language
            turn = self._turnsense.classify(text=self.text, audio=audio, sample_rate=SAMPLE_RATE)
            self.turn_state = str(turn.get("state") or "unknown")
            self.turn_confidence = float(turn.get("confidence") or 0.0)
            self.turn_source = str(turn.get("source") or "")
        return self._event("final" if final else "partial", final=final)

    def finish(self) -> dict[str, Any]:
        with self._lock:
            audio_size = self._audio.size
        if audio_size < int(SAMPLE_RATE * self.settings.min_audio_sec):
            return self._event("final", final=True)
        if self._streaming_state is not None:
            try:
                self.model.finish_streaming_transcribe(self._streaming_state)
                with self._lock:
                    self.text = (getattr(self._streaming_state, "text", "") or "").strip()
                    self.language = getattr(self._streaming_state, "language", "") or ""
                    turn = self._turnsense.classify(text=self.text, audio=self._audio.copy(), sample_rate=SAMPLE_RATE)
                    self.turn_state = str(turn.get("state") or "unknown")
                    self.turn_confidence = float(turn.get("confidence") or 0.0)
                    self.turn_source = str(turn.get("source") or "")
                return self._event("final", final=True)
            except Exception as exc:
                return {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
        return self.transcribe(final=True)

    def _event(self, event_type: str, *, final: bool) -> dict[str, Any]:
        return {
            "type": event_type,
            "text": self.text,
            "language": self.language,
            "final": final,
            "model": self.settings.model,
            "vad": "silero-vad" if self._vad.available else "energy-fallback",
            "audioVad": {
                "eventSource": "silero-vad" if self._vad.available else "energy-fallback",
                "sileroThreshold": self.settings.audio_vad_silero_threshold,
                "energyThreshold": self.settings.audio_vad_energy_threshold,
                "minSilenceMs": self.settings.audio_vad_min_silence_ms,
                "preSpeechBufferMs": self.settings.pre_speech_buffer_ms,
                "maxUtteranceSec": self.settings.max_utterance_sec,
                "rms": self._vad.last_rms,
                "inSpeech": self._vad.in_speech,
            },
            "semanticVad": {
                "state": self.turn_state,
                "confidence": self.turn_confidence,
                "source": self.turn_source,
            },
        }

    def _trim_audio_buffer(self) -> None:
        if self._has_speech:
            keep_samples = int(SAMPLE_RATE * max(3.0, self.settings.max_utterance_sec))
        else:
            keep_samples = int(SAMPLE_RATE * max(0.1, self.settings.pre_speech_buffer_ms / 1000))
        if keep_samples > 0 and self._audio.size > keep_samples:
            self._audio = self._audio[-keep_samples:]


class _SileroPauseDetector:
    def __init__(
        self,
        *,
        silero_silence_ms: int,
        silero_threshold: float,
        energy_threshold: float,
        energy_silence_ms: int,
    ) -> None:
        self.available = False
        self._iterator: Any | None = None
        self._pending = np.zeros(0, dtype=np.float32)
        self._in_speech = False
        self._silence_started: float | None = None
        self._energy_threshold = max(0.003, energy_threshold)
        self._min_silence_sec = max(0.25, energy_silence_ms / 1000)
        self.last_rms = 0.0
        try:
            from silero_vad import VADIterator, load_silero_vad

            self._torch = __import__("torch")
            self._iterator = VADIterator(
                load_silero_vad(),
                threshold=max(0.1, min(0.9, silero_threshold)),
                sampling_rate=SAMPLE_RATE,
                min_silence_duration_ms=silero_silence_ms,
                speech_pad_ms=80,
            )
            self.available = True
        except Exception:
            self._torch = None

    def push(self, chunk: np.ndarray) -> str | None:
        if self.available and self._iterator is not None:
            return self._push_silero(chunk)
        return self._push_energy(chunk)

    def _push_silero(self, chunk: np.ndarray) -> str | None:
        frame = 512
        event_type = None
        self.last_rms = _rms(chunk)
        self._pending = np.concatenate([self._pending, chunk.astype(np.float32, copy=False)])
        while self._pending.size >= frame:
            current = self._pending[:frame]
            self._pending = self._pending[frame:]
            event = self._iterator(self._torch.from_numpy(current), return_seconds=True)
            if isinstance(event, dict):
                if "start" in event:
                    self._in_speech = True
                    event_type = "start"
                if "end" in event and self._in_speech:
                    self._in_speech = False
                    event_type = "end"
        return event_type

    def _push_energy(self, chunk: np.ndarray) -> str | None:
        rms = _rms(chunk)
        self.last_rms = rms
        now = time.time()
        if rms > self._energy_threshold:
            self._in_speech = True
            self._silence_started = None
            return "start"
        if not self._in_speech:
            return None
        if self._silence_started is None:
            self._silence_started = now
            return None
        if now - self._silence_started >= self._min_silence_sec:
            self._in_speech = False
            self._silence_started = None
            return "end"
        return None

    @property
    def in_speech(self) -> bool:
        return self._in_speech


async def websocket_asr_loop(ws: Any, service: QwenAsrService) -> None:
    session: AsrStreamSession | None = None
    await ws.send(_json_event({"type": "status", **service.status()}))
    while True:
        message = await ws.recv()
        if message is None:
            break
        if isinstance(message, str):
            try:
                payload = __import__("json").loads(message)
            except Exception:
                await ws.send(_json_event({"type": "error", "error": "无效的 ASR 控制消息。"}))
                continue
            command = payload.get("type")
            if command == "start":
                try:
                    session = await asyncio.to_thread(service.create_session)
                except Exception as exc:
                    await ws.send(_json_event({"type": "error", "error": f"{type(exc).__name__}: {exc}"}))
                    continue
                await ws.send(_json_event({"type": "ready", **service.status()}))
            elif command == "finish":
                if session is not None:
                    event = await asyncio.to_thread(session.finish)
                    await ws.send(_json_event(event))
                break
            elif command == "cancel":
                await ws.send(_json_event({"type": "cancelled"}))
                break
            else:
                await ws.send(_json_event({"type": "error", "error": f"未知 ASR 指令：{command}"}))
            continue
        if session is None:
            await ws.send(_json_event({"type": "error", "error": "ASR 会话尚未启动。"}))
            continue
        chunk = np.frombuffer(message, dtype=np.float32).reshape(-1)
        event = await asyncio.to_thread(session.push_float32, chunk)
        await ws.send(_json_event(event))
        if event.get("type") == "final":
            break


def _json_event(payload: dict[str, Any]) -> str:
    return __import__("json").dumps(payload, ensure_ascii=False)


def _sanitize_audio(chunk: np.ndarray) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
    if chunk.size == 0:
        return chunk
    return np.nan_to_num(np.clip(chunk, -1.0, 1.0), copy=False)


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "y"}


def _transformers_device_kwargs(device: str, dtype: str) -> dict[str, Any]:
    import torch

    resolved_device = _resolve_device(device, torch)
    resolved_dtype = _resolve_dtype(dtype, resolved_device, torch)
    return {
        "torch_dtype": resolved_dtype,
        "device_map": {"": resolved_device} if resolved_device in {"mps", "cpu", "cuda"} else resolved_device,
    }


def _resolve_device(device: str, torch_module: Any) -> str:
    requested = (device or "auto").strip().lower()
    if requested in {"cpu", "mps", "cuda", "cuda:0"}:
        if requested == "cuda:0":
            return "cuda"
        return requested
    if requested != "auto":
        return "auto"
    try:
        if torch_module.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        if torch_module.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _resolve_dtype(dtype: str, device: str, torch_module: Any) -> Any:
    requested = (dtype or "auto").strip().lower()
    if requested in {"float16", "fp16", "half"}:
        return torch_module.float16
    if requested in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    if requested in {"float32", "fp32"}:
        return torch_module.float32
    if requested == "auto" and device == "mps":
        return torch_module.float16
    return "auto"


def _float_env(name: str, fallback: float) -> float:
    try:
        return float(os.getenv(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback


def _int_env(name: str, fallback: int) -> int:
    try:
        return int(os.getenv(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback


def _default_model_path() -> str:
    configured = (os.getenv("LANGCODE_ASR_MODEL") or "").strip()
    if configured:
        return configured
    local_model = Path.cwd() / ".langcode" / "asr-models" / "Qwen3-ASR-0.6B"
    if local_model.exists():
        return str(local_model)
    return "Qwen/Qwen3-ASR-0.6B"
