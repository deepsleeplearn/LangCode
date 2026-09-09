from __future__ import annotations

import asyncio
from dataclasses import dataclass, fields
from functools import lru_cache
import inspect
import os
from pathlib import Path
import threading
import time
import uuid
from typing import Any

import numpy as np

from .turnsense import MAX_AUDIO_SECONDS as TURN_WINDOW_SEC, TurnSenseService


SAMPLE_RATE = 16_000
LOAD_ERROR_COOLDOWN_SEC = 15.0
# How often the websocket loop looks for VAD speech events while a decode runs.
SPEECH_EVENT_POLL_SEC = 0.02
TURN_WINDOW_SAMPLES = SAMPLE_RATE * TURN_WINDOW_SEC
# Silero consumes fixed 512-sample frames; the near-field gate measures time in
# the same unit so both VAD paths mean the same thing by "one frame".
_VAD_FRAME_SAMPLES = 512
_VAD_FRAME_SEC = _VAD_FRAME_SAMPLES / SAMPLE_RATE
# Weight of one frame in the noise-floor EMA: the floor follows a change in the
# room over roughly a second, slow enough that speech pauses do not drag it.
_NOISE_FLOOR_ALPHA = 0.05
_NOISE_FLOOR_WARMUP_FRAMES = 5
_EMPTY_AUDIO = np.zeros(0, dtype=np.float32)
# ASCII punctuation after which a following ASCII word starts a new token.
_JOIN_SPACE_AFTER = ".?!,;"
_WARNED_ONCE: set[str] = set()


@dataclass(frozen=True)
class AsrSettings:
    """Single source of truth for every ASR/VAD threshold.

    Class-level defaults below are authoritative: ``from_env`` only overrides
    them from the environment, and every other module reads the values back
    from an ``AsrSettings`` instance instead of re-declaring literals.
    """

    model: str = "Qwen/Qwen3-ASR-0.6B"
    backend: str = "transformers"
    device: str = "auto"
    dtype: str = "auto"
    language: str = "Chinese"
    # Partial cadence drives how fast the UI sees what the user is saying, so it
    # stays close to the ~500 ms a mic chunk covers; the min-audio guard only has
    # to stop a decode from running on a sliver of audio.
    partial_interval_sec: float = 0.5
    min_audio_sec: float = 0.4
    final_silence_ms: int = 900
    audio_vad_silero_threshold: float = 0.55
    audio_vad_energy_threshold: float = 0.025
    audio_vad_min_silence_ms: int = 800
    # Near-field gate. Silero only says "this is a voice", never "this voice is
    # close", so loudness is the only thing that can tell the speaker at the mic
    # from someone talking across the room: a frame has to clear the noise floor
    # by ``snr_db`` and hold for ``min_speech_ms`` before speech starts.
    audio_vad_snr_db: float = 10.0
    audio_vad_min_speech_ms: int = 120
    pre_speech_buffer_ms: int = 350
    max_utterance_sec: float = 30.0
    max_decode_sec: float = 12.0
    chunk_size_sec: float = 1.0
    unfixed_chunk_num: int = 4
    unfixed_token_num: int = 5

    @classmethod
    def default(cls, name: str) -> Any:
        for item in fields(cls):
            if item.name == name:
                return item.default
        raise KeyError(name)

    @classmethod
    def from_env(cls) -> "AsrSettings":
        default = cls.default
        return cls(
            model=_default_model_path(),
            backend=_str_env("LANGCODE_ASR_BACKEND", default("backend")).lower(),
            device=_str_env("LANGCODE_ASR_DEVICE", default("device")).lower(),
            dtype=_str_env("LANGCODE_ASR_DTYPE", default("dtype")).lower(),
            language=_str_env("LANGCODE_ASR_LANGUAGE", default("language")),
            partial_interval_sec=_float_env("LANGCODE_ASR_PARTIAL_INTERVAL_SEC", default("partial_interval_sec")),
            min_audio_sec=_float_env("LANGCODE_ASR_MIN_AUDIO_SEC", default("min_audio_sec")),
            final_silence_ms=_int_env("LANGCODE_ASR_FINAL_SILENCE_MS", default("final_silence_ms")),
            audio_vad_silero_threshold=_float_env(
                "LANGCODE_AUDIO_VAD_SILERO_THRESHOLD", default("audio_vad_silero_threshold")
            ),
            audio_vad_energy_threshold=_float_env(
                "LANGCODE_AUDIO_VAD_ENERGY_THRESHOLD", default("audio_vad_energy_threshold")
            ),
            audio_vad_min_silence_ms=_int_env("LANGCODE_AUDIO_VAD_MIN_SILENCE_MS", default("audio_vad_min_silence_ms")),
            audio_vad_snr_db=_float_env("LANGCODE_AUDIO_VAD_SNR_DB", default("audio_vad_snr_db")),
            audio_vad_min_speech_ms=_int_env(
                "LANGCODE_AUDIO_VAD_MIN_SPEECH_MS", default("audio_vad_min_speech_ms")
            ),
            pre_speech_buffer_ms=_int_env("LANGCODE_ASR_PRE_SPEECH_BUFFER_MS", default("pre_speech_buffer_ms")),
            max_utterance_sec=_float_env("LANGCODE_ASR_MAX_UTTERANCE_SEC", default("max_utterance_sec")),
            max_decode_sec=_float_env("LANGCODE_ASR_MAX_DECODE_SEC", default("max_decode_sec")),
            chunk_size_sec=_float_env("LANGCODE_ASR_CHUNK_SIZE_SEC", default("chunk_size_sec")),
            unfixed_chunk_num=_int_env("LANGCODE_ASR_UNFIXED_CHUNK_NUM", default("unfixed_chunk_num")),
            unfixed_token_num=_int_env("LANGCODE_ASR_UNFIXED_TOKEN_NUM", default("unfixed_token_num")),
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
        self._load_error_at = 0.0
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
                    "snrDb": self.settings.audio_vad_snr_db,
                    "minSpeechMs": self.settings.audio_vad_min_speech_ms,
                    "preSpeechBufferMs": self.settings.pre_speech_buffer_ms,
                    "maxUtteranceSec": self.settings.max_utterance_sec,
                },
                "maxDecodeSec": self.settings.max_decode_sec,
                "semanticVad": self.turnsense.status(),
            }

    def load(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_error and time.monotonic() - self._load_error_at < LOAD_ERROR_COOLDOWN_SEC:
                raise RuntimeError(self._load_error)
            self._status.state = "loading"
            self._status.error = ""
        try:
            model = self._build_model()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._load_error = message
                self._load_error_at = time.monotonic()
                self._status.state = "error"
                self._status.error = message
            raise RuntimeError(message) from exc
        with self._lock:
            self._model = model
            self._load_error = ""
            self._load_error_at = 0.0
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


class _AudioBuffer:
    """Append-only audio window with O(1) appends and cheap head trimming.

    Samples are kept as a list of chunks so a streaming session never pays the
    O(n^2) cost of ``np.concatenate`` per pushed chunk; the array is only
    materialised (and cached) when a decode actually needs it. ``start``/``end``
    are absolute sample indices in the session's audio stream, which lets
    callers reason about what has already been committed or fed downstream.
    """

    __slots__ = ("_chunks", "start", "end")

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self.start = 0
        self.end = 0

    @property
    def size(self) -> int:
        return self.end - self.start

    def append(self, chunk: np.ndarray) -> None:
        if chunk.size:
            self._chunks.append(chunk)
            self.end += int(chunk.size)

    def drop_head(self, count: int) -> None:
        count = min(int(count), self.size)
        remaining = count
        while remaining > 0 and self._chunks:
            head = self._chunks[0]
            if head.size <= remaining:
                self._chunks.pop(0)
                remaining -= int(head.size)
            else:
                self._chunks[0] = head[remaining:]
                remaining = 0
        self.start += count - remaining

    def keep_last(self, keep_samples: int) -> None:
        keep_samples = max(0, int(keep_samples))
        if self.size > keep_samples:
            self.drop_head(self.size - keep_samples)

    def clear(self) -> None:
        self._chunks.clear()
        self.start = self.end

    def materialize(self) -> np.ndarray:
        if not self._chunks:
            return _EMPTY_AUDIO
        if len(self._chunks) > 1:
            self._chunks = [np.concatenate(self._chunks)]
        return self._chunks[0]


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
        self._buffer = _AudioBuffer()
        self._turn_audio = _AudioBuffer()
        self._committed_text = ""
        self._committed_samples = 0
        self._decoded_end = 0
        self._stream_fed = 0
        self._last_turn_text: str | None = None
        self._turn_stream_id = uuid.uuid4().hex
        self._last_partial_at = 0.0
        self._has_speech = False
        self._speech_transitions: list[str] = []
        self._lock = threading.RLock()
        self._vad = _SileroPauseDetector(
            silero_silence_ms=settings.final_silence_ms,
            silero_threshold=settings.audio_vad_silero_threshold,
            energy_threshold=settings.audio_vad_energy_threshold,
            energy_silence_ms=settings.audio_vad_min_silence_ms,
            snr_db=settings.audio_vad_snr_db,
            min_speech_ms=settings.audio_vad_min_speech_ms,
        )
        self._turnsense = turnsense or TurnSenseService()
        self._turn_supports_stream = _accepts_kwargs(self._turnsense.classify, ("audio_offset", "audio_stream"))
        self._streaming_state: Any | None = None
        if settings.backend == "vllm" and hasattr(model, "init_streaming_state"):
            self._streaming_state = model.init_streaming_state(
                context="",
                language=settings.language or None,
                unfixed_chunk_num=settings.unfixed_chunk_num,
                unfixed_token_num=settings.unfixed_token_num,
                chunk_size_sec=settings.chunk_size_sec,
            )

    @property
    def _audio(self) -> np.ndarray:
        """Uncommitted audio tail (the part that still gets re-decoded)."""
        with self._lock:
            return self._buffer.materialize()

    def push_float32(self, chunk: np.ndarray) -> dict[str, Any]:
        chunk = _sanitize_audio(chunk)
        if chunk.size == 0:
            return self._event("partial", final=False)
        with self._lock:
            self.updated_at = time.time()
            self._buffer.append(chunk)
            self._turn_audio.append(chunk)
            speech_event = self._vad.push(chunk)
            if speech_event == "start":
                self._has_speech = True
            if speech_event in ("start", "end"):
                self._speech_transitions.append(speech_event)
            self._trim_audio_buffer()
            # Length handling belongs to ``finish``/``transcribe``: requiring a
            # minimum buffer here swallowed Silero's ``end`` whenever the buffer
            # had just been emptied by a commit, so the turn never finalized.
            should_finalize = self._has_speech and speech_event == "end"
            now = time.time()
            should_partial = self._has_speech and now - self._last_partial_at >= self.settings.partial_interval_sec
        if should_finalize:
            # Silero reported end-of-speech: always re-run semantic endpointing.
            event = self.transcribe(final=False, force_turn=True)
            if event.get("type") == "error":
                return event
            # The event carries the turn state that was snapshotted under the
            # lock right after this decode's classification.
            turn_state = str(event.get("semanticVad", {}).get("state") or "unknown")
            return self._event("final", final=True) if turn_state in {"complete", "unknown"} else event
        if should_partial:
            return self.transcribe(final=False)
        return self._event("partial", final=False)

    def take_speech_events(self) -> list[dict[str, Any]]:
        """Drain the VAD speech transitions seen since the last call.

        These are their own websocket events, not a field on the transcript
        event, because the client needs "the user just started talking" as the
        barge-in signal *while* the decode of that same chunk is still running -
        a partial transcript is a decode too late to duck the assistant's voice.
        """
        with self._lock:
            events = [{"type": "speech", "state": state} for state in self._speech_transitions]
            self._speech_transitions.clear()
        return events

    def transcribe(self, *, final: bool, force_turn: bool = False) -> dict[str, Any]:
        with self._lock:
            # A final decode must never drop audio: the min-audio guard only
            # protects partials from decoding a sliver of speech.
            min_samples = 1 if final else int(SAMPLE_RATE * self.settings.min_audio_sec)
            if self._buffer.size < max(1, min_samples):
                return self._event("final" if final else "partial", final=final)
            audio = self._buffer.materialize()
            committed_text = self._committed_text
            stream_audio = self._pending_stream_audio(audio)
            # Absolute end of what this decode actually sees. Audio pushed while
            # the model runs lands after it and must survive the commit below.
            decoded_end = self._buffer.end
            self._last_partial_at = time.time()
        try:
            if self._streaming_state is not None:
                if stream_audio.size:
                    self.model.streaming_transcribe(stream_audio, self._streaming_state)
                    with self._lock:
                        self._stream_fed = decoded_end
                text = getattr(self._streaming_state, "text", "") or ""
                language = getattr(self._streaming_state, "language", "") or ""
            else:
                result = self.model.transcribe((audio, SAMPLE_RATE), language=self.settings.language or None)[0]
                text = _join_text(committed_text, (getattr(result, "text", "") or "").strip())
                language = getattr(result, "language", "") or ""
        except Exception as exc:
            return {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
        with self._lock:
            self.text = text.strip()
            self.language = language
            need_turn = force_turn or final or self.text != self._last_turn_text
            turn_audio = self._turn_audio.materialize() if need_turn else _EMPTY_AUDIO
            turn_offset = self._turn_audio.start
            turn_text = self.text
            self._decoded_end = max(self._decoded_end, decoded_end)
            decoded_size = max(0, decoded_end - self._buffer.start)
            if self._streaming_state is None and (final or decoded_size >= self._max_decode_samples()):
                self._commit_prefix(decoded_end)
        if need_turn:
            # TurnSense inference runs outside the session lock: it is the most
            # expensive step in the partial path and must not block audio pushes.
            self._classify_turn(turn_text, turn_audio, turn_offset)
        return self._event("final" if final else "partial", final=final)

    def finish(self) -> dict[str, Any]:
        with self._lock:
            audio_size = self._buffer.size
        if self._streaming_state is not None:
            try:
                self.model.finish_streaming_transcribe(self._streaming_state)
            except Exception as exc:
                return {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
            with self._lock:
                self.text = (getattr(self._streaming_state, "text", "") or "").strip()
                self.language = getattr(self._streaming_state, "language", "") or ""
                turn_audio = self._turn_audio.materialize()
                turn_offset = self._turn_audio.start
                turn_text = self.text
            self._classify_turn(turn_text, turn_audio, turn_offset)
            return self._event("final", final=True)
        # Whatever is still buffered has to be decoded, even when it is shorter
        # than ``min_audio_sec``: it is the tail of the user's last utterance.
        if audio_size <= 0:
            return self._event("final", final=True)
        return self.transcribe(final=True)

    def _max_decode_samples(self) -> int:
        # ``max_utterance_sec`` caps what the buffer may hold, so a larger
        # ``max_decode_sec`` would mean the commit never fires and the trimmed
        # head is dropped before it was ever transcribed.
        decode_sec = min(self.settings.max_decode_sec, self.settings.max_utterance_sec)
        return max(int(SAMPLE_RATE * self.settings.min_audio_sec), int(SAMPLE_RATE * decode_sec))

    def _commit_prefix(self, decoded_end: int | None = None) -> None:
        """Freeze the decoded prefix so later partials only decode the tail.

        ``decoded_end`` is the absolute end of the audio the finished decode
        actually saw. Only that prefix is dropped, so samples pushed while the
        model was running stay in the buffer and are decoded by the next pass.
        """
        end = self._buffer.end if decoded_end is None else min(int(decoded_end), self._buffer.end)
        dropped = max(0, end - self._buffer.start)
        self._committed_text = self.text
        self._committed_samples += dropped
        self._buffer.drop_head(dropped)

    def _pending_stream_audio(self, audio: np.ndarray) -> np.ndarray:
        """Audio not yet handed to the vLLM streaming state (it buffers itself)."""
        if self._stream_fed <= self._buffer.start:
            return audio
        offset = self._stream_fed - self._buffer.start
        return audio[offset:] if offset < audio.size else _EMPTY_AUDIO

    def _classify_turn(self, text: str, audio: np.ndarray, offset: int) -> None:
        kwargs: dict[str, Any] = {}
        if self._turn_supports_stream:
            kwargs = {"audio_offset": int(offset), "audio_stream": self._turn_stream_id}
        try:
            turn = self._turnsense.classify(text=text, audio=audio, sample_rate=SAMPLE_RATE, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive, TurnSense is best effort
            turn = {"state": "unknown", "confidence": 0.0, "source": f"error:{type(exc).__name__}"}
        with self._lock:
            self._last_turn_text = text
            self.turn_state = str(turn.get("state") or "unknown")
            self.turn_confidence = float(turn.get("confidence") or 0.0)
            self.turn_source = str(turn.get("source") or "")

    def _event(self, event_type: str, *, final: bool) -> dict[str, Any]:
        # Snapshot the mutable session state under the lock: ``_classify_turn``
        # and ``transcribe`` write these fields from other threads.
        with self._lock:
            text = self.text
            language = self.language
            turn_state = self.turn_state
            turn_confidence = self.turn_confidence
            turn_source = self.turn_source
        return {
            "type": event_type,
            "text": text,
            "language": language,
            "final": final,
            "model": self.settings.model,
            "vad": "silero-vad" if self._vad.available else "energy-fallback",
            "audioVad": {
                "eventSource": "silero-vad" if self._vad.available else "energy-fallback",
                "sileroThreshold": self.settings.audio_vad_silero_threshold,
                "energyThreshold": self.settings.audio_vad_energy_threshold,
                "minSilenceMs": self.settings.audio_vad_min_silence_ms,
                "snrDb": self.settings.audio_vad_snr_db,
                "minSpeechMs": self.settings.audio_vad_min_speech_ms,
                "preSpeechBufferMs": self.settings.pre_speech_buffer_ms,
                "maxUtteranceSec": self.settings.max_utterance_sec,
                "rms": self._vad.last_rms,
                # The level the room sits at when nobody near the mic is
                # talking - the number to compare ``rms`` against when tuning.
                "noiseFloor": self._vad.noise_floor,
                "levelThreshold": self._vad.level_threshold,
                "inSpeech": self._vad.in_speech,
            },
            "semanticVad": {
                "state": turn_state,
                "confidence": turn_confidence,
                "source": turn_source,
            },
        }

    def _trim_audio_buffer(self) -> None:
        if self._has_speech:
            keep_samples = int(SAMPLE_RATE * max(3.0, self.settings.max_utterance_sec))
            turn_keep = TURN_WINDOW_SAMPLES
        else:
            keep_samples = int(SAMPLE_RATE * max(0.1, self.settings.pre_speech_buffer_ms / 1000))
            turn_keep = keep_samples
        if self._buffer.size > keep_samples:
            # Freeze the decoded text before the head goes away, otherwise the
            # trimmed audio is lost without ever reaching the transcript. Only
            # the part a decode already saw may be committed.
            if self._has_speech and self._streaming_state is None and self._decoded_end > self._buffer.start:
                self._commit_prefix(self._decoded_end)
            self._buffer.keep_last(keep_samples)
        self._turn_audio.keep_last(min(turn_keep, TURN_WINDOW_SAMPLES))


class _SileroPauseDetector:
    """Speech start/end detection with a near-field gate in front of silero.

    Silero answers "is this a voice", never "is this voice close": someone
    talking two rooms away scores like the person at the microphone, so silero
    alone starts a turn - and ducks the assistant - for anybody in earshot. The
    level gate supplies the missing half. A frame counts as near-field only
    once it clears both the absolute ``energy_threshold`` and an ``snr_db``
    margin over the room's own noise floor, and ``start`` waits until that has
    held for ``min_speech_ms`` so a cough or a keyboard click cannot open it.
    """

    def __init__(
        self,
        *,
        silero_silence_ms: int,
        silero_threshold: float,
        energy_threshold: float,
        energy_silence_ms: int,
        snr_db: float,
        min_speech_ms: int,
    ) -> None:
        self.available = False
        self._iterator: Any | None = None
        self._pending = np.zeros(0, dtype=np.float32)
        # Silero's own verdict is tracked apart from the state the session sees:
        # while silero is active but the gate stays shut nothing is emitted, yet
        # every further frame is still judged, so a near-field speaker joining a
        # far-field segment already in progress still gets a ``start``.
        self._in_speech = False
        self._silero_active = False
        self._silence_started: float | None = None
        self._energy_threshold = max(0.003, energy_threshold)
        self._min_silence_sec = max(0.25, energy_silence_ms / 1000)
        self._snr_ratio = 10 ** (max(0.0, snr_db) / 20)
        self._min_speech_sec = max(0.0, min_speech_ms / 1000)
        self._noise_floor = 0.0
        self._noise_frames = 0
        self._loud_sec = 0.0
        self.last_rms = 0.0
        try:
            from silero_vad import VADIterator

            self._torch = __import__("torch")
            self._iterator = VADIterator(
                _silero_model(),
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
        event_type = None
        self.last_rms = _rms(chunk)
        self._pending = np.concatenate([self._pending, chunk.astype(np.float32, copy=False)])
        while self._pending.size >= _VAD_FRAME_SAMPLES:
            current = self._pending[:_VAD_FRAME_SAMPLES]
            self._pending = self._pending[_VAD_FRAME_SAMPLES:]
            event = self._iterator(self._torch.from_numpy(current), return_seconds=True)
            if isinstance(event, dict):
                if "start" in event:
                    self._silero_active = True
                if "end" in event:
                    self._silero_active = False
            # Judged on every frame, not only on silero's transitions, so the
            # gate can still open in the middle of a segment silero opened long
            # ago for a far-away voice.
            open_gate = self._advance_gate(_rms(current), _VAD_FRAME_SEC, candidate=self._silero_active)
            if self._silero_active and open_gate and not self._in_speech:
                self._in_speech = True
                event_type = "start"
            elif not self._silero_active and self._in_speech:
                self._in_speech = False
                event_type = "end"
        return event_type

    def _push_energy(self, chunk: np.ndarray) -> str | None:
        rms = _rms(chunk)
        self.last_rms = rms
        now = time.time()
        # No silero to ask here, so a loud frame is its own speech candidate;
        # the floor and the minimum duration then apply exactly as above.
        loud = rms >= self.level_threshold
        open_gate = self._advance_gate(rms, chunk.size / SAMPLE_RATE, candidate=loud)
        if loud:
            self._silence_started = None
            if open_gate and not self._in_speech:
                self._in_speech = True
                return "start"
            return None
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

    def _advance_gate(self, rms: float, duration_sec: float, *, candidate: bool) -> bool:
        """Feed one frame to the level gate and report whether it is open.

        ``candidate`` means "something already thinks this frame is speech" -
        silero's state, or simply a loud frame in the fallback. Only frames that
        are neither speech nor a candidate teach the noise floor, which is what
        stops a talker from slowly raising the very floor they have to clear.
        """
        if not (candidate or self._in_speech):
            self._observe_noise(rms, duration_sec)
            self._loud_sec = 0.0
            return False
        if rms < self.level_threshold:
            self._loud_sec = 0.0
            return False
        self._loud_sec += duration_sec
        return self._loud_sec >= self._min_speech_sec

    def _observe_noise(self, rms: float, duration_sec: float) -> None:
        frames = max(1.0, duration_sec / _VAD_FRAME_SEC)
        if self._noise_frames < _NOISE_FLOOR_WARMUP_FRAMES:
            # A running mean over the first frames, so the very first utterance
            # is judged against the real room instead of against zero.
            self._noise_frames += 1
            self._noise_floor += (rms - self._noise_floor) / self._noise_frames
            return
        # One EMA step per silero frame: a 1 s chunk from the energy fallback
        # then moves the floor as far as the 31 frames it contains would.
        weight = 1.0 - (1.0 - _NOISE_FLOOR_ALPHA) ** frames
        self._noise_floor += weight * (rms - self._noise_floor)

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    @property
    def level_threshold(self) -> float:
        """RMS a frame has to reach right now to count as near-field speech."""
        return max(self._energy_threshold, self._noise_floor * self._snr_ratio)


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
                    session = None
            elif command == "cancel":
                await ws.send(_json_event({"type": "cancelled"}))
                session = None
            else:
                await ws.send(_json_event({"type": "error", "error": f"未知 ASR 指令：{command}"}))
            continue
        if session is None:
            await ws.send(_json_event({"type": "error", "error": "ASR 会话尚未启动。"}))
            continue
        chunk = _decode_audio_frame(message)
        if chunk is None:
            continue
        event = await _push_audio(ws, session, chunk)
        await ws.send(_json_event(event))
        if event.get("type") == "final":
            session = None


async def _push_audio(ws: Any, session: AsrStreamSession, chunk: np.ndarray) -> dict[str, Any]:
    """Push one audio frame, forwarding VAD speech events as they happen.

    ``push_float32`` may run a decode that takes hundreds of milliseconds, and
    the speech ``start`` of that very frame is what the client needs to stop the
    assistant's audio. So the events are drained while the decode is still in
    its thread instead of after it returns. All sends stay on this one task -
    the session lock is released around inference, so draining never blocks the
    event loop.
    """
    push = asyncio.ensure_future(asyncio.to_thread(session.push_float32, chunk))
    try:
        while True:
            done, _ = await asyncio.wait({push}, timeout=SPEECH_EVENT_POLL_SEC)
            for speech_event in session.take_speech_events():
                await ws.send(_json_event(speech_event))
            if done:
                return push.result()
    finally:
        # A send that fails (client gone) leaves the decode running; drop the
        # future so it is not reported as an unretrieved task exception.
        push.cancel()


@lru_cache(maxsize=1)
def _silero_model() -> Any:
    from silero_vad import load_silero_vad

    return load_silero_vad()


def _json_event(payload: dict[str, Any]) -> str:
    return __import__("json").dumps(payload, ensure_ascii=False)


def _decode_audio_frame(message: Any) -> np.ndarray | None:
    """Decode one binary float32 frame, or ``None`` when it is malformed.

    A truncated websocket frame (length not a multiple of 4) used to raise out
    of the receive loop and kill the session; such frames are dropped instead.
    """
    try:
        return np.frombuffer(message, dtype=np.float32).reshape(-1)
    except (ValueError, TypeError, BufferError) as exc:
        _warn_once(f"asr-frame:{type(exc).__name__}", f"忽略无效的 ASR 音频帧：{exc}")
        return None


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    print(f"[langcode-asr] {message}")


def _sanitize_audio(chunk: np.ndarray) -> np.ndarray:
    chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
    if chunk.size == 0:
        return chunk
    return np.nan_to_num(np.clip(chunk, -1.0, 1.0), copy=False)


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0


def _join_text(prefix: str, tail: str) -> str:
    """Concatenate a committed prefix and a freshly decoded tail.

    A space is only ever inserted in front of an ASCII *letter*: a leading digit
    is far more likely to continue a number or identifier split across two
    decodes (``'abc' + '123'`` -> ``'abc123'``), and CJK never takes spaces.
    """
    prefix = (prefix or "").strip()
    tail = (tail or "").strip()
    if not prefix:
        return tail
    if not tail:
        return prefix
    last, first = prefix[-1], tail[0]
    if not (first.isascii() and first.isalpha()):
        return prefix + tail
    if last.isascii() and (last.isalnum() or last in _JOIN_SPACE_AFTER):
        return f"{prefix} {tail}"
    return prefix + tail


def _accepts_kwargs(fn: Any, names: tuple[str, ...]) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return True
    return all(name in parameters for name in names)


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


def _str_env(name: str, fallback: str) -> str:
    value = (os.getenv(name) or "").strip()
    return value or fallback


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
