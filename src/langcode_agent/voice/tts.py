from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import base64
import json
import mimetypes
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any
from urllib import parse, request

import numpy as np
import soundfile as sf

from .mlx_cosyvoice3 import DEFAULT_PREVIEW_TEXT, MlxCosyVoice3Service


DEFAULT_VOICE_ID = "xuefen"
SYSTEM_VOICE_ID = "default"
DEFAULT_REMOTE_MODEL = "remote-tts"
FALLBACK_NOTICE = "MLX 合成失败，已回退到系统语音"
# Server-side sentence chunking for /api/tts/stream: the first group is cut
# short so playback can start while the rest is still being synthesized.
FIRST_CHUNK_MAX_CHARS = 20
CHUNK_MAX_CHARS = 60
HARD_BOUNDARIES = "。！？!?；;\n"
SOFT_BOUNDARIES = "，,、"
# A cut may never land inside a URL or inside a quote/bracket pair.
_URL_RE = re.compile(r"https?://\S+")
_PAIRS = {"(": ")", "（": "）", "[": "]", "【": "】", "“": "”", "‘": "’", "「": "」", "《": "》"}
_OPENERS = set(_PAIRS)
_CLOSERS = {closing: opening for opening, closing in _PAIRS.items()}
_QUOTE_TOGGLES = {'"', "'"}
_TRAILING_WORD_RE = re.compile(r"[A-Za-z.]+$")
_ABBREVIATIONS = {"e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "prof", "st", "no", "fig", "al", "approx"}


@dataclass(frozen=True)
class TtsSettings:
    enabled: bool = True
    provider: str = "auto"
    model_repo: str = DEFAULT_REMOTE_MODEL
    base_url: str = ""
    voice: str = ""
    local_model_dir: str = field(default_factory=lambda: str(_default_mlx_model_dir()))
    voice_dir: str = field(default_factory=lambda: str(_default_voice_dir()))
    sample_dir: str = field(default_factory=lambda: str(Path.cwd()))
    preload: bool = True
    timeout_sec: float = 15.0
    worker_count: int = 1

    @classmethod
    def from_env(cls) -> "TtsSettings":
        return cls(
            enabled=_truthy(os.getenv("LANGCODE_TTS_ENABLED", "1")),
            provider=(os.getenv("LANGCODE_TTS_PROVIDER") or "auto").strip().lower(),
            model_repo=(os.getenv("LANGCODE_TTS_MODEL_REPO") or DEFAULT_REMOTE_MODEL).strip(),
            base_url=(os.getenv("LANGCODE_TTS_BASE_URL") or "").strip().rstrip("/"),
            voice=(os.getenv("LANGCODE_TTS_VOICE") or "").strip(),
            local_model_dir=(os.getenv("LANGCODE_TTS_MODEL_DIR") or str(_default_mlx_model_dir())).strip(),
            voice_dir=(os.getenv("LANGCODE_TTS_VOICE_DIR") or str(_default_voice_dir())).strip(),
            sample_dir=(os.getenv("LANGCODE_TTS_SAMPLE_DIR") or str(Path.cwd())).strip(),
            preload=_truthy(os.getenv("LANGCODE_TTS_PRELOAD", "1")),
            timeout_sec=_float_env("LANGCODE_TTS_TIMEOUT_SEC", 15.0),
            worker_count=_int_env("LANGCODE_TTS_WORKERS", 1),
        )


@dataclass
class TtsStatus:
    state: str = "idle"
    error: str = ""
    loaded_at: float | None = None


class TtsService:
    """TTS facade.

    Project-local MLX/CosyVoice3 is preferred for custom assistant playback.
    The model runs in a dedicated worker thread because MLX/Metal model objects
    are not safe to use from arbitrary request threads.
    """

    def __init__(self, settings: TtsSettings | None = None) -> None:
        self.settings = settings or TtsSettings.from_env()
        self._status = TtsStatus()
        self._mlx = MlxCosyVoice3Service(
            model_dir=Path(self.settings.local_model_dir).expanduser(),
            voice_dir=Path(self.settings.voice_dir).expanduser(),
            timeout_sec=max(self.settings.timeout_sec, 120.0),
            worker_count=self.settings.worker_count,
        )

    def start_preload(self) -> None:
        if not self.settings.enabled or not self.settings.preload:
            return
        if self._should_try_mlx():
            self._mlx.start()
            self._status.state = "loading"
            self._status.loaded_at = self._status.loaded_at or time.time()
            return
        if self.settings.base_url:
            self._status.state = "remote-ready"
            self._status.loaded_at = self._status.loaded_at or time.time()
            return
        if _say_available():
            self._status.state = "ready"
            self._status.loaded_at = self._status.loaded_at or time.time()

    def status(self) -> dict[str, Any]:
        assets = self.asset_status()
        mlx_status = self._mlx.status()
        state = "ready" if mlx_status.get("ready") else "loading" if mlx_status.get("loading") else self._status.state
        mlx_error_blocks_service = bool(mlx_status.get("error")) and not bool(mlx_status.get("ready"))
        return {
            "ok": self.settings.enabled and assets["ready"] and not mlx_error_blocks_service and self._status.state != "error",
            "enabled": self.settings.enabled,
            "provider": self.settings.provider,
            "modelRepo": self.settings.model_repo,
            "baseUrl": self.settings.base_url,
            "voice": self.settings.voice,
            "voiceId": SYSTEM_VOICE_ID,
            "localModelDir": str(self._mlx.model_dir),
            "voices": self.list_voices(),
            "mode": self._mode(),
            "state": state,
            "error": mlx_status.get("error") or self._status.error,
            "loadedAt": mlx_status.get("loadedAt") or self._status.loaded_at,
            "assets": assets,
            "fallback": "macos-say" if _say_available() else "",
            "previewText": DEFAULT_PREVIEW_TEXT,
            "workerCount": self.settings.worker_count,
            "mlx": mlx_status,
        }

    def asset_status(self, voice_id: str = "") -> dict[str, Any]:
        mlx_status = self._mlx.status()
        local_ready = bool(mlx_status.get("ready"))
        local_loading = bool(mlx_status.get("loading"))
        model_ready = self._mlx.model_dir.exists()
        voices = self._mlx.voices()
        profile_ready = all(bool(voice.get("profileReady")) for voice in voices)
        preview_ready = any(bool(voice.get("previewReady")) for voice in voices)
        remote_ready = bool(self.settings.base_url)
        say_ready = platform.system() == "Darwin" and _say_available()
        missing: list[str] = []
        if not model_ready:
            missing.append(str(self._mlx.model_dir))
        if not profile_ready:
            missing.append(str(Path(self.settings.voice_dir).expanduser() / "profiles"))
        if not local_ready and not local_loading and not remote_ready and not say_ready:
            missing.append("MLX CosyVoice3、LANGCODE_TTS_BASE_URL 或 macOS say")
        return {
            "ready": local_ready or local_loading or remote_ready or say_ready,
            "missing": missing,
            "localReady": local_ready,
            "localLoading": local_loading,
            "modelReady": model_ready,
            "profileReady": profile_ready,
            "remoteReady": remote_ready,
            "sayReady": say_ready,
            "previewReady": preview_ready,
        }

    def list_voices(self) -> list[dict[str, Any]]:
        voices: list[dict[str, Any]] = [self._system_voice_payload()]
        voices.extend(self._mlx.voices())
        existing = {_voice_identity(item) for item in voices}
        existing_ids = {str(item["id"]) for item in voices}
        for voice in self._local_voice_profiles():
            voice_id = str(voice.get("id") or "")
            identity = _voice_identity(voice)
            if voice_id and voice_id not in existing_ids and identity not in existing:
                existing_ids.add(voice_id)
                existing.add(identity)
                voices.append(voice)
        voices.extend(self._remote_voice_profiles(existing=existing_ids, existing_identities=existing))
        return voices

    def create_voice_profile(self, *, name: str, prompt_text: str, style: str, wav_bytes: bytes) -> dict[str, Any]:
        if not wav_bytes.startswith(b"RIFF"):
            raise ValueError("音色样本必须是 WAV/RIFF 音频")
        if len(wav_bytes) > 25 * 1024 * 1024:
            raise ValueError("音色样本过大，请控制在 25MB 以内")
        voice_dir = self._voice_dir()
        voice_dir.mkdir(parents=True, exist_ok=True)
        label = (name or style or "custom-voice").strip()[:40]
        voice_id = f"{_slugify(label)}-{uuid.uuid4().hex[:8]}"
        wav_path = voice_dir / f"{voice_id}.wav"
        metadata_path = voice_dir / f"{voice_id}.json"
        wav_path.write_bytes(wav_bytes)
        metadata = {
            "id": voice_id,
            "name": label or voice_id,
            "style": (style or "").strip()[:200],
            "promptText": (prompt_text or "").strip()[:1000],
            "promptWav": str(wav_path),
            "sourceAudio": str(wav_path),
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._voice_payload(metadata, metadata_path.parent) | {"builtIn": False}

    def voice_preview_path(self, voice_id: str) -> Path | None:
        voice_id = (voice_id or "").strip()
        if _is_system_voice_id(voice_id):
            return None
        preview_path = self._mlx.preview_path(voice_id or DEFAULT_VOICE_ID)
        if preview_path is not None:
            return preview_path
        for voice in self._local_voice_profiles():
            if voice.get("id") != voice_id:
                continue
            for key in ("previewWav", "sourceAudio", "promptWav"):
                path = Path(str(voice.get(key) or "")).expanduser()
                if path.exists() and path.is_file():
                    return path
        return None

    def voice_preview(self, voice_id: str) -> tuple[bytes, str]:
        voice_id = (voice_id or "").strip()
        if _is_system_voice_id(voice_id):
            return self.synthesize(DEFAULT_PREVIEW_TEXT, SYSTEM_VOICE_ID)
        path = self.voice_preview_path(voice_id)
        if path is None:
            raise FileNotFoundError(f"未找到音色试听文件：{voice_id}")
        path = path.expanduser().resolve()
        return path.read_bytes(), content_type_for_path(path)

    def synthesize(self, text: str, voice_id: str = "") -> tuple[bytes, str]:
        audio, content_type, _meta = self.synthesize_with_meta(text, voice_id=voice_id)
        return audio, content_type

    def synthesize_with_meta(self, text: str, voice_id: str = "") -> tuple[bytes, str, dict[str, Any]]:
        """Synthesize one segment and report which backend actually produced it.

        ``meta`` is ``{"provider": str, "fallback": str, "reason": str}``.
        ``fallback`` is non-empty only when the requested backend failed and the
        audio came from macOS ``say`` instead, so callers can surface a notice.
        """
        text = _clean_text(text)
        if not text:
            raise ValueError("TTS 文本为空")
        if not self.settings.enabled:
            raise RuntimeError("TTS 已禁用")
        if _is_system_voice_id(voice_id):
            if self._uses_system_say():
                audio = self._synthesize_system_say(text)
                return audio, "audio/wav", _tts_meta("macos-say")
            raise RuntimeError("系统默认音色需要 macOS say，但当前不可用")
        last_error = ""
        if self._should_try_mlx():
            try:
                audio, content_type = self._synthesize_mlx(text, voice_id or DEFAULT_VOICE_ID)
                return audio, content_type, _tts_meta("mlx-cosyvoice3")
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        if self._should_try_remote(voice_id):
            try:
                audio, content_type = self._synthesize_external(text, voice_id=voice_id)
                audio, content_type = _trim_speech_audio(audio, content_type)
                return audio, content_type, _tts_meta("remote")
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        if self._uses_system_say():
            audio = self._synthesize_system_say(text)
            return audio, "audio/wav", _tts_meta("macos-say", fallback="macos-say", reason=last_error)
        if last_error:
            raise RuntimeError(f"远程 TTS 不可用，且本机 say 不可用：{last_error}")
        raise RuntimeError("没有可用的 TTS 服务")

    def synthesize_chunks(
        self,
        text: str,
        voice_id: str = "",
        *,
        on_meta: Callable[[dict[str, Any]], None] | None = None,
    ) -> Iterator[tuple[bytes, str]]:
        """Yield one synthesized audio blob per sentence group.

        The text is split server-side on sentence boundaries so playback can
        start after the first (deliberately short) group instead of waiting for
        the whole answer. ``on_meta`` — when given — is called with the metadata
        of each segment *before* the audio is yielded, which lets a streaming
        caller emit a fallback notice ahead of the first audio event.
        """
        cleaned = _clean_text(text)
        if not cleaned:
            raise ValueError("TTS 文本为空")
        for index, part in enumerate(split_text_for_streaming(cleaned), start=1):
            audio, content_type, meta = self.synthesize_with_meta(part, voice_id=voice_id)
            if on_meta is not None:
                on_meta(dict(meta, index=index, text=part))
            yield audio, content_type

    def _local_voice_profiles(self) -> list[dict[str, Any]]:
        seen: set[str] = set()
        voices: list[dict[str, Any]] = []
        sample_dir = Path(self.settings.sample_dir).expanduser()
        if sample_dir.exists():
            for sample_path in sorted(sample_dir.glob("*.wav")):
                voice_id = sample_path.stem.strip()
                if not voice_id or voice_id in seen or voice_id == DEFAULT_VOICE_ID:
                    continue
                metadata = _load_sidecar_metadata(sample_path, self._candidate_voice_dirs())
                metadata.update(
                    {
                        "id": metadata.get("id") or voice_id,
                        "name": metadata.get("name") or voice_id,
                    }
                )
                metadata["sourceAudio"] = str(sample_path)
                metadata["promptWav"] = str(sample_path)
                seen.add(voice_id)
                voices.append(self._voice_payload(metadata, sample_dir))
        for voice_dir in self._candidate_voice_dirs():
            if not voice_dir.exists():
                continue
            for metadata_path in sorted(voice_dir.glob("*.json")):
                try:
                    data = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                voice = self._voice_payload(data, metadata_path.parent)
                voice_id = str(voice.get("id") or "").strip()
                if not voice_id or voice_id in seen or voice_id == DEFAULT_VOICE_ID:
                    continue
                seen.add(voice_id)
                voices.append(voice)
        return voices

    def _system_voice_payload(self) -> dict[str, Any]:
        return {
            "id": SYSTEM_VOICE_ID,
            "name": "默认音色",
            "style": "macOS 系统音色",
            "promptText": "",
            "promptWav": "",
            "sourceAudio": "",
            "previewWav": "",
            "previewUrl": f"/api/tts/voices/{SYSTEM_VOICE_ID}/preview",
            "previewText": DEFAULT_PREVIEW_TEXT,
            "previewReady": self._uses_system_say(),
            "profileReady": self._uses_system_say(),
            "builtIn": True,
            "provider": "macos-say",
        }

    def _voice_payload(self, data: dict[str, Any], voice_dir: Path) -> dict[str, Any]:
        voice_id = str(data.get("id") or "").strip()
        if not voice_id:
            voice_id = _slugify(str(data.get("name") or voice_dir.name or "voice"))
        prompt_wav = _first_existing_path(data.get("promptWav"), voice_dir / f"{voice_id}.wav")
        source_audio = _first_existing_path(data.get("sourceAudio"), voice_dir / f"{voice_id}.m4a", prompt_wav)
        preview_path = _first_existing_path(data.get("previewWav"), source_audio, prompt_wav)
        preview_ready = preview_path is not None and preview_path.exists()
        payload = {
            "id": voice_id,
            "name": str(data.get("name") or voice_id),
            "style": str(data.get("style") or ""),
            "promptText": str(data.get("promptText") or ""),
            "promptWav": str(prompt_wav) if prompt_wav else "",
            "sourceAudio": str(source_audio) if source_audio else "",
            "previewWav": str(preview_path) if preview_ready else "",
            "previewUrl": f"/api/tts/voices/{parse.quote(voice_id)}/preview" if preview_ready else "",
            "previewText": DEFAULT_PREVIEW_TEXT if preview_ready else "",
            "previewReady": preview_ready,
            "createdAt": data.get("createdAt"),
            "builtIn": False,
        }
        return payload

    def _remote_voice_profiles(self, *, existing: set[str], existing_identities: set[str] | None = None) -> list[dict[str, Any]]:
        if not self.settings.base_url:
            return []
        existing_identities = existing_identities if existing_identities is not None else set()
        voices: list[dict[str, Any]] = []
        for endpoint in ("/api/tts/voices", "/v1/audio/voices", "/voices"):
            try:
                req = request.Request(f"{self.settings.base_url}{endpoint}", method="GET")
                with request.urlopen(req, timeout=min(self.settings.timeout_sec, 3.0)) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except Exception:
                continue
            items = payload.get("voices") or payload.get("data") or []
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, str):
                    item = {"id": item, "name": item}
                if not isinstance(item, dict):
                    continue
                voice_id = str(item.get("id") or item.get("voice") or "").strip()
                identity = _voice_identity({"id": voice_id, "name": item.get("name") or voice_id})
                if not voice_id or voice_id in existing or identity in existing_identities:
                    continue
                existing.add(voice_id)
                existing_identities.add(identity)
                voices.append(
                    {
                        "id": voice_id,
                        "name": str(item.get("name") or voice_id),
                        "style": str(item.get("style") or "远程音色"),
                        "builtIn": False,
                        "remote": True,
                        "previewReady": False,
                    }
                )
            if voices:
                return voices
        return voices

    def _synthesize_mlx(self, text: str, voice_id: str) -> tuple[bytes, str]:
        """Synthesize through MLX/CosyVoice3, trimming in the float32 domain.

        ``synthesize_samples`` hands back the waveform the worker already has,
        so the leading/trailing silence is cut on the array and the WAV is
        encoded once — instead of encode -> decode -> trim -> re-encode.
        """
        samples_fn = getattr(self._mlx, "synthesize_samples", None)
        if callable(samples_fn):
            samples, sample_rate = samples_fn(text, voice_id)
            samples = np.asarray(samples, dtype=np.float32)
            if sample_rate <= 0:
                raise RuntimeError(f"MLX TTS 返回了无效的采样率：{sample_rate}")
            # An empty waveform is a result, not a reason to pay for the whole
            # synthesis again through the WAV API.
            if not samples.size:
                return _wav_buffer(samples.reshape(-1), int(sample_rate)).getvalue(), "audio/wav"
            return _wav_buffer(_trim_speech_samples(samples, int(sample_rate)), int(sample_rate)).getvalue(), "audio/wav"
        audio, content_type = self._mlx.synthesize(text, voice_id)
        return _trim_speech_audio(audio, content_type)

    def _synthesize_external(self, text: str, voice_id: str = "") -> tuple[bytes, str]:
        voice = voice_id or self.settings.voice or DEFAULT_VOICE_ID
        payload = {
            "text": text,
            "input": text,
            "model": self.settings.model_repo,
            "voice": voice,
            "voiceId": voice,
            "stream": False,
            "language": "zh",
        }
        endpoints = [
            f"{self.settings.base_url}/api/tts/speech",
            f"{self.settings.base_url}/v1/audio/speech",
            f"{self.settings.base_url}/tts",
            f"{self.settings.base_url}/generate",
        ]
        last_error = ""
        for url in endpoints:
            try:
                req = request.Request(
                    url,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "audio/wav,audio/mpeg,application/json,application/octet-stream",
                    },
                    method="POST",
                )
                with request.urlopen(req, timeout=self.settings.timeout_sec) as resp:
                    data = resp.read()
                    content_type = (resp.headers.get("Content-Type") or "audio/wav").split(";")[0]
                    if not data:
                        continue
                    if content_type == "application/json":
                        return _audio_from_json_response(data)
                    return data, content_type
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(f"远程 TTS 请求失败：{last_error}")

    def _synthesize_system_say(self, text: str) -> bytes:
        if platform.system() != "Darwin" or not _say_available():
            raise RuntimeError("当前系统没有可用的 macOS say TTS")
        voice = self.settings.voice or os.getenv("LANGCODE_TTS_SAY_VOICE") or "Tingting"
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
            output_path = Path(tmp.name)
        try:
            subprocess.run(
                ["say", "-v", voice, "-o", str(output_path), text],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=max(5.0, min(self.settings.timeout_sec, 60.0)),
            )
            audio, sample_rate = sf.read(output_path, dtype="float32")
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # Trim while the float32 array is still in hand: no decode -> trim ->
            # re-encode round trip for the macOS say path.
            return _wav_buffer(_trim_speech_samples(audio.reshape(-1), int(sample_rate)), int(sample_rate)).getvalue()
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"macOS say TTS 失败：{detail or exc}") from exc
        finally:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _should_try_remote(self, voice_id: str = "") -> bool:
        provider = self.settings.provider.strip().lower()
        if provider in {"say", "macos-say", "system-say", "mlx", "mlx-cosyvoice3", "local"}:
            return False
        return bool(self.settings.base_url)

    def _should_try_mlx(self) -> bool:
        provider = self.settings.provider.strip().lower()
        return provider in {"", "auto", "mlx", "mlx-cosyvoice3", "local"} and self._mlx.model_dir.exists()

    def _uses_system_say(self) -> bool:
        return platform.system() == "Darwin" and _say_available()

    def _mode(self) -> str:
        provider = self.settings.provider.strip().lower()
        mlx_status = self._mlx.status()
        if self._should_try_mlx():
            if mlx_status.get("ready"):
                return "mlx-cosyvoice3"
            if mlx_status.get("loading"):
                return "mlx-cosyvoice3-loading"
            return "mlx-cosyvoice3-with-fallback"
        if self.settings.base_url and provider not in {"say", "macos-say", "system-say"}:
            return "remote-with-say-fallback" if _say_available() else "remote"
        if self._uses_system_say():
            return "system-say"
        return "unavailable"

    def _voice_dir(self) -> Path:
        return Path(self.settings.voice_dir).expanduser()

    def _candidate_voice_dirs(self) -> list[Path]:
        paths = [Path(self.settings.voice_dir).expanduser()]
        unique: list[Path] = []
        for path in paths:
            if path not in unique:
                unique.append(path)
        return unique


def _audio_from_json_response(data: bytes) -> tuple[bytes, str]:
    payload = json.loads(data.decode("utf-8"))
    content_type = str(payload.get("contentType") or payload.get("mimeType") or "audio/wav")
    audio_value = payload.get("audio") or payload.get("audioBase64") or payload.get("data")
    if isinstance(audio_value, dict):
        content_type = str(audio_value.get("contentType") or content_type)
        audio_value = audio_value.get("base64") or audio_value.get("audio")
    if not isinstance(audio_value, str) or not audio_value:
        raise RuntimeError("远程 TTS JSON 响应缺少 audio/audioBase64 字段")
    if "," in audio_value and audio_value.strip().lower().startswith("data:"):
        header, audio_value = audio_value.split(",", 1)
        if ";" in header:
            content_type = header[5:].split(";", 1)[0] or content_type
    return base64.b64decode(audio_value), content_type


def _wav_buffer(audio: np.ndarray, sample_rate: int) -> BytesIO:
    buffer = BytesIO()
    sf.write(buffer, audio.astype(np.float32, copy=False), sample_rate, format="WAV")
    return buffer


def _frame_rms(mono: np.ndarray, frame_size: int) -> np.ndarray:
    """Per-frame RMS, vectorized (the trailing partial frame is kept)."""
    if mono.size == 0:
        return np.zeros(0, dtype=np.float32)
    full_frames = mono.size // frame_size
    values: list[np.ndarray] = []
    if full_frames:
        block = mono[: full_frames * frame_size].reshape(full_frames, frame_size)
        values.append(np.sqrt(np.mean(np.square(block, dtype=np.float64), axis=1)))
    tail = mono[full_frames * frame_size :]
    if tail.size:
        values.append(np.sqrt(np.mean(np.square(tail, dtype=np.float64)))[None])
    if not values:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(values)


def _speech_bounds(mono: np.ndarray, sample_rate: int) -> tuple[int, int] | None:
    """Return the [start, end) sample range holding speech, or None."""
    if mono.size == 0 or sample_rate <= 0:
        return None
    frame_size = max(1, int(sample_rate * 0.02))
    rms = _frame_rms(mono, frame_size)
    if rms.size == 0:
        return None
    peak = float(rms.max())
    if peak <= 1e-6:
        return None
    # Absolute floor: a relative-only gate turns room noise into "speech" when
    # the peak is quiet. The wider lead pad keeps soft attacks (fricatives, a
    # rising vowel onset) that sit below the gate in front of the first frame.
    threshold = max(peak * 0.04, 1e-3)
    voiced = np.flatnonzero(rms >= threshold)
    if voiced.size == 0:
        return None
    lead_pad = int(sample_rate * 0.05)
    trail_pad = int(sample_rate * 0.08)
    start = max(0, int(voiced[0]) * frame_size - lead_pad)
    end = min(int(mono.size), (int(voiced[-1]) + 1) * frame_size + trail_pad)
    if end <= start or (start == 0 and end == mono.size):
        return None
    if end - start < int(sample_rate * 0.2):
        return None
    return start, end


def _trim_speech_samples(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    mono = samples if samples.ndim == 1 else samples.mean(axis=1)
    bounds = _speech_bounds(mono, sample_rate)
    if bounds is None:
        return samples
    start, end = bounds
    return samples[start:end]


def _trim_speech_audio(audio: bytes, content_type: str) -> tuple[bytes, str]:
    content_type = content_type or "audio/wav"
    if not audio or not audio.startswith(b"RIFF") or "wav" not in content_type.lower():
        return audio, content_type
    try:
        samples, sample_rate = sf.read(BytesIO(audio), dtype="float32", always_2d=True)
    except Exception:
        return audio, content_type
    if samples.size == 0 or sample_rate <= 0:
        return audio, content_type
    bounds = _speech_bounds(samples.mean(axis=1), int(sample_rate))
    if bounds is None:
        return audio, content_type
    start, end = bounds
    trimmed = samples[start:end]
    return _wav_buffer(trimmed.squeeze() if samples.shape[1] == 1 else trimmed, sample_rate).getvalue(), content_type


_MD_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")
_MD_RULE_RE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_MD_TABLE_DIVIDER_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-+:?\s*)+\|?\s*$")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*")
_MD_BULLET_RE = re.compile(r"^\s{0,3}[-*+]\s+")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_BOLD_STAR_RE = re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*", re.S)
_MD_BOLD_UNDERSCORE_RE = re.compile(r"(?<![A-Za-z0-9_])__(?!\s)(.+?)(?<!\s)__(?![A-Za-z0-9_])", re.S)
_MD_ITALIC_STAR_RE = re.compile(r"(?<![A-Za-z0-9_*])\*(?!\s)([^*]+?)(?<!\s)\*(?![A-Za-z0-9_*])")
_MD_ITALIC_UNDERSCORE_RE = re.compile(r"(?<![A-Za-z0-9_])_(?!\s)([^_]+?)(?<!\s)_(?![A-Za-z0-9_])")


def _clean_text(text: str) -> str:
    return " ".join(_strip_markdown(text).split())[:2000]


def _strip_markdown(text: str) -> str:
    """Turn a Markdown answer into something worth speaking aloud.

    Fenced code blocks, images and horizontal rules are dropped; headings,
    bullets, emphasis markers and inline-code backticks lose the marker but keep
    their content; ``[label](url)`` keeps the label; a table row becomes its
    cells joined with ``，`` so the row is read as one phrase.
    """
    lines: list[str] = []
    in_fence = False
    for raw_line in str(text or "").splitlines():
        if _MD_FENCE_RE.match(raw_line):
            in_fence = not in_fence
            continue
        if in_fence or _MD_RULE_RE.match(raw_line) or _MD_TABLE_DIVIDER_RE.match(raw_line):
            continue
        line = _MD_HEADING_RE.sub("", raw_line)
        line = _MD_BULLET_RE.sub("", line)
        line = _MD_IMAGE_RE.sub("", line)
        line = _MD_LINK_RE.sub(r"\1", line)
        line = line.replace("`", "")
        if _is_table_row(line):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            line = "，".join(cell for cell in cells if cell)
        line = _MD_BOLD_STAR_RE.sub(r"\1", line)
        line = _MD_BOLD_UNDERSCORE_RE.sub(r"\1", line)
        line = _MD_ITALIC_STAR_RE.sub(r"\1", line)
        line = _MD_ITALIC_UNDERSCORE_RE.sub(r"\1", line)
        if line.strip():
            lines.append(line.strip())
    return "\n".join(lines)


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2


def _tts_meta(provider: str, *, fallback: str = "", reason: str = "") -> dict[str, Any]:
    return {"provider": provider, "fallback": fallback, "reason": reason}


def split_text_for_streaming(
    text: str,
    *,
    first_max_chars: int = FIRST_CHUNK_MAX_CHARS,
    max_chars: int = CHUNK_MAX_CHARS,
) -> list[str]:
    """Split text into sentence groups for incremental TTS synthesis.

    Hard sentence boundaries (``。！？!?；;``, newlines and an English ``.``
    followed by whitespace) end a group. Groups are packed up to ``max_chars``
    (``first_max_chars`` for the first one, so playback starts early); an
    over-long group is cut at the last soft boundary (``，,、``) that fits, then
    at whitespace, and only then hard-cut.

    Cuts never land inside a URL or inside paired quotes/brackets, a group that
    holds nothing speakable is merged into a neighbour instead of being sent to
    the synthesizer, and every group is re-split against its own limit after
    those merges.
    """
    text = str(text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    current = ""

    def limit() -> int:
        return max(1, first_max_chars if not chunks else max_chars)

    def flush(value: str) -> None:
        value = value.strip()
        if value:
            chunks.append(value)

    for unit in _split_sentence_units(text):
        if not current:
            current = unit
        elif len(current) + len(unit) <= limit():
            current += unit
        else:
            flush(current)
            current = unit
        while len(current) > limit():
            cut = _soft_cut(current, limit())
            if cut >= len(current):
                break
            flush(current[:cut])
            current = current[cut:].lstrip()
    flush(current)
    return _merge_and_resplit(chunks, first_max_chars, max_chars)


def _merge_and_resplit(chunks: list[str], first_max_chars: int, max_chars: int) -> list[str]:
    """Fold unspeakable groups into a neighbour, then re-apply the size limits."""
    merged: list[str] = []
    pending = ""
    for chunk in chunks:
        chunk = pending + chunk
        pending = ""
        if _has_speakable(chunk):
            merged.append(chunk)
        elif merged:
            merged[-1] = merged[-1] + chunk
        else:
            pending = chunk  # nothing to merge back into yet: carry it forward
    if pending and merged:
        merged[-1] = merged[-1] + pending
    result: list[str] = []
    for chunk in merged:
        limit = max(1, first_max_chars if not result else max_chars)
        while len(chunk) > limit:
            cut = _soft_cut(chunk, limit)
            if cut >= len(chunk):
                break
            piece = chunk[:cut].strip()
            if piece:
                result.append(piece)
            chunk = chunk[cut:].lstrip()
            limit = max(1, first_max_chars if not result else max_chars)
        if chunk.strip():
            result.append(chunk.strip())
    if len(result) > 1 and not _has_speakable(result[0]):
        result[1] = result[0] + result[1]
        del result[0]
    return result if any(_has_speakable(chunk) for chunk in result) else []


def _split_sentence_units(text: str) -> list[str]:
    protected, depth = _scan_text(text)
    units: list[str] = []
    current = ""
    for index, char in enumerate(text):
        current += char
        if not _cut_allowed(index, protected, depth):
            continue
        if char in HARD_BOUNDARIES or (char == "." and _is_english_sentence_end(text, index)):
            units.append(current)
            current = ""
    if current:
        units.append(current)
    return units


def _scan_text(text: str) -> tuple[list[bool], list[int]]:
    """Map out where ``text`` must not be cut.

    Returns a per-character ``protected`` flag (the character belongs to a URL)
    and a per-character nesting ``depth`` counting only quote/bracket pairs that
    actually close, so a stray apostrophe or bracket never freezes the splitter.
    """
    size = len(text)
    protected = [False] * size
    for match in _URL_RE.finditer(text):
        for index in range(match.start(), match.end()):
            protected[index] = True
    stack: list[tuple[str, int]] = []
    pairs: list[tuple[int, int]] = []
    for index, char in enumerate(text):
        if protected[index]:
            continue
        if char in _OPENERS:
            stack.append((char, index))
            continue
        opener = _CLOSERS.get(char)
        if opener is None and char in _QUOTE_TOGGLES and not _is_apostrophe(text, index):
            opener = char
        if opener is None:
            continue
        for position in range(len(stack) - 1, -1, -1):
            if stack[position][0] == opener:
                pairs.append((stack[position][1], index))
                del stack[position:]
                break
        else:
            if char in _QUOTE_TOGGLES:
                stack.append((char, index))
    delta = [0] * (size + 1)
    for start, end in pairs:
        delta[start] += 1
        delta[end] -= 1
    depth: list[int] = []
    running = 0
    for index in range(size):
        running += delta[index]
        depth.append(running)
    return protected, depth


def _cut_allowed(index: int, protected: list[bool], depth: list[int]) -> bool:
    """True when the text may be cut right after ``index``."""
    if index < 0 or index >= len(depth):
        return False
    if depth[index] > 0:
        return False
    return not (protected[index] and index + 1 < len(protected) and protected[index + 1])


def _is_apostrophe(text: str, index: int) -> bool:
    if text[index] != "'":
        return False
    before = text[index - 1] if index > 0 else ""
    after = text[index + 1] if index + 1 < len(text) else ""
    return before.isalnum() and after.isalnum()


def _is_english_sentence_end(text: str, index: int) -> bool:
    """True for a ``.`` that ends a sentence (not ``e.g.``, ``3.14`` or ``v1.2``)."""
    if index + 1 >= len(text) or not text[index + 1].isspace():
        return False
    if index > 0 and text[index - 1].isdigit():
        return False
    word = _TRAILING_WORD_RE.search(text[:index])
    return not (word and word.group(0).strip(".").lower() in _ABBREVIATIONS)


def _soft_cut(text: str, limit: int) -> int:
    """Best cut position at or before ``limit`` (may exceed it to keep a URL whole)."""
    protected, depth = _scan_text(text)
    window = min(limit, len(text))
    for index in range(window, 0, -1):
        char = text[index - 1]
        if (char in SOFT_BOUNDARIES or char in HARD_BOUNDARIES) and _cut_allowed(index - 1, protected, depth):
            return index
    for index in range(window, 0, -1):
        if text[index - 1].isspace() and _cut_allowed(index - 1, protected, depth):
            return index
    for index in range(window, 0, -1):
        if _cut_allowed(index - 1, protected, depth):
            return index
    for index in range(window + 1, len(text) + 1):
        if _cut_allowed(index - 1, protected, depth):
            return index
    return len(text)


def _has_speakable(text: str) -> bool:
    return any(char not in HARD_BOUNDARIES and char not in SOFT_BOUNDARIES and not char.isspace() for char in text)


def _first_existing_path(*values: Any) -> Path | None:
    for value in values:
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.exists():
            return path
    return None


def _load_sidecar_metadata(sample_path: Path, candidate_dirs: list[Path]) -> dict[str, Any]:
    candidates = [sample_path.with_suffix(".json")]
    candidates.extend(path / f"{sample_path.stem}.json" for path in candidate_dirs)
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            return dict(data)
    return {}


def content_type_for_path(path: Path) -> str:
    guessed, _encoding = mimetypes.guess_type(str(path))
    return guessed or "audio/wav"


def _say_available() -> bool:
    return shutil.which("say") is not None


def _is_system_voice_id(voice_id: str = "") -> bool:
    return (voice_id or "").strip().lower() in {SYSTEM_VOICE_ID, "system", "say", "macos-say", "system-say"}


_VOICE_ID_ALIASES = {
    "汪菊": "wangju",
    "雪芬": "xuefen",
}


def _voice_identity(voice: dict[str, Any]) -> str:
    voice_id = str(voice.get("id") or "").strip()
    name = str(voice.get("name") or "").strip()
    return (_VOICE_ID_ALIASES.get(voice_id) or _VOICE_ID_ALIASES.get(name) or voice_id or name).casefold()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff_-]+", "-", value.strip()).strip("-_").lower()
    return slug[:40] or "voice"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "y"}


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


def _default_voice_dir() -> Path:
    return Path.cwd() / ".langcode" / "tts-voices"


def _default_mlx_model_dir() -> Path:
    return Path.cwd() / ".langcode" / "tts-models" / "Fun-CosyVoice3-0.5B-2512-8bit"
