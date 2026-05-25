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
from collections.abc import Iterator
from typing import Any
from urllib import parse, request

import numpy as np
import soundfile as sf

from .mlx_cosyvoice3 import DEFAULT_PREVIEW_TEXT, MlxCosyVoice3Service


DEFAULT_VOICE_ID = "xuefen"
SYSTEM_VOICE_ID = "default"
DEFAULT_REMOTE_MODEL = "remote-tts"


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
        voices.extend(self._remote_voice_profiles(existing={str(item["id"]) for item in voices}))
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
        text = _clean_text(text)
        if not text:
            raise ValueError("TTS 文本为空")
        if not self.settings.enabled:
            raise RuntimeError("TTS 已禁用")
        if _is_system_voice_id(voice_id):
            if self._uses_system_say():
                return _trim_speech_audio(self._synthesize_system_say(text), "audio/wav")
            raise RuntimeError("系统默认音色需要 macOS say，但当前不可用")
        last_error = ""
        if self._should_try_mlx():
            try:
                audio, content_type = self._mlx.synthesize(text, voice_id or DEFAULT_VOICE_ID)
                return _trim_speech_audio(audio, content_type)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        if self._should_try_remote(voice_id):
            try:
                audio, content_type = self._synthesize_external(text, voice_id=voice_id)
                return _trim_speech_audio(audio, content_type)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        if self._uses_system_say():
            return _trim_speech_audio(self._synthesize_system_say(text), "audio/wav")
        if last_error:
            raise RuntimeError(f"远程 TTS 不可用，且本机 say 不可用：{last_error}")
        raise RuntimeError("没有可用的 TTS 服务")

    def synthesize_chunks(self, text: str, voice_id: str = "") -> Iterator[tuple[bytes, str]]:
        yield self.synthesize(text, voice_id=voice_id)

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

    def _remote_voice_profiles(self, *, existing: set[str]) -> list[dict[str, Any]]:
        if not self.settings.base_url:
            return []
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
                if not voice_id or voice_id in existing:
                    continue
                existing.add(voice_id)
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
            return _wav_buffer(audio.reshape(-1), int(sample_rate)).getvalue()
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
    mono = samples.mean(axis=1)
    frame_size = max(1, int(sample_rate * 0.02))
    rms: list[float] = []
    for start in range(0, len(mono), frame_size):
        frame = mono[start : start + frame_size]
        if frame.size:
            rms.append(float(np.sqrt(np.mean(frame * frame))))
    if not rms:
        return audio, content_type
    peak = max(rms)
    if peak <= 1e-6:
        return audio, content_type
    threshold = max(peak * 0.04, 1e-4)
    voiced = [index for index, value in enumerate(rms) if value >= threshold]
    if not voiced:
        return audio, content_type
    lead_pad = int(sample_rate * 0.04)
    trail_pad = int(sample_rate * 0.08)
    start = max(0, voiced[0] * frame_size - lead_pad)
    end = min(len(samples), (voiced[-1] + 1) * frame_size + trail_pad)
    if end <= start or (start == 0 and end == len(samples)):
        return audio, content_type
    if end - start < int(sample_rate * 0.2):
        return audio, content_type
    return _wav_buffer(samples[start:end].squeeze() if samples.shape[1] == 1 else samples[start:end], sample_rate).getvalue(), content_type


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").split())[:2000]


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
