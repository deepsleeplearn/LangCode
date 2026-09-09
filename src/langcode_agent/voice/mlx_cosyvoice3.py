from __future__ import annotations

import io
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


DEFAULT_MODEL_DIR = Path.cwd() / ".langcode" / "tts-models" / "Fun-CosyVoice3-0.5B-2512-8bit"
DEFAULT_VOICE_DIR = Path.cwd() / ".langcode" / "tts-voices"
DEFAULT_PREVIEW_TEXT = "欢迎使用LangCode，你的最后一个智能体。"


BUILTIN_VOICES: dict[str, dict[str, str]] = {
    "wangju": {
        "id": "wangju",
        "name": "汪菊",
        "style": "自定义音色：汪菊",
        "sample": "wangju.wav",
        "ref_text": "美国能源，但中方没有明确确认这个信息。说明这个信息差说明什么？是双方谈判口径不同，还是没有达成协议？现在没有达成协议。",
    },
    "xuefen": {
        "id": "xuefen",
        "name": "雪芬",
        "style": "自定义音色：雪芬",
        "sample": "xuefen.wav",
        "ref_text": "考个牛逼的，那同学你要想考牛逼高校的话你觉得你考这点分还够用吗？就不够用了，因为在咱们中国有一帮牛逼高校，他们在考研的时候有一个特权，他们可以干嘛？",
    },
}


@dataclass(frozen=True)
class VoiceProfile:
    voice_id: str
    name: str
    style: str
    ref_audio: str
    ref_text: str
    sample_rate: int
    arrays: dict[str, Any]


class MlxCosyVoice3Service:
    """MLX/CosyVoice3 runtime pool.

    MLX Metal objects are thread-affine in practice. Each runtime worker loads
    and uses its own model instance; callers enqueue synthesis requests onto a
    shared queue.
    """

    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        voice_dir: Path | None = None,
        timeout_sec: float = 120.0,
        worker_count: int = 1,
    ) -> None:
        self.model_dir = (model_dir or DEFAULT_MODEL_DIR).expanduser().resolve()
        self.voice_dir = (voice_dir or DEFAULT_VOICE_DIR).expanduser().resolve()
        self.samples_dir = self.voice_dir / "samples"
        self.profiles_dir = self.voice_dir / "profiles"
        self.previews_dir = self.voice_dir / "previews"
        self.timeout_sec = timeout_sec
        self.worker_count = max(1, min(int(worker_count or 1), 4))

        self._requests: queue.Queue[Any] = queue.Queue()
        self._lock = threading.Lock()
        self._profile_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._ready = False
        self._loading = False
        self._error = ""
        self._loaded_at: float | None = None
        self._voices: list[dict[str, Any]] = []
        self._ready_workers = 0
        self._loading_workers = 0
        self._worker_errors: dict[int, str] = {}

    def start(self) -> None:
        with self._lock:
            if self._threads:
                return
            self._loading = True
            self._error = ""
            self._ready_workers = 0
            self._loading_workers = self.worker_count
            self._worker_errors = {}
            self._threads = [
                threading.Thread(
                    target=self._worker_loop,
                    args=(index,),
                    name=f"langcode-mlx-cosyvoice3-{index + 1}",
                    daemon=True,
                )
                for index in range(self.worker_count)
            ]
            for thread in self._threads:
                thread.start()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ready": self._ready,
                "loading": self._loading,
                "error": self._error,
                "loadedAt": self._loaded_at,
                "modelDir": str(self.model_dir),
                "voiceDir": str(self.voice_dir),
                "voices": [dict(item) for item in self._voices],
                "workerCount": self.worker_count,
                "readyWorkers": self._ready_workers,
                "loadingWorkers": self._loading_workers,
                "queueSize": self._requests.qsize(),
                "workerErrors": dict(self._worker_errors),
            }

    def voices(self) -> list[dict[str, Any]]:
        voices = self._discover_voice_payloads()
        with self._lock:
            if voices:
                self._voices = voices
            return [dict(item) for item in self._voices or voices]

    def preview_path(self, voice_id: str) -> Path | None:
        voice_id = _normalize_voice_id(voice_id)
        for suffix in ("wav", "mp3", "m4a"):
            path = self.previews_dir / f"{voice_id}.{suffix}"
            if path.exists():
                return path
        sample = self.samples_dir / f"{voice_id}.wav"
        return sample if sample.exists() else None

    def synthesize_samples(self, text: str, voice_id: str) -> tuple[np.ndarray, int]:
        """Synthesize one segment and return the raw float32 samples.

        The worker already produces a float32 array; callers that post-process
        the waveform (silence trimming) should use this instead of
        ``synthesize`` so the audio is not encoded to WAV just to be decoded
        again.
        """
        text = " ".join(str(text or "").split())
        if not text:
            raise ValueError("TTS 文本为空")
        self.start()
        result_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._requests.put(("synthesize", _normalize_voice_id(voice_id), text, result_queue))
        try:
            result = result_queue.get(timeout=max(5.0, self.timeout_sec))
        except queue.Empty as exc:
            raise TimeoutError("MLX CosyVoice3 合成超时") from exc
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or "MLX CosyVoice3 合成失败"))
        return np.asarray(result["audio"], dtype=np.float32), int(result["sample_rate"])

    def synthesize(self, text: str, voice_id: str) -> tuple[bytes, str]:
        audio, sample_rate = self.synthesize_samples(text, voice_id)
        return _wav_bytes(audio, sample_rate), "audio/wav"

    def prepare_assets(self, *, generate_previews: bool = True) -> dict[str, Any]:
        """Prepare profiles and optional preview wavs in the current thread."""
        model = _load_model(self.model_dir)
        prepared = []
        for voice_id in BUILTIN_VOICES:
            profile = _build_voice_profile(model, voice_id, self.voice_dir)
            _save_voice_profile(profile, self.profiles_dir)
            if generate_previews:
                audio = _synthesize_with_profile(model, profile, DEFAULT_PREVIEW_TEXT)
                self.previews_dir.mkdir(parents=True, exist_ok=True)
                sf.write(self.previews_dir / f"{voice_id}.wav", audio, profile.sample_rate)
            prepared.append(voice_id)
        return {"ok": True, "voices": prepared, "modelDir": str(self.model_dir), "voiceDir": str(self.voice_dir)}

    def _worker_loop(self, worker_index: int) -> None:
        try:
            model = _load_model(self.model_dir)
            profiles = {
                voice_id: self._ensure_voice_profile_once(model, voice_id)
                for voice_id in BUILTIN_VOICES
            }
            voices = self._discover_voice_payloads()
            with self._lock:
                self._ready_workers += 1
                self._loading_workers = max(0, self._loading_workers - 1)
                self._ready = True
                self._loading = self._loading_workers > 0
                self._loaded_at = self._loaded_at or time.time()
                self._voices = voices
        except Exception as exc:
            with self._lock:
                self._loading_workers = max(0, self._loading_workers - 1)
                self._loading = self._loading_workers > 0
                self._ready = self._ready_workers > 0
                self._worker_errors[worker_index] = f"{type(exc).__name__}: {exc}"
                self._error = "; ".join(self._worker_errors.values())
            return

        while True:
            task = self._requests.get()
            if task is None:
                return
            kind, voice_id, text, result_queue = task
            try:
                if kind != "synthesize":
                    raise ValueError(f"未知任务类型：{kind}")
                profile = profiles.get(_normalize_voice_id(voice_id)) or profiles["xuefen"]
                audio = _synthesize_with_profile(model, profile, text)
                result_queue.put({"ok": True, "audio": audio, "sample_rate": profile.sample_rate})
            except Exception as exc:
                result_queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def _ensure_voice_profile_once(self, model: Any, voice_id: str) -> VoiceProfile:
        with self._profile_lock:
            return _ensure_voice_profile(model, voice_id, self.voice_dir)

    def _discover_voice_payloads(self) -> list[dict[str, Any]]:
        voices: list[dict[str, Any]] = []
        for voice_id, spec in BUILTIN_VOICES.items():
            sample = self.samples_dir / spec["sample"]
            preview = self.previews_dir / f"{voice_id}.wav"
            profile_npz = self.profiles_dir / f"{voice_id}.npz"
            voices.append(
                {
                    "id": voice_id,
                    "name": spec["name"],
                    "style": spec["style"],
                    "promptText": spec["ref_text"],
                    "promptWav": str(sample) if sample.exists() else "",
                    "sourceAudio": str(sample) if sample.exists() else "",
                    "previewWav": str(preview) if preview.exists() else "",
                    "previewUrl": f"/api/tts/voices/{voice_id}/preview" if preview.exists() else "",
                    "previewText": DEFAULT_PREVIEW_TEXT if preview.exists() else "",
                    "previewReady": preview.exists(),
                    "profileReady": profile_npz.exists(),
                    "builtIn": True,
                    "provider": "mlx-cosyvoice3",
                }
            )
        return voices


def _load_model(model_dir: Path):
    if not model_dir.exists():
        raise FileNotFoundError(f"未找到 MLX CosyVoice3 模型目录：{model_dir}")
    from mlx_audio.tts.utils import load_model

    model = load_model(model_path=model_dir)
    model._ensure_model_loaded()
    model._ensure_tokenizers_loaded()
    return model


def _ensure_voice_profile(model: Any, voice_id: str, voice_dir: Path) -> VoiceProfile:
    try:
        return _load_voice_profile(voice_id, voice_dir / "profiles")
    except FileNotFoundError:
        profile = _build_voice_profile(model, voice_id, voice_dir)
        _save_voice_profile(profile, voice_dir / "profiles")
        return profile


def _build_voice_profile(model: Any, voice_id: str, voice_dir: Path) -> VoiceProfile:
    import mlx.core as mx
    from mlx_audio.codec.models.s3gen.mel import mel_spectrogram as cosyvoice_mel_spectrogram
    from mlx_audio.codec.models.s3tokenizer import log_mel_spectrogram_compat as log_mel_spectrogram
    from mlx_audio.tts.generate import load_audio
    from scipy.signal import resample

    spec = BUILTIN_VOICES[_normalize_voice_id(voice_id)]
    ref_audio_path = voice_dir / "samples" / spec["sample"]
    if not ref_audio_path.exists():
        raise FileNotFoundError(f"未找到音色样本：{ref_audio_path}")

    ref_audio = load_audio(str(ref_audio_path), sample_rate=model.sample_rate)
    ref_audio_np = np.array(ref_audio)
    max_ref_samples = int(30 * model.sample_rate)
    if len(ref_audio_np) > max_ref_samples:
        ref_audio_np = ref_audio_np[:max_ref_samples]

    import librosa

    ref_audio_np, _ = librosa.effects.trim(
        ref_audio_np,
        top_db=60,
        frame_length=int(0.025 * model.sample_rate),
        hop_length=int(0.0125 * model.sample_rate),
    )
    ref_audio_16k = resample(ref_audio_np, int(len(ref_audio_np) * 16000 / model.sample_rate))
    ref_audio_16k = mx.array(ref_audio_16k, dtype=mx.float32)

    mel_128 = log_mel_spectrogram(ref_audio_16k, n_mels=128)
    mel_128 = mx.expand_dims(mel_128, 0)
    speech_tokens, speech_token_lens = model._s3_tokenizer(mel_128, mx.array([mel_128.shape[2]]))

    ref_audio_24k = mx.array(ref_audio_np, dtype=mx.float32)
    mel_80 = cosyvoice_mel_spectrogram(
        ref_audio_24k,
        n_fft=1920,
        num_mels=80,
        sampling_rate=24000,
        hop_size=480,
        win_size=1920,
        fmin=0,
        fmax=8000,
        center=False,
    )
    mel_80 = mx.swapaxes(mel_80, 1, 2)

    token_len = int(speech_token_lens[0])
    max_mel_len = int(mel_80.shape[1])
    if max_mel_len < token_len * 2:
        token_len = max_mel_len // 2
    mel_len = token_len * 2

    prompt_text_tokens = model._tokenizer.encode(spec["ref_text"], add_special_tokens=False)
    arrays = {
        "prompt_text": mx.array([prompt_text_tokens], dtype=mx.int32),
        "prompt_text_len": mx.array([len(prompt_text_tokens)], dtype=mx.int32),
        "prompt_speech_token": speech_tokens[:, :token_len],
        "prompt_speech_token_len": mx.array([token_len], dtype=mx.int32),
        "prompt_mel": mel_80[:, :mel_len, :],
        "prompt_mel_len": mx.array([mel_len], dtype=mx.int32),
        "speaker_embedding": model._speaker_encoder(ref_audio_16k, sample_rate=16000),
    }
    return VoiceProfile(
        voice_id=spec["id"],
        name=spec["name"],
        style=spec["style"],
        ref_audio=str(ref_audio_path),
        ref_text=spec["ref_text"],
        sample_rate=model.sample_rate,
        arrays=arrays,
    )


def _save_voice_profile(profile: VoiceProfile, profiles_dir: Path) -> None:
    profiles_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(profiles_dir / f"{profile.voice_id}.npz", **{key: np.array(value) for key, value in profile.arrays.items()})
    (profiles_dir / f"{profile.voice_id}.json").write_text(
        json.dumps(
            {
                "id": profile.voice_id,
                "name": profile.name,
                "style": profile.style,
                "refAudio": profile.ref_audio,
                "refText": profile.ref_text,
                "sampleRate": profile.sample_rate,
                "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_voice_profile(voice_id: str, profiles_dir: Path) -> VoiceProfile:
    import mlx.core as mx

    voice_id = _normalize_voice_id(voice_id)
    npz_path = profiles_dir / f"{voice_id}.npz"
    json_path = profiles_dir / f"{voice_id}.json"
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"未找到音色 profile：{voice_id}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    arrays_np = np.load(npz_path)
    return VoiceProfile(
        voice_id=voice_id,
        name=str(metadata.get("name") or voice_id),
        style=str(metadata.get("style") or ""),
        ref_audio=str(metadata.get("refAudio") or ""),
        ref_text=str(metadata.get("refText") or ""),
        sample_rate=int(metadata.get("sampleRate") or 24000),
        arrays={key: mx.array(arrays_np[key]) for key in arrays_np.files},
    )


def _synthesize_with_profile(model: Any, profile: VoiceProfile, text: str) -> np.ndarray:
    import mlx.core as mx

    text_tokens = model._tokenizer.encode(text, add_special_tokens=False)
    text_array = mx.array([text_tokens], dtype=mx.int32)
    text_len = mx.array([len(text_tokens)], dtype=mx.int32)
    audio = model._model.synthesize(
        text=text_array,
        text_len=text_len,
        prompt_text=profile.arrays["prompt_text"],
        prompt_text_len=profile.arrays["prompt_text_len"],
        prompt_speech_token=profile.arrays["prompt_speech_token"],
        prompt_speech_token_len=profile.arrays["prompt_speech_token_len"],
        prompt_mel=profile.arrays["prompt_mel"],
        prompt_mel_len=profile.arrays["prompt_mel_len"],
        speaker_embedding=profile.arrays["speaker_embedding"],
        sampling=25,
        max_token_text_ratio=20.0,
        min_token_text_ratio=2.0,
    )
    return np.array(audio.squeeze()).astype(np.float32)


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio.astype(np.float32, copy=False), sample_rate, format="WAV")
    return buffer.getvalue()


def _normalize_voice_id(voice_id: str) -> str:
    value = str(voice_id or "").strip()
    return value if value in BUILTIN_VOICES else "xuefen"
