from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
from typing import Any, Literal

import numpy as np


TurnState = Literal["complete", "incomplete", "invalid", "unknown"]
LABELS: list[TurnState] = ["complete", "incomplete", "invalid"]
SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 8


@dataclass(frozen=True)
class TurnSenseSettings:
    enabled: bool = True
    model: str = "brgroup/TurnSense"
    backend: str = "auto"
    min_chars: int = 2

    @classmethod
    def from_env(cls) -> "TurnSenseSettings":
        return cls(
            enabled=_truthy(os.getenv("LANGCODE_TURNSENSE_ENABLED", "1")),
            model=(os.getenv("LANGCODE_TURNSENSE_MODEL") or _default_model_path()).strip(),
            backend=(os.getenv("LANGCODE_TURNSENSE_BACKEND") or "auto").strip().lower(),
            min_chars=_int_env("LANGCODE_TURNSENSE_MIN_CHARS", 2),
        )


@dataclass
class TurnSenseStatus:
    state: str = "idle"
    model: str = ""
    backend: str = ""
    error: str = ""


class TurnSenseService:
    """Best-effort semantic endpointing wrapper.

    The real TurnSense model is optional because deployments may not have the
    ONNX/model files locally yet. When it is unavailable, the service falls back
    to a conservative Chinese text heuristic so ASR remains usable.
    """

    def __init__(self, settings: TurnSenseSettings | None = None) -> None:
        self.settings = settings or TurnSenseSettings.from_env()
        self._status = TurnSenseStatus(
            state="disabled" if not self.settings.enabled else "idle",
            model=self.settings.model,
            backend=self.settings.backend,
        )
        self._lock = threading.RLock()
        self._model: Any | None = None
        self._load_attempted = False

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ok": self._status.state != "error",
                "enabled": self.settings.enabled,
                "state": self._status.state,
                "model": self._status.model,
                "backend": self._status.backend,
                "error": self._status.error,
            }

    def classify(self, *, text: str, audio: np.ndarray | None = None, sample_rate: int = 16000) -> dict[str, Any]:
        text = (text or "").strip()
        if not self.settings.enabled:
            return _result("unknown", 0.0, "disabled")
        model = self._load_optional()
        if model is not None:
            try:
                return model.classify(text=text, audio=audio, sample_rate=sample_rate)
            except Exception as exc:
                with self._lock:
                    self._status.state = "error"
                    self._status.error = f"{type(exc).__name__}: {exc}"
        return self._heuristic(text)

    def _load_optional(self) -> Any | None:
        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_attempted:
                return None
            self._load_attempted = True
            self._status.state = "loading"
        try:
            model = _build_onnx_turnsense(self.settings)
        except Exception as exc:
            with self._lock:
                self._status.state = "fallback"
                self._status.error = f"{type(exc).__name__}: {exc}"
            return None
        with self._lock:
            self._model = model
            self._status.state = "ready"
            self._status.error = ""
            return model

    def _heuristic(self, text: str) -> dict[str, Any]:
        compact = "".join(text.split())
        if not compact or compact in {"嗯", "啊", "呃", "哦", "额", "唔", "好的", "好"}:
            return _result("invalid", 0.72, "heuristic")
        if len(compact) < self.settings.min_chars:
            return _result("invalid", 0.65, "heuristic")
        if compact.endswith(("，", ",", "、", "和", "或者", "然后", "因为", "所以", "但是", "如果", "比如", "就是")):
            return _result("incomplete", 0.68, "heuristic")
        return _result("complete", 0.62, "heuristic")


class _OnnxTurnSense:
    def __init__(self, session: Any, labels: list[str]) -> None:
        self.session = session
        self.labels = labels
        self.frontend = _AudioFrontend()

    def classify(self, *, text: str, audio: np.ndarray | None = None, sample_rate: int = 16000) -> dict[str, Any]:
        if audio is None or audio.size == 0:
            return _result("unknown", 0.0, "onnx")
        waveform = _normalize_audio(audio)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"TurnSense 需要 {SAMPLE_RATE}Hz 音频，当前为 {sample_rate}Hz")
        if waveform.size == 0 or float(np.sqrt(np.mean(np.square(waveform)))) < 1e-5:
            return _result("invalid", 0.7, "turnsense")
        waveform = waveform[-SAMPLE_RATE * MAX_AUDIO_SECONDS :]
        feats, feat_len = self.frontend.extract_features(waveform)
        outputs = self.session.run(
            None,
            {
                "feats": feats[None, ...].astype(np.float32),
                "feat_lengths": np.asarray([int(feat_len)], dtype=np.int64),
            },
        )
        probs = _softmax_np(np.asarray(outputs[0], dtype=np.float32), axis=1)
        pred_id = int(np.argmax(probs[0]))
        label = self.labels[pred_id] if 0 <= pred_id < len(self.labels) else "unknown"
        confidence = float(probs[0][pred_id])
        return _result(label, confidence, "turnsense")


def _build_onnx_turnsense(settings: TurnSenseSettings) -> _OnnxTurnSense:
    import onnxruntime as ort

    model_path = Path(settings.model).expanduser()
    if not model_path.exists():
        raise FileNotFoundError("TurnSense local model directory not found")
    onnx_file = _select_onnx_file(model_path)
    if onnx_file is None:
        raise FileNotFoundError("TurnSense ONNX file not found")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_file), sess_options=session_options, providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}
    if not {"feats", "feat_lengths"}.issubset(input_names):
        raise RuntimeError(f"TurnSense ONNX 输入不匹配：{sorted(input_names)}")
    return _OnnxTurnSense(session, LABELS)


class _AudioFrontend:
    def __init__(
        self,
        *,
        fs: int = SAMPLE_RATE,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        lfr_m: int = 7,
        lfr_n: int = 6,
        dither: float = 0.0,
    ) -> None:
        import kaldi_native_fbank as knf

        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = fs
        opts.frame_opts.dither = dither
        opts.frame_opts.window_type = window
        opts.frame_opts.frame_shift_ms = float(frame_shift)
        opts.frame_opts.frame_length_ms = float(frame_length)
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = n_mels
        opts.mel_opts.debug_mel = False
        opts.energy_floor = 0
        self._knf = knf
        self.opts = opts
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n

    def extract_features(self, waveform: np.ndarray) -> tuple[np.ndarray, int]:
        waveform = waveform.astype(np.float32, copy=False) * (1 << 15)
        fbank = self._knf.OnlineFbank(self.opts)
        fbank.accept_waveform(self.opts.frame_opts.samp_freq, waveform.tolist())
        frame_count = fbank.num_frames_ready
        if frame_count <= 0:
            return np.zeros((1, self.opts.mel_opts.num_bins * self.lfr_m), dtype=np.float32), 1
        feat = np.empty((frame_count, self.opts.mel_opts.num_bins), dtype=np.float32)
        for index in range(frame_count):
            feat[index, :] = fbank.get_frame(index)
        feat = self._apply_lfr(feat, self.lfr_m, self.lfr_n)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat.astype(np.float32, copy=False), int(feat.shape[0])

    @staticmethod
    def _apply_lfr(inputs: np.ndarray, lfr_m: int, lfr_n: int) -> np.ndarray:
        if lfr_m == 1 and lfr_n == 1:
            return inputs.astype(np.float32, copy=False)
        lfr_inputs = []
        frame_count = inputs.shape[0]
        lfr_frame_count = int(np.ceil(frame_count / lfr_n))
        left_padding = np.tile(inputs[0], ((lfr_m - 1) // 2, 1))
        padded = np.vstack((left_padding, inputs))
        padded_count = frame_count + (lfr_m - 1) // 2
        for index in range(lfr_frame_count):
            start = index * lfr_n
            if lfr_m <= padded_count - start:
                lfr_inputs.append(padded[start : start + lfr_m].reshape(1, -1))
            else:
                padding_count = lfr_m - (padded_count - start)
                frame = padded[start:].reshape(-1)
                for _ in range(padding_count):
                    frame = np.hstack((frame, padded[-1]))
                lfr_inputs.append(frame.reshape(1, -1))
        return np.vstack(lfr_inputs).astype(np.float32)


def _select_onnx_file(model_path: Path) -> Path | None:
    if model_path.is_file() and model_path.suffix == ".onnx":
        return model_path
    candidates = sorted(model_path.rglob("*.onnx"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (0 if "int8" in path.name.lower() else 1, len(path.parts), str(path)))[0]


def _result(state: TurnState, confidence: float, source: str) -> dict[str, Any]:
    return {"state": state, "confidence": confidence, "source": source}


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0 if waveform.shape[0] <= 8 else 1)
    waveform = waveform.reshape(-1)
    if waveform.size == 0:
        return waveform
    return np.nan_to_num(np.clip(waveform, -1.0, 1.0), copy=False)


def _softmax_np(value: np.ndarray, axis: int = -1) -> np.ndarray:
    value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    shifted = value - np.max(value, axis=axis, keepdims=True)
    exp_value = np.exp(shifted)
    return exp_value / np.sum(exp_value, axis=axis, keepdims=True)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "y"}


def _int_env(name: str, fallback: int) -> int:
    try:
        return int(os.getenv(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback


def _default_model_path() -> str:
    local_model = Path.cwd() / ".langcode" / "turnsense-models" / "TurnSense"
    if local_model.exists():
        return str(local_model)
    return "brgroup/TurnSense"
