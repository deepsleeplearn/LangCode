import numpy as np

from langcode_agent.voice.asr import (
    AsrSettings,
    AsrStreamSession,
    QwenAsrService,
    SAMPLE_RATE,
    _transformers_device_kwargs,
)
from langcode_agent.voice.turnsense import TurnSenseService, TurnSenseSettings


class FakeResult:
    language = "Chinese"
    text = "你好，世界"


class FakeModel:
    def transcribe(self, audio, language=None):
        wav, sr = audio
        assert sr == SAMPLE_RATE
        assert wav.dtype == np.float32
        assert language == "Chinese"
        return [FakeResult()]


class CountingFakeModel(FakeModel):
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, language=None):
        self.calls += 1
        return super().transcribe(audio, language=language)


class FakeIncompleteResult:
    language = "Chinese"
    text = "我想说的是，"


class FakeIncompleteModel:
    def transcribe(self, audio, language=None):
        return [FakeIncompleteResult()]


def test_asr_status_does_not_load_model(monkeypatch):
    monkeypatch.setenv("LANGCODE_ASR_PRELOAD", "0")
    service = QwenAsrService(AsrSettings(model="fake-model"))

    status = service.status()

    assert status["state"] == "idle"
    assert status["model"] == "fake-model"
    assert status["audioVad"]["energyThreshold"] == 0.014
    assert status["semanticVad"]["enabled"] is True


def test_asr_status_reports_device_and_dtype():
    service = QwenAsrService(AsrSettings(model="fake-model", device="mps", dtype="float16"))

    status = service.status()

    assert status["device"] == "mps"
    assert status["dtype"] == "float16"


def test_asr_transformers_kwargs_can_force_mps_float16():
    kwargs = _transformers_device_kwargs("mps", "float16")

    assert kwargs["device_map"] == {"": "mps"}
    assert str(kwargs["torch_dtype"]).endswith("float16")


def test_asr_stream_session_final_transcribes_float32_audio():
    settings = AsrSettings(model="fake-model", min_audio_sec=0.1)
    turnsense = TurnSenseService(TurnSenseSettings(model="missing-local-model"))
    session = AsrStreamSession(settings, FakeModel(), turnsense=turnsense)
    chunk = np.ones(int(SAMPLE_RATE * 0.2), dtype=np.float32) * 0.02

    session.push_float32(chunk)
    event = session.finish()

    assert event["type"] == "final"
    assert event["text"] == "你好，世界"
    assert event["language"] == "Chinese"
    assert event["semanticVad"]["state"] == "complete"


def test_asr_stream_session_does_not_transcribe_noise_before_vad_start():
    settings = AsrSettings(
        model="fake-model",
        min_audio_sec=0.1,
        partial_interval_sec=0.0,
        audio_vad_energy_threshold=0.05,
    )
    model = CountingFakeModel()
    turnsense = TurnSenseService(TurnSenseSettings(model="missing-local-model"))
    session = AsrStreamSession(settings, model, turnsense=turnsense)
    session._vad.available = False
    noise = np.ones(int(SAMPLE_RATE * 0.3), dtype=np.float32) * 0.01

    event = session.push_float32(noise)

    assert event["type"] == "partial"
    assert event["text"] == ""
    assert model.calls == 0


def test_asr_stream_session_trims_pre_speech_noise_before_speech_start():
    settings = AsrSettings(
        model="fake-model",
        min_audio_sec=0.1,
        pre_speech_buffer_ms=100,
        partial_interval_sec=0.0,
        audio_vad_energy_threshold=0.05,
    )
    model = CountingFakeModel()
    turnsense = TurnSenseService(TurnSenseSettings(model="missing-local-model"))
    session = AsrStreamSession(settings, model, turnsense=turnsense)
    session._vad.available = False
    noise = np.ones(int(SAMPLE_RATE * 1.0), dtype=np.float32) * 0.01
    speech = np.ones(int(SAMPLE_RATE * 0.2), dtype=np.float32) * 0.06

    session.push_float32(noise)
    assert session._audio.size <= int(SAMPLE_RATE * 0.11)

    session.push_float32(speech)

    assert session._has_speech is True
    assert session._audio.size <= int(SAMPLE_RATE * 0.31)


def test_turnsense_text_fallback_marks_short_fillers_invalid():
    service = TurnSenseService(TurnSenseSettings(model="missing-local-model"))

    result = service.classify(text="嗯")

    assert result["state"] == "invalid"
    assert result["source"] == "heuristic"


def test_asr_stream_session_keeps_incomplete_turn_partial():
    settings = AsrSettings(model="fake-model", min_audio_sec=0.1)
    turnsense = TurnSenseService(TurnSenseSettings(model="missing-local-model"))
    session = AsrStreamSession(settings, FakeIncompleteModel(), turnsense=turnsense)
    chunk = np.ones(int(SAMPLE_RATE * 0.2), dtype=np.float32) * 0.02

    session.push_float32(chunk)
    event = session.transcribe(final=False)

    assert event["semanticVad"]["state"] == "incomplete"


def test_turnsense_loads_local_onnx_when_available():
    service = TurnSenseService()
    result = service.classify(text="测试", audio=np.zeros(SAMPLE_RATE, dtype=np.float32), sample_rate=SAMPLE_RATE)

    if service.status()["state"] == "ready":
        assert result["source"] == "turnsense"
        assert result["state"] in {"complete", "incomplete", "invalid"}
    else:
        assert result["source"] == "heuristic"
