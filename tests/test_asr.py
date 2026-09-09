import threading
import time

import numpy as np
import pytest

from langcode_agent.voice.asr import (
    AsrSettings,
    AsrStreamSession,
    LOAD_ERROR_COOLDOWN_SEC,
    QwenAsrService,
    SAMPLE_RATE,
    _VAD_FRAME_SAMPLES,
    _decode_audio_frame,
    _join_text,
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
    assert status["audioVad"]["energyThreshold"] == 0.025
    assert status["audioVad"]["energyThreshold"] == AsrSettings.default("audio_vad_energy_threshold")
    assert status["audioVad"]["sileroThreshold"] == AsrSettings.default("audio_vad_silero_threshold")
    assert status["maxDecodeSec"] == AsrSettings.default("max_decode_sec")
    assert status["semanticVad"]["enabled"] is True


def test_asr_settings_defaults_are_the_single_source(monkeypatch):
    for name in (
        "LANGCODE_AUDIO_VAD_ENERGY_THRESHOLD",
        "LANGCODE_AUDIO_VAD_SILERO_THRESHOLD",
        "LANGCODE_ASR_MAX_DECODE_SEC",
        "LANGCODE_ASR_FINAL_SILENCE_MS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = AsrSettings.from_env()

    assert settings.audio_vad_energy_threshold == AsrSettings.default("audio_vad_energy_threshold") == 0.025
    assert settings.audio_vad_silero_threshold == AsrSettings.default("audio_vad_silero_threshold") == 0.55
    assert settings.max_decode_sec == AsrSettings.default("max_decode_sec") == 12.0
    assert settings.final_silence_ms == AsrSettings.default("final_silence_ms") == 900

    monkeypatch.setenv("LANGCODE_AUDIO_VAD_ENERGY_THRESHOLD", "0.04")
    monkeypatch.setenv("LANGCODE_ASR_MAX_DECODE_SEC", "5")

    overridden = AsrSettings.from_env()

    assert overridden.audio_vad_energy_threshold == 0.04
    assert overridden.max_decode_sec == 5.0
    assert overridden.audio_vad_silero_threshold == AsrSettings.default("audio_vad_silero_threshold")


def test_asr_load_failure_fast_fails_until_cooldown_expires(monkeypatch):
    assert LOAD_ERROR_COOLDOWN_SEC == 15.0
    service = QwenAsrService(AsrSettings(model="fake-model"))
    attempts = []

    def boom():
        attempts.append(1)
        raise FileNotFoundError("no local weights")

    monkeypatch.setattr(service, "_build_model", boom)

    with pytest.raises(RuntimeError, match="no local weights"):
        service.create_session()
    with pytest.raises(RuntimeError, match="no local weights"):
        service.create_session()

    assert len(attempts) == 1
    assert service.status()["state"] == "error"

    service._load_error_at = time.monotonic() - (LOAD_ERROR_COOLDOWN_SEC + 1.0)

    with pytest.raises(RuntimeError, match="no local weights"):
        service.create_session()

    assert len(attempts) == 2


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


class CountingTurnSense:
    def __init__(self, state: str = "incomplete") -> None:
        self.calls = 0
        self.texts: list[str] = []
        self.offsets: list[int | None] = []
        self.state = state

    def classify(self, *, text, audio=None, sample_rate=16000, audio_offset=None, audio_stream=""):
        self.calls += 1
        self.texts.append(text)
        self.offsets.append(audio_offset)
        return {"state": self.state, "confidence": 0.5, "source": "stub"}

    def status(self):
        return {"ok": True, "enabled": True, "state": "stub"}


class RecordingModel:
    def __init__(self, text: str = "你好") -> None:
        self.lengths: list[int] = []
        self.text = text

    def transcribe(self, audio, language=None):
        wav, sample_rate = audio
        assert sample_rate == SAMPLE_RATE
        self.lengths.append(int(wav.size))
        result = type("R", (), {"text": self.text, "language": "Chinese"})
        return [result()]


def _speech_session(turnsense, model=None, **overrides):
    options = {
        "model": "fake-model",
        "min_audio_sec": 0.1,
        "partial_interval_sec": 0.0,
        "audio_vad_energy_threshold": 0.05,
    }
    options.update(overrides)
    settings = AsrSettings(**options)
    session = AsrStreamSession(settings, model or RecordingModel(), turnsense=turnsense)
    session._vad.available = False
    return session


def test_asr_classifies_turn_only_when_text_changes_or_speech_ends():
    turnsense = CountingTurnSense()
    session = _speech_session(turnsense)
    speech = np.ones(int(SAMPLE_RATE * 0.5), dtype=np.float32) * 0.06

    for _ in range(5):
        session.push_float32(speech)

    assert turnsense.calls == 1
    assert turnsense.texts == ["你好"]

    session._vad._silence_started = time.time() - 10
    silence = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
    session.push_float32(silence)

    assert turnsense.calls == 2


def test_asr_websocket_sends_speech_start_while_the_decode_is_still_running():
    """Item: a barge-in signal that waited for the decode would be a decode too late."""
    import asyncio
    import json

    from langcode_agent.voice.asr import _push_audio

    class SlowSession:
        """A push whose decode only returns after the client saw the speech event."""

        def __init__(self) -> None:
            self.decoded = threading.Event()
            self.drained = False

        def push_float32(self, _chunk):
            self.decoded.wait(2.0)
            return {"type": "partial", "text": "你好"}

        def take_speech_events(self):
            if self.drained:
                return []
            self.drained = True
            return [{"type": "speech", "state": "start"}]

    class FakeWs:
        def __init__(self, session) -> None:
            self.sent: list[dict] = []
            self.session = session

        async def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))
            self.session.decoded.set()

    session = SlowSession()
    ws = FakeWs(session)

    event = asyncio.run(_push_audio(ws, session, np.zeros(160, dtype=np.float32)))

    assert ws.sent == [{"type": "speech", "state": "start"}]
    assert event == {"type": "partial", "text": "你好"}


def test_asr_reports_speech_start_and_end_as_drainable_events():
    """The barge-in signal the websocket forwards on its own, ahead of partials."""
    session = _speech_session(CountingTurnSense())
    speech = np.ones(int(SAMPLE_RATE * 0.5), dtype=np.float32) * 0.06
    silence = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)

    session.push_float32(speech)
    started = session.take_speech_events()
    session._vad._silence_started = time.time() - 10
    session.push_float32(silence)
    ended = session.take_speech_events()

    assert started == [{"type": "speech", "state": "start"}]
    # Draining is destructive, so the same transition is never sent twice.
    assert ended == [{"type": "speech", "state": "end"}]
    assert session.take_speech_events() == []


class FakeSileroIterator:
    """Scripted stand-in for silero's ``VADIterator`` - no torch, no weights.

    The test flips ``speaking``; the fake emits the same ``{"start": t}`` /
    ``{"end": t}`` dicts on the transitions that the real iterator emits, and
    stays silent in between.
    """

    def __init__(self, speaking: bool = True) -> None:
        self.speaking = speaking
        self.frames = 0
        self._active = False

    def __call__(self, frame, return_seconds=True):
        self.frames += 1
        if self.speaking == self._active:
            return None
        self._active = self.speaking
        key = "start" if self.speaking else "end"
        return {key: round(self.frames * _VAD_FRAME_SAMPLES / SAMPLE_RATE, 3)}


class FakeTorch:
    """``_push_silero`` only calls ``from_numpy``; the fake iterator takes the array as is."""

    @staticmethod
    def from_numpy(array):
        return array


def _silero_session(speaking: bool = True, **overrides):
    """A session whose VAD runs the silero code path against the scripted fake."""
    options = {"model": "fake-model", "min_audio_sec": 0.1, "partial_interval_sec": 0.0}
    options.update(overrides)
    session = AsrStreamSession(AsrSettings(**options), RecordingModel(), turnsense=CountingTurnSense())
    session._vad.available = True
    session._vad._torch = FakeTorch
    session._vad._iterator = FakeSileroIterator(speaking)
    return session


def _frames(level: float, count: int = 1):
    return np.ones(_VAD_FRAME_SAMPLES * count, dtype=np.float32) * level


def test_asr_far_field_voice_passes_silero_but_never_starts_a_turn():
    """Silero is loudness agnostic: without the level gate the room starts turns."""
    import asyncio
    import json

    from langcode_agent.voice.asr import _push_audio

    class FakeWs:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send(self, payload: str) -> None:
            self.sent.append(json.loads(payload))

    session = _silero_session()
    ws = FakeWs()

    for _ in range(60):  # ~2 s, far past min_speech_ms
        asyncio.run(_push_audio(ws, session, _frames(0.01)))

    assert session._vad._iterator.frames == 60  # silero saw speech on every frame
    assert session._vad.in_speech is False
    assert session._has_speech is False
    assert ws.sent == []


def test_asr_near_field_voice_starts_a_turn_after_min_speech_ms():
    session = _silero_session()

    session.push_float32(_frames(0.08))

    assert session.take_speech_events() == []  # one 32 ms frame is not speech yet

    events = []
    for _ in range(5):
        session.push_float32(_frames(0.08))
        events += session.take_speech_events()

    assert events == [{"type": "speech", "state": "start"}]
    assert session._vad.in_speech is True

    session._vad._iterator.speaking = False
    session.push_float32(_frames(0.0))

    assert session.take_speech_events() == [{"type": "speech", "state": "end"}]


def test_asr_near_field_voice_starts_mid_far_field_segment():
    """A close speaker joining a segment silero opened for the room still gets a start."""
    session = _silero_session()

    for _ in range(16):  # ~500 ms of a far-away voice
        session.push_float32(_frames(0.01))

    assert session.take_speech_events() == []

    started_at = 0
    for index in range(1, 9):
        session.push_float32(_frames(0.1))
        if session.take_speech_events():
            started_at = index
            break

    assert started_at == 4  # 120 ms of min speech = 4 frames after the jump


def test_asr_noise_floor_lifts_the_gate_above_a_loud_room():
    session = _silero_session(speaking=False)

    event = None
    for _ in range(63):  # ~2 s of background, above the absolute 0.025 floor
        event = session.push_float32(_frames(0.03))

    assert session._vad.noise_floor == pytest.approx(0.03)
    assert event["audioVad"]["noiseFloor"] == pytest.approx(0.03)
    assert session._vad.level_threshold == pytest.approx(0.03 * 10 ** 0.5)

    session._vad._iterator.speaking = True
    for _ in range(30):  # a voice above the absolute floor but under floor + 10 dB
        session.push_float32(_frames(0.035))

    assert session.take_speech_events() == []
    assert session._vad.noise_floor == pytest.approx(0.03)  # frozen while silero is active

    for _ in range(4):
        session.push_float32(_frames(0.12))

    assert session.take_speech_events() == [{"type": "speech", "state": "start"}]


def test_asr_single_loud_frame_is_a_click_not_speech():
    session = _silero_session()

    for _ in range(6):
        session.push_float32(_frames(0.005))
        session.push_float32(_frames(0.6))

    assert session._vad.in_speech is False
    assert session.take_speech_events() == []


def test_asr_energy_fallback_also_waits_for_min_speech_ms():
    session = _speech_session(CountingTurnSense(), audio_vad_min_speech_ms=200)

    session.push_float32(np.ones(int(SAMPLE_RATE * 0.05), dtype=np.float32) * 0.06)

    assert session._has_speech is False

    session.push_float32(np.ones(int(SAMPLE_RATE * 0.2), dtype=np.float32) * 0.06)

    assert session._has_speech is True


def test_asr_near_field_settings_come_from_env(monkeypatch):
    for name in ("LANGCODE_AUDIO_VAD_SNR_DB", "LANGCODE_AUDIO_VAD_MIN_SPEECH_MS"):
        monkeypatch.delenv(name, raising=False)

    settings = AsrSettings.from_env()

    assert settings.audio_vad_snr_db == AsrSettings.default("audio_vad_snr_db") == 10.0
    assert settings.audio_vad_min_speech_ms == AsrSettings.default("audio_vad_min_speech_ms") == 120

    monkeypatch.setenv("LANGCODE_AUDIO_VAD_SNR_DB", "16")
    monkeypatch.setenv("LANGCODE_AUDIO_VAD_MIN_SPEECH_MS", "240")
    overridden = AsrSettings.from_env()

    assert overridden.audio_vad_snr_db == 16.0
    assert overridden.audio_vad_min_speech_ms == 240

    audio_vad = QwenAsrService(overridden).status()["audioVad"]

    assert audio_vad["snrDb"] == 16.0
    assert audio_vad["minSpeechMs"] == 240


def test_asr_classifies_turn_again_when_text_changes():
    turnsense = CountingTurnSense()
    model = RecordingModel("你好")
    session = _speech_session(turnsense, model)
    speech = np.ones(int(SAMPLE_RATE * 0.5), dtype=np.float32) * 0.06

    session.push_float32(speech)
    session.push_float32(speech)
    assert turnsense.calls == 1

    model.text = "你好世界"
    session.push_float32(speech)

    assert turnsense.calls == 2
    assert turnsense.texts[-1].endswith("你好世界")


def test_asr_turnsense_runs_outside_the_session_lock():
    class LockProbingTurnSense(CountingTurnSense):
        def __init__(self, session_holder):
            super().__init__()
            self.session_holder = session_holder
            self.locked_during_classify = None

        def classify(self, **kwargs):
            session = self.session_holder[0]
            # RLock is re-entrant for the same thread, so probe the raw counter.
            self.locked_during_classify = session._lock._is_owned()
            return super().classify(**kwargs)

    holder: list = [None]
    turnsense = LockProbingTurnSense(holder)
    session = _speech_session(turnsense)
    holder[0] = session
    session.push_float32(np.ones(int(SAMPLE_RATE * 0.5), dtype=np.float32) * 0.06)

    assert turnsense.calls == 1
    assert turnsense.locked_during_classify is False


def test_asr_partial_decode_length_stays_bounded_and_final_keeps_whole_text():
    turnsense = CountingTurnSense()
    model = RecordingModel("你好")
    session = _speech_session(turnsense, model, max_decode_sec=2.0)
    chunk = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.06

    for _ in range(20):
        event = session.push_float32(chunk)

    assert max(model.lengths) <= int(SAMPLE_RATE * 3.0)
    assert sum(model.lengths) < 20 * SAMPLE_RATE * 5
    assert event["text"] == "你好" * 10

    final_event = session.finish()

    assert final_event["type"] == "final"
    assert final_event["text"] == "你好" * 10


def test_asr_partial_text_keeps_committed_prefix_and_tail():
    turnsense = CountingTurnSense()
    model = RecordingModel("你好")
    session = _speech_session(turnsense, model, max_decode_sec=1.0)
    chunk = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.06

    first = session.push_float32(chunk)
    assert first["text"] == "你好"
    assert session._committed_text == "你好"
    assert session._audio.size == 0

    second = session.push_float32(chunk)

    assert second["text"] == "你好你好"
    assert model.lengths[-1] == SAMPLE_RATE


def test_asr_finish_transcribes_trailing_speech_shorter_than_min_audio():
    turnsense = CountingTurnSense()
    model = RecordingModel("尾巴")
    session = _speech_session(turnsense, model, min_audio_sec=1.0, max_decode_sec=1.0)
    speech = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.06
    tail = np.ones(int(SAMPLE_RATE * 0.8), dtype=np.float32) * 0.06

    session.push_float32(speech)

    assert session._committed_text == "尾巴"
    assert session._audio.size == 0

    session.push_float32(tail)

    assert model.lengths == [SAMPLE_RATE]  # too short for a partial decode

    event = session.finish()

    assert model.lengths[-1] == tail.size
    assert event["type"] == "final"
    assert event["final"] is True
    assert event["text"] == "尾巴尾巴"


def test_asr_finalizes_when_speech_ends_right_after_a_commit():
    turnsense = CountingTurnSense(state="complete")
    model = RecordingModel("你好")
    session = _speech_session(turnsense, model, max_decode_sec=1.0)
    speech = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.06

    session.push_float32(speech)

    assert session._audio.size == 0  # the commit emptied the buffer

    session._vad._silence_started = time.time() - 10
    event = session.push_float32(np.zeros(int(SAMPLE_RATE * 0.05), dtype=np.float32))

    assert event["type"] == "final"
    assert event["final"] is True
    assert event["text"] == "你好"


def test_asr_decode_window_never_exceeds_max_utterance():
    turnsense = CountingTurnSense()
    model = RecordingModel("你好")
    session = _speech_session(turnsense, model, max_utterance_sec=5.0, max_decode_sec=12.0)
    chunk = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.06

    for _ in range(10):
        event = session.push_float32(chunk)

    assert max(model.lengths) <= int(SAMPLE_RATE * 5.0)
    assert session._committed_text == "你好你好"  # both 5s halves reached the transcript
    assert event["text"] == "你好你好"


def test_asr_trim_commits_decoded_text_before_dropping_the_head():
    turnsense = CountingTurnSense()
    model = RecordingModel("你好")
    session = _speech_session(
        turnsense, model, max_utterance_sec=3.0, max_decode_sec=12.0, partial_interval_sec=60.0
    )
    speech = np.ones(SAMPLE_RATE * 2, dtype=np.float32) * 0.06
    session._last_partial_at = time.time()

    session.push_float32(speech)
    session.transcribe(final=False)

    assert session._committed_text == ""
    assert model.lengths == [SAMPLE_RATE * 2]

    session._last_partial_at = time.time()
    session.push_float32(speech)

    assert session._committed_text == "你好"
    assert session._audio.size == SAMPLE_RATE * 2  # the un-decoded tail survives


def test_asr_commit_keeps_audio_pushed_during_inference():
    turnsense = CountingTurnSense()
    started = threading.Event()
    release = threading.Event()

    class SlowModel(RecordingModel):
        def transcribe(self, audio, language=None):
            started.set()
            assert release.wait(5.0)
            return super().transcribe(audio, language=language)

    model = SlowModel("你好")
    session = _speech_session(turnsense, model, max_decode_sec=1.0, partial_interval_sec=60.0)
    session._last_partial_at = time.time()
    session.push_float32(np.ones(SAMPLE_RATE, dtype=np.float32) * 0.06)

    worker = threading.Thread(target=lambda: session.transcribe(final=False))
    worker.start()
    assert started.wait(5.0)
    late = np.ones(int(SAMPLE_RATE * 0.4), dtype=np.float32) * 0.06
    session.push_float32(late)
    release.set()
    worker.join(5.0)

    assert worker.is_alive() is False
    assert model.lengths == [SAMPLE_RATE]
    assert session._committed_text == "你好"
    assert session._audio.size == late.size


def test_asr_join_text_spacing_rules():
    assert _join_text("abc", "123") == "abc123"
    assert _join_text("Hello.", "World") == "Hello. World"
    assert _join_text("你好", "世界") == "你好世界"
    assert _join_text("Hello", "World") == "Hello World"
    assert _join_text("", "World") == "World"
    assert _join_text("你好", "") == "你好"


def test_asr_ignores_audio_frames_that_are_not_float32_aligned():
    assert _decode_audio_frame(b"\x00\x00\x00") is None
    assert _decode_audio_frame(b"") is not None
    decoded = _decode_audio_frame(np.ones(2, dtype=np.float32).tobytes())

    assert decoded is not None
    assert decoded.tolist() == [1.0, 1.0]


def test_turnsense_incremental_fbank_matches_full_recompute():
    pytest.importorskip("kaldi_native_fbank")
    from langcode_agent.voice.turnsense import _AudioFrontend

    rng = np.random.default_rng(7)
    waveform = (rng.standard_normal(SAMPLE_RATE * 3) * 0.05).astype(np.float32)
    reference, reference_len = _AudioFrontend().extract_features(waveform)

    frontend = _AudioFrontend()
    feats, feat_len = None, 0
    for seconds in (1, 2, 3):
        feats, feat_len = frontend.extract_features(waveform[: SAMPLE_RATE * seconds], offset=0, stream="stream-a")

    assert feat_len == reference_len
    assert np.allclose(feats, reference, atol=1e-3)

    other, _ = frontend.extract_features(waveform[: SAMPLE_RATE], offset=0, stream="stream-b")

    assert other.shape[0] < reference_len


def test_turnsense_incremental_window_matches_the_same_frames_of_a_full_recompute():
    pytest.importorskip("kaldi_native_fbank")
    from langcode_agent.voice.turnsense import MAX_AUDIO_SECONDS, _AudioFrontend

    frame_shift, frame_length = 160, 400  # 10ms / 25ms at 16kHz
    rng = np.random.default_rng(11)
    waveform = (rng.standard_normal(SAMPLE_RATE * 10) * 0.05).astype(np.float32)
    frontend = _AudioFrontend()

    assert frontend._max_frames == (MAX_AUDIO_SECONDS * SAMPLE_RATE - frame_length) // frame_shift + 1 == 798

    feats, feat_len = None, 0
    for seconds in range(1, 11):
        feats, feat_len = frontend.extract_features(waveform[: SAMPLE_RATE * seconds], offset=0, stream="long")

    # The rolling window must hold exactly the trailing 798 frames, so a full
    # recompute of the audio those frames cover has to match index for index.
    dropped = (waveform.size - frame_length) // frame_shift + 1 - frontend._max_frames
    reference, reference_len = _AudioFrontend().extract_features(waveform[dropped * frame_shift :])

    assert dropped > 0
    assert feat_len == reference_len
    assert np.allclose(feats, reference, atol=1e-3)


def test_turnsense_loads_local_onnx_when_available():
    service = TurnSenseService()
    result = service.classify(text="测试", audio=np.zeros(SAMPLE_RATE, dtype=np.float32), sample_rate=SAMPLE_RATE)

    if service.status()["state"] == "ready":
        assert result["source"] == "turnsense"
        assert result["state"] in {"complete", "incomplete", "invalid"}
    else:
        assert result["source"] == "heuristic"
