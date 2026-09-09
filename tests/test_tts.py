from io import BytesIO
from pathlib import Path
import time

import numpy as np
import pytest
import soundfile as sf

from langcode_agent.voice.stream import TtsTurnRegistry, iter_tts_events
from langcode_agent.voice.tts import (
    CHUNK_MAX_CHARS,
    FALLBACK_NOTICE,
    FIRST_CHUNK_MAX_CHARS,
    TtsService,
    TtsSettings,
    _clean_text,
    _has_speakable,
    _speech_bounds,
    _trim_speech_audio,
    _trim_speech_samples,
    _wav_buffer,
    split_text_for_streaming,
)


def test_tts_status_uses_system_say_when_no_remote(monkeypatch):
    service = TtsService(TtsSettings(provider="auto", base_url="", local_model_dir="/missing/local-tts-model", preload=False))
    monkeypatch.setattr("langcode_agent.voice.tts.platform.system", lambda: "Darwin")
    monkeypatch.setattr("langcode_agent.voice.tts._say_available", lambda: True)

    status = service.status()

    assert status["ok"] is True
    assert status["mode"] == "system-say"
    assert status["assets"]["sayReady"] is True


def test_tts_status_reports_remote_with_say_fallback(monkeypatch):
    service = TtsService(
        TtsSettings(provider="auto", base_url="http://tts.example", local_model_dir="/missing/local-tts-model", preload=False)
    )
    monkeypatch.setattr("langcode_agent.voice.tts._say_available", lambda: True)

    status = service.status()

    assert status["ok"] is True
    assert status["mode"] == "remote-with-say-fallback"
    assert status["assets"]["remoteReady"] is True


def test_tts_worker_count_comes_from_settings():
    service = TtsService(TtsSettings(worker_count=2, preload=False))

    assert service.status()["workerCount"] == 2
    assert service.status()["mlx"]["workerCount"] == 2


def test_tts_worker_count_comes_from_env(monkeypatch):
    monkeypatch.setenv("LANGCODE_TTS_WORKERS", "3")

    settings = TtsSettings.from_env()
    service = TtsService(settings)

    assert settings.worker_count == 3
    assert service.status()["mlx"]["workerCount"] == 3


def test_tts_lists_root_wav_samples_with_preview(tmp_path: Path, monkeypatch):
    sample = tmp_path / "汪菊.wav"
    sample.write_bytes(b"RIFF-sample")
    sample.with_suffix(".json").write_text(
        '{"id":"汪菊","name":"汪菊","promptText":"样本文本","sourceAudio":"' + str(sample) + '"}',
        encoding="utf-8",
    )
    service = TtsService(
        TtsSettings(
            sample_dir=str(tmp_path),
            voice_dir=str(tmp_path / "voices"),
            preload=False,
        )
    )
    monkeypatch.setattr(service._mlx, "voices", lambda: [])

    voices = service.list_voices()
    custom = next(voice for voice in voices if voice["id"] == "汪菊")

    assert custom["name"] == "汪菊"
    assert custom["previewReady"] is True
    assert custom["previewUrl"] == "/api/tts/voices/%E6%B1%AA%E8%8F%8A/preview"
    assert service.voice_preview_path("汪菊") == sample


def test_tts_preview_falls_back_to_original_sample(tmp_path: Path, monkeypatch):
    sample = tmp_path / "雪芬.wav"
    sample.write_bytes(b"RIFF-sample")
    service = TtsService(
        TtsSettings(
            sample_dir=str(tmp_path),
            voice_dir=str(tmp_path / "voices"),
            preload=False,
        )
    )
    monkeypatch.setattr(service._mlx, "voices", lambda: [])
    monkeypatch.setattr(service._mlx, "preview_path", lambda _voice_id: None)

    assert service.voice_preview_path("雪芬") == sample


def test_tts_root_sample_overrides_sidecar_source_audio(tmp_path: Path, monkeypatch):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    sample = sample_dir / "雪芬.wav"
    stale_audio = sample_dir / "old-雪芬.m4a"
    sample.write_bytes(b"RIFF-root")
    stale_audio.write_bytes(b"legacy")
    (sample_dir / "雪芬.json").write_text(
        '{"id":"雪芬","name":"雪芬","promptText":"旧文本","sourceAudio":"' + str(stale_audio) + '"}',
        encoding="utf-8",
    )
    service = TtsService(
        TtsSettings(
            sample_dir=str(sample_dir),
            voice_dir=str(tmp_path / "voices"),
            preload=False,
        )
    )
    monkeypatch.setattr(service._mlx, "voices", lambda: [])
    monkeypatch.setattr(service._mlx, "preview_path", lambda _voice_id: None)

    voice = next(item for item in service.list_voices() if item["id"] == "雪芬")

    assert voice["sourceAudio"] == str(sample)
    assert service.voice_preview_path("雪芬") == sample


def test_tts_list_voices_dedupes_builtin_voice_aliases(tmp_path: Path, monkeypatch):
    (tmp_path / "汪菊.wav").write_bytes(b"RIFF-wangju")
    (tmp_path / "雪芬.wav").write_bytes(b"RIFF-xuefen")
    service = TtsService(
        TtsSettings(
            sample_dir=str(tmp_path),
            voice_dir=str(tmp_path / "voices"),
            preload=False,
        )
    )
    monkeypatch.setattr(
        service._mlx,
        "voices",
        lambda: [
            {"id": "wangju", "name": "汪菊", "provider": "mlx-cosyvoice3", "previewReady": True, "profileReady": True},
            {"id": "xuefen", "name": "雪芬", "provider": "mlx-cosyvoice3", "previewReady": True, "profileReady": True},
        ],
    )

    voices = service.list_voices()

    assert [voice["name"] for voice in voices].count("汪菊") == 1
    assert [voice["name"] for voice in voices].count("雪芬") == 1
    assert next(voice for voice in voices if voice["name"] == "汪菊")["id"] == "wangju"
    assert next(voice for voice in voices if voice["name"] == "雪芬")["id"] == "xuefen"


def test_tts_remote_success(monkeypatch):
    service = TtsService(TtsSettings(base_url="http://tts.example", local_model_dir="/missing/local-tts-model", timeout_sec=1, preload=False))

    def fake_external(text, voice_id=""):
        assert text == "你好"
        assert voice_id == "custom"
        return b"RIFF-remote", "audio/wav"

    monkeypatch.setattr(service, "_synthesize_external", fake_external)
    monkeypatch.setattr(service, "_synthesize_system_say", lambda text: (_ for _ in ()).throw(AssertionError("no fallback")))

    audio, content_type = service.synthesize("你好", voice_id="custom")

    assert audio == b"RIFF-remote"
    assert content_type == "audio/wav"


def test_tts_remote_failure_falls_back_to_say(monkeypatch):
    service = TtsService(TtsSettings(base_url="http://tts.example", local_model_dir="/missing/local-tts-model", timeout_sec=1, preload=False))
    monkeypatch.setattr("langcode_agent.voice.tts.platform.system", lambda: "Darwin")
    monkeypatch.setattr("langcode_agent.voice.tts._say_available", lambda: True)
    monkeypatch.setattr(service, "_synthesize_external", lambda text, voice_id="": (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(service, "_synthesize_system_say", lambda text: b"RIFF-say")

    audio, content_type = service.synthesize("你好", voice_id="custom")

    assert audio == b"RIFF-say"
    assert content_type == "audio/wav"


def test_tts_disabled_raises():
    service = TtsService(TtsSettings(enabled=False, preload=False))

    with pytest.raises(RuntimeError, match="TTS 已禁用"):
        service.synthesize("你好")


def test_split_text_for_streaming_starts_short_and_keeps_order():
    text = "你好，我是你的语音助手。今天天气不错，我们可以出门走走；晚上回来继续写代码。最后一句话。"

    chunks = split_text_for_streaming(text)

    assert "".join(chunks) == text
    assert len(chunks[0]) <= FIRST_CHUNK_MAX_CHARS
    assert all(len(chunk) <= CHUNK_MAX_CHARS for chunk in chunks[1:])
    assert chunks[0] == "你好，我是你的语音助手。"
    assert chunks[1] == "今天天气不错，我们可以出门走走；晚上回来继续写代码。最后一句话。"
    assert chunks[-1].endswith("最后一句话。")


def test_split_text_for_streaming_uses_soft_boundaries_for_long_sentences():
    text = "第一段话，" + "很长的内容继续说下去" * 8 + "，结束了"

    chunks = split_text_for_streaming(text)

    assert "".join(chunks) == text
    assert len(chunks[0]) <= FIRST_CHUNK_MAX_CHARS
    assert all(len(chunk) <= CHUNK_MAX_CHARS for chunk in chunks[1:])
    assert chunks[0] == "第一段话，"


def test_split_text_for_streaming_hard_cuts_when_no_punctuation():
    text = "无标点内容" * 30

    chunks = split_text_for_streaming(text)

    assert "".join(chunks) == text
    assert len(chunks[0]) == FIRST_CHUNK_MAX_CHARS
    assert all(len(chunk) <= CHUNK_MAX_CHARS for chunk in chunks[1:])


def test_split_text_for_streaming_resplits_merged_punctuation_runs():
    text = "你好。" + "，" * 70

    chunks = split_text_for_streaming(text)

    assert "".join(chunks) == text
    assert len(chunks[0]) <= FIRST_CHUNK_MAX_CHARS
    assert all(len(chunk) <= CHUNK_MAX_CHARS for chunk in chunks)
    assert all(_has_speakable(chunk) for chunk in chunks[:1])


def test_split_text_for_streaming_never_starts_with_a_punctuation_only_chunk():
    chunks = split_text_for_streaming("。。。你好世界。", first_max_chars=3, max_chars=10)

    assert "".join(chunks) == "。。。你好世界。"
    assert _has_speakable(chunks[0])
    assert chunks[0].startswith("。。。你好")


def test_split_text_for_streaming_drops_text_without_anything_speakable():
    assert split_text_for_streaming("。。。！！！") == []


def test_split_text_for_streaming_keeps_paired_quotes_together():
    text = "他说：“今天很好。明天也不错。”然后走了。"

    chunks = split_text_for_streaming(text)

    assert "".join(chunks) == text
    assert all(chunk.count("“") == chunk.count("”") for chunk in chunks)
    assert "“今天很好。明天也不错。”" in chunks[0]


def test_split_text_for_streaming_never_cuts_a_url():
    url = "https://example.com/a?b=1&c=2"
    text = f"参考 {url} 这个链接。"

    chunks = split_text_for_streaming(text)

    assert any(url in chunk for chunk in chunks)
    assert all(chunk.count("https://") <= 1 for chunk in chunks)


def test_split_text_for_streaming_uses_english_sentence_boundaries():
    text = "This is one. This is two. e.g. keep it, 3.14 stays."

    chunks = split_text_for_streaming(text)

    assert chunks[0] == "This is one."
    assert any("e.g. keep it" in chunk for chunk in chunks)
    assert any("3.14 stays." in chunk for chunk in chunks)
    assert all(len(chunk) <= CHUNK_MAX_CHARS for chunk in chunks[1:])


def test_clean_text_strips_markdown():
    assert _clean_text("**加粗**和*斜体*") == "加粗和斜体"
    assert _clean_text("## 标题") == "标题"
    assert _clean_text("看 `code` 这里") == "看 code 这里"
    assert _clean_text("见 [文档](https://x.com/a)") == "见 文档"
    assert _clean_text("![图](https://x.com/i.png)你好") == "你好"
    assert _clean_text("前\n```py\ncode = drop\n```\n后") == "前 后"
    assert _clean_text("上\n---\n下") == "上 下"
    assert _clean_text("| 姓名 | 年龄 |\n| --- | --- |\n| 小明 | 3 |") == "姓名，年龄 小明，3"
    assert _clean_text("- 第一项\n- 第二项") == "第一项 第二项"
    assert _clean_text("变量 snake_case 保持不变") == "变量 snake_case 保持不变"


def test_tts_synthesize_chunks_yields_one_blob_per_sentence_group(monkeypatch):
    service = TtsService(TtsSettings(local_model_dir="/missing/local-tts-model", preload=False))
    seen: list[str] = []

    def fake_synthesize(text, voice_id=""):
        seen.append(text)
        return text.encode("utf-8"), "audio/wav", {"provider": "stub", "fallback": "", "reason": ""}

    monkeypatch.setattr(service, "synthesize_with_meta", fake_synthesize)

    chunks = list(service.synthesize_chunks("你好，我是助手。今天天气不错，我们出去走走。再见。"))

    assert [audio.decode("utf-8") for audio, _ct in chunks] == seen
    assert "".join(seen) == "你好，我是助手。今天天气不错，我们出去走走。再见。"
    assert len(seen[0]) <= FIRST_CHUNK_MAX_CHARS
    assert all(content_type == "audio/wav" for _audio, content_type in chunks)


def test_tts_say_fallback_path_is_also_chunked(monkeypatch):
    service = TtsService(TtsSettings(base_url="", local_model_dir="/missing/local-tts-model", preload=False))
    monkeypatch.setattr("langcode_agent.voice.tts.platform.system", lambda: "Darwin")
    monkeypatch.setattr("langcode_agent.voice.tts._say_available", lambda: True)
    monkeypatch.setattr(service, "_should_try_mlx", lambda: True)
    monkeypatch.setattr(
        service._mlx, "synthesize_samples", lambda text, voice: (_ for _ in ()).throw(RuntimeError("mlx down"))
    )
    spoken: list[str] = []

    def fake_say(text):
        spoken.append(text)
        return b"RIFF" + text.encode("utf-8")

    monkeypatch.setattr(service, "_synthesize_system_say", fake_say)
    metas: list[dict] = []
    text = "第一句。" * 30

    chunks = list(service.synthesize_chunks(text, voice_id="custom", on_meta=metas.append))

    assert "".join(spoken) == text
    assert [len(part) for part in spoken] == [FIRST_CHUNK_MAX_CHARS, CHUNK_MAX_CHARS, 40]
    assert len(chunks) == len(spoken) == 3
    assert [meta["fallback"] for meta in metas] == ["macos-say"] * 3
    assert "mlx down" in metas[0]["reason"]


class StubTts:
    def __init__(self, parts, *, fallback: str = "", error: str = "", fallback_from: int = 1) -> None:
        self.parts = parts
        self.fallback = fallback
        self.error = error
        self.fallback_from = fallback_from

    def synthesize_chunks(self, text, voice_id="", *, on_meta=None):
        for index, part in enumerate(self.parts, start=1):
            fallback = self.fallback if index >= self.fallback_from else ""
            if on_meta is not None:
                on_meta({"provider": "macos-say" if fallback else "mlx", "fallback": fallback, "reason": ""})
            yield part, "audio/wav"
        if self.error:
            raise RuntimeError(self.error)


def test_iter_tts_events_emits_audio_then_done():
    events = list(iter_tts_events(StubTts([b"one", b"two"]), "你好"))

    assert [event["type"] for event in events] == ["audio", "audio", "done"]
    assert [event["index"] for event in events[:2]] == [1, 2]
    assert events[0]["contentType"] == "audio/wav"
    assert events[0]["audio"] == "b25l"
    assert events[-1] == {"type": "done", "ok": True, "seq": 2}


def test_iter_tts_events_emits_fallback_notice_before_first_audio():
    events = list(iter_tts_events(StubTts([b"one", b"two"], fallback="macos-say"), "你好"))

    assert [event["type"] for event in events] == ["notice", "audio", "audio", "done"]
    assert events[0] == {"type": "notice", "kind": "tts_fallback", "message": FALLBACK_NOTICE, "seq": 0}
    assert events[1]["index"] == 1


def test_iter_tts_events_marks_a_notice_that_arrives_after_audio_as_late():
    events = list(iter_tts_events(StubTts([b"one", b"two"], fallback="macos-say", fallback_from=2), "你好"))

    assert [event["type"] for event in events] == ["audio", "notice", "audio", "done"]
    assert events[1] == {
        "type": "notice",
        "kind": "tts_fallback",
        "message": FALLBACK_NOTICE,
        "late": True,
        "seq": 1,
    }
    assert [event["index"] for event in events if event["type"] == "audio"] == [1, 2]


def test_iter_tts_events_numbers_every_event_and_times_the_first_audio():
    events = list(
        iter_tts_events(
            StubTts([b"one", b"two"], fallback="macos-say"),
            "你好",
            started_at=time.perf_counter() - 0.25,
        )
    )

    # ``seq`` counts every event of the request, notices included, so a client
    # can tell a dropped event from a slow one.
    assert [event["seq"] for event in events] == [0, 1, 2, 3]
    # ``firstAudioMs`` is measured from ``started_at`` (the request), not from
    # the moment synthesis happened to start, and only the first audio has it.
    assert events[1]["firstAudioMs"] >= 250
    assert "firstAudioMs" not in events[2]


def test_iter_tts_events_stops_at_the_next_chunk_once_should_stop_flips():
    stop: list[bool] = []
    stream = iter_tts_events(
        StubTts([b"one", b"two", b"three"]),
        "你好",
        should_stop=lambda: bool(stop),
        turn_id="turn-1",
    )

    first = next(stream)
    stop.append(True)
    rest = list(stream)

    assert first["type"] == "audio"
    # One chunk was already in flight when the flag flipped; it is dropped
    # instead of sent, and the third is never synthesized at all.
    assert rest == [{"type": "cancelled", "turnId": "turn-1", "seq": 1}]


def test_iter_tts_events_that_starts_cancelled_never_calls_the_synthesizer():
    class ExplodingTts:
        def synthesize_chunks(self, text, voice_id="", **_kwargs):
            raise AssertionError("synthesis must not start for a cancelled turn")

    events = list(iter_tts_events(ExplodingTts(), "你好", should_stop=lambda: True, turn_id="turn-2"))

    assert events == [{"type": "cancelled", "turnId": "turn-2", "seq": 0}]


def test_tts_turn_registry_supersedes_older_turns_of_the_same_session():
    registry = TtsTurnRegistry()

    registry.claim("s1", "turn-1")
    assert registry.is_stale("s1", "turn-1") is False

    registry.claim("s1", "turn-2")
    assert registry.is_stale("s1", "turn-1") is True
    assert registry.is_stale("s1", "turn-2") is False
    # Another session is untouched, and a request without ids is never stale.
    assert registry.is_stale("s2", "turn-1") is False
    assert registry.is_stale("", "") is False


def test_tts_turn_registry_cancel_stops_the_current_turn_and_stays_bounded():
    registry = TtsTurnRegistry()
    registry.claim("s1", "turn-1")

    assert registry.cancel("s1", "turn-1") is True
    assert registry.cancel("s1", "") is False
    assert registry.is_stale("s1", "turn-1") is True

    # Only the last 32 cancelled turns per session are remembered; forgetting an
    # old one is harmless because its producer is long gone. ``s2`` never claimed
    # a turn, so here only the cancel bookkeeping can make a turn stale.
    for index in range(40):
        registry.cancel("s2", f"old-{index}")
    assert registry.is_stale("s2", "old-39") is True
    assert registry.is_stale("s2", "old-0") is False


def test_iter_tts_events_reports_an_error_when_no_audio_was_produced():
    events = list(iter_tts_events(StubTts([]), "你好"))

    assert [event["type"] for event in events] == ["error"]
    assert events[-1]["ok"] is False
    assert "TTS 未产生任何音频" in events[-1]["error"]


def test_iter_tts_events_reports_error_without_done():
    events = list(iter_tts_events(StubTts([b"one"], error="boom"), "你好"))

    assert [event["type"] for event in events] == ["audio", "error"]
    assert events[-1]["ok"] is False
    assert events[-1]["error"] == "RuntimeError: boom"


def _legacy_trim_bounds(mono: np.ndarray, sample_rate: int):
    frame_size = max(1, int(sample_rate * 0.02))
    rms: list[float] = []
    for start in range(0, len(mono), frame_size):
        frame = mono[start : start + frame_size]
        if frame.size:
            rms.append(float(np.sqrt(np.mean(frame * frame))))
    if not rms:
        return None
    peak = max(rms)
    if peak <= 1e-6:
        return None
    threshold = max(peak * 0.04, 1e-3)
    voiced = [index for index, value in enumerate(rms) if value >= threshold]
    if not voiced:
        return None
    lead_pad = int(sample_rate * 0.05)
    trail_pad = int(sample_rate * 0.08)
    start = max(0, voiced[0] * frame_size - lead_pad)
    end = min(len(mono), (voiced[-1] + 1) * frame_size + trail_pad)
    if end <= start or (start == 0 and end == len(mono)):
        return None
    if end - start < int(sample_rate * 0.2):
        return None
    return start, end


def test_trim_speech_audio_matches_legacy_frame_loop():
    sample_rate = 16000
    silence = np.zeros(int(sample_rate * 0.7), dtype=np.float32)
    tone = (0.4 * np.sin(2 * np.pi * 220 * np.arange(int(sample_rate * 1.3)) / sample_rate)).astype(np.float32)
    signal = np.concatenate([silence, tone, silence])
    wav = _wav_buffer(signal, sample_rate).getvalue()

    trimmed, content_type = _trim_speech_audio(wav, "audio/wav")
    decoded, decoded_rate = sf.read(BytesIO(trimmed), dtype="float32")
    bounds = _legacy_trim_bounds(signal, sample_rate)

    assert content_type == "audio/wav"
    assert bounds is not None
    assert decoded_rate == sample_rate
    assert decoded.shape[0] == bounds[1] - bounds[0]
    # _wav_buffer writes 16-bit PCM, so allow one quantization step of drift.
    assert np.allclose(decoded, signal[bounds[0] : bounds[1]], atol=1e-4)
    assert decoded.shape[0] < signal.shape[0]


def test_tts_mlx_path_trims_float32_without_wav_round_trip(monkeypatch):
    sample_rate = 24000
    silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    tone = (0.4 * np.sin(2 * np.pi * 220 * np.arange(int(sample_rate * 1.0)) / sample_rate)).astype(np.float32)
    signal = np.concatenate([silence, tone, silence])
    service = TtsService(TtsSettings(local_model_dir="/missing/local-tts-model", preload=False))
    monkeypatch.setattr(service, "_should_try_mlx", lambda: True)
    monkeypatch.setattr(service._mlx, "synthesize_samples", lambda text, voice: (signal, sample_rate))
    monkeypatch.setattr(
        service._mlx, "synthesize", lambda text, voice: (_ for _ in ()).throw(AssertionError("no WAV round trip"))
    )

    audio, content_type = service.synthesize("你好", voice_id="xuefen")
    decoded, decoded_rate = sf.read(BytesIO(audio), dtype="float32")
    expected = _trim_speech_samples(signal, sample_rate)

    assert content_type == "audio/wav"
    assert decoded_rate == sample_rate
    assert decoded.shape[0] == expected.shape[0] < signal.shape[0]
    assert np.allclose(decoded, expected, atol=1e-4)


def test_tts_mlx_empty_samples_do_not_re_run_the_wav_synthesis(monkeypatch):
    service = TtsService(TtsSettings(local_model_dir="/missing/local-tts-model", preload=False))
    monkeypatch.setattr(service, "_should_try_mlx", lambda: True)
    monkeypatch.setattr(service._mlx, "synthesize_samples", lambda text, voice: (np.zeros(0, dtype=np.float32), 24000))
    monkeypatch.setattr(
        service._mlx, "synthesize", lambda text, voice: (_ for _ in ()).throw(AssertionError("no second synthesis"))
    )

    audio, content_type = service.synthesize("你好", voice_id="xuefen")
    decoded, decoded_rate = sf.read(BytesIO(audio), dtype="float32")

    assert content_type == "audio/wav"
    assert decoded_rate == 24000
    assert decoded.shape[0] == 0


def test_tts_trim_keeps_a_soft_attack_before_the_first_voiced_frame():
    sample_rate = 16000
    silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    tone = (0.4 * np.sin(2 * np.pi * 220 * np.arange(int(sample_rate * 1.0)) / sample_rate)).astype(np.float32)
    signal = np.concatenate([silence, tone, silence])

    trimmed = _trim_speech_samples(signal, sample_rate)
    bounds = _speech_bounds(signal, sample_rate)

    assert trimmed.shape[0] < signal.shape[0]
    # 50ms of pre-roll is kept ahead of the first voiced 20ms frame.
    assert bounds is not None
    assert bounds[0] == silence.shape[0] - int(sample_rate * 0.05)


def test_tts_mlx_path_falls_back_to_wav_api_when_samples_api_missing(monkeypatch):
    service = TtsService(TtsSettings(local_model_dir="/missing/local-tts-model", preload=False))
    monkeypatch.setattr(service, "_should_try_mlx", lambda: True)
    monkeypatch.delattr(type(service._mlx), "synthesize_samples", raising=True)
    monkeypatch.setattr(service._mlx, "synthesize", lambda text, voice: (b"RIFF-legacy", "audio/wav"))

    audio, content_type = service.synthesize("你好", voice_id="xuefen")

    assert audio == b"RIFF-legacy"
    assert content_type == "audio/wav"


def test_trim_speech_audio_keeps_pure_silence_untouched():
    sample_rate = 16000
    wav = _wav_buffer(np.zeros(sample_rate, dtype=np.float32), sample_rate).getvalue()

    trimmed, content_type = _trim_speech_audio(wav, "audio/wav")

    assert trimmed == wav
    assert content_type == "audio/wav"
