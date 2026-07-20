from pathlib import Path

import pytest

from langcode_agent.voice.tts import TtsService, TtsSettings


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
