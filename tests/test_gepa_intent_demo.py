from __future__ import annotations

import inspect
from types import SimpleNamespace

from trial import gepa_intent_demo


class FakeDspy:
    def __init__(self) -> None:
        self.configured_lm = None
        self.created_lms = []

    def LM(self, model: str, **kwargs):
        lm = SimpleNamespace(model=model, kwargs=kwargs)
        self.created_lms.append(lm)
        return lm

    def configure(self, *, lm):
        self.configured_lm = lm


def clear_model_env(monkeypatch) -> None:
    for key in (
        "LANGCODE_GEPA_MODEL",
        "LANGCODE_GEPA_REFLECTION_MODEL",
        "LANGCODE_MODEL",
        "LANGCODE_OPENAI_GATEWAY",
        "AIMP_GLM_API_KEY",
        "AIMP_GLM_BASE_URL",
        "AIMP_GLM_USER",
        "AIMP_DEEPSEEK_V4_API_KEY",
        "AIMP_DEEPSEEK_V4_BASE_URL",
        "AIMP_DEEPSEEK_V4_USER",
        "AIMP_GPT4O_API_KEY",
        "AIMP_GPT4O_BASE_URL",
        "AIMP_GPT4O_USER",
        "AIGC_USER",
    ):
        monkeypatch.delenv(key, raising=False)


def test_model_config_maps_glm5_to_aimp_glm_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(gepa_intent_demo, "PROJECT_ROOT", tmp_path)
    clear_model_env(monkeypatch)
    monkeypatch.setenv("AIMP_GLM_API_KEY", "glm-key")
    monkeypatch.setenv("AIMP_GLM_USER", "guojian34")

    cfg = gepa_intent_demo.model_config("glm-5")

    assert cfg == {
        "model": "glm-5",
        "api_base": "https://aimpapi.midea.com/t-aigc/aimp-glm/v1",
        "api_key": "glm-key",
        "headers": {"AIGC-USER": "guojian34"},
    }


def test_model_config_maps_deepseek_to_aimp_deepseek_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(gepa_intent_demo, "PROJECT_ROOT", tmp_path)
    clear_model_env(monkeypatch)
    monkeypatch.setenv("AIMP_DEEPSEEK_V4_API_KEY", "deepseek-key")
    monkeypatch.setenv("AIMP_DEEPSEEK_V4_USER", "guojian34")

    cfg = gepa_intent_demo.model_config("deepseek-v4-pro")

    assert cfg["model"] == "deepseek-v4-pro"
    assert cfg["api_base"] == "https://aimpapi.midea.com/t-aigc/aimp-deepseek-v4-pro/v1"
    assert cfg["api_key"] == "deepseek-key"
    assert cfg["headers"] == {"AIGC-USER": "guojian34"}


def test_model_config_maps_gpt4o_to_aimp_gpt4o_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(gepa_intent_demo, "PROJECT_ROOT", tmp_path)
    clear_model_env(monkeypatch)
    monkeypatch.setenv("AIMP_GPT4O_API_KEY", "gpt-key")
    monkeypatch.setenv("AIMP_GPT4O_USER", "guojian34")

    cfg = gepa_intent_demo.model_config("gpt-4o")

    assert cfg["model"] == "gpt-4o"
    assert cfg["api_base"] == "https://aimpapi.midea.com/t-aigc/mip-chat-app/openai/standard/v1"
    assert cfg["api_key"] == "gpt-key"
    assert cfg["headers"] == {"Aimp-Biz-Id": "gpt-4o", "AIGC-USER": "guojian34"}


def test_make_lm_sets_non_streaming_dspy_lm() -> None:
    fake = FakeDspy()
    cfg = {
        "model": "glm-5",
        "api_base": "https://example.test/v1",
        "api_key": "key",
        "headers": {"AIGC-USER": "user"},
    }

    lm = gepa_intent_demo.make_lm(
        fake,
        cfg,
        request_timeout=30,
        num_retries=0,
        max_tokens=128,
        temperature=0.0,
    )

    assert lm.model == "openai/glm-5"
    assert lm.kwargs["stream"] is False
    assert lm.kwargs["api_base"] == "https://example.test/v1"
    assert lm.kwargs["extra_headers"] == {"AIGC-USER": "user"}


def test_configure_dspy_returns_main_and_reflection_lms(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(gepa_intent_demo, "PROJECT_ROOT", tmp_path)
    clear_model_env(monkeypatch)
    monkeypatch.setenv("AIMP_GLM_API_KEY", "glm-key")
    monkeypatch.setenv("AIMP_DEEPSEEK_V4_API_KEY", "deepseek-key")
    fake = FakeDspy()

    main_lm, reflection_lm = gepa_intent_demo.configure_dspy(
        fake,
        model_name="glm-5",
        reflection_model_name="deepseek-v4-pro",
        request_timeout=30,
        num_retries=0,
        max_tokens=128,
        reflection_max_tokens=2048,
        reflection_temperature=1.0,
    )

    assert fake.configured_lm is main_lm
    assert main_lm.model == "openai/glm-5"
    assert reflection_lm.model == "openai/deepseek-v4-pro"
    assert reflection_lm.kwargs["max_tokens"] == 2048
    assert reflection_lm.kwargs["temperature"] == 1.0


def test_progress_metric_accepts_current_gepa_signature(capsys) -> None:
    example = SimpleNamespace(label="memory_update")
    pred = SimpleNamespace(intent="memory_update")
    metric = gepa_intent_demo.progress_metric(gepa_intent_demo.intent_metric, max_calls=3)

    inspect.signature(metric).bind(None, None, None, None, None)
    result = metric(example, pred, None, "intent", None)

    assert result.score == 1.0
    output = capsys.readouterr().out
    assert "[GEPA metric 1/3] pred=intent expected=memory_update predicted=memory_update score=1.00" in output
