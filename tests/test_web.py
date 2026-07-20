import concurrent.futures
import json
from pathlib import Path
import sqlite3

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage

from langcode_agent.interfaces.web import WebApp, _repair_tool_history, _tool_result_event
from langcode_agent.interfaces.web import _VoiceModeOutputFilter, _sanitize_voice_mode_output, _voice_mode_messages


def test_web_chat_without_api_key_returns_actionable_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LANGCODE_PROVIDER", raising=False)
    monkeypatch.delenv("LANGCODE_MODEL", raising=False)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AIMP_GPT4O_API_KEY", raising=False)
    app = WebApp(tmp_path, tmp_path)

    response = app.chat({"sessionId": "no-key", "message": "hello"})

    assert response["ok"] is False
    assert "provider=zhipu" in response["error"]


def test_web_local_memory_command_does_not_require_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = WebApp(tmp_path, tmp_path)

    response = app.chat({"sessionId": "memory", "message": "# 记住先跑快速测试"})

    assert response["ok"] is True
    assert "已写入项目记忆" in response["messages"][0]["content"]
    assert "先跑快速测试" in (tmp_path / ".langcode" / "memories" / "MEMORY.md").read_text(encoding="utf-8")


def test_web_tts_preview_decodes_encoded_voice_id(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LANGCODE_ASR_PRELOAD", "0")
    monkeypatch.setenv("LANGCODE_TTS_SAMPLE_DIR", str(tmp_path))
    monkeypatch.setenv("LANGCODE_TTS_PREVIEW_DIR", str(tmp_path / "previews"))
    (tmp_path / "汪菊.wav").write_bytes(b"RIFF-sample")
    app = WebApp(tmp_path, tmp_path)

    audio, content_type = app.tts_voice_preview("%E6%B1%AA%E8%8F%8A")

    assert audio == b"RIFF-sample"
    assert content_type == "audio/x-wav"


def test_web_chat_events_refreshes_old_session_system_prompt_to_hermes_memory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    (tmp_path / ".langcode" / "memories").mkdir(parents=True)
    (tmp_path / ".langcode" / "memories" / "MEMORY.md").write_text("新 Hermes 热记忆", encoding="utf-8")
    (tmp_path / ".langcode" / "MEMORY.md").write_text("旧版记忆不应加载", encoding="utf-8")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("old-session")
    session.messages = [SystemMessage(content="你是一个谨慎的代码 Agent，旧 system prompt"), HumanMessage(content="旧问题")]
    session.display_messages = list(session.messages)
    model = FakeCaptureMessagesStreamingModel()
    session.model = model

    events = list(app.chat_events({"sessionId": "old-session", "message": "继续"}))

    assert events[-1] == {"type": "done", "ok": True}
    system_text = str(model.messages[0].content)
    assert "新 Hermes 热记忆" in system_text
    assert "旧版记忆不应加载" not in system_text


def test_web_chat_events_auto_reflects_once_and_persists_cursor(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("reflect-session")
    session.model = FakeCaptureMessagesStreamingModel()

    events = list(app.chat_events({"sessionId": "reflect-session", "message": "以后默认先跑快速测试。"}))

    assert events[-1] == {"type": "done", "ok": True}
    assert "以后默认先跑快速测试" in (tmp_path / ".langcode" / "memories" / "USER.md").read_text(encoding="utf-8")
    stored = app.store.load_session("reflect-session")
    assert stored is not None
    assert stored["state"]["last_reflected_count"] > 0
    before = (tmp_path / ".langcode" / "evolution" / "reflections.jsonl").read_text(encoding="utf-8")

    app._save_history(session)
    after = (tmp_path / ".langcode" / "evolution" / "reflections.jsonl").read_text(encoding="utf-8")
    assert after == before


def test_voice_mode_prompt_forbids_meta_format_preamble() -> None:
    prompt = str(_voice_mode_messages(True)[0].content)

    assert "不要把这些格式要求" in prompt
    assert "第一句话必须直接回答用户问题" in prompt


def test_voice_mode_output_sanitizer_removes_leading_meta_control() -> None:
    content = _sanitize_voice_mode_output("回答控制在 300 字内。嗯，我是 Hermes，一个编程助手。")

    assert content == "嗯，我是 Hermes，一个编程助手。"


def test_voice_mode_output_sanitizer_removes_bare_meta_control() -> None:
    content = _sanitize_voice_mode_output("控制在两百字以内。我是 Hermes，一个编程助手。")

    assert content == "我是 Hermes，一个编程助手。"


def test_voice_mode_output_filter_buffers_split_meta_control() -> None:
    output_filter = _VoiceModeOutputFilter()

    assert output_filter.push("回答控制在 ") == ""
    assert output_filter.push("300 字内。嗯，我是 Hermes。") == "嗯，我是 Hermes。"
    assert output_filter.push("继续回答。") == "继续回答。"


def test_web_voice_worker_mode_does_not_load_local_voice_services(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LANGCODE_VOICE_WORKER_URL", "http://127.0.0.1:8879")
    app = WebApp(tmp_path, tmp_path)
    assert app.asr is None
    assert app.tts is None
    assert app.turnsense is None
    assert app.voice_worker is not None
    monkeypatch.setattr(
        app.voice_worker,
        "status",
        lambda: {
            "ok": True,
            "asr": {"ok": True, "state": "ready"},
            "turnsense": {"ok": True, "state": "ready"},
            "tts": {"ok": True, "state": "ready"},
        },
    )

    status = app.status()

    assert status["voiceWorker"] == {"enabled": True, "ok": True, "url": "http://127.0.0.1:8879"}
    assert status["asr"]["state"] == "ready"
    assert status["tts"]["state"] == "ready"


def test_web_compact_command_archives_and_saves_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("compact")
    from langchain_core.messages import AIMessage

    session.messages = [AIMessage(content=f"older {index}") for index in range(12)]
    session.display_messages = list(session.messages)

    response = app.chat({"sessionId": "compact", "message": "/compact"})

    assert response["ok"] is True
    assert "已压缩" in response["messages"][0]["content"]
    assert any("compact summary" in str(message.content) for message in session.messages)
    assert list((tmp_path / ".langcode" / "compactions").glob("compact-*.json"))


def test_web_compact_preserves_visible_history_after_reload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("compact-display")
    session.messages = [HumanMessage(content="用户早期需求"), AIMessage(content="早期回答")]
    session.display_messages = list(session.messages)
    app._save_history(session)

    response = app.chat({"sessionId": "compact-display", "message": "/compact 压缩上下文"})

    assert response["ok"] is True
    assert any("compact summary" in str(message.content) for message in session.messages)
    reloaded = WebApp(tmp_path, tmp_path)
    view = reloaded.session_view("compact-display")
    contents = [message["content"] for message in view["messages"]]
    assert contents == [
        "用户早期需求",
        "早期回答",
        "/compact 压缩上下文",
        response["messages"][0]["content"],
    ]


def test_web_recovers_legacy_compacted_visible_history_from_archive(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = WebApp(tmp_path, tmp_path)
    archive_dir = tmp_path / ".langcode" / "compactions"
    archive_dir.mkdir(parents=True)
    archive = archive_dir / "legacy-0003.json"
    compact_reply = "已压缩当前会话上下文。较早消息已汇总为一条系统摘要，最近消息会继续保留；完整压缩归档已写入本地状态目录。"
    archive.write_text(
        json.dumps(
            {
                "summary": "LangCode context compact summary.",
                "instructions": "压缩上下文",
                "messages": [
                    {"role": "human", "content": "压缩前用户问题"},
                    {"role": "ai", "content": "压缩前回答"},
                    {"role": "human", "content": "/compact 压缩上下文"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    app.store.save_messages(
        "legacy",
        str(tmp_path),
        [
            {"role": "system", "content": "LangCode context compact summary."},
            {"role": "human", "content": "/compact 压缩上下文"},
            {"role": "ai", "content": compact_reply},
        ],
        state={"todos": []},
    )

    reloaded = WebApp(tmp_path, tmp_path)
    view = reloaded.session_view("legacy")
    contents = [message["content"] for message in view["messages"]]

    assert contents == [
        "压缩前用户问题",
        "压缩前回答",
        "/compact 压缩上下文",
        compact_reply,
    ]


def test_web_chat_events_local_compact_emits_done_without_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = WebApp(tmp_path, tmp_path)

    events = list(app.chat_events({"sessionId": "stream-compact", "message": "/compact"}))

    assert events[0]["type"] == "delta"
    assert "已压缩" in events[0]["content"]
    assert events[-1] == {"type": "done", "ok": True}


def test_web_chat_appends_user_messages_after_model_is_initialized(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("multi-turn")
    session.model = FakeTwoTurnModel()

    first = app.chat({"sessionId": "multi-turn", "message": "第一句"})
    second = app.chat({"sessionId": "multi-turn", "message": "第二句"})

    assert first["ok"] is True
    assert second["ok"] is True
    human_messages = [message.content for message in session.messages if isinstance(message, HumanMessage)]
    assert human_messages == ["第一句", "第二句"]


class FakeStreamingModel:
    def stream(self, _messages):
        yield AIMessageChunk(content="hel")
        yield AIMessageChunk(content="lo")


class FakeCaptureMessagesStreamingModel:
    def __init__(self) -> None:
        self.messages = []

    def stream(self, messages):
        self.messages = list(messages)
        yield AIMessageChunk(content="已停下来")


class FakeShouldNotStreamModel:
    def stream(self, _messages):
        raise AssertionError("stream should not be called after cancellation")


class FakeTwoChunkStreamingModel:
    def stream(self, _messages):
        yield AIMessageChunk(content="partial")
        yield AIMessageChunk(content="tail")


class FakeThinkingStreamingModel:
    def stream(self, _messages):
        yield AIMessageChunk(content="", additional_kwargs={"reasoning_content": "先分析需求。"})
        yield AIMessageChunk(content=[{"type": "reasoning", "text": "再检查约束。"}])
        yield AIMessageChunk(content="正式回答")


class FakeRawThinkingStreamingModel:
    def stream(self, _messages):
        yield AIMessageChunk(content="<think>先想一下")
        yield AIMessageChunk(content="，再确认。</think>")
        yield AIMessageChunk(content="正式回答")


class FakeStrayThinkCloseStreamingModel:
    def stream(self, _messages):
        yield AIMessageChunk(content="</think>好的，收到。")


class FakeTwoTurnModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        return AIMessage(content=f"answer {self.calls}")


class FakeSimpleAnswerModel:
    def invoke(self, _messages):
        return AIMessage(content="普通回答")


class FakeRawThinkingInvokeModel:
    def invoke(self, _messages):
        return AIMessage(content="<think>内部推理</think>最终回答")


class FakeInterruptedStreamingModel:
    def stream(self, _messages):
        yield AIMessageChunk(content="partial")
        raise RuntimeError("stream interrupted")


class FakeNoReplyStreamingModel:
    def stream(self, _messages):
        raise RuntimeError("model failed before replying")


class FakeStreamingToolModel:
    def __init__(self) -> None:
        self.calls = 0

    def stream(self, _messages):
        self.calls += 1
        if self.calls == 1:
            yield AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": "README.md"},
                        "id": "read-call-1",
                    }
                ],
            )
            return
        yield AIMessageChunk(content="done")


class FakeDiagramStreamingModel:
    def __init__(self) -> None:
        self.calls = 0

    def stream(self, _messages):
        self.calls += 1
        if self.calls == 1:
            yield AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "diagram",
                        "args": {
                            "title": "协作逻辑",
                            "diagram_type": "collaboration",
                            "direction": "LR",
                            "nodes": [
                                {"id": "main_agent", "label": "主 Agent"},
                                {"id": "reviewer", "label": "审查子 Agent"},
                            ],
                            "edges": [
                                {"source": "main_agent", "target": "reviewer", "label": "委派审查"},
                            ],
                        },
                        "id": "diagram-call-1",
                    }
                ],
            )
            return
        yield AIMessageChunk(content="图已生成")


class FakeAgentDebateStreamingModel:
    def __init__(self) -> None:
        self.calls = 0

    def stream(self, _messages):
        self.calls += 1
        if self.calls == 1:
            yield AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "agent_debate",
                        "args": {
                            "debate_id": "debate-test",
                            "topic": "先有鸡还是先有蛋",
                            "agents": [
                                {"id": "egg", "name": "先有蛋", "stance": "先有蛋"},
                                {"id": "chicken", "name": "先有鸡", "stance": "先有鸡"},
                            ],
                            "rounds": 1,
                        },
                        "id": "debate-call-1",
                    }
                ],
            )
            return
        yield AIMessageChunk(content="辩论已汇总")


class FakeTasksStreamingModel:
    def __init__(self) -> None:
        self.calls = 0

    def stream(self, _messages):
        self.calls += 1
        if self.calls == 1:
            yield AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "task_create",
                        "args": {"content": "读取代码", "status": "completed"},
                        "id": "task-create-1",
                    },
                    {
                        "name": "task_create",
                        "args": {"content": "实现功能", "status": "in_progress"},
                        "id": "task-create-2",
                    }
                ],
            )
            return
        yield AIMessageChunk(content="继续执行")


class FakePartialTaskUpdateModel:
    def __init__(self) -> None:
        self.calls = 0

    def stream(self, _messages):
        self.calls += 1
        if self.calls == 1:
            yield AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "task_update",
                        "args": {"task_id": "1", "status": "completed"},
                        "id": "partial-task-update-1",
                    }
                ],
            )
            return
        yield AIMessageChunk(content="继续。")


class FakePlanOnlyThenWorkModel:
    def __init__(self) -> None:
        self.calls = 0

    def stream(self, _messages):
        self.calls += 1
        if self.calls == 1:
            yield AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "task_create",
                        "args": {"content": "读取 README", "status": "pending"},
                        "id": "plan-call-1",
                    }
                ],
            )
            return
        if self.calls == 2:
            yield AIMessageChunk(content="计划写好了。")
            return
        if self.calls == 3:
            yield AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": "README.md"},
                        "id": "read-call-1",
                    }
                ],
            )
            return
        yield AIMessageChunk(content="已完成读取。")


class FakeTaskToolStreamingModel:
    def __init__(self) -> None:
        self.calls = 0

    def stream(self, _messages):
        self.calls += 1
        if self.calls == 1:
            yield AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "task_create",
                        "args": {"content": "读取代码", "status": "pending"},
                        "id": "task-create-1",
                    }
                ],
            )
            return
        if self.calls == 2:
            yield AIMessageChunk(content="任务已创建。")
            return
        if self.calls == 3:
            yield AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "task_update",
                        "args": {"task_id": "1", "status": "completed"},
                        "id": "task-update-1",
                    }
                ],
            )
            return
        yield AIMessageChunk(content="任务完成。")


class FakeTaskLimitContinuationModel:
    def __init__(self) -> None:
        self.calls = 0

    def stream(self, _messages):
        self.calls += 1
        if self.calls == 1:
            yield AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "task_create",
                        "args": {"content": "完成长任务", "status": "in_progress"},
                        "id": "task-create-limit-1",
                    }
                ],
            )
            return
        if self.calls == 2:
            yield AIMessageChunk(content="阶段性整理，继续推进。")
            return
        if self.calls == 3:
            yield AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "task_update",
                        "args": {"task_id": "1", "status": "completed"},
                        "id": "task-update-limit-1",
                    }
                ],
            )
            return
        yield AIMessageChunk(content="任务已经完成。")


class FakeWebSearchLimitStreamingModel:
    def __init__(self) -> None:
        self.calls = 0

    def stream(self, _messages):
        self.calls += 1
        if self.calls == 1:
            yield AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "第一次搜索"},
                        "id": "web-search-1",
                    }
                ],
            )
            return
        if self.calls == 2:
            yield AIMessageChunk(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "第二次搜索"},
                        "id": "web-search-2",
                    }
                ],
            )
            return
        yield AIMessageChunk(content="基于第一次搜索结果回答。")


class FakeApprovalToolModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": "created.txt", "content": "approved"},
                        "id": "call-1",
                    }
                ],
            )
        return AIMessage(content="done")


class FakeApprovalStreamingModel(FakeApprovalToolModel):
    def stream(self, _messages):
        yield AIMessageChunk(content="done")


class FakeShellApprovalModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "shell",
                        "args": {"command": "mkdir output", "timeout_seconds": 5},
                        "id": "shell-call-1",
                    }
                ],
            )
        return AIMessage(content="done")


def test_web_chat_events_streams_model_deltas(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("stream")
    session.model = FakeStreamingModel()

    events = list(app.chat_events({"sessionId": "stream", "message": "hi"}))

    assert events == [
        {"type": "delta", "content": "hel"},
        {"type": "delta", "content": "lo"},
        {"type": "done", "ok": True},
    ]


def test_web_chat_events_streams_thinking_separately_from_answer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("thinking-stream")
    session.model = FakeThinkingStreamingModel()

    events = list(app.chat_events({"sessionId": "thinking-stream", "message": "hi"}))

    assert events == [
        {"type": "thinking_delta", "content": "先分析需求。"},
        {"type": "thinking_delta", "content": "再检查约束。"},
        {"type": "delta", "content": "正式回答"},
        {"type": "done", "ok": True},
    ]
    stored = app.store.load_session("thinking-stream")
    assert stored["messages"][-1]["content"] == "正式回答"


def test_web_chat_events_strips_raw_think_tags_from_visible_answer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("raw-thinking-stream")
    session.model = FakeRawThinkingStreamingModel()

    events = list(app.chat_events({"sessionId": "raw-thinking-stream", "message": "hi"}))

    assert events == [
        {"type": "thinking_delta", "content": "先想一下"},
        {"type": "thinking_delta", "content": "，再确认。"},
        {"type": "delta", "content": "正式回答"},
        {"type": "done", "ok": True},
    ]
    stored = app.store.load_session("raw-thinking-stream")
    assert stored["messages"][-1]["content"] == "正式回答"


def test_web_chat_events_drops_stray_raw_think_close_tag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("stray-thinking-close")
    session.model = FakeStrayThinkCloseStreamingModel()

    events = list(app.chat_events({"sessionId": "stray-thinking-close", "message": "hi"}))

    assert events == [
        {"type": "delta", "content": "好的，收到。"},
        {"type": "done", "ok": True},
    ]
    stored = app.store.load_session("stray-thinking-close")
    assert stored["messages"][-1]["content"] == "好的，收到。"


def test_web_chat_strips_raw_think_tags_from_non_stream_answer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("raw-thinking-chat")
    session.model = FakeRawThinkingInvokeModel()

    response = app.chat({"sessionId": "raw-thinking-chat", "message": "hi"})

    assert response["ok"] is True
    assert response["messages"] == [{"role": "assistant", "content": "最终回答", "kind": "message"}]
    stored = app.store.load_session("raw-thinking-chat")
    assert stored["messages"][-1]["content"] == "最终回答"


def test_web_chat_events_voice_interrupt_uses_hidden_tool_context(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("voice-interrupt")
    session.messages = [HumanMessage(content="用一句话介绍一下你自己。"), AIMessage(content="我是一个代码助手。")]
    session.display_messages = list(session.messages)
    model = FakeCaptureMessagesStreamingModel()
    session.model = model

    events = list(
        app.chat_events(
            {
                "sessionId": "voice-interrupt",
                "message": "等一下",
                "voiceInterrupt": {
                    "spokenText": "等一下",
                    "previousUserText": "用一句话介绍一下你自己。",
                    "assistantDisplayedText": "我是一个代码助手。",
                },
            }
        )
    )

    assert events == [{"type": "delta", "content": "已停下来"}, {"type": "done", "ok": True}]
    assert isinstance(model.messages[-3], HumanMessage)
    assert model.messages[-3].content == "等一下"
    assert isinstance(model.messages[-2], AIMessage)
    assert model.messages[-2].tool_calls[0]["name"] == "voice_interrupt"
    assert isinstance(model.messages[-1], ToolMessage)
    tool_payload = json.loads(model.messages[-1].content)
    assert tool_payload["spoken_text"] == "等一下"
    assert tool_payload["previous_user_text"] == "用一句话介绍一下你自己。"
    visible_contents = [message["content"] for message in app.session_view("voice-interrupt")["messages"]]
    assert visible_contents[-2:] == ["等一下", "已停下来"]
    assert all("打断前 assistant 已展示内容" not in content for content in visible_contents)


def test_web_chat_events_persists_user_and_partial_ai_during_stream(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("partial-stream")
    session.model = FakeInterruptedStreamingModel()

    events = list(app.chat_events({"sessionId": "partial-stream", "message": "keep this"}))

    assert events[-1]["type"] == "error"
    assert "stream interrupted" in events[-1]["error"]
    stored = app.store.load_session("partial-stream")
    assert stored is not None
    assert [message["role"] for message in stored["messages"]] == ["system", "human", "ai"]
    assert stored["messages"][1]["content"] == "keep this"
    assert stored["messages"][2]["content"] == "partial"


def test_web_chat_events_drops_failed_turn_when_model_never_replies(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("no-reply")
    session.model = FakeNoReplyStreamingModel()

    events = list(app.chat_events({"sessionId": "no-reply", "message": "不要留下这个失败问题"}))

    assert events[-1]["type"] == "error"
    stored = app.store.load_session("no-reply")
    assert stored is not None
    assert [message["role"] for message in stored["messages"]] == ["system"]
    assert app.session_view("no-reply")["messages"] == []


def test_web_chat_events_emits_tool_progress(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("progress")
    session.model = FakeStreamingToolModel()

    events = list(app.chat_events({"sessionId": "progress", "message": "read readme"}))

    progress_events = [event for event in events if event["type"] == "progress"]
    assert progress_events[0]["status"] == "running"
    assert progress_events[0]["toolName"] == "read_file"
    assert progress_events[0]["target"] == "README.md"
    assert progress_events[1]["status"] == "completed"
    assert progress_events[-1]["status"] == "summary"
    assert any(event["type"] == "delta" and event["content"] == "done" for event in events)
    assert not any(event["type"] == "tool_result" for event in events)


def test_web_chat_events_emits_and_persists_diagram(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("diagram")
    session.model = FakeDiagramStreamingModel()

    events = list(app.chat_events({"sessionId": "diagram", "message": "画出协作逻辑"}))

    diagram_events = [event for event in events if event["type"] == "tool_result" and event.get("kind") == "diagram"]
    assert diagram_events
    assert diagram_events[0]["title"] == "协作逻辑"
    assert diagram_events[0]["content"].startswith("flowchart LR")
    assert "委派审查" in diagram_events[0]["content"]

    viewed = app.session_view("diagram")
    diagram_messages = [message for message in viewed["messages"] if message["kind"] == "diagram"]
    assert diagram_messages
    assert diagram_messages[0]["title"] == "协作逻辑"


def test_web_chat_events_emits_and_persists_agent_dialogue(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")

    def fake_debate_events(_workspace_root, **tool_input):
        yield {
            "ok": True,
            "kind": "agent_dialogue",
            "dialogue_type": "debate",
            "thread_id": tool_input["debate_id"],
            "title": "辩论：先有鸡还是先有蛋",
            "participants": [
                {"id": "egg", "name": "先有蛋", "stance": "先有蛋"},
                {"id": "chicken", "name": "先有鸡", "stance": "先有鸡"},
            ],
            "messages": [
                {"agent_id": "egg", "agent_name": "先有蛋", "role": "assistant", "round": 1, "content": "蛋方发言"},
            ],
        }
        yield {
            "ok": True,
            "kind": "agent_dialogue",
            "dialogue_type": "debate",
            "thread_id": tool_input["debate_id"],
            "title": "辩论：先有鸡还是先有蛋",
            "participants": [
                {"id": "egg", "name": "先有蛋", "stance": "先有蛋"},
                {"id": "chicken", "name": "先有鸡", "stance": "先有鸡"},
            ],
            "messages": [
                {"agent_id": "egg", "agent_name": "先有蛋", "role": "assistant", "round": 1, "content": "蛋方发言"},
                {"agent_id": "chicken", "agent_name": "先有鸡", "role": "assistant", "round": 1, "content": "鸡方发言"},
            ],
        }

    monkeypatch.setattr("langcode_agent.interfaces.web.iter_agent_debate_events", fake_debate_events)
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("agent-dialogue")
    session.model = FakeAgentDebateStreamingModel()

    events = list(app.chat_events({"sessionId": "agent-dialogue", "message": "辩论一下"}))

    dialogue_events = [event for event in events if event["type"] == "tool_result" and event.get("kind") == "agent_dialogue"]
    assert len(dialogue_events) == 2
    assert dialogue_events[0]["threadId"] == "debate-test"
    assert len(dialogue_events[0]["messages"]) == 1
    assert len(dialogue_events[1]["messages"]) == 2
    viewed = app.session_view("agent-dialogue")
    dialogue_messages = [message for message in viewed["messages"] if message["kind"] == "agent_dialogue"]
    assert dialogue_messages
    assert dialogue_messages[0]["messages"][0]["content"] == "蛋方发言"


def test_web_chat_events_emits_and_persists_todos(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("todos")
    session.model = FakeTasksStreamingModel()

    events = list(app.chat_events({"sessionId": "todos", "message": "做一个计划"}))

    todo_events = [event for event in events if event["type"] == "todos"]
    assert todo_events
    assert todo_events[-1]["todos"][1]["status"] == "in_progress"
    assert app.session_view("todos")["todos"][1]["content"] == "实现功能"

    reloaded = WebApp(tmp_path, tmp_path)
    assert reloaded.session_view("todos")["todos"][0]["status"] == "completed"


def test_web_chat_events_saves_todos_before_yielding_todo_event(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("todos-early-save")
    session.model = FakeTasksStreamingModel()

    iterator = app.chat_events({"sessionId": "todos-early-save", "message": "做一个计划"})
    todo_event = None
    for event in iterator:
        if event["type"] == "todos" and len(event["todos"]) == 2:
            todo_event = event
            break

    assert todo_event is not None
    reloaded = WebApp(tmp_path, tmp_path)
    assert reloaded.session_view("todos-early-save")["todos"][1]["status"] == "in_progress"


def test_web_chat_events_merges_partial_todo_updates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("partial-todos")
    session.todos = [
        {"id": "1", "content": "创建项目结构", "status": "in_progress"},
        {"id": "2", "content": "实现基础模板", "status": "pending"},
    ]
    session.model = FakePartialTaskUpdateModel()

    events = list(app.chat_events({"sessionId": "partial-todos", "message": "继续"}))

    todo_events = [event for event in events if event["type"] == "todos"]
    assert todo_events
    assert todo_events[0]["todos"] == [
        {"id": "1", "content": "创建项目结构", "status": "completed"},
        {"id": "2", "content": "实现基础模板", "status": "pending"},
    ]
    assert not any(event["type"] == "tool_result" for event in events)


def test_web_chat_events_continues_when_model_only_writes_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("plan-then-work")
    session.model = FakePlanOnlyThenWorkModel()

    events = list(app.chat_events({"sessionId": "plan-then-work", "message": "先计划再执行"}))

    assert any(
        event["type"] == "progress" and "继续执行第一项任务" in event.get("label", "")
        for event in events
    )
    assert any(event["type"] == "progress" and event.get("toolName") == "read_file" for event in events)
    assert events[-1] == {"type": "done", "ok": True}


def test_web_chat_events_supports_incremental_task_tools(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("task-tools")
    session.model = FakeTaskToolStreamingModel()

    events = list(app.chat_events({"sessionId": "task-tools", "message": "按任务工具执行"}))

    todo_events = [event for event in events if event["type"] == "todos"]
    assert [event["todos"][0]["status"] for event in todo_events] == ["pending", "completed"]
    assert app.session_view("task-tools")["todos"] == [
        {"id": "1", "content": "读取代码", "status": "completed"}
    ]
    assert any(event["type"] == "progress" and event.get("toolName") == "task_create" for event in events)
    assert any(event["type"] == "progress" and event.get("toolName") == "task_update" for event in events)
    assert not any(event["type"] == "tool_result" for event in events)


def test_web_chat_events_keeps_working_without_tool_round_limit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("task-limit-continuation")
    session.model = FakeTaskLimitContinuationModel()

    events = list(app.chat_events({"sessionId": "task-limit-continuation", "message": "执行长任务"}))

    assert events[-1] == {"type": "done", "ok": True}
    assert any(
        event["type"] == "progress" and "继续执行第一项任务" in event.get("label", "")
        for event in events
    )
    assert any(event["type"] == "progress" and event.get("toolName") == "task_update" for event in events)
    assert app.session_view("task-limit-continuation")["todos"] == [
        {"id": "1", "content": "完成长任务", "status": "completed"}
    ]


def test_web_chat_events_limits_external_web_search_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    monkeypatch.setenv("LANGCODE_WEB_SEARCH_LIMIT", "1")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("web-search-limit")
    session.model = FakeWebSearchLimitStreamingModel()
    session.agent.request_tool = lambda tool_call, thread_id: {
        "tool_result": {"ok": True, "results": {"query": tool_call.args["query"], "results": []}}
    }

    events = list(app.chat_events({"sessionId": "web-search-limit", "message": "查网页"}))

    assert {"type": "delta", "content": "基于第一次搜索结果回答。"} in events
    assert events[-1] == {"type": "done", "ok": True}
    assert any(
        event["type"] == "progress" and event.get("toolName") == "web_search" and event.get("ok") is False
        for event in events
    )
    assert any(
        event["type"] == "progress" and "外部网页搜索次数已达到上限" in event.get("label", "")
        for event in events
    )


def test_web_cancelled_stream_does_not_start_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("cancel-before-start")
    session.model = FakeShouldNotStreamModel()
    app.cancel_run({"sessionId": "cancel-before-start", "runId": "run-1"})

    events = list(app.chat_events({"sessionId": "cancel-before-start", "message": "stop", "runId": "run-1"}))

    assert events == [{"type": "done", "ok": False, "cancelled": True}]
    assert app.session_view("cancel-before-start")["messages"] == []


def test_web_cancelled_stream_saves_partial_and_stops_before_later_chunks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("cancel-mid-stream")
    session.model = FakeTwoChunkStreamingModel()
    events = app.chat_events({"sessionId": "cancel-mid-stream", "message": "stop later", "runId": "run-2"})

    assert next(events) == {"type": "delta", "content": "partial"}
    app.cancel_run({"sessionId": "cancel-mid-stream", "runId": "run-2"})

    assert list(events) == [{"type": "done", "ok": False, "cancelled": True}]
    contents = [message["content"] for message in app.session_view("cancel-mid-stream")["messages"]]
    assert contents == ["stop later", "partial"]


def test_web_approval_flow_does_not_return_empty_tool_call_assistant(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("approval-ui")
    session.model = FakeApprovalToolModel()

    pending = app.chat({"sessionId": "approval-ui", "message": "create a file"})

    assert pending["ok"] is True
    assert pending["messages"] == []
    assert pending["pendingApproval"]["toolName"] == "write_file"

    approved = app.approve({"sessionId": "approval-ui", "approval": {"type": "accept"}})

    assert approved["ok"] is True
    assert approved["messages"] == [{"role": "assistant", "content": "done", "kind": "message"}]
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "approved"


def test_web_approval_events_streams_resume_result(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("approval-stream-ui")
    session.model = FakeApprovalStreamingModel()

    pending = app.chat({"sessionId": "approval-stream-ui", "message": "create a file"})
    events = list(
        app.approval_events(
            {"sessionId": "approval-stream-ui", "runId": "approval-run-1", "approval": {"type": "accept"}}
        )
    )

    assert pending["pendingApproval"]["toolName"] == "write_file"
    assert any(event["type"] == "progress" and event.get("status") == "completed" for event in events)
    assert {"type": "delta", "content": "done"} in events
    assert events[-1] == {"type": "done", "ok": True}
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "approved"


def test_web_chat_does_not_return_stale_todos_when_turn_does_not_update_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("stale-todos")
    session.todos = [{"id": "1", "content": "旧任务", "status": "completed"}]
    session.model = FakeSimpleAnswerModel()

    response = app.chat({"sessionId": "stale-todos", "message": "后续普通问题"})

    assert response["ok"] is True
    assert response["messages"] == [{"role": "assistant", "content": "普通回答", "kind": "message"}]
    assert "todos" not in response
    assert app.session_view("stale-todos")["todos"] == session.todos


def test_web_shell_approval_can_be_remembered(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("remember-shell")
    session.model = FakeShellApprovalModel()

    pending = app.chat({"sessionId": "remember-shell", "message": "make output dir"})

    assert pending["pendingApproval"]["toolName"] == "shell"
    approved = app.approve(
        {"sessionId": "remember-shell", "approval": {"type": "accept", "remember": True}}
    )

    assert approved["ok"] is True
    assert (tmp_path / "output").is_dir()
    settings_text = (tmp_path / ".langcode" / "settings.json").read_text(encoding="utf-8")
    assert "Bash(mkdir output)" in settings_text

    second = app.get_session("remember-shell-next")
    second.model = FakeShellApprovalModel()
    response = app.chat({"sessionId": "remember-shell-next", "message": "make output dir again"})

    assert response["ok"] is True
    assert response["pendingApproval"] is None


def test_web_workspace_switch_preserves_existing_session_roots(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    app = WebApp(first, tmp_path)

    session = app.get_session("switch")
    assert session.agent.workspace_root == first

    response = app.set_workspace({"workspace": str(second)})

    assert response["ok"] is True
    assert response["workspace"] == str(second)
    assert app.store.path == first / ".langcode" / "web.sqlite"
    assert app.get_session("switch").agent.workspace_root == first
    assert app.get_session("new-after-switch").agent.workspace_root == second


def test_web_workspace_switch_keeps_left_sidebar_sessions(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    app = WebApp(first, tmp_path)
    old_session = app.get_session("old")
    old_session.messages = [HumanMessage(content="old"), AIMessage(content="answer")]
    app._save_history(old_session)

    app.set_workspace({"workspace": str(second)})
    new_session = app.get_session("new")
    new_session.messages = [HumanMessage(content="new")]
    app._save_history(new_session)

    listed = app.list_sessions()
    listed_ids = {item["id"] for item in listed["sessions"]}
    assert {"old", "new"} <= listed_ids
    assert app.session_view("old")["workspace"] == str(first)
    assert app.session_view("new")["workspace"] == str(second)


def test_web_create_session_binds_session_to_selected_workspace(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    app = WebApp(first, tmp_path)

    response = app.create_session({"sessionId": "bound", "workspace": str(second)})

    assert response["ok"] is True
    assert any(item["id"] == "bound" and item["workspace"] == str(second) for item in response["sessions"])
    session = app.get_session("bound")
    assert session.workspace_root == second
    assert app.store.load_session("bound")["workspace"] == str(second)


def test_web_pending_approval_persists_across_app_reload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("pending-reload")
    session.model = FakeApprovalToolModel()

    pending = app.chat({"sessionId": "pending-reload", "message": "create a file"})

    assert pending["pendingApproval"]["toolName"] == "write_file"
    reloaded = WebApp(tmp_path, tmp_path)
    viewed = reloaded.session_view("pending-reload")
    assert viewed["pendingApproval"]["toolName"] == "write_file"
    assert viewed["pendingApproval"]["toolInput"] == {"path": "created.txt", "content": "approved"}


def test_web_settings_updates_model_and_unbinds_sessions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LANGCODE_PROVIDER", "openai")
    monkeypatch.setenv("LANGCODE_OPENAI_GATEWAY", "aimp-glm")
    monkeypatch.setenv("LANGCODE_MODEL", "glm-5")
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("settings")
    session.model = object()

    response = app.set_settings({"provider": "openai", "model": "gpt-4o", "gateway": "aimp", "thinking": True})

    assert response["ok"] is True
    assert response["provider"] == "openai"
    assert response["model"] == "gpt-4o"
    assert response["gateway"] == "aimp"
    assert response["thinking"] is True
    assert session.model is None


def test_web_settings_updates_deepseek_thinking(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AIMP_DEEPSEEK_V4_API_KEY", "deepseek-key")
    app = WebApp(tmp_path, tmp_path)

    response = app.set_settings(
        {
            "provider": "openai",
            "model": "deepseek-v4-pro",
            "gateway": "aimp-deepseek-v4-pro",
            "thinking": True,
        }
    )

    assert response["ok"] is True
    assert response["model"] == "deepseek-v4-pro"
    assert response["gateway"] == "aimp-deepseek-v4-pro"
    assert response["thinking"] is True
    assert response["hasApiKey"] is True


def test_web_directories_lists_child_directories(tmp_path: Path) -> None:
    (tmp_path / "beta").mkdir()
    (tmp_path / "alpha").mkdir()
    (tmp_path / "file.txt").write_text("not a directory", encoding="utf-8")
    app = WebApp(tmp_path, tmp_path)

    response = app.directories(str(tmp_path))

    assert response["ok"] is True
    assert response["path"] == str(tmp_path)
    names = [item["name"] for item in response["directories"]]
    assert "alpha" in names
    assert "beta" in names
    assert "file.txt" not in names


def test_web_session_list_and_delete_removes_history(tmp_path: Path) -> None:
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("keep")
    session.messages = []
    app._save_history(session)
    checkpoint_path = tmp_path / ".langcode" / "checkpoints.sqlite"
    with sqlite3.connect(checkpoint_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint BLOB,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            );
            CREATE TABLE IF NOT EXISTS writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT,
                value BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            );
            """
        )
        conn.execute("INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id) VALUES ('keep', '', 'c1')")
        conn.execute(
            "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel) VALUES ('keep', '', 'c1', 't1', 0, 'x')"
        )

    listed = app.list_sessions()
    assert listed["ok"] is True
    assert any(item["id"] == "keep" for item in listed["sessions"])

    deleted = app.delete_session({"sessionId": "keep"})

    assert deleted["ok"] is True
    assert not any(item["id"] == "keep" for item in deleted["sessions"])
    with sqlite3.connect(app.store.path) as conn:
        message_count = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = 'keep'").fetchone()[0]
        events = [
            row[0]
            for row in conn.execute(
                "SELECT event_type FROM session_events WHERE session_id = 'keep' ORDER BY id"
            ).fetchall()
        ]
    with sqlite3.connect(checkpoint_path) as conn:
        checkpoint_count = conn.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'keep'").fetchone()[0]
        write_count = conn.execute("SELECT COUNT(*) FROM writes WHERE thread_id = 'keep'").fetchone()[0]
    assert message_count == 0
    assert checkpoint_count == 0
    assert write_count == 0
    assert "delete" in events


def test_web_session_rename_persists_metadata(tmp_path: Path) -> None:
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("rename-me")
    session.messages = []
    app._save_history(session)

    renamed = app.rename_session({"sessionId": "rename-me", "title": "自定义名称"})

    assert renamed["ok"] is True
    assert any(
        item["id"] == "rename-me" and item["title"] == "自定义名称"
        for item in renamed["sessions"]
    )

    reloaded = WebApp(tmp_path, tmp_path)
    listed = reloaded.list_sessions()
    viewed = reloaded.session_view("rename-me")

    assert any(
        item["id"] == "rename-me" and item["title"] == "自定义名称"
        for item in listed["sessions"]
    )
    assert viewed["title"] == "自定义名称"
    with sqlite3.connect(reloaded.store.path) as conn:
        event = conn.execute(
            "SELECT event_type FROM session_events WHERE session_id = 'rename-me' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert event[0] == "rename"


def test_web_session_clear_keeps_session_and_removes_interactions(tmp_path: Path) -> None:
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("clear-me")
    session.messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    session.display_messages = list(session.messages)
    session.todos = [{"id": "t1", "content": "旧任务", "status": "completed"}]
    app._save_history(session)
    app.rename_session({"sessionId": "clear-me", "title": "保留名称"})
    checkpoint_path = tmp_path / ".langcode" / "checkpoints.sqlite"
    with sqlite3.connect(checkpoint_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                type TEXT,
                checkpoint BLOB,
                metadata BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
            );
            CREATE TABLE IF NOT EXISTS writes (
                thread_id TEXT NOT NULL,
                checkpoint_ns TEXT NOT NULL DEFAULT '',
                checkpoint_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                type TEXT,
                value BLOB,
                PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
            );
            """
        )
        conn.execute("INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id) VALUES ('clear-me', '', 'c1')")
        conn.execute(
            "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel) VALUES ('clear-me', '', 'c1', 't1', 0, 'x')"
        )

    cleared = app.clear_session({"sessionId": "clear-me"})

    assert cleared["ok"] is True
    assert any(item["id"] == "clear-me" and item["title"] == "保留名称" for item in cleared["sessions"])
    viewed = app.session_view("clear-me")
    assert viewed["messages"] == []
    assert viewed["todos"] == []
    assert viewed["title"] == "保留名称"
    with sqlite3.connect(app.store.path) as conn:
        message_count = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = 'clear-me'").fetchone()[0]
        state_json = conn.execute("SELECT state_json FROM sessions WHERE id = 'clear-me'").fetchone()[0]
        event = conn.execute(
            "SELECT event_type FROM session_events WHERE session_id = 'clear-me' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    with sqlite3.connect(checkpoint_path) as conn:
        checkpoint_count = conn.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'clear-me'").fetchone()[0]
        write_count = conn.execute("SELECT COUNT(*) FROM writes WHERE thread_id = 'clear-me'").fetchone()[0]
    assert message_count == 0
    assert state_json is None
    assert checkpoint_count == 0
    assert write_count == 0
    assert event[0] == "clear"


def test_web_session_history_uses_sqlite_not_json_files(tmp_path: Path) -> None:
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("sqlite-only")
    session.messages = [HumanMessage(content="hi"), AIMessage(content="hello")]

    app._save_history(session)

    assert (tmp_path / ".langcode" / "web.sqlite").exists()
    assert not (tmp_path / ".langcode" / "web-sessions" / "sqlite-only.json").exists()
    viewed = app.session_view("sqlite-only")
    assert viewed["messages"] == [
        {"role": "user", "kind": "message", "content": "hi"},
        {"role": "assistant", "kind": "message", "content": "hello"},
    ]


def test_web_session_store_migrates_legacy_json_sessions(tmp_path: Path) -> None:
    sessions_dir = tmp_path / ".langcode" / "web-sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "legacy.json").write_text(
        json.dumps(
            [
                {"role": "human", "content": "old question"},
                {"role": "ai", "content": "old answer"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / ".langcode" / "web-session-metadata.json").write_text(
        json.dumps({"legacy": {"title": "旧会话"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    app = WebApp(tmp_path, tmp_path)

    listed = app.list_sessions()
    viewed = app.session_view("legacy")
    assert any(item["id"] == "legacy" and item["title"] == "旧会话" for item in listed["sessions"])
    assert viewed["messages"] == [
        {"role": "user", "kind": "message", "content": "old question"},
        {"role": "assistant", "kind": "message", "content": "old answer"},
    ]


def test_web_session_store_handles_concurrent_writes(tmp_path: Path) -> None:
    app = WebApp(tmp_path, tmp_path)

    def write_session(index: int) -> str:
        session = app.get_session(f"parallel-{index}")
        session.messages = [HumanMessage(content=f"hi {index}"), AIMessage(content=f"hello {index}")]
        app._save_history(session)
        app.rename_session({"sessionId": session.id, "title": f"并发 {index}"})
        return session.id

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        session_ids = list(executor.map(write_session, range(12)))

    listed = app.list_sessions()
    listed_ids = {item["id"] for item in listed["sessions"]}
    assert set(session_ids).issubset(listed_ids)
    with sqlite3.connect(app.store.path) as conn:
        message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        rename_count = conn.execute(
            "SELECT COUNT(*) FROM session_events WHERE event_type = 'rename'"
        ).fetchone()[0]
    assert message_count == 24
    assert rename_count == 12


def test_web_session_store_round_trips_ai_tool_calls(tmp_path: Path) -> None:
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("tool-history")
    session.messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"path": "README.md"},
                    "id": "call-1",
                }
            ],
        ),
        ToolMessage(content='{"ok": true}', tool_call_id="call-1"),
    ]

    app._save_history(session)

    reloaded = WebApp(tmp_path, tmp_path)
    messages = reloaded.get_session("tool-history").messages
    assert len(messages) == 2
    assert messages[0].tool_calls[0]["id"] == "call-1"
    assert messages[1].tool_call_id == "call-1"


def test_web_tool_result_event_normalizes_bytes() -> None:
    event = _tool_result_event("shell", {"ok": False, "stdout": b"partial", "stderr": b"late"})

    assert event is not None
    assert event["role"] == "tool"
    assert json.loads(event["content"]) == {"ok": False, "stdout": "partial", "stderr": "late"}


def test_web_repair_tool_history_drops_unanswered_tool_calls() -> None:
    messages = [
        HumanMessage(content="first"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "shell",
                    "args": {"command": "echo hi"},
                    "id": "missing-call",
                }
            ],
        ),
        HumanMessage(content="next"),
    ]

    repaired = _repair_tool_history(messages)

    assert [message.type for message in repaired] == ["human", "human"]


def test_web_repair_tool_history_preserves_complete_tool_call_pairs() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "shell",
                    "args": {"command": "echo hi"},
                    "id": "call-1",
                }
            ],
        ),
        ToolMessage(content='{"ok": true}', tool_call_id="call-1"),
    ]

    repaired = _repair_tool_history(messages)

    assert len(repaired) == 2
    assert isinstance(repaired[0], AIMessage)
    assert isinstance(repaired[1], ToolMessage)


def test_web_session_view_maps_history_messages(tmp_path: Path) -> None:
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("view")

    session.messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    app._save_history(session)

    response = app.session_view("view")

    assert response["ok"] is True
    assert response["messages"] == [
        {"role": "user", "kind": "message", "content": "hi"},
        {"role": "assistant", "kind": "message", "content": "hello"},
    ]


def test_web_session_view_hides_successful_tool_results(tmp_path: Path) -> None:
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("view-tool-success")

    session.messages = [
        HumanMessage(content="run ls"),
        ToolMessage(
            content=json.dumps(
                {
                    "ok": True,
                    "result": {
                        "exit_code": 0,
                        "stdout": "backend details",
                        "stderr": "",
                        "timed_out": False,
                    },
                }
            ),
            tool_call_id="shell-call",
        ),
        ToolMessage(content=json.dumps({"ok": False, "error": "failed"}), tool_call_id="failed-call"),
        AIMessage(content="done"),
    ]
    app._save_history(session)

    response = app.session_view("view-tool-success")

    assert response["messages"] == [
        {"role": "user", "kind": "message", "content": "run ls"},
        {"role": "tool", "kind": "tool_result", "content": '{"ok": false, "error": "failed"}'},
        {"role": "assistant", "kind": "message", "content": "done"},
    ]


def test_web_session_view_hides_empty_tool_call_assistant_messages(tmp_path: Path) -> None:
    app = WebApp(tmp_path, tmp_path)
    session = app.get_session("view-empty-tool-call")

    session.messages = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"path": "created.txt", "content": "approved"},
                    "id": "call-1",
                }
            ],
        ),
        AIMessage(content="done"),
    ]
    app._save_history(session)

    response = app.session_view("view-empty-tool-call")

    assert response["messages"] == [
        {"role": "user", "kind": "message", "content": "hi"},
        {"role": "assistant", "kind": "message", "content": "done"},
    ]


def test_web_native_directory_picker_can_be_cancelled(tmp_path: Path, monkeypatch) -> None:
    app = WebApp(tmp_path, tmp_path)
    monkeypatch.setattr("langcode_agent.interfaces.web.platform.system", lambda: "Darwin")

    class Completed:
        returncode = 0
        stdout = "\n"
        stderr = ""

    monkeypatch.setattr("langcode_agent.interfaces.web.subprocess.run", lambda *_args, **_kwargs: Completed())

    response = app.choose_directory({"start": str(tmp_path)})

    assert response == {"ok": True, "cancelled": True, "path": None}


def test_web_native_directory_picker_returns_selected_directory(tmp_path: Path, monkeypatch) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    app = WebApp(tmp_path, tmp_path)
    monkeypatch.setattr("langcode_agent.interfaces.web.platform.system", lambda: "Darwin")
    captured = {}

    class Completed:
        returncode = 0
        stdout = f"{selected}\n"
        stderr = ""

    def fake_run(args, **_kwargs):
        captured["args"] = args
        return Completed()

    monkeypatch.setattr("langcode_agent.interfaces.web.subprocess.run", fake_run)

    response = app.choose_directory({"start": str(tmp_path), "prompt": "选择工作目录"})

    assert response == {"ok": True, "cancelled": False, "path": str(selected)}
    assert captured["args"][-2:] == [str(tmp_path), "选择工作目录"]
    assert "item 2 of argv" in captured["args"][2]
