from pathlib import Path
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langcode_agent.core.context_management import (
    CONTEXT_SUMMARY_ACK,
    CONTEXT_SUMMARY_PREFIX,
    compact_history_if_needed,
    count_messages_tokens,
)

from langcode_agent.runtime.agent import CodeAgent
from langcode_agent.runtime.chat import (
    TOOL_ROUND_LIMIT_MOCK_USER,
    ChatSession,
    default_system_prompt,
    messages_from_json,
    model_settings_from_env,
    tool_schemas,
)
from langcode_agent.runtime.delegation import (
    TOOL_ROUND_LIMIT_MOCK_USER as DELEGATE_ROUND_LIMIT_USER,
    _delegate_system_prompt,
    run_delegate_agent,
)
from langcode_agent.runtime.multi_agent import run_agent_debate, run_parallel_delegate_agents
from langcode_agent.storage.session_store import SessionStore
from langcode_agent.memory.project import compact_messages, serialize_message


class FakeToolCallingModel:
    def __init__(self) -> None:
        self.calls = 0
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"path": "README.md", "content": "from chat"},
                        "id": "call-1",
                    }
                ],
            )
        return AIMessage(content="done")


def test_tool_schemas_include_core_code_tools() -> None:
    names = {schema["function"]["name"] for schema in tool_schemas()}

    assert {
        "read_file",
        "ls",
        "glob",
        "search",
        "web_search",
        "web_fetch",
        "memory",
        "soul",
        "self_evolve",
        "cron",
        "session_search",
        "skill",
        "diagram",
        "write_file",
        "edit_file",
        "shell",
        "sandbox_shell",
        "task_create",
        "task_update",
        "task_list",
        "task_get",
        "task_cancel",
        "delegate_agent",
        "delegate_agents",
        "agent_debate",
    } <= names
    assert "write_todos" not in names


def test_model_facing_prompts_and_tool_descriptions_are_chinese(tmp_path: Path) -> None:
    english_markers = [
        "You are",
        "Use ",
        "Read ",
        "Write ",
        "Search ",
        "Run ",
        "Fetch ",
        "Create ",
        "Update ",
        "Return ",
        "Task:",
        "write_todos",
        "workspace",
        "pending",
        "in_progress",
        "completed",
    ]

    model_facing_text = [
        default_system_prompt(str(tmp_path)),
        _delegate_system_prompt(tmp_path, "researcher"),
        *[schema["function"]["description"] for schema in tool_schemas()],
    ]

    for text in model_facing_text:
        assert any("\u4e00" <= char <= "\u9fff" for char in text), text
        for marker in english_markers:
            assert marker not in text


def test_default_prompt_requires_valid_markdown_tables(tmp_path: Path) -> None:
    prompt = default_system_prompt(str(tmp_path))

    assert "合法 GFM 表格" in prompt
    assert "表头、分隔行和每一行都必须各占一行" in prompt
    assert "分隔行列数必须与表头一致" in prompt
    assert "不要把多行表格压缩到同一行" in prompt


def test_chat_session_executes_model_tool_call_with_approval(tmp_path: Path) -> None:
    model = FakeToolCallingModel()
    approvals = []

    def approve(payload: dict) -> dict:
        approvals.append(payload)
        return {"type": "accept"}

    session = ChatSession(
        agent=CodeAgent(tmp_path),
        model=model,
        approval_callback=approve,
        thread_id="chat-test",
    )

    response = session.send("write a readme")

    assert response == "done"
    assert model.bound_tools
    assert approvals[0]["tool_name"] == "write_file"
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "from chat"


def test_aimp_glm_is_default_openai_compatible_provider(monkeypatch) -> None:
    monkeypatch.delenv("LANGCODE_PROVIDER", raising=False)
    monkeypatch.delenv("LANGCODE_OPENAI_GATEWAY", raising=False)
    monkeypatch.delenv("LANGCODE_MODEL", raising=False)
    monkeypatch.delenv("AIMP_GLM_BASE_URL", raising=False)
    monkeypatch.setenv("AIMP_GLM_API_KEY", "glm-key")
    monkeypatch.setenv("AIMP_GLM_USER", "user-glm")
    monkeypatch.setenv("LANGCODE_THINKING", "true")

    settings = model_settings_from_env()

    assert settings.provider == "openai"
    assert settings.base_url == "https://aimpapi.midea.com/t-aigc/aimp-glm/v1"
    assert settings.model == "glm-5"
    assert settings.api_key == "glm-key"
    assert settings.default_headers == {"AIGC-USER": "user-glm"}
    assert settings.extra_body == {
        "chat_template_kwargs": {"thinking": True},
        "separate_reasoning": True,
    }


def test_openai_provider_can_still_be_selected(monkeypatch) -> None:
    monkeypatch.setenv("LANGCODE_PROVIDER", "openai")
    monkeypatch.setenv("LANGCODE_OPENAI_GATEWAY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LANGCODE_MODEL", "custom-model")

    settings = model_settings_from_env()

    assert settings.provider == "openai"
    assert settings.base_url == "https://example.test/v1"
    assert settings.model == "custom-model"
    assert settings.api_key == "openai-key"


def test_aimp_gpt4o_provider_adds_gateway_headers(monkeypatch) -> None:
    monkeypatch.setenv("LANGCODE_PROVIDER", "openai")
    monkeypatch.setenv("LANGCODE_OPENAI_GATEWAY", "aimp")
    monkeypatch.delenv("LANGCODE_MODEL", raising=False)
    monkeypatch.delenv("AIMP_GPT4O_BASE_URL", raising=False)
    monkeypatch.setenv("AIMP_GPT4O_API_KEY", "aimp-key")
    monkeypatch.setenv("AIMP_GPT4O_USER", "user-1")

    settings = model_settings_from_env()

    assert settings.provider == "openai"
    assert settings.base_url == "https://aimpapi.midea.com/t-aigc/mip-chat-app/openai/standard/v1"
    assert settings.model == "gpt-4o"
    assert settings.api_key == "aimp-key"
    assert settings.default_headers == {
        "Aimp-Biz-Id": "gpt-4o",
        "AIGC-USER": "user-1",
    }


def test_aimp_base_url_accepts_full_chat_completions_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("LANGCODE_PROVIDER", "openai")
    monkeypatch.setenv("LANGCODE_OPENAI_GATEWAY", "aimp")
    monkeypatch.setenv("LANGCODE_MODEL", "gpt-4o")
    monkeypatch.setenv(
        "AIMP_GPT4O_BASE_URL",
        "https://aimpapi.midea.com/t-aigc/mip-chat-app/openai/standard/v1/chat/completions",
    )
    monkeypatch.setenv("AIMP_GPT4O_API_KEY", "aimp-key")

    settings = model_settings_from_env()

    assert settings.provider == "openai"
    assert settings.base_url == "https://aimpapi.midea.com/t-aigc/mip-chat-app/openai/standard/v1"


def test_aimp_deepseek_v4_provider_adds_headers_and_thinking_body(monkeypatch) -> None:
    monkeypatch.setenv("LANGCODE_PROVIDER", "openai")
    monkeypatch.setenv("LANGCODE_OPENAI_GATEWAY", "aimp-deepseek-v4-pro")
    monkeypatch.delenv("LANGCODE_MODEL", raising=False)
    monkeypatch.delenv("AIMP_DEEPSEEK_V4_BASE_URL", raising=False)
    monkeypatch.setenv("AIMP_DEEPSEEK_V4_API_KEY", "deepseek-key")
    monkeypatch.setenv("AIMP_DEEPSEEK_V4_USER", "user-2")
    monkeypatch.setenv("LANGCODE_THINKING", "true")

    settings = model_settings_from_env()

    assert settings.provider == "openai"
    assert settings.base_url == "https://aimpapi.midea.com/t-aigc/aimp-deepseek-v4-pro/v1"
    assert settings.model == "deepseek-v4-pro"
    assert settings.api_key == "deepseek-key"
    assert settings.default_headers == {"AIGC-USER": "user-2"}
    assert settings.extra_body == {"chat_template_kwargs": {"thinking": True}}


def test_aimp_deepseek_v4_base_url_accepts_full_chat_completions_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("LANGCODE_PROVIDER", "openai")
    monkeypatch.setenv("LANGCODE_OPENAI_GATEWAY", "aimp-deepseek-v4-pro")
    monkeypatch.setenv("LANGCODE_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv(
        "AIMP_DEEPSEEK_V4_BASE_URL",
        "https://aimpapi.midea.com/t-aigc/aimp-deepseek-v4-pro/v1/chat/completions",
    )
    monkeypatch.setenv("AIMP_DEEPSEEK_V4_API_KEY", "deepseek-key")

    settings = model_settings_from_env()

    assert settings.base_url == "https://aimpapi.midea.com/t-aigc/aimp-deepseek-v4-pro/v1"


def test_aimp_glm_base_url_accepts_full_chat_completions_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("LANGCODE_PROVIDER", "openai")
    monkeypatch.setenv("LANGCODE_OPENAI_GATEWAY", "aimp-glm")
    monkeypatch.setenv("LANGCODE_MODEL", "glm-5")
    monkeypatch.setenv(
        "AIMP_GLM_BASE_URL",
        "https://aimpapi.midea.com/t-aigc/aimp-glm/v1/chat/completions",
    )
    monkeypatch.setenv("AIMP_GLM_API_KEY", "glm-key")
    monkeypatch.setenv("LANGCODE_THINKING", "false")

    settings = model_settings_from_env()

    assert settings.provider == "openai"
    assert settings.base_url == "https://aimpapi.midea.com/t-aigc/aimp-glm/v1"
    assert settings.model == "glm-5"
    assert settings.extra_body == {
        "chat_template_kwargs": {"thinking": False},
        "separate_reasoning": True,
    }


def test_messages_from_json_preserves_ai_tool_calls() -> None:
    messages = messages_from_json(
        [
            {
                "role": "ai",
                "content": "",
                "tool_calls": [
                    {
                        "name": "read_file",
                        "args": {"path": "README.md"},
                        "id": "call-1",
                    }
                ],
            },
            {"role": "tool", "content": '{"ok": true}', "tool_call_id": "call-1"},
        ]
    )

    assert len(messages) == 2
    assert isinstance(messages[0], AIMessage)
    assert messages[0].tool_calls[0]["id"] == "call-1"
    assert isinstance(messages[1], ToolMessage)
    assert messages[1].tool_call_id == "call-1"


def test_messages_from_json_drops_orphan_tool_messages_for_strict_openai() -> None:
    messages = messages_from_json(
        [
            {"role": "ai", "content": "done"},
            {"role": "tool", "content": '{"ok": true}', "tool_call_id": "missing"},
            {"role": "human", "content": "next"},
        ]
    )

    assert [message.type for message in messages] == ["ai", "human"]


def test_chat_session_local_compact_command_replaces_old_context(tmp_path: Path) -> None:
    session = ChatSession(
        agent=CodeAgent(tmp_path),
        model=FakeToolCallingModel(),
        approval_callback=lambda _payload: {"type": "accept"},
        thread_id="compact",
    )
    for index in range(12):
        session.messages.append(AIMessage(content=f"old message {index}"))

    response = session.send("/compact 保留关键决策")

    assert "已压缩" in response
    assert len(session.messages) < 16
    assert any("LangCode context compact summary" in str(message.content) for message in session.messages)


def test_chat_session_hash_command_writes_project_memory(tmp_path: Path) -> None:
    session = ChatSession(
        agent=CodeAgent(tmp_path),
        model=FakeToolCallingModel(),
        approval_callback=lambda _payload: {"type": "accept"},
        thread_id="memory",
    )

    response = session.send("# 以后默认先读 AGENT_MEMORY.md")

    assert "已写入项目记忆" in response
    assert "以后默认先读" in (tmp_path / ".langcode" / "memories" / "MEMORY.md").read_text(encoding="utf-8")


def test_load_project_context_creates_empty_hermes_memory_files(tmp_path: Path) -> None:
    from langcode_agent.memory.project import load_project_context

    context = load_project_context(tmp_path)

    assert "SOUL.md" in context
    assert "LangCode" in context
    assert (tmp_path / ".langcode" / "memories" / "MEMORY.md").is_file()
    assert (tmp_path / ".langcode" / "memories" / "USER.md").is_file()
    assert (tmp_path / ".langcode" / "SOUL.md").is_file()


def test_compact_messages_drops_orphan_tool_messages_and_preserves_tool_calls() -> None:
    messages = [
        AIMessage(content=f"old {index}") for index in range(8)
    ] + [
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "README.md"}, "id": "call-1"}],
        ),
        ToolMessage(content='{"ok": true}', tool_call_id="call-1"),
        AIMessage(content="after tool"),
    ]

    compacted, _summary = compact_messages(messages, keep_recent=2)

    assert all(not isinstance(message, ToolMessage) for message in compacted)
    serialized = serialize_message(messages[-3])
    assert serialized["tool_calls"][0]["id"] == "call-1"


class FakeDelegateModel:
    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, _messages):
        return AIMessage(content="子 Agent 结论")


class FakeDebateModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        return AIMessage(content=f"发言 {self.calls}")


def test_delegate_agent_uses_independent_read_only_model(tmp_path: Path) -> None:
    model = FakeDelegateModel()

    result = run_delegate_agent(
        tmp_path,
        role="researcher",
        task="总结项目",
        model=model,
    )

    assert result["ok"] is True
    assert result["summary"] == "子 Agent 结论"
    assert {schema["function"]["name"] for schema in model.tools} == {
        "read_file",
        "ls",
        "glob",
        "search",
        "web_search",
        "web_fetch",
    }


def test_delegate_agent_defaults_to_researcher_when_role_omitted(tmp_path: Path) -> None:
    result = run_delegate_agent(
        tmp_path,
        task="总结项目",
        model=FakeDelegateModel(),
    )

    assert result["ok"] is True
    assert result["role"] == "researcher"


class BadDelegateModel:
    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, _messages):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"path": "owned.txt", "content": "bad"},
                    "id": "bad-call",
                }
            ],
        )


def test_delegate_agent_rejects_mutating_tool_calls(tmp_path: Path) -> None:
    result = run_delegate_agent(
        tmp_path,
        role="researcher",
        task="try mutation",
        model=BadDelegateModel(),
    )

    assert result["ok"] is False
    assert "只读" in result["last_tool_result"]["error"]


class NetworkSandboxDelegateModel:
    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, _messages):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "sandbox_shell",
                    "args": {"command": "curl https://example.com"},
                    "id": "sandbox-network",
                }
            ],
        )


def test_delegate_agent_respects_sandbox_shell_permission_policy(tmp_path: Path) -> None:
    result = run_delegate_agent(
        tmp_path,
        role="verifier",
        task="try network",
        model=NetworkSandboxDelegateModel(),
    )

    assert result["ok"] is False
    assert "权限=ask" in result["last_tool_result"]["error"]
    assert not (tmp_path / "owned.txt").exists()


def test_parallel_delegate_agents_persists_dialogue(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / ".langcode" / "web.sqlite")
    store.ensure_session("session-1", str(tmp_path))

    def fake_runner(_workspace_root, *, role, task, context=""):
        return {"ok": True, "role": role, "summary": f"{role}:{task}:{context}"}

    result = run_parallel_delegate_agents(
        tmp_path,
        agents=[
            {"id": "a", "name": "甲", "role": "researcher", "task": "查事实"},
            {"id": "b", "name": "乙", "role": "reviewer", "task": "审风险"},
        ],
        _current_session_id="session-1",
        _session_store_path=str(store.path),
        delegate_runner=fake_runner,
    )

    assert result["ok"] is True
    assert result["kind"] == "agent_dialogue"
    assert [message["agent_name"] for message in result["messages"]] == ["甲", "乙"]
    saved = store.load_agent_thread(result["thread_id"])
    assert saved is not None
    assert saved["messages"][0]["content"].startswith("researcher:查事实")


def test_parallel_delegate_agents_isolates_one_agent_failure(tmp_path: Path) -> None:
    def fake_runner(_workspace_root, *, role, task, context=""):
        if task == "fail":
            raise RuntimeError("boom")
        return {"ok": True, "role": role, "summary": task}

    result = run_parallel_delegate_agents(
        tmp_path,
        agents=[
            {"id": "a", "name": "甲", "task": "pass"},
            {"id": "b", "name": "乙", "task": "fail"},
        ],
        delegate_runner=fake_runner,
    )

    assert result["results"][0]["ok"] is True
    assert result["results"][1] == {"ok": False, "error": "RuntimeError: boom"}


def test_agent_debate_persists_and_continues_transcript(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / ".langcode" / "web.sqlite")
    store.ensure_session("session-1", str(tmp_path))
    model = FakeDebateModel()

    first = run_agent_debate(
        tmp_path,
        topic="先有鸡还是先有蛋",
        agents=[
            {"id": "a", "name": "先有蛋", "stance": "先有蛋"},
            {"id": "b", "name": "先有鸡", "stance": "先有鸡"},
        ],
        rounds=1,
        debate_id="debate-1",
        _current_session_id="session-1",
        _session_store_path=str(store.path),
        model=model,
    )
    second = run_agent_debate(
        tmp_path,
        topic="先有鸡还是先有蛋",
        agents=[
            {"id": "a", "name": "先有蛋", "stance": "先有蛋"},
            {"id": "b", "name": "先有鸡", "stance": "先有鸡"},
        ],
        rounds=1,
        debate_id="debate-1",
        _current_session_id="session-1",
        _session_store_path=str(store.path),
        model=model,
    )

    assert first["messages"][0]["round"] == 1
    assert second["messages"][0]["round"] == 1
    assert second["messages"][-1]["round"] == 4
    saved = store.load_agent_thread("debate-1")
    assert saved is not None
    assert len(saved["messages"]) == 6


class LoopingDelegateModel:
    """A sub-agent model that keeps asking for tools until told to stop."""

    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls += 1
        if any(getattr(message, "content", "") == DELEGATE_ROUND_LIMIT_USER for message in messages):
            return AIMessage(content="基于现有信息的最终结论")
        return AIMessage(
            content="",
            tool_calls=[{"name": "ls", "args": {"path": "."}, "id": f"loop-{self.calls}"}],
        )


def test_delegate_agent_stops_at_explicit_round_limit(tmp_path: Path) -> None:
    model = LoopingDelegateModel()

    result = run_delegate_agent(tmp_path, task="无限循环", max_rounds=3, model=model)

    assert result["ok"] is True
    assert result["round_limit_reached"] is True
    assert result["max_rounds"] == 3
    assert result["summary"] == "基于现有信息的最终结论"
    assert model.calls == 4


def test_delegate_agent_round_limit_defaults_to_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LANGCODE_SUBAGENT_MAX_ROUNDS", "2")
    model = LoopingDelegateModel()

    result = run_delegate_agent(tmp_path, task="无限循环", model=model)

    assert result["round_limit_reached"] is True
    assert model.calls == 3


def test_delegate_agent_feeds_tool_failure_back_instead_of_aborting(tmp_path: Path) -> None:
    class FailingThenAnsweringModel:
        def __init__(self) -> None:
            self.calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "read_file", "args": {"path": "missing.txt"}, "id": "bad-1"}],
                )
            return AIMessage(content="工具失败后我继续作答")

    model = FailingThenAnsweringModel()
    result = run_delegate_agent(tmp_path, task="读不存在的文件", model=model)

    assert result["ok"] is True
    assert result["summary"] == "工具失败后我继续作答"
    assert model.calls == 2


def test_parallel_delegate_agents_forwards_max_rounds(tmp_path: Path) -> None:
    seen: list[int | None] = []

    def fake_runner(_workspace_root, *, role, task, context="", max_rounds=None):
        seen.append(max_rounds)
        return {"ok": True, "role": role, "summary": task}

    run_parallel_delegate_agents(
        tmp_path,
        agents=[{"id": "a", "name": "甲", "task": "查事实"}],
        max_rounds=5,
        delegate_runner=fake_runner,
    )

    assert seen == [5]


class AlwaysToolCallingModel:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.calls += 1
        if any(getattr(message, "content", "") == TOOL_ROUND_LIMIT_MOCK_USER for message in messages):
            return AIMessage(content="到达上限后的最终回答")
        return AIMessage(
            content="",
            tool_calls=[{"name": "ls", "args": {"path": "."}, "id": f"chat-loop-{self.calls}"}],
        )


def test_chat_session_stops_at_tool_round_limit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LANGCODE_MAX_TOOL_ROUNDS", "4")
    model = AlwaysToolCallingModel()
    session = ChatSession(
        agent=CodeAgent(tmp_path),
        model=model,
        approval_callback=lambda _payload: {"type": "accept"},
        thread_id="round-limit",
    )

    response = session.send("一直调用工具")

    assert response == "到达上限后的最终回答"
    assert model.calls == 5
    assert any(
        getattr(message, "content", "") == TOOL_ROUND_LIMIT_MOCK_USER for message in session.messages
    )


class FakeSummaryModel:
    model_name = "gpt-4o-mini"

    def __init__(self, summary: str = "摘要：讨论了 A 与 B，待办是 C。") -> None:
        self.calls = 0
        self.summary = summary

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.summary)


def _long_history(turns: int = 12) -> list:
    filler = "细节内容 " * 40
    history: list = [SystemMessage(content="你是一个谨慎的代码 Agent")]
    for index in range(turns):
        history.append(HumanMessage(content=f"问题 {index} {filler}"))
        history.append(
            AIMessage(content="", tool_calls=[{"name": "ls", "args": {"path": "."}, "id": f"t{index}"}])
        )
        history.append(ToolMessage(content=f"结果 {index} {filler}", tool_call_id=f"t{index}"))
        history.append(AIMessage(content=f"回答 {index} {filler}"))
    return history


def test_count_messages_tokens_grows_with_history() -> None:
    history = _long_history(4)

    assert count_messages_tokens(history[:5], "gpt-4o-mini") < count_messages_tokens(history, "gpt-4o-mini")
    assert count_messages_tokens([], "gpt-4o-mini") == 0


def test_compact_history_leaves_short_history_untouched() -> None:
    history = _long_history(1)
    model = FakeSummaryModel()

    result, compacted = compact_history_if_needed(history, model=model, max_tokens=100000)

    assert compacted is False
    assert result == history
    assert model.calls == 0


def test_compact_history_summarizes_older_segment() -> None:
    history = _long_history()
    model = FakeSummaryModel()

    result, compacted = compact_history_if_needed(
        history, model=model, max_tokens=2000, keep_recent_tokens=600
    )

    assert compacted is True
    assert model.calls == 1
    assert isinstance(result[0], SystemMessage)
    assert result[1].content.startswith(CONTEXT_SUMMARY_PREFIX)
    assert "摘要：讨论了 A 与 B" in result[1].content
    assert result[2].content == CONTEXT_SUMMARY_ACK
    assert len(result) < len(history)
    assert count_messages_tokens(result, "gpt-4o-mini") <= 2000


def test_compact_history_keeps_tool_messages_paired() -> None:
    history = _long_history()

    result, _ = compact_history_if_needed(
        history, model=FakeSummaryModel(), max_tokens=2000, keep_recent_tokens=600
    )

    for index, message in enumerate(result):
        if isinstance(message, ToolMessage):
            previous = result[index - 1]
            assert isinstance(previous, AIMessage)
            assert previous.tool_calls


def test_compact_history_keeps_full_history_when_summary_raises() -> None:
    class BrokenSummaryModel:
        model_name = "gpt-4o-mini"

        def invoke(self, _messages):
            raise RuntimeError("summary backend down")

    history = _long_history()

    result, compacted = compact_history_if_needed(
        history, model=BrokenSummaryModel(), max_tokens=2000, keep_recent_tokens=600
    )

    # A failed summary must never silently delete the older segment.
    assert compacted is False
    assert result == history


def test_compact_history_keeps_full_history_when_summary_is_empty() -> None:
    class EmptySummaryModel:
        model_name = "gpt-4o-mini"

        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            return AIMessage(content="   \n  ")

    history = _long_history()
    model = EmptySummaryModel()

    result, compacted = compact_history_if_needed(
        history, model=model, max_tokens=2000, keep_recent_tokens=600
    )

    assert model.calls == 1
    assert compacted is False
    assert result == history


def test_compact_history_summarizes_with_the_unbound_model() -> None:
    class Binding:
        """Stands in for the RunnableBinding that ``bind_tools`` returns."""

        def __init__(self, bound) -> None:
            self.bound = bound

        def invoke(self, _messages):
            # A tool-bound model answers with a tool call, not prose.
            return AIMessage(content="", tool_calls=[{"name": "ls", "args": {}, "id": "x"}])

    inner = FakeSummaryModel()
    inner.model_name = "gpt-4o-mini"

    result, compacted = compact_history_if_needed(
        _long_history(), model=Binding(inner), max_tokens=2000, keep_recent_tokens=600
    )

    assert compacted is True
    assert inner.calls == 1
    assert result[1].content.startswith(CONTEXT_SUMMARY_PREFIX)


def test_large_inline_image_parts_are_counted_towards_the_budget() -> None:
    blob = "A" * 400_000
    message = HumanMessage(
        content=[
            {"type": "text", "text": "看这张图"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{blob}"}},
        ]
    )

    tokens = count_messages_tokens([message], "gpt-4o-mini")

    # ~len/4 for the blob; before the fix the image part cost ~0 tokens.
    assert tokens > 90_000

    history = [SystemMessage(content="系统"), HumanMessage(content="问题"), message, HumanMessage(content="继续")]
    _, compacted = compact_history_if_needed(
        history, model=FakeSummaryModel(), max_tokens=2000, keep_recent_tokens=600
    )

    assert compacted is True


def test_token_counts_are_consistent_between_window_split_and_budget() -> None:
    from langcode_agent.core.context_management import _TokenCounter, _recent_window_start

    body = _long_history(6)[1:]
    counter = _TokenCounter("gpt-4o-mini")
    split = _recent_window_start(body, 600, counter)

    assert count_messages_tokens(body[split:], "gpt-4o-mini") == counter.total(body[split:])
    # Tool-calling AIMessages carry tool_calls tokens the old split ignored.
    assert counter.total(body) > sum(
        _TokenCounter("gpt-4o-mini").message_tokens(message) for message in body if not getattr(message, "tool_calls", None)
    )


def test_compact_history_handles_list_content_parts() -> None:
    history = [SystemMessage(content="系统")]
    for index in range(12):
        history.append(HumanMessage(content=[{"type": "text", "text": "问题 " * 200}]))
        history.append(AIMessage(content=[{"type": "text", "text": "回答 " * 200}]))

    result, compacted = compact_history_if_needed(
        history, model=FakeSummaryModel(), max_tokens=2000, keep_recent_tokens=600
    )

    assert compacted is True
    assert count_messages_tokens(result, "gpt-4o-mini") <= 2000


def test_compact_history_reads_budgets_from_env(monkeypatch) -> None:
    monkeypatch.setenv("LANGCODE_CONTEXT_MAX_TOKENS", "2000")
    monkeypatch.setenv("LANGCODE_CONTEXT_KEEP_RECENT_TOKENS", "600")
    model = FakeSummaryModel()

    _, compacted = compact_history_if_needed(_long_history(), model=model)

    assert compacted is True
    assert model.calls == 1


def test_load_project_context_is_cached_until_a_file_changes(tmp_path: Path, monkeypatch) -> None:
    from langcode_agent.memory import project as project_module

    project_module.reset_project_context_cache()
    (tmp_path / "CLAUDE.md").write_text("项目说明 v1", encoding="utf-8")
    first = project_module.load_project_context(tmp_path)
    assert "项目说明 v1" in first

    reads = {"count": 0}
    original_read_text = Path.read_text

    def counting_read_text(self, *args, **kwargs):
        reads["count"] += 1
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    assert project_module.load_project_context(tmp_path) == first
    assert reads["count"] == 0

    time.sleep(0.01)
    (tmp_path / "CLAUDE.md").write_text("项目说明 v2", encoding="utf-8")
    refreshed = project_module.load_project_context(tmp_path)

    assert reads["count"] > 0
    assert "项目说明 v2" in refreshed


def test_skill_catalog_cache_invalidates_when_a_skill_is_added(tmp_path: Path) -> None:
    from langcode_agent.memory import project as project_module

    project_module.reset_project_context_cache()
    skills_dir = tmp_path / ".langcode" / "skills" / "deploy"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("---\nname: deploy\ndescription: 发布流程\n---\n步骤", encoding="utf-8")

    assert [item["name"] for item in project_module.load_skill_catalog(tmp_path)] == ["deploy"]

    other = tmp_path / ".langcode" / "skills" / "review"
    other.mkdir(parents=True)
    (other / "SKILL.md").write_text("---\nname: review\ndescription: 评审流程\n---\n步骤", encoding="utf-8")

    assert sorted(item["name"] for item in project_module.load_skill_catalog(tmp_path)) == ["deploy", "review"]


def test_ensure_hermes_memory_files_runs_its_syscalls_once_per_workspace(tmp_path: Path, monkeypatch) -> None:
    from langcode_agent.memory import project as project_module

    project_module.reset_project_context_cache()
    project_module.ensure_hermes_memory_files(tmp_path)

    mkdirs = {"count": 0}
    original_mkdir = Path.mkdir

    def counting_mkdir(self, *args, **kwargs):
        mkdirs["count"] += 1
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", counting_mkdir)
    project_module.ensure_hermes_memory_files(tmp_path)
    project_module.ensure_hermes_memory_files(tmp_path)

    assert mkdirs["count"] == 0


def test_long_tool_descriptions_stay_compact() -> None:
    descriptions = {
        schema["function"]["name"]: schema["function"]["description"] for schema in tool_schemas()
    }

    assert len(descriptions["diagram"]) <= 75
    assert len(descriptions["self_evolve"]) <= 50
    assert len(descriptions["skill"]) <= 55
    assert "Mermaid" in descriptions["diagram"]
    assert "记忆" in descriptions["self_evolve"]
    assert "技能" in descriptions["skill"]
