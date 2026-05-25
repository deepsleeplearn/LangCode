from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from langcode_agent.runtime.agent import CodeAgent
from langcode_agent.runtime.chat import ChatSession, default_system_prompt, messages_from_json, model_settings_from_env, tool_schemas
from langcode_agent.runtime.delegation import _delegate_system_prompt, run_delegate_agent
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
    monkeypatch.delenv("LANGCODE_OPENAI_GATEWAY", raising=False)
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

    assert context == ""
    assert (tmp_path / ".langcode" / "memories" / "MEMORY.md").is_file()
    assert (tmp_path / ".langcode" / "memories" / "USER.md").is_file()


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
