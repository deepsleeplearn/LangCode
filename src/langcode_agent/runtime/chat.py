from collections.abc import Callable
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from ..core._compat import patch_langchain_debug
from .agent import CodeAgent
from ..core.config import load_env_files
from ..core.context_management import compact_tool_result, make_json_safe
from .delegation import delegate_tool_schema, run_delegate_agent
from .multi_agent import (
    agent_debate_tool_schema,
    delegate_agents_tool_schema,
    run_agent_debate,
    run_parallel_delegate_agents,
)
from .deep_harness import cancel_task, create_task, deepagents_capability_summary, get_task, list_tasks, update_task
from ..memory.project import handle_local_command, load_project_context
from .permissions import ToolCall

ApprovalCallback = Callable[[dict], dict]
DEFAULT_WEB_SEARCH_LIMIT = 8
WEB_SEARCH_LIMIT_MOCK_USER = (
    "不要再调用任何工具，不要继续搜索网页。请只基于本轮已经找到的工具结果和对话内容，"
    "直接回答我最初的问题；如果信息不足，也请说明依据和不确定性。"
)
WEB_SEARCH_LIMIT_ERROR = "外部网页搜索次数已达到上限。请基于已获得的搜索结果回答。"


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model: str
    base_url: str | None
    api_key: str | None
    default_headers: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None


class ChatSession:
    def __init__(
        self,
        agent: CodeAgent,
        model: Any,
        approval_callback: ApprovalCallback,
        *,
        thread_id: str,
        system_prompt: str | None = None,
        history: list[BaseMessage] | None = None,
    ) -> None:
        self.agent = agent
        self.model = model.bind_tools(tool_schemas())
        self.approval_callback = approval_callback
        self.thread_id = thread_id
        self.todos: list[dict] = []
        self.messages: list[BaseMessage] = list(history or [])
        if system_prompt:
            self._ensure_system_prompt(system_prompt)

    def _ensure_system_prompt(self, system_prompt: str) -> None:
        current = SystemMessage(content=system_prompt)
        for index, message in enumerate(self.messages):
            if not isinstance(message, SystemMessage):
                continue
            if str(message.content or "").startswith("你是一个谨慎的代码 Agent"):
                self.messages[index] = current
            else:
                self.messages.insert(0, current)
            return
        self.messages.insert(0, current)

    def send(self, user_text: str) -> str:
        local_reply = handle_local_command(self.agent.workspace_root, self.messages, user_text)
        if local_reply is not None:
            return local_reply

        self.messages.append(HumanMessage(content=user_text))

        web_search_count = 0
        while True:
            web_search_limit_reached = False
            ai_message = self.model.invoke(self.messages)
            self.messages.append(ai_message)
            tool_calls = _message_tool_calls(ai_message)
            if not tool_calls:
                return str(ai_message.content)

            for tool_call in tool_calls:
                if tool_call["name"] == "web_search":
                    if web_search_count >= _web_search_limit():
                        result = _web_search_limit_result(dict(tool_call.get("args", {})), web_search_count)
                        self.messages.append(
                            ToolMessage(
                                content=json.dumps(make_json_safe(result), ensure_ascii=False),
                                tool_call_id=tool_call.get("id") or tool_call["name"],
                            )
                        )
                        web_search_limit_reached = True
                        continue
                    web_search_count += 1
                result = self._run_tool_call(tool_call)
                self.messages.append(
                    ToolMessage(
                        content=json.dumps(make_json_safe(result), ensure_ascii=False),
                        tool_call_id=tool_call.get("id") or tool_call["name"],
                    )
                )
            if web_search_limit_reached:
                self.messages.append(HumanMessage(content=WEB_SEARCH_LIMIT_MOCK_USER))

    def _run_tool_call(self, raw_tool_call: dict) -> dict:
        if raw_tool_call["name"] == "delegate_agent":
            return make_json_safe(run_delegate_agent(
                self.agent.workspace_root,
                **dict(raw_tool_call.get("args", {})),
            ))
        if raw_tool_call["name"] == "delegate_agents":
            return make_json_safe(run_parallel_delegate_agents(
                self.agent.workspace_root,
                _current_session_id=self.thread_id,
                **dict(raw_tool_call.get("args", {})),
            ))
        if raw_tool_call["name"] == "agent_debate":
            return make_json_safe(run_agent_debate(
                self.agent.workspace_root,
                _current_session_id=self.thread_id,
                **dict(raw_tool_call.get("args", {})),
            ))
        tool_input = dict(raw_tool_call.get("args", {}))
        if raw_tool_call["name"] == "self_evolve":
            tool_input.setdefault("_current_session_id", self.thread_id)
            tool_input.setdefault("messages", self.export_history())
            tool_input.setdefault("todos", list(self.todos))
        tool_call = ToolCall(raw_tool_call["name"], tool_input)
        result = self.agent.request_tool(tool_call, thread_id=self.thread_id)
        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            approval = self.approval_callback(payload)
            result = self.agent.resume(self.thread_id, approval)
        tool_result = result.get("tool_result", result)
        if isinstance(tool_result, dict):
            if raw_tool_call["name"] in {"task_create", "task_update", "task_list", "task_get", "task_cancel"}:
                return self._run_task_tool(raw_tool_call["name"], dict(raw_tool_call.get("args", {})), tool_result)
            return compact_tool_result(self.agent.workspace_root, self.thread_id, raw_tool_call["name"], tool_result)
        return make_json_safe(tool_result)

    def _run_task_tool(self, tool_name: str, tool_input: dict, tool_result: dict) -> dict:
        if tool_result.get("ok") is False:
            return tool_result
        try:
            if tool_name == "task_create":
                result = create_task(
                    self.todos,
                    str(tool_input.get("content") or ""),
                    status=str(tool_input.get("status") or "pending"),
                    task_id=str(tool_input["task_id"]) if tool_input.get("task_id") else None,
                )
            elif tool_name == "task_update":
                result = update_task(
                    self.todos,
                    str(tool_input.get("task_id") or ""),
                    content=str(tool_input["content"]) if tool_input.get("content") is not None else None,
                    status=str(tool_input["status"]) if tool_input.get("status") is not None else None,
                )
            elif tool_name == "task_list":
                return list_tasks(self.todos, status=str(tool_input["status"]) if tool_input.get("status") else None)
            elif tool_name == "task_get":
                return get_task(self.todos, str(tool_input.get("task_id") or ""))
            else:
                result = cancel_task(
                    self.todos,
                    str(tool_input.get("task_id") or ""),
                    reason=str(tool_input["reason"]) if tool_input.get("reason") else None,
                )
        except Exception as exc:
            return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        self.todos = list(result.get("todos") or self.todos)
        return result

    def export_history(self) -> list[dict]:
        return [_serialize_message(message) for message in self.messages]


def build_openai_model() -> ChatOpenAI:
    patch_langchain_debug()
    settings = model_settings_from_env()
    kwargs: dict[str, Any] = {"model": settings.model}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    if settings.api_key:
        kwargs["api_key"] = settings.api_key
    if settings.default_headers:
        kwargs["default_headers"] = settings.default_headers
    if settings.extra_body:
        kwargs["extra_body"] = settings.extra_body
    return ChatOpenAI(**kwargs)


def _web_search_limit() -> int:
    raw = os.getenv("LANGCODE_WEB_SEARCH_LIMIT", str(DEFAULT_WEB_SEARCH_LIMIT))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_WEB_SEARCH_LIMIT
    return max(1, value)


def _web_search_limit_result(tool_input: dict, count: int) -> dict:
    return {
        "ok": False,
        "error": WEB_SEARCH_LIMIT_ERROR,
        "error_type": "web_search_limit",
        "searches_used": count,
        "blocked_query": str(tool_input.get("query") or ""),
        "instruction": "不要继续调用 web_search；请基于本轮已有搜索结果回答用户问题。",
    }


def model_settings_from_env() -> ModelSettings:
    load_env_files(Path.cwd())
    provider = (os.getenv("LANGCODE_PROVIDER") or "zhipu").lower()
    if provider == "zhipu":
        return ModelSettings(
            provider="zhipu",
            model=os.getenv("LANGCODE_MODEL") or "glm-5.1",
            base_url=os.getenv("ZHIPU_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4",
            api_key=os.getenv("ZHIPU_API_KEY") or os.getenv("OPENAI_API_KEY"),
        )
    model = os.getenv("LANGCODE_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    if provider == "openai" and _openai_gateway_from_env() == "aimp":
        aimp_model = os.getenv("AIMP_GPT4O_MODEL") or (model if model.startswith("gpt-") else "gpt-4o")
        return ModelSettings(
            provider="openai",
            model=aimp_model,
            base_url=_aimp_base_url(),
            api_key=os.getenv("AIMP_GPT4O_API_KEY"),
            default_headers=_aimp_headers(aimp_model),
        )
    if provider == "openai" and _openai_gateway_from_env() == "aimp-deepseek-v4-pro":
        deepseek_model = os.getenv("AIMP_DEEPSEEK_V4_MODEL") or (
            model if model == "deepseek-v4-pro" else "deepseek-v4-pro"
        )
        return ModelSettings(
            provider="openai",
            model=deepseek_model,
            base_url=_aimp_deepseek_base_url(),
            api_key=os.getenv("AIMP_DEEPSEEK_V4_API_KEY"),
            default_headers=_aimp_deepseek_headers(),
            extra_body=_thinking_extra_body(),
        )
    if provider == "openai" and _openai_gateway_from_env() == "aimp-glm":
        glm_model = os.getenv("AIMP_GLM_MODEL") or (model if model == "glm-5" else "glm-5")
        return ModelSettings(
            provider="openai",
            model=glm_model,
            base_url=_aimp_glm_base_url(),
            api_key=os.getenv("AIMP_GLM_API_KEY"),
            default_headers=_aimp_glm_headers(),
            extra_body=_glm_extra_body(),
        )
    return ModelSettings(
        provider="openai",
        model=model,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _aimp_base_url() -> str:
    configured = (
        os.getenv("AIMP_GPT4O_BASE_URL")
        or "https://aimpapi.midea.com/t-aigc/mip-chat-app/openai/standard/v1"
    ).rstrip("/")
    suffix = "/chat/completions"
    if configured.endswith(suffix):
        return configured[: -len(suffix)]
    return configured


def _aimp_deepseek_base_url() -> str:
    configured = (
        os.getenv("AIMP_DEEPSEEK_V4_BASE_URL")
        or "https://aimpapi.midea.com/t-aigc/aimp-deepseek-v4-pro/v1"
    ).rstrip("/")
    suffix = "/chat/completions"
    if configured.endswith(suffix):
        return configured[: -len(suffix)]
    return configured


def _aimp_glm_base_url() -> str:
    configured = (
        os.getenv("AIMP_GLM_BASE_URL")
        or "https://aimpapi.midea.com/t-aigc/aimp-glm/v1"
    ).rstrip("/")
    suffix = "/chat/completions"
    if configured.endswith(suffix):
        return configured[: -len(suffix)]
    return configured


def _openai_gateway_from_env() -> str:
    return (os.getenv("LANGCODE_OPENAI_GATEWAY") or "").lower()


def _aimp_headers(model: str) -> dict[str, str]:
    headers = {"Aimp-Biz-Id": model}
    user = os.getenv("AIMP_GPT4O_USER") or ""
    if user:
        headers["AIGC-USER"] = user
    return headers


def _aimp_deepseek_headers() -> dict[str, str]:
    user = os.getenv("AIMP_DEEPSEEK_V4_USER") or os.getenv("AIGC_USER") or ""
    return {"AIGC-USER": user} if user else {}


def _aimp_glm_headers() -> dict[str, str]:
    user = os.getenv("AIMP_GLM_USER") or os.getenv("AIGC_USER") or "guojian34"
    return {"AIGC-USER": user} if user else {}


def _thinking_extra_body() -> dict[str, Any]:
    return {"chat_template_kwargs": {"thinking": _env_bool("LANGCODE_THINKING")}}


def _glm_extra_body() -> dict[str, Any]:
    return {
        "chat_template_kwargs": {"thinking": _env_bool("LANGCODE_THINKING", default=True)},
        "separate_reasoning": True,
    }


def _env_bool(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    return value in {"1", "true", "yes", "on", "y"}


def tool_schemas(*, include_delegation: bool = True) -> list[dict]:
    schemas = [
        _function_schema(
            "read_file",
            "读取工作区内的 UTF-8 文本文件。",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        _function_schema(
            "search",
            "在工作区内搜索文本；可用时优先使用 rg。",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "max_results": {"type": "integer", "default": 50},
                },
                "required": ["query"],
            },
        ),
        _function_schema(
            "ls",
            "列出工作区内的文件和目录。",
            {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
            },
        ),
        _function_schema(
            "glob",
            "按类似 glob 的文件名模式查找工作区文件。",
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "max_results": {"type": "integer", "default": 100},
                },
                "required": ["pattern"],
            },
        ),
        _function_schema(
            "web_search",
            "使用 Tavily 搜索公开网页；适合查询最新文档、新闻、接口资料和外部参考。",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "default": "basic",
                    },
                    "include_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "exclude_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "topic": {
                        "type": "string",
                        "enum": ["general", "news", "finance"],
                        "default": "general",
                    },
                },
                "required": ["query"],
            },
        ),
        _function_schema(
            "web_fetch",
            "使用 Tavily 抓取指定公开 http(s) URL，并提取可读的 Markdown 内容。",
            {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "extract_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "default": "basic",
                    },
                    "max_chars": {"type": "integer", "default": 12000, "minimum": 1000, "maximum": 50000},
                },
                "required": ["url"],
            },
        ),
        _function_schema(
            "voice_interrupt",
            "内部语音打断事件工具。系统会自动写入，模型不要主动调用；用于记录用户在语音播报过程中打断时的上下文。",
            {
                "type": "object",
                "properties": {
                    "spoken_text": {"type": "string", "description": "用户打断时实际说出的文本。"},
                    "previous_user_text": {"type": "string", "description": "打断前最近一次用户问题。"},
                    "assistant_displayed_text": {"type": "string", "description": "打断前 assistant 已经展示或播报的内容。"},
                },
                "required": ["spoken_text"],
            },
        ),
        _function_schema(
            "memory",
            "管理 Hermes 风格的有界长期记忆。memory 保存项目事实、决策和经验；user 保存稳定用户偏好。记忆会作为后续会话系统上下文的一部分读取。",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "add", "replace", "remove"],
                        "default": "read",
                    },
                    "target": {
                        "type": "string",
                        "enum": ["memory", "user"],
                        "default": "memory",
                    },
                    "content": {"type": "string", "description": "add 的新增内容，replace 的新内容，或 remove 的目标内容。"},
                    "old": {"type": "string", "description": "replace/remove 要匹配的唯一原始片段。"},
                },
                "required": ["action"],
            },
        ),
        _function_schema(
            "soul",
            "管理 LangCode 的 Hermes 风格长期身份文件 SOUL.md。SOUL 定义 Agent 稳定身份、语气和默认行为；不要写入项目临时规则。",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "write"],
                        "default": "read",
                    },
                    "content": {"type": "string", "description": "write 时的新 SOUL 内容。"},
                },
                "required": ["action"],
            },
        ),
        _function_schema(
            "self_evolve",
            "运行可审计的自进化流程：查看状态、反思当前会话、归档经验、生成技能/提示词/工具描述改进提案。高置信偏好可写入记忆，低置信内容只生成候选。",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "status",
                            "reflect_session",
                            "list_reflections",
                            "propose",
                            "list_proposals",
                            "read_soul",
                            "update_soul",
                        ],
                        "default": "status",
                    },
                    "session_id": {"type": "string", "description": "需要反思的会话 ID，默认当前会话。"},
                    "apply": {"type": "boolean", "default": True, "description": "是否自动应用高置信候选。"},
                    "title": {"type": "string", "description": "propose 时的提案标题。"},
                    "target": {"type": "string", "description": "propose 时的优化对象，如 skill、prompt、tool_description、code。"},
                    "content": {"type": "string", "description": "提案正文或 SOUL 新内容。"},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["action"],
            },
        ),
        _function_schema(
            "cron",
            "管理 LangCode 本地定时任务。任务可绑定 skills，支持创建、查看、暂停、恢复、删除、查询到期任务和记录运行结果。",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "create", "update", "delete", "pause", "resume", "run_due", "run"],
                        "default": "list",
                    },
                    "job_id": {"type": "string"},
                    "name": {"type": "string"},
                    "prompt": {"type": "string"},
                    "schedule": {"type": "string", "description": "例如 every 60 minutes、every 2 hours、daily 09:00。"},
                    "skills": {"type": "array", "items": {"type": "string"}, "default": []},
                    "status": {"type": "string", "enum": ["active", "paused"]},
                    "result": {"type": "string", "description": "run 时记录的执行结果。"},
                },
                "required": ["action"],
            },
        ),
        _function_schema(
            "session_search",
            "搜索或浏览本地 SQLite 会话历史；用于回忆过去对话、查找已完成方案、定位某条历史消息附近上下文。",
            {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["search", "recent", "around"],
                        "default": "search",
                    },
                    "query": {"type": "string", "description": "search 模式的关键词或 FTS 查询。"},
                    "limit": {"type": "integer", "default": 8, "minimum": 1, "maximum": 20},
                    "session_id": {"type": "string", "description": "around 模式要查看的会话 ID；为空时使用当前会话。"},
                    "message_id": {"type": "integer", "description": "around 模式的中心消息序号。"},
                    "before": {"type": "integer", "default": 3},
                    "after": {"type": "integer", "default": 3},
                },
            },
        ),
        _function_schema(
            "skill",
            "管理可复用技能记忆。用于列出、读取、创建或更新技能文件；复杂任务完成后，如果形成了稳定流程或踩坑经验，应沉淀为技能，后续相似任务可先读取技能。",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "read", "upsert", "remove"],
                        "default": "list",
                    },
                    "name": {"type": "string", "description": "技能名称，使用简短稳定的英文或拼音标识。"},
                    "description": {"type": "string", "description": "一句话说明该技能适用于什么场景。"},
                    "content": {"type": "string", "description": "技能正文，建议包含适用场景、步骤、验证方式和注意事项。"},
                    "scope": {
                        "type": "string",
                        "enum": ["project", "global"],
                        "default": "project",
                        "description": "project 写入当前项目技能；global 写入用户级 Hermes 技能目录。",
                    },
                },
                "required": ["action"],
            },
        ),
        _function_schema(
            "diagram",
            "生成可在前端渲染的 Mermaid 图。任何流程、关系、架构、调用链、状态转换、数据流、审批流或协作逻辑需要清晰展示时使用；结构化节点边会通过 LangChain Graph.draw_mermaid 生成。",
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "图标题。"},
                    "diagram_type": {
                        "type": "string",
                        "enum": ["flowchart", "collaboration", "sequence", "architecture", "state"],
                        "default": "flowchart",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["TD", "TB", "BT", "LR", "RL"],
                        "default": "TD",
                    },
                    "nodes": {
                        "type": "array",
                        "description": "flowchart/collaboration 图的节点列表。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "label": {"type": "string"},
                                "role": {"type": "string"},
                            },
                            "required": ["id"],
                        },
                    },
                    "edges": {
                        "type": "array",
                        "description": "flowchart/collaboration 图的边列表。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "label": {"type": "string"},
                            },
                            "required": ["source", "target"],
                        },
                    },
                    "mermaid": {
                        "type": "string",
                        "description": "可选。已有 Mermaid DSL 时直接传入，例如 sequenceDiagram 或 stateDiagram。",
                    },
                },
                "required": ["title"],
            },
        ),
        _function_schema(
            "write_file",
            "在工作区内写入 UTF-8 文件；需要人工审批。",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        ),
        _function_schema(
            "edit_file",
            "替换工作区内 UTF-8 文件中的文本；需要人工审批。",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "replace_all": {"type": "boolean", "default": False},
                },
                "required": ["path", "old", "new"],
            },
        ),
        _function_schema(
            "shell",
            "在工作区内执行带超时的 shell 命令；低风险本地命令可自动执行，危险命令需要人工审批。",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "default": 30},
                },
                "required": ["command"],
            },
        ),
        _function_schema(
            "sandbox_shell",
            "在工作区副本沙箱中执行 shell 命令；适合有风险的验证或实验。",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "default": 30},
                    "copy_workspace": {"type": "boolean", "default": True},
                },
                "required": ["command"],
            },
        ),
        _function_schema(
            "task_create",
            "创建一个新的结构化任务，并返回稳定任务标识；适合 Claude Code 风格的增量任务跟踪。",
            {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "blocked", "cancelled"],
                        "default": "pending",
                    },
                    "task_id": {"type": "string"},
                },
                "required": ["content"],
            },
        ),
        _function_schema(
            "task_update",
            "按任务标识增量更新任务内容或状态；同一任务组内最多保留一个进行中任务。",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "content": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "blocked", "cancelled"],
                    },
                },
                "required": ["task_id"],
            },
        ),
        _function_schema(
            "task_list",
            "读取当前会话的结构化任务列表，可按状态筛选。",
            {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "blocked", "cancelled"],
                    }
                },
            },
        ),
        _function_schema(
            "task_get",
            "按任务标识读取单个任务。",
            {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        ),
        _function_schema(
            "task_cancel",
            "取消指定任务；用于用户打断、需求变化或任务不再需要的情况。",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["task_id"],
            },
        ),
    ]
    if include_delegation:
        schemas.append(delegate_tool_schema())
        schemas.append(delegate_agents_tool_schema())
        schemas.append(agent_debate_tool_schema())
    return schemas


def default_system_prompt(workspace_root: str) -> str:
    prompt = (
        "你是一个谨慎的代码 Agent，正在终端环境中工作。"
        f"当前工作区根目录是 {workspace_root}。"
        "请默认使用中文与用户沟通；任务清单、阶段总结、工具调用说明和最终回答都使用中文。"
        "需要查看文件、编辑文件、搜索工作区、搜索或抓取公开网页、管理记忆、检索历史会话、执行 shell 命令时，请使用工具。"
        f"{deepagents_capability_summary()} "
        "多步骤任务必须使用 task_create、task_update、task_list、task_get 维护任务清单。"
        "执行任务时只能保留一个正在进行的任务；"
        "任务完成后要立刻标记为已完成；只要还有待办或正在进行的任务，就不要声称整体任务已经完成。"
        "有风险的实验先用 sandbox_shell 验证。"
        "当独立调研、审查、规划或验证有帮助时，使用 delegate_agent。"
        "需要多视角时，使用 delegate_agents 并行调用多个只读子 Agent，然后由你汇总。"
        "需要辩论、博弈、角色对话或子 Agent 通讯时，使用 agent_debate；"
        "Debate Manager 会维护 transcript，并让 A/B/Judge 按轮次发言。"
        "只要需要解释流程、关系、架构、调用链、状态转换、数据流、审批流、任务依赖或协作逻辑，就使用 diagram 生成可视化图。"
        "长期记忆遵循 Hermes 风格：MEMORY 保存项目事实/决策/经验，USER 保存稳定用户偏好；"
        "SOUL 保存你的长期身份、语气和默认行为；必要时使用 soul 或 memory 工具增删改查，使用 session_search 回忆历史会话。"
        "遇到相似复杂任务时，先用 skill 列出并读取相关技能；"
        "复杂任务完成后，如果得到可复用流程、验证方法或踩坑经验，应在最终回答前使用 skill 沉淀或更新技能。"
        "任务完成、用户纠正、发现稳定偏好、踩坑后恢复、形成可复用流程时，使用 self_evolve 反思会话；"
        "高置信用户偏好或工程经验可以写入 USER/MEMORY，低置信技能、提示词、工具描述或代码优化只能生成可审计提案。"
        "用户要求周期性检查、提醒、监控或例行任务时，使用 cron 创建或管理本地定时任务，可绑定相关 skills。"
        "本地命令：/compact 用于压缩较早的对话上下文，/memory 用于查看项目记忆，"
        "/agents 用于列出内置子 Agent，/skills 查看技能，/evolve 查看自进化状态，/cron 查看定时任务，以 # 开头的行会保存项目记忆。"
        "向用户说明进展时保持简洁，并严格遵守人工审批结果。"
        "如果需要输出 Markdown 表格，必须使用合法 GFM 表格：表头、分隔行和每一行都必须各占一行；"
        "分隔行列数必须与表头一致；每个数据行列数也必须与表头一致；不要把多行表格压缩到同一行。"
    )
    context = load_project_context(workspace_root)
    if context:
        prompt += "\n\n项目记忆和指令：\n" + context
    return prompt


def messages_from_json(items: list[dict]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    pending_tool_call_ids: set[str] = set()
    for item in items:
        role = item.get("role")
        content = item.get("content", "")
        if role == "system":
            messages.append(SystemMessage(content=content))
            pending_tool_call_ids.clear()
        elif role == "human":
            messages.append(HumanMessage(content=content))
            pending_tool_call_ids.clear()
        elif role == "ai":
            tool_calls = item.get("tool_calls")
            if isinstance(tool_calls, list):
                messages.append(AIMessage(content=content, tool_calls=tool_calls))
                pending_tool_call_ids = {
                    str(tool_call.get("id"))
                    for tool_call in tool_calls
                    if isinstance(tool_call, dict) and tool_call.get("id")
                }
            else:
                messages.append(AIMessage(content=content))
                pending_tool_call_ids.clear()
        elif role == "tool":
            tool_call_id = str(item.get("tool_call_id") or "")
            if tool_call_id and tool_call_id in pending_tool_call_ids:
                messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
                pending_tool_call_ids.remove(tool_call_id)
    return messages


def _function_schema(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _message_tool_calls(message: AIMessage) -> list[dict]:
    return list(getattr(message, "tool_calls", None) or [])


def _serialize_message(message: BaseMessage) -> dict:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": str(message.content)}
    if isinstance(message, HumanMessage):
        return {"role": "human", "content": str(message.content)}
    if isinstance(message, AIMessage):
        item = {"role": "ai", "content": str(message.content)}
        tool_calls = _message_tool_calls(message)
        if tool_calls:
            item["tool_calls"] = tool_calls
        return item
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": str(message.content),
            "tool_call_id": message.tool_call_id,
        }
    return {"role": "unknown", "content": str(message.content)}
