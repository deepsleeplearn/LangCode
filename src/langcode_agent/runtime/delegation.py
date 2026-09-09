from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..core.context_management import compact_tool_result, make_json_safe
from .permissions import ApprovalMode, ToolCall, permission_for_tool
from ..tooling.tools import execute_tool


ROLE_TOOL_NAMES = {
    "researcher": {"read_file", "search", "ls", "glob", "web_search", "web_fetch"},
    "reviewer": {"read_file", "search", "ls", "glob", "web_search", "web_fetch"},
    "planner": {"read_file", "search", "ls", "glob", "web_search", "web_fetch"},
    "verifier": {"read_file", "search", "ls", "glob", "sandbox_shell", "web_search", "web_fetch"},
}
READ_ONLY_TOOL_NAMES = ROLE_TOOL_NAMES["researcher"]
DEFAULT_WEB_SEARCH_LIMIT = 8
WEB_SEARCH_LIMIT_MOCK_USER = (
    "不要再调用任何工具，不要继续搜索网页。请只基于本轮已经找到的工具结果和对话内容，"
    "直接回答我最初的问题；如果信息不足，也请说明依据和不确定性。"
)
WEB_SEARCH_LIMIT_ERROR = "外部网页搜索次数已达到上限。请基于已获得的搜索结果回答。"
DEFAULT_SUBAGENT_MAX_ROUNDS = 8
TOOL_ROUND_LIMIT_MOCK_USER = "已达工具调用轮次上限，请基于现有信息直接作答"


def _subagent_max_rounds(max_rounds: int | None = None) -> int:
    if max_rounds is not None:
        try:
            return max(1, int(max_rounds))
        except (TypeError, ValueError):
            pass
    raw = os.getenv("LANGCODE_SUBAGENT_MAX_ROUNDS", str(DEFAULT_SUBAGENT_MAX_ROUNDS))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_SUBAGENT_MAX_ROUNDS
    return max(1, value)


def delegate_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "delegate_agent",
            "description": (
                "运行一个拥有独立短上下文的角色化子 Agent；"
                "适合仓库调研、代码审查、任务规划或沙箱验证。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["researcher", "reviewer", "planner", "verifier"],
                        "default": "researcher",
                    },
                    "task": {"type": "string"},
                    "context": {"type": "string", "default": ""},
                },
                "required": ["task"],
            },
        },
    }


def run_delegate_agent(
    workspace_root: str | Path,
    *,
    task: str,
    role: str = "researcher",
    context: str = "",
    max_rounds: int | None = None,
    model: Any | None = None,
) -> dict:
    if model is None:
        from .chat import build_openai_model

        model = build_openai_model()
    role = role if role in ROLE_TOOL_NAMES else "researcher"
    allowed_tools = ROLE_TOOL_NAMES[role]
    bound_model = model.bind_tools(role_tool_schemas(role))
    messages = [
        SystemMessage(content=_delegate_system_prompt(Path(workspace_root), role)),
        HumanMessage(content=_delegate_user_prompt(task, context)),
    ]
    last_tool_result: dict | None = None
    web_search_count = 0
    rounds_used = 0
    round_limit = _subagent_max_rounds(max_rounds)
    session_id = f"delegate-{role}"

    for _ in range(round_limit):
        web_search_limit_reached = False
        ai_message = bound_model.invoke(messages)
        rounds_used += 1
        messages.append(ai_message)
        tool_calls = list(getattr(ai_message, "tool_calls", None) or [])
        if not tool_calls:
            return {
                "ok": True,
                "role": role,
                "summary": str(ai_message.content),
                "rounds": rounds_used,
            }
        for raw_tool_call in tool_calls:
            tool_name = raw_tool_call["name"]
            tool_input = dict(raw_tool_call.get("args") or {})
            if tool_name == "web_search" and web_search_count >= _web_search_limit():
                tool_result = _web_search_limit_result(tool_input, web_search_count)
                messages.append(
                    _tool_message(
                        workspace_root,
                        session_id,
                        tool_name,
                        tool_result,
                        raw_tool_call.get("id") or tool_name,
                    )
                )
                web_search_limit_reached = True
                last_tool_result = tool_result
                continue
            if tool_name == "web_search":
                web_search_count += 1
            if tool_name not in allowed_tools:
                tool_result = {
                    "ok": False,
                    "error": f"子 Agent 角色 {role} 是只读或受限角色，不能使用工具 {tool_name}",
                }
            else:
                try:
                    permission = permission_for_tool(ToolCall(tool_name, tool_input), workspace_root=workspace_root)
                    if permission is not ApprovalMode.ALLOW:
                        tool_result = {
                            "ok": False,
                            "error": f"子 Agent 角色 {role} 不能自动执行 {tool_name}；权限={permission.value}",
                        }
                    else:
                        tool_result = execute_tool(workspace_root, tool_name, tool_input)
                except Exception as exc:
                    tool_result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            last_tool_result = tool_result
            # A failing tool must still be answered with a ToolMessage, otherwise
            # the transcript keeps an unpaired tool_call and the provider rejects
            # the next request. The round bound above stops runaway retries.
            messages.append(
                _tool_message(
                    workspace_root,
                    session_id,
                    tool_name,
                    tool_result,
                    raw_tool_call.get("id") or tool_name,
                )
            )
        if web_search_limit_reached:
            messages.append(HumanMessage(content=WEB_SEARCH_LIMIT_MOCK_USER))

    messages.append(HumanMessage(content=TOOL_ROUND_LIMIT_MOCK_USER))
    final_message = bound_model.invoke(messages)
    rounds_used += 1
    summary = _message_text(final_message)
    if not summary and isinstance(last_tool_result, dict) and last_tool_result.get("ok") is False:
        return {
            "ok": False,
            "role": role,
            "error": str(last_tool_result.get("error") or "子 Agent 工具调用失败。"),
            "last_tool_result": last_tool_result,
            "rounds": rounds_used,
            "round_limit_reached": True,
            "max_rounds": round_limit,
        }
    return {
        "ok": True,
        "role": role,
        "summary": summary,
        "rounds": rounds_used,
        "round_limit_reached": True,
        "max_rounds": round_limit,
    }


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts).strip()
    return str(content or "").strip()


def _tool_message(
    workspace_root: str | Path,
    session_id: str,
    tool_name: str,
    tool_result: Any,
    tool_call_id: str,
) -> ToolMessage:
    if isinstance(tool_result, dict):
        payload = compact_tool_result(workspace_root, session_id, tool_name, tool_result)
    else:
        payload = make_json_safe(tool_result)
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        tool_call_id=tool_call_id,
    )


def read_only_tool_schemas() -> list[dict]:
    return role_tool_schemas("researcher")


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


def role_tool_schemas(role: str) -> list[dict]:
    from .chat import tool_schemas

    allowed = ROLE_TOOL_NAMES.get(role, ROLE_TOOL_NAMES["researcher"])
    return [schema for schema in tool_schemas(include_delegation=False) if schema["function"]["name"] in allowed]


def _delegate_system_prompt(workspace_root: Path, role: str) -> str:
    role_guidance = {
        "researcher": "查找具体的仓库事实，并引用相关文件或命令证据。",
        "reviewer": "重点查找功能风险、边界条件和缺失验证，避免只提代码风格意见。",
        "planner": "把任务拆成可执行步骤，说明依赖关系和验证点。",
        "verifier": "在沙箱中运行验证，并报告明确的通过或失败证据，不修改真实工作区。",
    }
    return (
        f"你是工作区 {workspace_root} 的只读 {role} 子 Agent。"
        f"{role_guidance[role]} "
        "只能使用当前角色暴露出来的工具。"
        "不要修改真实工作区；verifier 的 shell 命令只能在沙箱中运行。"
        "请用中文向主 Agent 返回简洁、结构化的结果。"
    )


def _delegate_user_prompt(task: str, context: str) -> str:
    if context.strip():
        return f"任务：\n{task.strip()}\n\n父级上下文：\n{context.strip()}"
    return f"任务：\n{task.strip()}"
