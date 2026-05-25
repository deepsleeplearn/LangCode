from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.context_management import make_json_safe
from ..storage.session_store import SessionStore
from .delegation import run_delegate_agent


DelegateRunner = Callable[..., dict]


def delegate_agents_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "delegate_agents",
            "description": (
                "并行运行多个只读或沙箱验证子 Agent，并把各自结果交给主 Agent 汇总；"
                "适合多视角调研、方案对比、审查与验证。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agents": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
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
                    "parallel": {"type": "boolean", "default": True},
                    "title": {"type": "string", "default": "多视角子 Agent 协作"},
                },
                "required": ["agents"],
            },
        },
    }


def agent_debate_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "agent_debate",
            "description": (
                "启动或继续一个由 Debate Manager 管理的多 Agent 辩论、博弈或角色对话；"
                "A/B/Judge 按轮次发言，transcript 会写入当前会话。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "debate_id": {"type": "string", "description": "可选。继续已有辩论时传入。"},
                    "topic": {"type": "string"},
                    "agents": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "stance": {"type": "string"},
                                "role_prompt": {"type": "string"},
                            },
                            "required": ["name", "stance"],
                        },
                    },
                    "rounds": {"type": "integer", "default": 2},
                    "judge": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "default": "judge"},
                            "name": {"type": "string", "default": "裁判"},
                            "role_prompt": {"type": "string"},
                        },
                    },
                    "max_words_per_turn": {"type": "integer", "default": 180},
                },
                "required": ["topic", "agents"],
            },
        },
    }


def run_parallel_delegate_agents(
    workspace_root: str | Path,
    *,
    agents: list[dict],
    max_rounds: int | None = None,
    parallel: bool = True,
    title: str = "多视角子 Agent 协作",
    _current_session_id: str = "",
    _session_store_path: str = "",
    delegate_runner: DelegateRunner | None = None,
) -> dict:
    normalized = [_normalize_delegate_agent(agent, index) for index, agent in enumerate(agents or [], start=1)]
    if not normalized:
        return {"ok": False, "error": "至少需要提供一个子 Agent。"}

    runner = delegate_runner or run_delegate_agent

    def run_one(agent: dict) -> dict:
        result = runner(
            workspace_root,
            role=agent["role"],
            task=agent["task"],
            context=agent.get("context", ""),
        )
        return {
            "agent_id": agent["id"],
            "agent_name": agent["name"],
            "role": "assistant",
            "round": 1,
            "content": _delegate_result_text(result),
            "result": make_json_safe(result),
        }

    if parallel and len(normalized) > 1:
        messages: list[dict] = []
        with ThreadPoolExecutor(max_workers=min(len(normalized), 4)) as executor:
            future_to_index = {executor.submit(run_one, agent): index for index, agent in enumerate(normalized)}
            collected: dict[int, dict] = {}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                collected[index] = future.result()
            messages = [collected[index] for index in sorted(collected)]
    else:
        messages = [run_one(agent) for agent in normalized]

    thread_id = f"parallel-{uuid4().hex[:10]}"
    payload = {
        "ok": True,
        "kind": "agent_dialogue",
        "dialogue_type": "parallel_delegate",
        "thread_id": thread_id,
        "title": title or "多视角子 Agent 协作",
        "participants": [
            {"id": agent["id"], "name": agent["name"], "role": agent["role"], "mode": "read_only"}
            for agent in normalized
        ],
        "messages": [{key: value for key, value in item.items() if key != "result"} for item in messages],
        "results": [item["result"] for item in messages],
    }
    _persist_dialogue(payload, _current_session_id, _session_store_path)
    return payload


def run_agent_debate(
    workspace_root: str | Path,
    *,
    topic: str,
    agents: list[dict],
    debate_id: str = "",
    rounds: int = 2,
    judge: dict | None = None,
    max_words_per_turn: int = 180,
    _current_session_id: str = "",
    _session_store_path: str = "",
    model: Any | None = None,
) -> dict:
    last_payload: dict | None = None
    for payload in iter_agent_debate_events(
        workspace_root,
        topic=topic,
        agents=agents,
        debate_id=debate_id,
        rounds=rounds,
        judge=judge,
        max_words_per_turn=max_words_per_turn,
        _current_session_id=_current_session_id,
        _session_store_path=_session_store_path,
        model=model,
    ):
        last_payload = payload
    return last_payload or {"ok": False, "error": "辩论没有产生任何发言。"}


def iter_agent_debate_events(
    workspace_root: str | Path,
    *,
    topic: str,
    agents: list[dict],
    debate_id: str = "",
    rounds: int = 2,
    judge: dict | None = None,
    max_words_per_turn: int = 180,
    _current_session_id: str = "",
    _session_store_path: str = "",
    model: Any | None = None,
):
    participants = [_normalize_debate_agent(agent, index) for index, agent in enumerate(agents or [], start=1)]
    if len(participants) < 2:
        yield {"ok": False, "error": "辩论或博弈至少需要两个参与 Agent。"}
        return
    if model is None:
        from .chat import build_openai_model

        model = build_openai_model()

    thread_id = str(debate_id or "").strip() or f"debate-{uuid4().hex[:10]}"
    existing = _load_dialogue(_session_store_path, thread_id)
    transcript = list(existing.get("messages") or []) if existing else []
    start_round = _next_round(transcript)
    rounds = max(1, min(int(rounds or 2), 6))
    max_words = max(60, min(int(max_words_per_turn or 180), 500))

    for round_index in range(start_round, start_round + rounds):
        for participant in participants:
            content = _invoke_agent_turn(
                model,
                topic=topic,
                participant=participant,
                transcript=transcript,
                round_index=round_index,
                max_words=max_words,
            )
            transcript.append(
                {
                    "agent_id": participant["id"],
                    "agent_name": participant["name"],
                    "role": "assistant",
                    "round": round_index,
                    "content": content,
                }
            )
            payload = _debate_payload(thread_id, topic, participants, _normalize_judge(judge), transcript)
            _persist_dialogue(payload, _current_session_id, _session_store_path)
            yield payload

    judge_spec = _normalize_judge(judge)
    judge_content = _invoke_judge_turn(model, topic=topic, judge=judge_spec, transcript=transcript, max_words=max_words)
    transcript.append(
        {
            "agent_id": judge_spec["id"],
            "agent_name": judge_spec["name"],
            "role": "assistant",
            "round": start_round + rounds,
            "content": judge_content,
        }
    )

    payload = _debate_payload(thread_id, topic, participants, judge_spec, transcript)
    _persist_dialogue(payload, _current_session_id, _session_store_path)
    yield payload


def _debate_payload(
    thread_id: str,
    topic: str,
    participants: list[dict],
    judge_spec: dict,
    transcript: list[dict],
) -> dict:
    return {
        "ok": True,
        "kind": "agent_dialogue",
        "dialogue_type": "debate",
        "thread_id": thread_id,
        "title": f"辩论：{topic}",
        "participants": [
            {"id": item["id"], "name": item["name"], "stance": item["stance"]} for item in participants
        ]
        + [{"id": judge_spec["id"], "name": judge_spec["name"], "stance": "裁判"}],
        "messages": list(transcript),
        "state": {"topic": topic, "next_round": _next_round(transcript)},
    }


def _normalize_delegate_agent(agent: dict, index: int) -> dict:
    role = str(agent.get("role") or "researcher")
    if role not in {"researcher", "reviewer", "planner", "verifier"}:
        role = "researcher"
    agent_id = str(agent.get("id") or role or f"agent_{index}").strip() or f"agent_{index}"
    return {
        "id": agent_id,
        "name": str(agent.get("name") or _role_name(role)).strip() or _role_name(role),
        "role": role,
        "task": str(agent.get("task") or "").strip(),
        "context": str(agent.get("context") or "").strip(),
    }


def _normalize_debate_agent(agent: dict, index: int) -> dict:
    agent_id = str(agent.get("id") or f"agent_{index}").strip() or f"agent_{index}"
    name = str(agent.get("name") or f"Agent {index}").strip() or f"Agent {index}"
    stance = str(agent.get("stance") or "").strip()
    return {
        "id": agent_id,
        "name": name,
        "stance": stance or name,
        "role_prompt": str(agent.get("role_prompt") or "").strip(),
    }


def _normalize_judge(judge: dict | None) -> dict:
    judge = judge or {}
    return {
        "id": str(judge.get("id") or "judge"),
        "name": str(judge.get("name") or "裁判"),
        "role_prompt": str(judge.get("role_prompt") or "你是中立裁判，负责总结各方论点并给出清晰裁决。"),
    }


def _role_name(role: str) -> str:
    return {
        "researcher": "研究员",
        "reviewer": "审查员",
        "planner": "规划员",
        "verifier": "验证员",
    }.get(role, role)


def _delegate_result_text(result: dict) -> str:
    if result.get("ok") is True:
        return str(result.get("summary") or "子 Agent 已完成，但没有返回摘要。")
    return f"子 Agent 失败：{result.get('error') or result}"


def _invoke_agent_turn(
    model: Any,
    *,
    topic: str,
    participant: dict,
    transcript: list[dict],
    round_index: int,
    max_words: int,
) -> str:
    system = (
        f"你是多 Agent 辩论中的 {participant['name']}。"
        f"你的立场是：{participant['stance']}。"
        f"{participant.get('role_prompt') or ''}\n"
        "请只代表自己的角色发言，不要代替其他 Agent 或裁判总结。"
        f"本轮回答不超过 {max_words} 个中文词，清晰、有针对性。"
    )
    prompt = (
        f"辩题：{topic}\n"
        f"当前轮次：第 {round_index} 轮\n\n"
        f"已有 transcript：\n{_render_transcript(transcript)}\n\n"
        "请给出你的下一段发言。"
    )
    return _invoke_text(model, [SystemMessage(content=system), HumanMessage(content=prompt)])


def _invoke_judge_turn(model: Any, *, topic: str, judge: dict, transcript: list[dict], max_words: int) -> str:
    system = (
        f"你是 {judge['name']}。{judge.get('role_prompt') or ''} "
        "请基于 transcript 做中立裁决，不要引入未出现的新事实。"
    )
    prompt = (
        f"辩题：{topic}\n\n"
        f"完整 transcript：\n{_render_transcript(transcript)}\n\n"
        f"请用不超过 {max_words} 个中文词总结双方核心分歧、优势弱点，并给出裁决。"
    )
    return _invoke_text(model, [SystemMessage(content=system), HumanMessage(content=prompt)])


def _invoke_text(model: Any, messages: list[Any]) -> str:
    response = model.invoke(messages)
    return str(getattr(response, "content", response) or "").strip()


def _render_transcript(messages: list[dict]) -> str:
    if not messages:
        return "（暂无历史发言）"
    lines = []
    for item in messages[-24:]:
        name = item.get("agent_name") or item.get("agent_id") or "Agent"
        round_index = item.get("round") or "-"
        content = " ".join(str(item.get("content") or "").split())
        lines.append(f"- 第 {round_index} 轮，{name}：{content}")
    return "\n".join(lines)


def _next_round(messages: list[dict]) -> int:
    rounds = [int(item.get("round") or 0) for item in messages if isinstance(item, dict)]
    return max(rounds, default=0) + 1


def _persist_dialogue(payload: dict, session_id: str, store_path: str) -> None:
    if not session_id or not store_path:
        return
    store = SessionStore(store_path)
    store.save_agent_dialogue(
        session_id,
        str(payload["thread_id"]),
        kind=str(payload.get("dialogue_type") or payload.get("kind") or "agent_dialogue"),
        title=str(payload.get("title") or payload["thread_id"]),
        participants=list(payload.get("participants") or []),
        messages=list(payload.get("messages") or []),
        state=dict(payload.get("state") or {}),
    )


def _load_dialogue(store_path: str, thread_id: str) -> dict | None:
    if not store_path or not thread_id:
        return None
    return SessionStore(store_path).load_agent_thread(thread_id)
