from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .project import memory_tool, skill_tool, soul_tool


EVOLUTION_DIR = ".langcode/evolution"
REFLECTIONS_FILE = "reflections.jsonl"
PROPOSALS_DIR = "proposals"
DEFAULT_MEMORY_AUTO_APPLY_THRESHOLD = 0.9
_LOGGER = logging.getLogger(__name__)


def _memory_auto_apply_enabled() -> bool:
    """Whether high-confidence reflections may write memory without approval."""

    raw = str(os.getenv("LANGCODE_MEMORY_AUTO_APPLY", "1")).strip().lower()
    return raw in {"1", "true", "yes", "on", "y"}


def _memory_auto_apply_threshold() -> float:
    try:
        value = float(os.getenv("LANGCODE_MEMORY_AUTO_APPLY_THRESHOLD", str(DEFAULT_MEMORY_AUTO_APPLY_THRESHOLD)))
    except (TypeError, ValueError):
        return DEFAULT_MEMORY_AUTO_APPLY_THRESHOLD
    return min(1.0, max(0.0, value))


@dataclass
class EvolutionCandidate:
    kind: str
    target: str
    content: str
    confidence: float
    reason: str
    apply: bool = False
    name: str = ""
    description: str = ""


def self_evolution_tool(workspace_root: str | Path, action: str, tool_input: dict) -> dict:
    root = Path(workspace_root).expanduser().resolve()
    normalized_action = str(action or "status").strip().lower()
    if normalized_action == "status":
        return evolution_status(root)
    if normalized_action in {"reflect", "reflect_session"}:
        return reflect_session(
            root,
            session_id=str(tool_input.get("session_id") or tool_input.get("_current_session_id") or "manual"),
            messages=tool_input.get("messages") if isinstance(tool_input.get("messages"), list) else [],
            todos=tool_input.get("todos") if isinstance(tool_input.get("todos"), list) else [],
            apply=bool(tool_input.get("apply", True)),
        )
    if normalized_action in {"list_reflections", "reflections"}:
        return list_reflections(root, limit=int(tool_input.get("limit", 20)))
    if normalized_action in {"propose", "proposal"}:
        return write_proposal(
            root,
            title=str(tool_input.get("title") or "自进化提案"),
            target=str(tool_input.get("target") or "skill"),
            content=str(tool_input.get("content") or ""),
        )
    if normalized_action in {"list_proposals", "proposals"}:
        return list_proposals(root)
    if normalized_action in {"read_soul", "soul"}:
        return soul_tool(root, "read")
    if normalized_action in {"update_soul", "write_soul"}:
        return soul_tool(root, "write", content=str(tool_input.get("content") or ""))
    return {"ok": False, "error": f"未知 self_evolve 操作：{action}"}


def evolution_status(workspace_root: str | Path) -> dict:
    root = Path(workspace_root).expanduser().resolve()
    evolution_dir = root / EVOLUTION_DIR
    reflections = _count_jsonl(evolution_dir / REFLECTIONS_FILE)
    proposals = list((evolution_dir / PROPOSALS_DIR).glob("*.md")) if (evolution_dir / PROPOSALS_DIR).exists() else []
    return {
        "ok": True,
        "layout": {
            "soul": ".langcode/SOUL.md",
            "memory": ".langcode/memories/MEMORY.md",
            "user": ".langcode/memories/USER.md",
            "skills": ".langcode/skills/",
            "reflections": f"{EVOLUTION_DIR}/{REFLECTIONS_FILE}",
            "proposals": f"{EVOLUTION_DIR}/{PROPOSALS_DIR}/",
        },
        "reflections": reflections,
        "proposals": len(proposals),
        "capabilities": [
            "热记忆注入",
            "会话 FTS 检索",
            "技能读写",
            "会话反思归档",
            "高置信偏好/经验自动写入",
            "低置信改进提案留档",
        ],
    }


def reflect_session(
    workspace_root: str | Path,
    *,
    session_id: str,
    messages: Iterable[Any],
    todos: list[dict] | None = None,
    apply: bool = True,
) -> dict:
    root = Path(workspace_root).expanduser().resolve()
    normalized = [_normalize_message(item) for item in messages]
    candidates = _build_candidates(normalized, todos or [])
    if not candidates:
        return {
            "ok": True,
            "session_id": session_id,
            "created_at": _now(),
            "message_count": len(normalized),
            "todo_count": len(todos or []),
            "applied": [],
            "staged": [],
            "skipped": "no_candidates",
        }
    applied: list[dict] = []
    staged: list[dict] = []
    auto_apply_enabled = _memory_auto_apply_enabled()
    threshold = _memory_auto_apply_threshold()
    for candidate in candidates:
        should_apply = (
            apply
            and auto_apply_enabled
            and candidate.apply
            and candidate.confidence >= threshold
        )
        if should_apply:
            result = _apply_candidate(root, candidate)
            item = {**asdict(candidate), "result": result}
            _LOGGER.info(
                "自进化自动应用记忆候选：session=%s kind=%s target=%s confidence=%.2f ok=%s",
                session_id,
                candidate.kind,
                candidate.target,
                candidate.confidence,
                result.get("ok") if isinstance(result, dict) else None,
            )
            applied.append(item)
        else:
            staged.append(asdict(candidate))
    record = {
        "session_id": session_id,
        "created_at": _now(),
        "message_count": len(normalized),
        "todo_count": len(todos or []),
        "applied": applied,
        "staged": staged,
    }
    _append_jsonl(root / EVOLUTION_DIR / REFLECTIONS_FILE, record)
    return {"ok": True, **record}


def list_reflections(workspace_root: str | Path, *, limit: int = 20) -> dict:
    path = Path(workspace_root).expanduser().resolve() / EVOLUTION_DIR / REFLECTIONS_FILE
    rows = _read_jsonl(path)
    return {"ok": True, "reflections": rows[-max(1, min(limit, 100)) :]}


def write_proposal(workspace_root: str | Path, *, title: str, target: str, content: str) -> dict:
    text = content.strip()
    if not text:
        return {"ok": False, "error": "自进化提案内容不能为空"}
    root = Path(workspace_root).expanduser().resolve()
    slug = _slug(title) or "proposal"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / EVOLUTION_DIR / PROPOSALS_DIR / f"{stamp}-{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        [
            f"# {title.strip()}",
            "",
            f"- 目标：{target.strip() or '未指定'}",
            f"- 创建时间：{_now()}",
            "",
            "## 提案内容",
            "",
            text,
            "",
            "## 审核原则",
            "",
            "- 需要通过测试或人工确认后才能改写提示词、工具描述或代码。",
            "- 禁止自动写入高风险代码变更；只能生成可审计提案。",
        ]
    )
    path.write_text(body, encoding="utf-8")
    return {"ok": True, "path": str(path.relative_to(root)), "title": title, "target": target}


def list_proposals(workspace_root: str | Path) -> dict:
    root = Path(workspace_root).expanduser().resolve()
    proposals_dir = root / EVOLUTION_DIR / PROPOSALS_DIR
    rows = []
    if proposals_dir.exists():
        for path in sorted(proposals_dir.glob("*.md"), reverse=True):
            rows.append({"path": str(path.relative_to(root)), "title": _first_heading(path)})
    return {"ok": True, "proposals": rows}


def _build_candidates(messages: list[dict], todos: list[dict]) -> list[EvolutionCandidate]:
    candidates: list[EvolutionCandidate] = []
    human_texts = [item["content"] for item in messages if item["role"] == "user"]
    assistant_texts = [item["content"] for item in messages if item["role"] == "assistant"]
    tool_texts = [item["content"] for item in messages if item["role"] == "tool"]
    last_user = human_texts[-1] if human_texts else ""
    last_assistant = assistant_texts[-1] if assistant_texts else ""

    for text in human_texts[-4:]:
        preference = _extract_preference(text)
        if preference:
            candidates.append(
                EvolutionCandidate(
                    kind="memory",
                    target="user",
                    content=f"用户偏好：{preference}",
                    confidence=0.9,
                    reason="用户使用了明确的长期偏好/记住类表达。",
                    apply=True,
                )
            )

    errors = [text for text in tool_texts if re.search(r"error|exception|traceback|报错|失败", text, re.I)]
    if errors and re.search(r"修复|解决|成功|完成|已", last_assistant):
        candidates.append(
            EvolutionCandidate(
                kind="memory",
                target="memory",
                content=f"经验：会话中遇到工具/运行错误后已恢复。下次类似任务先定位真实错误，再做最小修复。最近用户目标：{_clip(last_user, 120)}",
                confidence=0.84,
                reason="检测到工具错误后出现成功/完成类总结。",
                apply=True,
            )
        )

    completed = [todo for todo in todos if str(todo.get("status")) == "completed"]
    if len(completed) >= 2 and last_user and last_assistant:
        name = _slug(last_user)[:48] or "learned-workflow"
        description = f"复用本会话形成的工作流：{_clip(last_user, 80)}"
        content = _skill_from_session(last_user, completed, last_assistant)
        auto_apply = len(completed) >= 3 and _auto_skill_enabled()
        candidates.append(
            EvolutionCandidate(
                kind="skill",
                target="project",
                name=name,
                description=description,
                content=content,
                # `auto_apply` is already an explicit opt-in gate
                # (LANGCODE_AUTO_SKILL + >= 3 completed todos), so it clears the
                # raised auto-apply bar; without that gate it stays a candidate.
                confidence=0.9 if auto_apply else 0.78,
                reason=(
                    "任务清单完成项达到自动沉淀阈值，写入项目 skill。"
                    if auto_apply
                    else "任务清单有多项完成，适合沉淀为候选 skill；默认先归档候选，避免污染技能库。"
                ),
                apply=auto_apply,
            )
        )
    return _dedupe_candidates(candidates)


def _apply_candidate(workspace_root: Path, candidate: EvolutionCandidate) -> dict:
    if candidate.kind == "memory":
        return memory_tool(workspace_root, "add", target=candidate.target, content=candidate.content)
    if candidate.kind == "skill":
        return skill_tool(
            workspace_root,
            "upsert",
            name=candidate.name,
            description=candidate.description,
            content=candidate.content,
            scope=candidate.target or "project",
        )
    return {"ok": False, "error": f"未知候选类型：{candidate.kind}"}


def _normalize_message(item: Any) -> dict:
    if isinstance(item, HumanMessage):
        return {"role": "user", "content": str(item.content or "")}
    if isinstance(item, AIMessage):
        return {"role": "assistant", "content": str(item.content or "")}
    if isinstance(item, ToolMessage):
        return {"role": "tool", "content": str(item.content or "")}
    if isinstance(item, BaseMessage):
        return {"role": item.type, "content": str(item.content or "")}
    if isinstance(item, dict):
        role = str(item.get("role") or "")
        if role == "human":
            role = "user"
        if role == "ai":
            role = "assistant"
        return {"role": role, "content": str(item.get("content") or "")}
    return {"role": "unknown", "content": str(item or "")}


def _extract_preference(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) > 260:
        return ""
    if not re.search(r"以后|之后|记住|偏好|我希望|默认|每次|不要再|必须|总是", cleaned):
        return ""
    if cleaned.endswith("?") or cleaned.endswith("？"):
        return ""
    return cleaned


def _skill_from_session(user_goal: str, completed: list[dict], final_text: str) -> str:
    steps = "\n".join(f"{index}. {str(todo.get('content') or '').strip()}" for index, todo in enumerate(completed, 1))
    return (
        "## 适用场景\n\n"
        f"当用户提出类似任务时使用：{_clip(user_goal, 160)}\n\n"
        "## 执行步骤\n\n"
        f"{steps or '1. 先澄清目标，再按项目现有约定执行。'}\n\n"
        "## 验证方式\n\n"
        "- 运行与变更范围匹配的最小测试。\n"
        "- 检查用户可见行为是否符合原始目标。\n\n"
        "## 注意事项\n\n"
        f"- 本 skill 由会话反思候选生成，首次复用前应人工检查。\n"
        f"- 上次总结：{_clip(final_text, 240)}"
    )


def _dedupe_candidates(candidates: list[EvolutionCandidate]) -> list[EvolutionCandidate]:
    seen: set[tuple[str, str, str]] = set()
    result: list[EvolutionCandidate] = []
    for candidate in candidates:
        key = (candidate.kind, candidate.target, candidate.content)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _count_jsonl(path: Path) -> int:
    return len(_read_jsonl(path))


def _first_heading(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line.removeprefix("# ").strip()
    except OSError:
        pass
    return path.stem


def _slug(text: str) -> str:
    ascii_text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", text.strip().lower()).strip("-")
    if not ascii_text:
        return ""
    if re.search(r"[\u4e00-\u9fff]", ascii_text):
        return "workflow-" + hashlib_fallback(ascii_text)
    return ascii_text[:80]


def hashlib_fallback(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _clip(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "..."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _auto_skill_enabled() -> bool:
    raw = os.getenv("LANGCODE_AUTO_SKILL_EVOLUTION", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}
