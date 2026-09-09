from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import json
import re
import threading
from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


PROJECT_CONTEXT_FILES = (
    "CLAUDE.md",
    "AGENTS.md",
    "AGENT_MEMORY.md",
)

HERMES_SOUL_FILE = ".langcode/SOUL.md"
HERMES_MEMORY_DIR = ".langcode/memories"
HERMES_MEMORY_FILES = {"memory": "MEMORY.md", "user": "USER.md"}
PROJECT_SKILLS_DIR = ".langcode/skills"
GLOBAL_SKILLS_DIR = "~/.hermes/skills"
ENTRY_DELIMITER = "\n§\n"
MEMORY_CHAR_LIMIT = 2200
USER_CHAR_LIMIT = 1375
DEFAULT_MEMORY_MAX_CHARS = MEMORY_CHAR_LIMIT + USER_CHAR_LIMIT + 1200
DEFAULT_SOUL = (
    "你是 LangCode，一个长期运行的中文代码 Agent。"
    "你谨慎、务实、可审批、可记忆，优先帮助用户在当前工作区读代码、写代码、运行命令、搜索资料、沉淀经验。"
    "你不会训练或修改底层模型权重；你的自进化来自记忆、技能、会话归档、定时任务和可审计的改进提案。"
)
_MEMORY_LOCK = threading.RLock()
_SKILL_LOCK = threading.RLock()
_CACHE_LOCK = threading.RLock()
_PROJECT_CONTEXT_CACHE: dict[tuple[str, int], tuple[tuple, str]] = {}
_SKILL_CATALOG_CACHE: dict[str, tuple[tuple, list[dict]]] = {}
_ENSURED_MEMORY_ROOTS: set[str] = set()
_INVISIBLE_CHARS = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e"}
_MEMORY_THREAT_PATTERNS = [
    (r"ignore\s+(previous|all|above|prior)\s+instructions", "prompt_injection"),
    (r"you\s+are\s+now\s+", "role_hijack"),
    (r"do\s+not\s+tell\s+the\s+user", "deception_hide"),
    (r"system\s+prompt\s+override", "sys_prompt_override"),
    (r"disregard\s+(your|all|any)\s+(instructions|rules|guidelines)", "disregard_rules"),
    (r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_curl"),
    (r"wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_wget"),
    (r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)", "read_secrets"),
    (r"authorized_keys", "ssh_backdoor"),
    (r"\$HOME/\.ssh|\~/\.ssh", "ssh_access"),
]


def load_project_context(workspace_root: str | Path, *, max_chars: int = DEFAULT_MEMORY_MAX_CHARS) -> str:
    """Render the project system-context block, cached per workspace.

    The block goes into every system prompt, so it is memoized and only rebuilt
    when one of the contributing files (SOUL, memories, CLAUDE/AGENTS docs) or
    the skills directories changes, keyed by mtime + size.
    """

    root = Path(workspace_root).expanduser().resolve()
    ensure_hermes_memory_files(root)
    cache_key = (str(root), int(max_chars))
    fingerprint = _project_context_fingerprint(root)
    with _CACHE_LOCK:
        cached = _PROJECT_CONTEXT_CACHE.get(cache_key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
    rendered = _build_project_context(root, max_chars)
    with _CACHE_LOCK:
        _PROJECT_CONTEXT_CACHE[cache_key] = (fingerprint, rendered)
    return rendered


def _build_project_context(root: Path, max_chars: int) -> str:
    sections: list[str] = []
    remaining = max_chars
    soul = load_soul(root)
    if soul:
        clipped = soul[:remaining]
        sections.append(f"## SOUL.md（长期身份）\n{clipped}")
        remaining -= len(clipped)
    memory_snapshot = render_hermes_memory_snapshot(root)
    if memory_snapshot:
        clipped = memory_snapshot[:remaining]
        sections.append(clipped)
        remaining -= len(clipped)
    for relative_path in PROJECT_CONTEXT_FILES:
        if remaining <= 0:
            break
        path = root / relative_path
        if not path.exists() or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not content:
            continue
        clipped = content[:remaining]
        sections.append(f"## {relative_path}\n{clipped}")
        remaining -= len(clipped)
    skills = load_skill_catalog(root)
    if skills and remaining > 0:
        content = "可按需参考以下项目技能：\n" + "\n".join(
            f"- {item['name']}: {item['description']} ({item['path']})" for item in skills
        )
        clipped = content[:remaining]
        sections.append(f"## .langcode/skills\n{clipped}")
    return "\n\n".join(sections)


def load_skill_catalog(workspace_root: str | Path) -> list[dict]:
    root = Path(workspace_root).expanduser().resolve()
    fingerprint = _skill_catalog_fingerprint(root)
    with _CACHE_LOCK:
        cached = _SKILL_CATALOG_CACHE.get(str(root))
        if cached is not None and cached[0] == fingerprint:
            return [dict(item) for item in cached[1]]
    items = _build_skill_catalog(root)
    with _CACHE_LOCK:
        _SKILL_CATALOG_CACHE[str(root)] = (fingerprint, items)
    return [dict(item) for item in items]


def _build_skill_catalog(root: Path) -> list[dict]:
    items: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for scope, skills_root in _skill_roots(root):
        if not skills_root.exists() or not skills_root.is_dir():
            continue
        for skill_path in sorted(skills_root.glob("*/SKILL.md")):
            metadata = _parse_skill_frontmatter(skill_path)
            name = metadata.get("name") or skill_path.parent.name
            description = metadata.get("description") or "项目技能"
            key = (scope, name)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "name": name,
                    "description": description,
                    "scope": scope,
                    "path": _relative_skill_path(skill_path, root, scope),
                }
            )
    return items


def handle_local_command(
    workspace_root: str | Path,
    messages: list[BaseMessage],
    user_text: str,
    *,
    archive_path: Path | None = None,
) -> str | None:
    stripped = user_text.strip()
    if not stripped:
        return None

    if stripped.startswith("/compact"):
        instructions = stripped.removeprefix("/compact").strip()
        messages.append(HumanMessage(content=user_text))
        original_messages = list(messages)
        compacted, summary = compact_messages(messages, instructions=instructions)
        if archive_path is not None:
            archive_messages(archive_path, original_messages, summary=summary, instructions=instructions)
        messages[:] = compacted
        reply = (
            "已压缩当前会话上下文。较早消息已汇总为一条系统摘要，最近消息会继续保留；"
            "完整压缩归档已写入本地状态目录。"
        )
        messages.append(AIMessage(content=reply))
        return reply

    if stripped.startswith("#"):
        body = stripped.lstrip("#").strip()
        if not body:
            return None
        messages.append(HumanMessage(content=user_text))
        path = write_memory_note(workspace_root, "用户记忆", body)
        reply = f"已写入项目记忆：{path.relative_to(Path(workspace_root).expanduser().resolve())}"
        messages.append(AIMessage(content=reply))
        return reply

    if stripped == "/memory":
        messages.append(HumanMessage(content=user_text))
        context = load_project_context(workspace_root)
        reply = context or "当前没有可加载的项目记忆文件。"
        messages.append(AIMessage(content=reply))
        return reply

    if stripped == "/soul":
        messages.append(HumanMessage(content=user_text))
        soul = load_soul(workspace_root)
        reply = soul or "当前没有 SOUL 身份文件。"
        messages.append(AIMessage(content=reply))
        return reply

    if stripped == "/evolve":
        from .evolution import evolution_status

        messages.append(HumanMessage(content=user_text))
        status = evolution_status(workspace_root)
        reply = (
            "自进化状态：\n"
            f"- SOUL：{status['layout']['soul']}\n"
            f"- 热记忆：{status['layout']['memory']} / {status['layout']['user']}\n"
            f"- 技能目录：{status['layout']['skills']}\n"
            f"- 反思归档：{status['reflections']} 条\n"
            f"- 改进提案：{status['proposals']} 条"
        )
        messages.append(AIMessage(content=reply))
        return reply

    if stripped == "/cron":
        from .cron import cron_tool

        messages.append(HumanMessage(content=user_text))
        result = cron_tool(workspace_root, "list", {})
        jobs = result.get("jobs") if isinstance(result, dict) else []
        if not jobs:
            reply = "当前没有本地定时任务。"
        else:
            reply = "本地定时任务：\n" + "\n".join(
                f"- {job.get('name')}（{job.get('id')}）：{job.get('schedule')}，{job.get('status')}，下次 {job.get('next_run_at')}"
                for job in jobs
            )
        messages.append(AIMessage(content=reply))
        return reply

    if stripped == "/agents":
        messages.append(HumanMessage(content=user_text))
        reply = (
            "当前可用 Agent 能力：\n"
            "- 默认：单主 Agent 直接处理。\n"
            "- 辅助：`delegate_agent` 调用 researcher/reviewer/planner/verifier 中的一个短上下文子 Agent。\n"
            "- 多视角：`delegate_agents` 并行调用多个只读子 Agent 后由主 Agent 汇总。\n"
            "- 辩论/博弈：`agent_debate` 由 Debate Manager 维护 transcript，A/B/Judge 轮流发言。\n\n"
            "内置子 Agent：researcher、reviewer、planner、verifier。"
        )
        messages.append(AIMessage(content=reply))
        return reply

    if stripped == "/skills":
        messages.append(HumanMessage(content=user_text))
        skills = load_skill_catalog(workspace_root)
        if not skills:
            reply = "当前没有项目技能或全局 Hermes 技能。"
        else:
            reply = "项目技能：\n" + "\n".join(
                f"- {item['name']}：{item['description']}（{item['scope']}，{item['path']}）" for item in skills
            )
        messages.append(AIMessage(content=reply))
        return reply

    return None


def write_memory_note(workspace_root: str | Path, title: str, body: str) -> Path:
    root = Path(workspace_root).expanduser().resolve()
    memory_tool(root, "add", target="memory", content=f"{title.strip() or '用户记忆'}：{body.strip()}")
    return hermes_memory_path(root, "memory")


def hermes_memory_path(workspace_root: str | Path, target: str = "memory") -> Path:
    root = Path(workspace_root).expanduser().resolve()
    normalized = str(target or "memory").strip().lower()
    if normalized not in HERMES_MEMORY_FILES:
        raise ValueError("记忆目标必须是 memory 或 user")
    memory_dir = root / HERMES_MEMORY_DIR
    memory_dir.mkdir(parents=True, exist_ok=True)
    return memory_dir / HERMES_MEMORY_FILES[normalized]


def hermes_soul_path(workspace_root: str | Path) -> Path:
    root = Path(workspace_root).expanduser().resolve()
    path = root / HERMES_SOUL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_soul(workspace_root: str | Path) -> str:
    ensure_hermes_memory_files(workspace_root)
    path = hermes_soul_path(workspace_root)
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def soul_tool(workspace_root: str | Path, action: str, *, content: str = "") -> dict:
    normalized_action = str(action or "read").strip().lower()
    ensure_hermes_memory_files(workspace_root)
    path = hermes_soul_path(workspace_root)
    if normalized_action in {"read", "view", "list"}:
        return {
            "ok": True,
            "action": "read",
            "path": _relative_memory_path(path, workspace_root),
            "content": path.read_text(encoding="utf-8") if path.exists() else "",
        }
    if normalized_action in {"write", "replace", "update"}:
        text = str(content or "").strip()
        if not text:
            return {"ok": False, "error": "SOUL 内容不能为空"}
        scan_error = _scan_memory_content(text)
        if scan_error:
            return {"ok": False, "error": scan_error}
        with _MEMORY_LOCK:
            path.write_text(text, encoding="utf-8")
        return {
            "ok": True,
            "action": "write",
            "path": _relative_memory_path(path, workspace_root),
            "content": text,
        }
    return {"ok": False, "error": f"未知 soul 操作：{action}"}


def memory_tool(workspace_root: str | Path, action: str, *, target: str = "memory", content: str = "", old: str = "") -> dict:
    """Hermes-style bounded markdown memory tool.

    memory = project facts / lessons. user = stable user preferences.
    """

    normalized_action = str(action or "").strip().lower()
    if normalized_action == "append":
        normalized_action = "add"
    if normalized_action == "delete":
        normalized_action = "remove"
    path = hermes_memory_path(workspace_root, target)
    target = str(target or "memory").strip().lower()
    limit = USER_CHAR_LIMIT if target == "user" else MEMORY_CHAR_LIMIT

    with _MEMORY_LOCK:
        entries = _read_memory_entries(path)

    if normalized_action in {"read", "view", "list"}:
        return _memory_response(True, target, path, workspace_root, entries, "当前记忆状态。")

    if normalized_action == "add":
        text = str(content or "").strip()
        if not text:
            return {"ok": False, "error": "追加记忆内容不能为空"}
        scan_error = _scan_memory_content(text)
        if scan_error:
            return {"ok": False, "error": scan_error}
        with _MEMORY_LOCK:
            entries = _read_memory_entries(path)
            if text in entries:
                return _memory_response(True, target, path, workspace_root, entries, "Entry already exists (no duplicate added).")
            candidate = entries + [text]
            if _entries_char_count(candidate) > limit:
                return _memory_response(
                    False,
                    target,
                    path,
                    workspace_root,
                    entries,
                    f"Memory at {_entries_char_count(entries):,}/{limit:,} chars. Adding this entry ({len(text)} chars) would exceed the limit. Replace or remove existing entries first.",
                )
            _write_memory_entries(path, candidate)
        return _memory_response(True, target, path, workspace_root, candidate, "Entry added.")

    if normalized_action == "replace":
        old_text = str(old or "")
        new_text = str(content or "")
        if not old_text:
            return {"ok": False, "error": "replace 操作必须提供 old"}
        if not new_text:
            return {"ok": False, "error": "replace 操作的新内容不能为空；如需删除请使用 remove"}
        scan_error = _scan_memory_content(new_text)
        if scan_error:
            return {"ok": False, "error": scan_error}
        with _MEMORY_LOCK:
            entries = _read_memory_entries(path)
            matches = [(index, entry) for index, entry in enumerate(entries) if old_text in entry]
            if not matches:
                return _memory_response(False, target, path, workspace_root, entries, f"No entry matched '{old_text}'.")
            if len({entry for _index, entry in matches}) > 1:
                return _memory_response(False, target, path, workspace_root, entries, f"Multiple entries matched '{old_text}'. Be more specific.")
            index = matches[0][0]
            candidate = list(entries)
            candidate[index] = new_text
            if _entries_char_count(candidate) > limit:
                return _memory_response(False, target, path, workspace_root, entries, f"Replacement would exceed {limit:,} chars.")
            _write_memory_entries(path, candidate)
        return _memory_response(True, target, path, workspace_root, candidate, "Entry replaced.")

    if normalized_action == "remove":
        old_text = str(old or content or "")
        if not old_text:
            return {"ok": False, "error": "remove 操作必须提供 old 或 content"}
        with _MEMORY_LOCK:
            entries = _read_memory_entries(path)
            matches = [(index, entry) for index, entry in enumerate(entries) if old_text in entry]
            if not matches:
                return _memory_response(False, target, path, workspace_root, entries, f"No entry matched '{old_text}'.")
            if len({entry for _index, entry in matches}) > 1:
                return _memory_response(False, target, path, workspace_root, entries, f"Multiple entries matched '{old_text}'. Be more specific.")
            candidate = [entry for index, entry in enumerate(entries) if index != matches[0][0]]
            _write_memory_entries(path, candidate)
        return _memory_response(True, target, path, workspace_root, candidate, "Entry removed.")

    return {"ok": False, "error": f"未知 memory 操作：{action}"}


def skill_tool(
    workspace_root: str | Path,
    action: str,
    *,
    name: str = "",
    description: str = "",
    content: str = "",
    scope: str = "project",
) -> dict:
    """管理 Hermes 风格技能记忆，只允许读写固定技能目录中的 SKILL.md。"""

    root = Path(workspace_root).expanduser().resolve()
    normalized_action = str(action or "list").strip().lower()
    normalized_scope = _normalize_skill_scope(scope)

    if normalized_action == "list":
        return {"ok": True, "action": "list", "skills": load_skill_catalog(root)}

    if normalized_action in {"read", "get"}:
        slug = _skill_slug(name)
        if not slug:
            return {"ok": False, "error": "读取技能必须提供 name"}
        matches = _find_skill_paths(root, slug)
        if not matches:
            return {"ok": False, "error": f"未找到技能：{name}"}
        scope_name, path = matches[0]
        return {
            "ok": True,
            "action": "read",
            "scope": scope_name,
            "name": slug,
            "path": _relative_skill_path(path, root, scope_name),
            "content": path.read_text(encoding="utf-8"),
        }

    if normalized_action in {"upsert", "create", "update"}:
        slug = _skill_slug(name)
        if not slug:
            return {"ok": False, "error": "沉淀技能必须提供 name"}
        summary = str(description or "").strip()
        if not summary:
            return {"ok": False, "error": "沉淀技能必须提供 description"}
        body = str(content or "").strip()
        if not body:
            return {"ok": False, "error": "沉淀技能必须提供 content"}
        scan_error = _scan_skill_content(body)
        if scan_error:
            return {"ok": False, "error": scan_error}
        skill_path = _skill_root(root, normalized_scope) / slug / "SKILL.md"
        rendered = _render_skill_markdown(slug, summary, body)
        with _SKILL_LOCK:
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text(rendered, encoding="utf-8")
        return {
            "ok": True,
            "action": "upsert",
            "scope": normalized_scope,
            "name": slug,
            "path": _relative_skill_path(skill_path, root, normalized_scope),
            "message": "技能已沉淀。",
        }

    if normalized_action in {"delete", "remove"}:
        slug = _skill_slug(name)
        if not slug:
            return {"ok": False, "error": "删除技能必须提供 name"}
        skill_path = _skill_root(root, normalized_scope) / slug / "SKILL.md"
        if not skill_path.exists():
            return {"ok": False, "error": f"未找到技能：{name}"}
        with _SKILL_LOCK:
            skill_path.unlink()
            try:
                skill_path.parent.rmdir()
            except OSError:
                pass
        return {
            "ok": True,
            "action": "remove",
            "scope": normalized_scope,
            "name": slug,
            "path": _relative_skill_path(skill_path, root, normalized_scope),
            "message": "技能已删除。",
        }

    return {"ok": False, "error": f"未知 skill 操作：{action}"}


def render_hermes_memory_snapshot(workspace_root: str | Path) -> str:
    ensure_hermes_memory_files(workspace_root)
    blocks = []
    for target, title, limit in (
        ("memory", "MEMORY（Agent 个人笔记）", MEMORY_CHAR_LIMIT),
        ("user", "USER PROFILE（用户画像）", USER_CHAR_LIMIT),
    ):
        path = hermes_memory_path(workspace_root, target)
        entries = _read_memory_entries(path)
        if not entries:
            continue
        used = _entries_char_count(entries)
        percent = int(round((used / limit) * 100)) if limit else 0
        blocks.append(
            "\n".join(
                [
                    "══════════════════════════════════════════════",
                    f"{title} [{percent}% — {used:,}/{limit:,} chars]",
                    "══════════════════════════════════════════════",
                    ENTRY_DELIMITER.join(entries),
                ]
            )
        )
    return "\n\n".join(blocks)


def ensure_hermes_memory_files(workspace_root: str | Path) -> None:
    """Create the Hermes memory scaffolding once per (workspace, process).

    This runs on every system-prompt build; the mkdir/exists syscalls are pure
    overhead after the first call, so the result is memoized. Writers still call
    ``hermes_memory_path`` / ``hermes_soul_path``, which mkdir on their own.
    """

    root = Path(workspace_root).expanduser().resolve()
    key = str(root)
    with _CACHE_LOCK:
        if key in _ENSURED_MEMORY_ROOTS:
            return

    soul_path = root / HERMES_SOUL_FILE
    soul_path.parent.mkdir(parents=True, exist_ok=True)
    if not soul_path.exists():
        soul_path.write_text(DEFAULT_SOUL, encoding="utf-8")
    memory_dir = root / HERMES_MEMORY_DIR
    memory_dir.mkdir(parents=True, exist_ok=True)
    for filename in HERMES_MEMORY_FILES.values():
        path = memory_dir / filename
        if not path.exists():
            path.write_text("", encoding="utf-8")

    with _CACHE_LOCK:
        _ENSURED_MEMORY_ROOTS.add(key)


def _stat_signature(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (0, -1)
    return (stat.st_mtime_ns, stat.st_size)


def _project_context_fingerprint(root: Path) -> tuple:
    parts: list = []
    relatives = [
        HERMES_SOUL_FILE,
        *(f"{HERMES_MEMORY_DIR}/{filename}" for filename in HERMES_MEMORY_FILES.values()),
        *PROJECT_CONTEXT_FILES,
    ]
    for relative in relatives:
        parts.append((relative, _stat_signature(root / relative)))
    parts.append(_skill_catalog_fingerprint(root))
    return tuple(parts)


def _skill_catalog_fingerprint(root: Path) -> tuple:
    parts: list = []
    for scope, skills_root in _skill_roots(root):
        try:
            with os.scandir(skills_root) as entries:
                names = sorted(entry.name for entry in entries if entry.is_dir())
        except OSError:
            parts.append((scope, str(skills_root), None))
            continue
        signatures = tuple((name, _stat_signature(skills_root / name / "SKILL.md")) for name in names)
        parts.append((scope, str(skills_root), signatures))
    return tuple(parts)


def reset_project_context_cache() -> None:
    """Drop every memoized system-context artifact (used by tests)."""

    with _CACHE_LOCK:
        _PROJECT_CONTEXT_CACHE.clear()
        _SKILL_CATALOG_CACHE.clear()
        _ENSURED_MEMORY_ROOTS.clear()


def compact_messages(
    messages: list[BaseMessage],
    *,
    keep_recent: int = 8,
    instructions: str = "",
) -> tuple[list[BaseMessage], str]:
    system_messages = [message for message in messages if isinstance(message, SystemMessage)]
    non_system = [message for message in messages if not isinstance(message, SystemMessage)]
    recent = non_system[-keep_recent:]
    older = non_system[:-keep_recent]
    summary = _summarize_messages(older, recent, instructions=instructions)
    compacted: list[BaseMessage] = []
    compacted.extend(system_messages[:1])
    compacted.append(SystemMessage(content=summary))
    compacted.extend(recent)
    compacted = _drop_orphan_tool_messages(compacted)
    return compacted, summary


def archive_messages(path: Path, messages: Iterable[BaseMessage], *, summary: str, instructions: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "instructions": instructions,
        "messages": [serialize_message(message) for message in messages],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def serialize_message(message: BaseMessage) -> dict:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": str(message.content)}
    if isinstance(message, HumanMessage):
        return {"role": "human", "content": str(message.content)}
    if isinstance(message, AIMessage):
        item = {"role": "ai", "content": str(message.content)}
        tool_calls = list(getattr(message, "tool_calls", None) or [])
        if tool_calls:
            item["tool_calls"] = tool_calls
        return item
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": str(message.content),
            "tool_call_id": getattr(message, "tool_call_id", "tool"),
        }
    return {"role": "unknown", "content": str(message.content)}


def _summarize_messages(older: list[BaseMessage], recent: list[BaseMessage], *, instructions: str) -> str:
    lines = [
        "LangCode context compact summary.",
        "",
        "保留原则：记录用户目标、已经完成的实现、关键约束、未完成事项、最近工具结果；省略寒暄和冗余输出。",
    ]
    if instructions.strip():
        lines.extend(["", f"用户压缩指令：{instructions.strip()}"])

    if older:
        lines.extend(["", "## 已压缩的较早上下文"])
        for message in older[-20:]:
            lines.append(f"- {_message_label(message)}: {_clip(str(message.content), 280)}")
    else:
        lines.extend(["", "## 已压缩的较早上下文", "- 无较早消息需要压缩。"])

    if recent:
        lines.extend(["", "## 保留的最近上下文摘要"])
        for message in recent:
            lines.append(f"- {_message_label(message)}: {_clip(str(message.content), 220)}")

    return "\n".join(lines)


def _message_label(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, ToolMessage):
        return "tool"
    if isinstance(message, SystemMessage):
        return "system"
    return message.type


def _drop_orphan_tool_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    filtered: list[BaseMessage] = []
    pending_tool_call_ids: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            filtered.append(message)
            pending_tool_call_ids = {
                str(tool_call.get("id"))
                for tool_call in list(getattr(message, "tool_calls", None) or [])
                if isinstance(tool_call, dict) and tool_call.get("id")
            }
            continue
        if isinstance(message, ToolMessage):
            tool_call_id = str(getattr(message, "tool_call_id", "") or "")
            if tool_call_id and tool_call_id in pending_tool_call_ids:
                filtered.append(message)
                pending_tool_call_ids.remove(tool_call_id)
            continue
        filtered.append(message)
        if not isinstance(message, SystemMessage):
            pending_tool_call_ids.clear()
    return filtered


def _clip(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "..."


def _read_memory_entries(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    return list(dict.fromkeys(entry.strip() for entry in text.split(ENTRY_DELIMITER) if entry.strip()))


def _write_memory_entries(path: Path, entries: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ENTRY_DELIMITER.join(entry.strip() for entry in entries if entry.strip()), encoding="utf-8")


def _entries_char_count(entries: list[str]) -> int:
    if not entries:
        return 0
    return len(ENTRY_DELIMITER.join(entries))


def _memory_response(ok: bool, target: str, path: Path, workspace_root: str | Path, entries: list[str], message: str) -> dict:
    limit = USER_CHAR_LIMIT if target == "user" else MEMORY_CHAR_LIMIT
    used = _entries_char_count(entries)
    return {
        "ok": ok,
        "success": ok,
        "target": target,
        "path": _relative_memory_path(path, workspace_root),
        "message": message,
        "entries": entries,
        "usage": f"{used:,}/{limit:,}",
        "remaining": max(0, limit - used),
        "content": ENTRY_DELIMITER.join(entries),
    }


def _scan_memory_content(content: str) -> str | None:
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Blocked: content contains invisible unicode character U+{ord(char):04X}."
    for pattern, pattern_id in _MEMORY_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return f"Blocked: content matches threat pattern '{pattern_id}'."
    return None


def _relative_memory_path(path: Path, workspace_root: str | Path) -> str:
    root = Path(workspace_root).expanduser().resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _parse_skill_frontmatter(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    metadata: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in {"name", "description"}:
            metadata[key] = value.strip().strip("\"'")
    return metadata


def _skill_roots(workspace_root: Path) -> list[tuple[str, Path]]:
    return [
        ("project", _skill_root(workspace_root, "project")),
        ("global", _skill_root(workspace_root, "global")),
    ]


def _skill_root(workspace_root: Path, scope: str) -> Path:
    if scope == "global":
        configured = os.getenv("LANGCODE_GLOBAL_SKILLS_DIR") or GLOBAL_SKILLS_DIR
        return Path(configured).expanduser().resolve()
    return workspace_root / PROJECT_SKILLS_DIR


def _normalize_skill_scope(scope: str) -> str:
    normalized = str(scope or "project").strip().lower()
    if normalized in {"global", "user"}:
        return "global"
    return "project"


def _find_skill_paths(workspace_root: Path, slug: str) -> list[tuple[str, Path]]:
    matches: list[tuple[str, Path]] = []
    for scope, root in _skill_roots(workspace_root):
        path = root / slug / "SKILL.md"
        if path.exists() and path.is_file():
            matches.append((scope, path))
    return matches


def _skill_slug(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(name or "").strip().lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-_")
    return normalized[:80]


def _render_skill_markdown(name: str, description: str, content: str) -> str:
    body = content.strip()
    if body.startswith("---"):
        return body + "\n"
    return "\n".join(
        [
            "---",
            f"name: {name}",
            f"description: {description.strip()}",
            "---",
            "",
            f"# {name}",
            "",
            body,
            "",
        ]
    )


def _scan_skill_content(content: str) -> str | None:
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Blocked: content contains invisible unicode character U+{ord(char):04X}."
    for pattern, pattern_id in _MEMORY_THREAT_PATTERNS[:5]:
        if re.search(pattern, content, re.IGNORECASE):
            return f"Blocked: skill content matches threat pattern '{pattern_id}'."
    return None


def _relative_skill_path(path: Path, workspace_root: Path, scope: str) -> str:
    if scope == "project":
        try:
            return str(path.relative_to(workspace_root))
        except ValueError:
            return str(path)
    return str(path)
