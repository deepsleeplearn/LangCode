from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import shlex
import subprocess

from ..core.paths import WorkspaceViolation, resolve_workspace_path
from .web_tools import web_fetch, web_search
from .diagram import diagram_tool
from ..memory.cron import cron_tool
from ..memory.evolution import self_evolution_tool
from ..memory.project import memory_tool, skill_tool, soul_tool
from ..storage.session_store import SessionStore


@dataclass(frozen=True)
class ShellResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def read_file(
    workspace_root: str | Path,
    path: str | Path,
    *,
    offset: int = 0,
    limit: int = 2000,
    allow_workspace_escape: bool = False,
) -> str:
    return _read_file_page(
        workspace_root,
        path,
        offset=offset,
        limit=limit,
        allow_workspace_escape=allow_workspace_escape,
    )["content"]


def _read_file_page(
    workspace_root: str | Path,
    path: str | Path,
    *,
    offset: int = 0,
    limit: int = 2000,
    allow_workspace_escape: bool = False,
) -> dict:
    target = resolve_workspace_path(workspace_root, path, allow_workspace_escape=allow_workspace_escape)
    lines = target.read_text(encoding="utf-8").splitlines()
    start = max(0, int(offset))
    stop = start + max(1, min(int(limit), 5000))
    shown = lines[start:stop]
    shown_start = start + 1 if shown else 0
    shown_end = start + len(shown)
    content = "\n".join(f"{index + 1}: {line}" for index, line in enumerate(shown, start=start))
    content += f"\n\n[共 {len(lines)} 行，已显示 {shown_start}-{shown_end} 行]"
    return {
        "content": content,
        "total_lines": len(lines),
        "shown_range": [shown_start, shown_end],
        "truncated": shown_end < len(lines),
    }


def write_file(
    workspace_root: str | Path,
    path: str | Path,
    content: str,
    *,
    allow_workspace_escape: bool = False,
) -> None:
    target = resolve_workspace_path(workspace_root, path, allow_workspace_escape=allow_workspace_escape)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def edit_file(
    workspace_root: str | Path,
    path: str | Path,
    old: str,
    new: str,
    *,
    replace_all: bool = False,
    allow_workspace_escape: bool = False,
) -> None:
    target = resolve_workspace_path(workspace_root, path, allow_workspace_escape=allow_workspace_escape)
    content = target.read_text(encoding="utf-8")
    if old not in content:
        raise ValueError(f"未在 {path} 中找到要替换的文本")
    count = -1 if replace_all else 1
    target.write_text(content.replace(old, new, count), encoding="utf-8")


def search(
    workspace_root: str | Path,
    query: str,
    path: str | Path = ".",
    *,
    max_results: int = 50,
    allow_workspace_escape: bool = False,
) -> list[dict]:
    target = resolve_workspace_path(workspace_root, path, allow_workspace_escape=allow_workspace_escape)
    if shutil.which("rg"):
        command = [
            "rg",
            "--line-number",
            "--no-heading",
            "--max-count",
            str(max_results),
            "--max-columns",
            "400",
            "--",
            query,
            str(target),
        ]
    else:
        command = ["grep", "-R", "-n", query, str(target)]

    completed = subprocess.run(
        command,
        cwd=Path(workspace_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(completed.stderr.strip() or "搜索命令执行失败")

    results: list[dict] = []
    for line in completed.stdout.splitlines()[:max_results]:
        file_name, line_number, text = _split_search_line(line)
        results.append({"path": file_name, "line": line_number, "text": text})
    return results


def shell(
    workspace_root: str | Path,
    command: str,
    *,
    timeout_seconds: int = 30,
    allow_workspace_escape: bool = False,
) -> ShellResult:
    root = Path(workspace_root).expanduser().resolve()
    if not allow_workspace_escape:
        _reject_shell_workspace_escapes(root, command)
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        return ShellResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        return ShellResult(
            exit_code=124,
            stdout=_output_text(exc.stdout),
            stderr=_output_text(exc.stderr) or f"命令执行超过 {timeout_seconds} 秒后超时",
            timed_out=True,
        )


def execute_tool(
    workspace_root: str | Path,
    tool_name: str,
    tool_input: dict,
    *,
    allow_workspace_escape: bool = False,
) -> dict:
    from ..runtime.backends import WorkspaceBackend

    backend = WorkspaceBackend(workspace_root)
    if tool_name == "read_file":
        return backend.read_file(
            tool_input["path"],
            offset=int(tool_input.get("offset", 0)),
            limit=int(tool_input.get("limit", 2000)),
            allow_workspace_escape=allow_workspace_escape,
        )
    if tool_name == "write_file":
        return backend.write_file(
            tool_input["path"],
            tool_input["content"],
            allow_workspace_escape=allow_workspace_escape,
        )
    if tool_name == "edit_file":
        return backend.edit_file(
            tool_input["path"],
            tool_input["old"],
            tool_input["new"],
            replace_all=bool(tool_input.get("replace_all", False)),
            allow_workspace_escape=allow_workspace_escape,
        )
    if tool_name == "search":
        return backend.search(
            tool_input["query"],
            tool_input.get("path", "."),
            max_results=int(tool_input.get("max_results", 50)),
            allow_workspace_escape=allow_workspace_escape,
        )
    if tool_name == "ls":
        return backend.ls(tool_input.get("path", "."), allow_workspace_escape=allow_workspace_escape)
    if tool_name == "glob":
        return backend.glob(
            tool_input["pattern"],
            tool_input.get("path", "."),
            max_results=int(tool_input.get("max_results", 100)),
        )
    if tool_name == "shell":
        return backend.shell(
            tool_input["command"],
            timeout_seconds=int(tool_input.get("timeout_seconds", 30)),
            allow_workspace_escape=allow_workspace_escape,
        )
    if tool_name == "sandbox_shell":
        return backend.sandbox_shell(
            tool_input["command"],
            timeout_seconds=int(tool_input.get("timeout_seconds", 30)),
            copy_workspace=bool(tool_input.get("copy_workspace", True)),
        )
    if tool_name in {"task_create", "task_update", "task_list", "task_get", "task_cancel"}:
        return {"ok": True, "operation": tool_name, "input": dict(tool_input)}
    if tool_name == "memory":
        return memory_tool(
            workspace_root,
            str(tool_input.get("action") or "read"),
            target=str(tool_input.get("target") or "memory"),
            content=str(tool_input.get("content") or ""),
            old=str(tool_input.get("old") or ""),
        )
    if tool_name == "soul":
        return soul_tool(
            workspace_root,
            str(tool_input.get("action") or "read"),
            content=str(tool_input.get("content") or ""),
        )
    if tool_name == "self_evolve":
        return self_evolution_tool(
            workspace_root,
            str(tool_input.get("action") or "status"),
            dict(tool_input),
        )
    if tool_name == "cron":
        return cron_tool(
            workspace_root,
            str(tool_input.get("action") or "list"),
            dict(tool_input),
        )
    if tool_name == "session_search":
        return session_search_tool(workspace_root, tool_input)
    if tool_name == "skill":
        return skill_tool(
            workspace_root,
            str(tool_input.get("action") or "list"),
            name=str(tool_input.get("name") or ""),
            description=str(tool_input.get("description") or ""),
            content=str(tool_input.get("content") or ""),
            scope=str(tool_input.get("scope") or "project"),
        )
    if tool_name == "diagram":
        return diagram_tool(workspace_root, tool_input)
    if tool_name == "delegate_agents":
        from ..runtime.multi_agent import run_parallel_delegate_agents

        return run_parallel_delegate_agents(workspace_root, **dict(tool_input))
    if tool_name == "agent_debate":
        from ..runtime.multi_agent import run_agent_debate

        return run_agent_debate(workspace_root, **dict(tool_input))
    if tool_name == "web_search":
        return {
            "ok": True,
            "results": web_search(
                tool_input["query"],
                max_results=int(tool_input.get("max_results", 5)),
                search_depth=str(tool_input.get("search_depth", "basic")),
                include_domains=_string_list(tool_input.get("include_domains")),
                exclude_domains=_string_list(tool_input.get("exclude_domains")),
                topic=str(tool_input.get("topic", "general")),
            ),
        }
    if tool_name == "web_fetch":
        return {
            "ok": True,
            "result": web_fetch(
                tool_input["url"],
                extract_depth=str(tool_input.get("extract_depth", "basic")),
                max_chars=int(tool_input.get("max_chars", 12000)),
            ),
        }
    return {"ok": False, "error": f"未知工具：{tool_name}"}


def session_search_tool(workspace_root: str | Path, tool_input: dict) -> dict:
    raw_store_path = str(tool_input.get("_session_store_path") or "").strip()
    if raw_store_path:
        store_path = Path(raw_store_path).expanduser()
    else:
        store_path = Path(workspace_root).expanduser().resolve() / ".langcode" / "web.sqlite"
    store = SessionStore(store_path)
    mode = str(tool_input.get("mode") or "search").strip().lower()
    limit = int(tool_input.get("limit", 8))
    if mode == "recent":
        return {"ok": True, "mode": "recent", "sessions": store.recent_sessions(limit=limit)}
    if mode == "around":
        session_id = str(tool_input.get("session_id") or "").strip()
        if not session_id:
            session_id = str(tool_input.get("_current_session_id") or "").strip()
        message_id = int(tool_input.get("message_id", 0))
        return {
            "ok": True,
            "mode": "around",
            "session_id": session_id,
            "messages": store.messages_around(
                session_id,
                message_id,
                before=int(tool_input.get("before", 3)),
                after=int(tool_input.get("after", 3)),
            ),
        }
    query = str(tool_input.get("query") or "").strip()
    return {
        "ok": True,
        "mode": "search",
        "results": store.search_messages(
            query,
            current_session_id=str(tool_input.get("_current_session_id") or ""),
            limit=limit,
        ),
    }


def detect_workspace_escape(workspace_root: str | Path, tool_name: str, tool_input: dict) -> dict | None:
    try:
        _detect_workspace_escape_or_raise(workspace_root, tool_name, tool_input)
    except WorkspaceViolation as exc:
        return {"dangerous": True, "reason": str(exc)}
    return None


def _detect_workspace_escape_or_raise(workspace_root: str | Path, tool_name: str, tool_input: dict) -> None:
    if tool_name in {"read_file", "write_file", "edit_file"}:
        resolve_workspace_path(workspace_root, tool_input["path"])
        return
    if tool_name in {"search", "ls", "glob"}:
        resolve_workspace_path(workspace_root, tool_input.get("path", "."))
        return
    if tool_name == "shell":
        _reject_shell_workspace_escapes(Path(workspace_root).expanduser().resolve(), str(tool_input.get("command", "")))
        return
    if tool_name in _MEMORY_TOOL_WRITE_ACTIONS:
        _reject_memory_tool_escapes(workspace_root, tool_name, tool_input)


_MEMORY_TOOL_WRITE_ACTIONS = {
    "skill": {"upsert", "create", "update", "delete", "remove"},
    "soul": {"write", "replace", "update"},
    "memory": {"add", "replace", "remove"},
}


def _reject_memory_tool_escapes(workspace_root: str | Path, tool_name: str, tool_input: dict) -> None:
    """Flag memory/soul/skill writes that land outside the workspace.

    ``skill(scope=global)`` writes to ``~/.hermes/skills``, which is outside the
    workspace; without this the approval prompt reported a generic risk instead
    of a workspace escape.
    """

    action = str(tool_input.get("action") or "").strip().lower()
    if action not in _MEMORY_TOOL_WRITE_ACTIONS[tool_name]:
        return
    for target in _memory_tool_write_targets(tool_name, workspace_root, tool_input):
        resolve_workspace_path(workspace_root, target)


def _memory_tool_write_targets(tool_name: str, workspace_root: str | Path, tool_input: dict) -> list[Path]:
    from ..memory.project import (
        HERMES_MEMORY_DIR,
        HERMES_MEMORY_FILES,
        HERMES_SOUL_FILE,
        _normalize_skill_scope,
        _skill_root,
        _skill_slug,
    )

    root = Path(workspace_root).expanduser().resolve()
    if tool_name == "soul":
        return [root / HERMES_SOUL_FILE]
    if tool_name == "memory":
        target = str(tool_input.get("target") or "memory").strip().lower()
        filename = HERMES_MEMORY_FILES.get(target)
        if filename is None:
            return []
        return [root / HERMES_MEMORY_DIR / filename]
    slug = _skill_slug(str(tool_input.get("name") or ""))
    if not slug:
        return []
    scope = _normalize_skill_scope(str(tool_input.get("scope") or "project"))
    return [_skill_root(root, scope) / slug / "SKILL.md"]


def _string_list(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    return [str(item) for item in value]


def _output_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _split_search_line(line: str) -> tuple[str, int, str]:
    parts = line.split(":", 2)
    if len(parts) != 3:
        return line, 0, ""
    file_name, line_number, text = parts
    try:
        parsed_line = int(line_number)
    except ValueError:
        parsed_line = 0
    return file_name, parsed_line, text


_SHELL_OPERATOR_TOKENS = {"&&", "||", "|", ";", ";;", "&", "(", ")", "<", "<<", ">", ">>"}


def _tokenize_shell_command(command: str) -> list[str]:
    """Tokenize a command with control operators as their own tokens.

    ``shlex.split`` keeps ``..;`` glued together, so ``cd ..; cat .env`` hid the
    parent-directory hop from the path scan below while ``cd .. && cat .env``
    was caught. ``punctuation_chars`` splits ``;``, ``&&``, ``|``, ``>`` and
    friends off, so both forms are checked identically.
    """

    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError as exc:
        raise WorkspaceViolation(f"无法安全解析 shell 命令：{exc}") from exc


def _reject_shell_workspace_escapes(workspace_root: Path, command: str) -> None:
    tokens = _tokenize_shell_command(command)

    _reject_nested_shell_escapes(workspace_root, tokens)
    _reject_redirect_escapes(workspace_root, tokens)

    for token in tokens:
        candidate = _path_from_shell_token(workspace_root, token)
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if resolved != workspace_root and workspace_root not in resolved.parents:
            raise WorkspaceViolation(f"Shell 命令引用了工作区外路径：{candidate}")


def _path_from_shell_token(workspace_root: Path, token: str) -> Path | None:
    if "$" in token:
        raise WorkspaceViolation(f"Shell 路径包含不可静态解析的环境变量：{token}")
    stripped = token
    while stripped and stripped[0] in "><":
        stripped = stripped[1:]
    if not stripped or stripped.startswith("-") or stripped in _SHELL_OPERATOR_TOKENS or "://" in stripped:
        return None
    candidate = Path(stripped).expanduser()
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    if "/" not in stripped and stripped not in {".", "..", "~"} and not candidate.exists():
        return None
    return candidate


def _reject_nested_shell_escapes(workspace_root: Path, tokens: list[str]) -> None:
    if not tokens:
        return
    shell_name = Path(tokens[0]).name or tokens[0]
    if shell_name == "eval":
        # ``eval "cat ../secret"`` hides the path inside a single quoted token;
        # re-scan the reconstructed command string.
        if len(tokens) > 1:
            _reject_shell_workspace_escapes(workspace_root, " ".join(tokens[1:]))
        return
    if shell_name not in {"sh", "bash", "zsh"}:
        return

    for index, token in enumerate(tokens[1:], start=1):
        if token == "-c" and index + 1 < len(tokens):
            _reject_shell_workspace_escapes(workspace_root, tokens[index + 1])
            return
        if token.startswith("-") and "c" in token[1:] and index + 1 < len(tokens):
            _reject_shell_workspace_escapes(workspace_root, tokens[index + 1])
            return


def _reject_redirect_escapes(workspace_root: Path, tokens: list[str]) -> None:
    redirect_ops = {">", ">>", "<", "<<"}
    for index, token in enumerate(tokens):
        target: str | None = None
        if token in redirect_ops and index + 1 < len(tokens):
            target = tokens[index + 1]
        elif token.startswith((">>", "<<")) and len(token) > 2:
            target = token[2:]
        elif token.startswith((">", "<")) and len(token) > 1:
            target = token[1:]

        if target is None or target.startswith("&"):
            continue
        resolve_workspace_path(workspace_root, target)
