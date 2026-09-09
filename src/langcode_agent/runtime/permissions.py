from dataclasses import dataclass
from enum import Enum
import fnmatch
import json
from pathlib import Path
import shlex
import threading


class ApprovalMode(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "args": self.args}

    @classmethod
    def from_dict(cls, value: dict) -> "ToolCall":
        return cls(name=value["name"], args=dict(value.get("args", {})))


@dataclass(frozen=True)
class ShellRisk:
    dangerous: bool
    reason: str


TASK_TOOLS = {"task_create", "task_update", "task_list", "task_get", "task_cancel"}
AUTO_ALLOWED_TOOLS = {
    "read_file",
    "search",
    "ls",
    "glob",
    "web_search",
    "web_fetch",
    "session_search",
    "diagram",
    *TASK_TOOLS,
}
APPROVAL_REQUIRED_TOOLS = {"write_file", "edit_file"}
SHELL_TOOL_NAME = "shell"
SANDBOX_SHELL_TOOL_NAME = "sandbox_shell"
_SETTINGS_LOCK = threading.RLock()

_SHELL_MUTATING_COMMANDS = {
    "chmod",
    "chown",
    "cp",
    "install",
    "ln",
    "mkdir",
    "mv",
    "rm",
    "rmdir",
    "tee",
    "touch",
    "truncate",
}

_SHELL_NETWORK_OR_PRIVILEGE_COMMANDS = {
    "brew",
    "curl",
    "gh",
    "pip",
    "pip3",
    "pnpm",
    "scp",
    "ssh",
    "sudo",
    "wget",
}

_SHELL_READ_ONLY_PIPE_COMMANDS = {
    "cat",
    "column",
    "cut",
    "grep",
    "head",
    "less",
    "more",
    "nl",
    "sort",
    "tail",
    "tr",
    "uniq",
    "wc",
}

_SHELL_INTERPRETER_EVAL_COMMANDS = {
    "bash": {"-c"},
    "node": {"-e", "--eval"},
    "osascript": {"-e"},
    "perl": {"-e"},
    "python": {"-c"},
    "python3": {"-c"},
    "ruby": {"-e"},
    "sh": {"-c"},
    "zsh": {"-c"},
}

# Interpreters whose short flags may be bundled (``perl -ne '...'``), so any
# cluster containing ``e`` still evaluates inline code.
_SHELL_CLUSTERED_EVAL_INTERPRETERS = {"perl", "ruby"}

# Builtins that re-dispatch an arbitrary command string. Their argument is
# never visible to the static workspace-escape checker, so they always ask.
_SHELL_COMMAND_DISPATCH_COMMANDS = {
    ".",
    "builtin",
    "command",
    "eval",
    "exec",
    "source",
}

# Commands that print the process environment, i.e. every API key the agent
# was started with. They ask with or without arguments.
_SHELL_ENVIRONMENT_DUMP_COMMANDS = {
    "declare",
    "env",
    "export",
    "printenv",
    "set",
    "typeset",
}

# Credential stores / secret managers.
_SHELL_CREDENTIAL_COMMANDS = {
    "defaults",
    "keychain",
    "op",
    "pass",
    "security",
    "ssh-add",
}

# ``awk`` dialects take their program as a positional argument, so the
# interpreter-eval flag check never sees it.
_SHELL_INLINE_PROGRAM_COMMANDS = {"awk", "gawk", "mawk", "nawk"}

# An inline program touching the filesystem or spawning a process. The rule is
# deliberately blunt: a program text without ``/``, ``(`` or a backtick cannot
# open a path or shell out, so it stays allowed.
_INLINE_PROGRAM_RISK_MARKERS = ("getline", "system(", "open(", "File.", "`", "/", "(")

_FIND_ACTION_FLAGS = {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprintf", "-fls"}

_DEFAULT_SHELL_ASK_PATTERNS = [
    "git push*",
    "git reset --hard*",
    "git clean*",
    "npm install*",
    "npm add*",
    "pnpm install*",
    "pnpm add*",
    "yarn install*",
    "yarn add*",
]


def permission_for_tool(tool_call: ToolCall, *, workspace_root: str | Path | None = None) -> ApprovalMode:
    if tool_call.name in AUTO_ALLOWED_TOOLS:
        return ApprovalMode.ALLOW
    if tool_call.name in {"memory", "soul", "self_evolve", "cron", "skill"}:
        return _memory_tool_permission(tool_call)
    if tool_call.name in APPROVAL_REQUIRED_TOOLS:
        return ApprovalMode.ASK
    if tool_call.name == SHELL_TOOL_NAME:
        return permission_for_shell(str(tool_call.args.get("command", "")), workspace_root=workspace_root)
    if tool_call.name == SANDBOX_SHELL_TOOL_NAME:
        return permission_for_sandbox_shell(str(tool_call.args.get("command", "")), workspace_root=workspace_root)
    return ApprovalMode.DENY


def _memory_tool_permission(tool_call: ToolCall) -> ApprovalMode:
    defaults = {"memory": "read", "soul": "read", "self_evolve": "status", "cron": "list", "skill": "list"}
    action = str(tool_call.args.get("action") or defaults[tool_call.name]).strip().lower()
    read_actions = {
        "memory": {"read", "view", "list"},
        "soul": {"read", "view", "list"},
        "self_evolve": {"status", "list_reflections", "reflections", "list_proposals", "proposals", "read_soul", "soul"},
        "cron": {"list", "status", "due", "run_due"},
        "skill": {"list", "read", "get"},
    }
    return ApprovalMode.ALLOW if action in read_actions.get(tool_call.name, set()) else ApprovalMode.ASK


def permission_for_shell(command: str, *, workspace_root: str | Path | None = None) -> ApprovalMode:
    rules = load_permission_rules(workspace_root) if workspace_root is not None else {}
    if _matches_any_rule(command, rules.get("deny", [])):
        return ApprovalMode.DENY
    if _matches_any_rule(command, rules.get("allow", [])):
        return ApprovalMode.ALLOW
    if _matches_any_rule(command, rules.get("ask", [])):
        return ApprovalMode.ASK

    if _matches_any_rule(command, _DEFAULT_SHELL_ASK_PATTERNS):
        return ApprovalMode.ASK

    risk = classify_shell_risk(command)
    if risk.dangerous:
        return ApprovalMode.ASK
    return ApprovalMode.ALLOW


def permission_for_sandbox_shell(command: str, *, workspace_root: str | Path | None = None) -> ApprovalMode:
    rules = load_permission_rules(workspace_root) if workspace_root is not None else {}
    if _matches_any_rule(command, rules.get("deny", [])):
        return ApprovalMode.DENY
    if _matches_any_rule(command, rules.get("allow", [])):
        return ApprovalMode.ALLOW
    if _matches_any_rule(command, rules.get("ask", [])):
        return ApprovalMode.ASK

    risk = classify_shell_risk(command)
    if risk.dangerous:
        return ApprovalMode.ASK
    return ApprovalMode.ALLOW


def load_permission_rules(workspace_root: str | Path | None) -> dict[str, list[str]]:
    if workspace_root is None:
        return {"allow": [], "ask": [], "deny": []}

    settings_path = Path(workspace_root).expanduser().resolve() / ".langcode" / "settings.json"
    if not settings_path.exists():
        return {"allow": [], "ask": [], "deny": []}

    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"allow": [], "ask": [], "deny": []}

    permissions = data.get("permissions") if isinstance(data, dict) else {}
    if not isinstance(permissions, dict):
        return {"allow": [], "ask": [], "deny": []}
    return {
        key: [str(item) for item in permissions.get(key, []) if isinstance(item, str)]
        for key in ("allow", "ask", "deny")
    }


def remember_shell_permission(workspace_root: str | Path, command: str, mode: str = "allow") -> Path:
    if mode not in {"allow", "ask", "deny"}:
        raise ValueError(f"Unsupported permission rule mode: {mode}")

    settings_path = Path(workspace_root).expanduser().resolve() / ".langcode" / "settings.json"
    rule = f"Bash({command})"
    with _SETTINGS_LOCK:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {}
        if settings_path.exists():
            try:
                loaded = json.loads(settings_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except (OSError, json.JSONDecodeError):
                data = {}

        permissions = data.setdefault("permissions", {})
        if not isinstance(permissions, dict):
            permissions = {}
            data["permissions"] = permissions
        rules = permissions.setdefault(mode, [])
        if not isinstance(rules, list):
            rules = []
            permissions[mode] = rules
        if rule not in rules:
            rules.append(rule)
        settings_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return settings_path


def classify_shell_risk(command: str) -> ShellRisk:
    """Classify a whole command line, chain segment by chain segment.

    ``a && b``, ``a || b`` and ``a; b`` used to be dangerous purely because of
    the operator, which made ``python3 -m pytest -q && echo done`` ask for
    approval. Each segment is classified on its own instead: the command asks
    when *any* segment is dangerous, and is allowed only when every segment is.
    """

    chains = [segment for segment in _scan_shell_syntax(command).chains if segment.strip()]
    if len(chains) > 1:
        for segment in chains:
            risk = _classify_single_command(segment)
            if risk.dangerous:
                return risk
        return ShellRisk(False, "No high-risk shell pattern detected")
    return _classify_single_command(command)


def _classify_single_command(command: str) -> ShellRisk:
    scan = _scan_shell_syntax(command)
    if scan.expansions:
        return ShellRisk(True, f"Command uses shell control syntax: {scan.expansions[0]}")
    blocking = [operator for operator in scan.operators if operator != "|"]
    if blocking:
        return ShellRisk(True, f"Command uses shell control syntax: {blocking[0]}")
    if "|" in scan.operators and not _is_read_only_pipeline(scan.segments):
        return ShellRisk(True, "Command uses shell control syntax: |")

    # Downstream pipeline stages are read-only whitelisted commands; the risk of
    # the pipeline is therefore the risk of its upstream command.
    head_command = scan.segments[0] if scan.segments else command

    lowered = command.lower()
    dangerous_patterns = [
        "rm -rf",
        "rm -fr",
        "mkfs",
        "dd if=",
        "shutdown",
        "reboot",
        ":(){",
        "chmod -r 777",
        "chown -r",
        "git reset --hard",
        "git clean -fd",
    ]
    for pattern in dangerous_patterns:
        if pattern in lowered:
            return ShellRisk(True, f"Command contains dangerous pattern: {pattern}")

    try:
        parts = shlex.split(head_command)
    except ValueError as exc:
        return ShellRisk(True, f"Command cannot be parsed safely: {exc}")

    if parts and parts[0] == "rm" and _rm_has_recursive_force_flags(parts[1:]):
        return ShellRisk(True, "Command removes files recursively and forcibly")
    if _has_shell_redirection(parts):
        return ShellRisk(True, "Command uses shell redirection and may read or write files")
    if parts:
        command_name = Path(parts[0]).name.lower() or parts[0].lower()
        if command_name in _SHELL_COMMAND_DISPATCH_COMMANDS:
            return ShellRisk(True, f"Command re-dispatches an unchecked command string: {command_name}")
        if command_name in _SHELL_ENVIRONMENT_DUMP_COMMANDS:
            return ShellRisk(True, f"Command can dump environment secrets: {command_name}")
        if command_name in _SHELL_CREDENTIAL_COMMANDS:
            return ShellRisk(True, f"Command reads a credential store: {command_name}")
        if command_name == "gpg" and any(arg.startswith("--export") for arg in parts[1:]):
            return ShellRisk(True, "Command exports private key material: gpg --export")
        if _is_interpreter_eval(command_name, parts[1:]):
            return ShellRisk(True, f"Command evaluates inline code: {command_name}")
        if command_name in _SHELL_INTERPRETER_EVAL_COMMANDS and not parts[1:]:
            # A bare ``sh``/``python3`` opens an interactive interpreter whose
            # input is never seen by any static check.
            return ShellRisk(True, f"Command starts an interactive interpreter: {command_name}")
        if _has_risky_inline_program(command_name, parts[1:]):
            return ShellRisk(True, f"Command runs an inline program that can touch the filesystem: {command_name}")
        if command_name == "find" and any(arg in _FIND_ACTION_FLAGS for arg in parts[1:]):
            return ShellRisk(True, "Command runs an action for every matched file: find")
        if command_name in _SHELL_MUTATING_COMMANDS:
            return ShellRisk(True, f"Command may mutate files: {command_name}")
        if command_name in _SHELL_NETWORK_OR_PRIVILEGE_COMMANDS:
            return ShellRisk(True, f"Command may use network, credentials, or elevated privileges: {command_name}")

    return ShellRisk(False, "No high-risk shell pattern detected")


@dataclass(frozen=True)
class _ShellScan:
    """Quote-aware view of a shell command's control syntax."""

    operators: list[str]
    expansions: list[str]
    segments: list[str]
    chains: list[str]


def _scan_shell_syntax(command: str) -> _ShellScan:
    """Find control operators that are *not* neutralized by quoting.

    Operators inside quotes (``echo 'a -> b'``) are literal text and are not
    reported. Parameter/command expansion (``$``, ``${``, ``$(``, backticks) is
    reported unless it sits inside single quotes, because a double-quoted
    ``"$HOME"`` is still expanded by the shell.

    ``segments`` splits on ``|`` (pipeline stages); ``chains`` splits on ``&&``,
    ``||`` and ``;`` (independently executed commands, each classified on its
    own by :func:`classify_shell_risk`).
    """

    operators: list[str] = []
    expansions: list[str] = []
    segments: list[str] = []
    chains: list[str] = []
    current: list[str] = []
    chain_current: list[str] = []
    quote: str | None = None
    index = 0
    length = len(command)

    def emit(text: str) -> None:
        current.append(text)
        chain_current.append(text)

    while index < length:
        char = command[index]
        if quote == "'":
            if char == "'":
                quote = None
            emit(char)
            index += 1
            continue
        if char == "\\" and index + 1 < length:
            emit(command[index : index + 2])
            index += 2
            continue
        if char in "$`":
            if command.startswith("${", index):
                expansions.append("${")
            elif command.startswith("$(", index):
                expansions.append("$(")
            else:
                expansions.append(char)
            emit(char)
            index += 1
            continue
        if quote == '"':
            if char == '"':
                quote = None
            emit(char)
            index += 1
            continue
        if char in "'\"":
            quote = char
            emit(char)
            index += 1
            continue
        if command.startswith("&&", index) or command.startswith("||", index):
            operators.append(command[index : index + 2])
            current.append(command[index : index + 2])
            chains.append("".join(chain_current))
            chain_current = []
            index += 2
            continue
        if char == "|":
            operators.append("|")
            segments.append("".join(current))
            current = []
            chain_current.append("|")
            index += 1
            continue
        if char == ";":
            operators.append(char)
            current.append(char)
            chains.append("".join(chain_current))
            chain_current = []
            index += 1
            continue
        if char == "&":
            operators.append(char)
            emit(char)
            index += 1
            continue
        if char == ">":
            operators.append(">>" if command.startswith(">>", index) else ">")
            emit(char)
            index += 1
            continue
        emit(char)
        index += 1

    segments.append("".join(current))
    chains.append("".join(chain_current))
    return _ShellScan(operators=operators, expansions=expansions, segments=segments, chains=chains)


def _is_read_only_pipeline(segments: list[str]) -> bool:
    """True when every downstream stage of a pipeline is a read-only filter."""

    if len(segments) < 2:
        return False
    for segment in segments[1:]:
        try:
            parts = shlex.split(segment)
        except ValueError:
            return False
        if not parts:
            return False
        if Path(parts[0]).name.lower() not in _SHELL_READ_ONLY_PIPE_COMMANDS:
            return False
    return True


def _rm_has_recursive_force_flags(args: list[str]) -> bool:
    flags: set[str] = set()
    for arg in args:
        if not arg.startswith("-") or arg == "-":
            continue
        for char in arg[1:]:
            flags.add(char.lower())
    return "r" in flags and "f" in flags


def _has_shell_redirection(parts: list[str]) -> bool:
    redirect_ops = {">", ">>", "<", "<<", "2>", "2>>", "&>", "&>>"}
    for part in parts:
        if part in redirect_ops:
            return True
        if part.startswith((">", ">>", "<", "<<", "2>", "2>>", "&>", "&>>")):
            return True
    return False


def _is_interpreter_eval(command_name: str, args: list[str]) -> bool:
    flags = _SHELL_INTERPRETER_EVAL_COMMANDS.get(command_name)
    if not flags:
        return False
    if any(arg in flags for arg in args):
        return True
    if command_name not in _SHELL_CLUSTERED_EVAL_INTERPRETERS:
        return False
    # ``perl -ne '...'`` / ``perl -lane '...'`` bundle the eval flag with others.
    return any(
        arg.startswith("-") and not arg.startswith("--") and "e" in arg[1:].lower() and arg[1:].isalpha()
        for arg in args
    )


def _has_risky_inline_program(command_name: str, args: list[str]) -> bool:
    """True when an ``awk``-style positional program can reach the filesystem."""

    if command_name not in _SHELL_INLINE_PROGRAM_COMMANDS:
        return False
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in {"-f", "--file", "--source"}:
            # The program lives in a file we cannot inspect statically.
            return True
        if arg in {"-v", "--assign", "-F", "--field-separator"}:
            skip_next = True
            continue
        if arg.startswith("-") and arg != "-":
            continue
        return any(marker in arg for marker in _INLINE_PROGRAM_RISK_MARKERS)
    return False


def _matches_any_rule(command: str, rules: list[str]) -> bool:
    normalized_command = _normalize_shell_command(command)
    for rule in rules:
        pattern = _extract_bash_rule_pattern(rule)
        if pattern is None:
            continue
        normalized_pattern = _normalize_shell_command(pattern)
        if normalized_pattern == normalized_command:
            return True
        if fnmatch.fnmatchcase(normalized_command, normalized_pattern):
            return True
    return False


def _extract_bash_rule_pattern(rule: str) -> str | None:
    stripped = rule.strip()
    if stripped.startswith("Bash(") and stripped.endswith(")"):
        return stripped[5:-1]
    return stripped or None


def _normalize_shell_command(command: str) -> str:
    return " ".join(command.strip().split())
