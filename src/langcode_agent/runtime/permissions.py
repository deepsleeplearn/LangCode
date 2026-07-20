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
MEMORY_TOOLS = {"memory", "soul", "self_evolve", "cron", "session_search", "skill", "diagram"}
AUTO_ALLOWED_TOOLS = {"read_file", "search", "ls", "glob", "web_search", "web_fetch", *TASK_TOOLS, *MEMORY_TOOLS}
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
    if tool_call.name in APPROVAL_REQUIRED_TOOLS:
        return ApprovalMode.ASK
    if tool_call.name == SHELL_TOOL_NAME:
        return permission_for_shell(str(tool_call.args.get("command", "")), workspace_root=workspace_root)
    if tool_call.name == SANDBOX_SHELL_TOOL_NAME:
        return permission_for_sandbox_shell(str(tool_call.args.get("command", "")), workspace_root=workspace_root)
    return ApprovalMode.DENY


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
        parts = shlex.split(command)
    except ValueError as exc:
        return ShellRisk(True, f"Command cannot be parsed safely: {exc}")

    if parts and parts[0] == "rm" and _rm_has_recursive_force_flags(parts[1:]):
        return ShellRisk(True, "Command removes files recursively and forcibly")
    if _has_shell_redirection(parts):
        return ShellRisk(True, "Command uses shell redirection and may read or write files")
    if parts:
        command_name = Path(parts[0]).name.lower()
        if _is_interpreter_eval(command_name, parts[1:]):
            return ShellRisk(True, f"Command evaluates inline code: {command_name}")
        if command_name in _SHELL_MUTATING_COMMANDS:
            return ShellRisk(True, f"Command may mutate files: {command_name}")
        if command_name in _SHELL_NETWORK_OR_PRIVILEGE_COMMANDS:
            return ShellRisk(True, f"Command may use network, credentials, or elevated privileges: {command_name}")

    return ShellRisk(False, "No high-risk shell pattern detected")


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
    return any(arg in flags for arg in args)


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
