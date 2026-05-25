from __future__ import annotations

from pathlib import Path
import shutil
import shlex
from uuid import uuid4

from ..tooling.tools import ShellResult, shell


EXCLUDED_SANDBOX_NAMES = {
    ".git",
    ".langcode",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
}


def run_sandbox_shell(
    workspace_root: str | Path,
    command: str,
    *,
    timeout_seconds: int = 30,
    copy_workspace: bool = True,
) -> dict:
    _reject_unsafe_sandbox_command(command)
    root = Path(workspace_root).expanduser().resolve()
    sandbox_root = root / ".langcode" / "sandboxes" / uuid4().hex
    work_dir = sandbox_root / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    if copy_workspace:
        _copy_workspace(root, work_dir)

    result = shell(work_dir, command, timeout_seconds=timeout_seconds)
    return {
        "ok": True,
        "sandbox": str(work_dir.relative_to(root)),
        "copied_workspace": copy_workspace,
        "result": result.to_dict() if isinstance(result, ShellResult) else result,
    }


def _copy_workspace(source: Path, target: Path) -> None:
    for child in source.iterdir():
        if child.name in EXCLUDED_SANDBOX_NAMES:
            continue
        destination = target / child.name
        if child.is_dir():
            shutil.copytree(child, destination, ignore=shutil.ignore_patterns(*EXCLUDED_SANDBOX_NAMES))
        elif child.is_file():
            shutil.copy2(child, destination)


def _reject_unsafe_sandbox_command(command: str) -> None:
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"Sandbox command cannot be parsed safely: {exc}") from exc
    if not parts:
        return
    command_name = Path(parts[0]).name.lower()
    inline_eval_flags = {
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
    if any(arg in inline_eval_flags.get(command_name, set()) for arg in parts[1:]):
        raise ValueError(
            "sandbox_shell does not allow inline interpreter commands because this local sandbox is directory-based, not OS-isolated."
        )
