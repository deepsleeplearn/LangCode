from __future__ import annotations

from pathlib import Path
import os
import shutil
import shlex
import tempfile
import threading
import time

from ..tooling.tools import ShellResult, shell


EXCLUDED_SANDBOX_NAMES = {
    ".git",
    ".langcode",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
}

SANDBOX_DIR_PREFIX = "langcode-sandbox-"
SANDBOX_GC_MAX_AGE_SECONDS = 6 * 60 * 60
_GC_LOCK = threading.Lock()
_GC_DONE = False
# Sandboxes this process is currently running in. The GC deletes stale trees by
# mtime, and a sandbox_shell that runs longer than the max age (or whose mtime
# never advances) would otherwise have its own working tree pulled out from
# under it by a concurrent call.
_ACTIVE_SANDBOXES: set[str] = set()
_ACTIVE_LOCK = threading.Lock()


def run_sandbox_shell(
    workspace_root: str | Path,
    command: str,
    *,
    timeout_seconds: int = 30,
    copy_workspace: bool = True,
) -> dict:
    _reject_unsafe_sandbox_command(command)
    _gc_stale_sandboxes()
    root = Path(workspace_root).expanduser().resolve()

    # The sandbox lives in the OS temp dir, never under the workspace: a copy
    # inside the workspace pollutes searches and git status, feeds the next
    # copytree, and used to be left behind forever.
    sandbox_root = Path(tempfile.mkdtemp(prefix=SANDBOX_DIR_PREFIX))
    with _ACTIVE_LOCK:
        _ACTIVE_SANDBOXES.add(str(sandbox_root.resolve()))
    try:
        work_dir = sandbox_root / "work"
        work_dir.mkdir(parents=True, exist_ok=True)

        if copy_workspace:
            _copy_workspace(root, work_dir)

        result = shell(work_dir, command, timeout_seconds=timeout_seconds)
        return {
            "ok": True,
            "sandbox": str(work_dir),
            "copied_workspace": copy_workspace,
            "result": result.to_dict() if isinstance(result, ShellResult) else result,
        }
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_SANDBOXES.discard(str(sandbox_root.resolve()))
        shutil.rmtree(sandbox_root, ignore_errors=True)


def _gc_stale_sandboxes() -> int:
    """Collect sandboxes a crashed process left behind. Runs once per process."""

    global _GC_DONE
    with _GC_LOCK:
        if _GC_DONE:
            return 0
        _GC_DONE = True
    return gc_sandboxes()


def gc_sandboxes(*, max_age_seconds: int = SANDBOX_GC_MAX_AGE_SECONDS) -> int:
    """Delete abandoned sandbox trees. Only ever touches directories that are

    (a) direct children of the temp dir, (b) named ``langcode-sandbox-*``,
    (c) older than ``max_age_seconds`` by mtime, and (d) not currently in use by
    this process.
    """

    cutoff = time.time() - max(0, max_age_seconds)
    try:
        with os.scandir(tempfile.gettempdir()) as entries:
            candidates = [entry.path for entry in entries if entry.name.startswith(SANDBOX_DIR_PREFIX)]
    except OSError:
        return 0
    with _ACTIVE_LOCK:
        active = set(_ACTIVE_SANDBOXES)
    removed = 0
    for path in candidates:
        try:
            if os.path.islink(path):
                continue
            if str(Path(path).resolve()) in active:
                continue
            if not Path(path).is_dir() or os.stat(path).st_mtime > cutoff:
                continue
        except OSError:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed


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
