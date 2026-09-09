from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import fnmatch

from ..tooling import tools
from .deep_harness import cancel_task, create_task, get_task, list_tasks, update_task


@dataclass
class WorkspaceBackend:
    """DeepAgents-style backend boundary for LangCode workspace operations."""

    workspace_root: Path

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()

    def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int = 2000,
        allow_workspace_escape: bool = False,
    ) -> dict:
        return {
            "ok": True,
            **tools._read_file_page(
                self.workspace_root,
                path,
                offset=offset,
                limit=limit,
                allow_workspace_escape=allow_workspace_escape,
            ),
        }

    def write_file(self, path: str, content: str, *, allow_workspace_escape: bool = False) -> dict:
        tools.write_file(self.workspace_root, path, content, allow_workspace_escape=allow_workspace_escape)
        return {"ok": True}

    def edit_file(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
        allow_workspace_escape: bool = False,
    ) -> dict:
        tools.edit_file(
            self.workspace_root,
            path,
            old,
            new,
            replace_all=replace_all,
            allow_workspace_escape=allow_workspace_escape,
        )
        return {"ok": True}

    def search(
        self,
        query: str,
        path: str = ".",
        *,
        max_results: int = 50,
        allow_workspace_escape: bool = False,
    ) -> dict:
        return {
            "ok": True,
            "results": tools.search(
                self.workspace_root,
                query,
                path,
                max_results=max_results,
                allow_workspace_escape=allow_workspace_escape,
            ),
        }

    def ls(self, path: str = ".", *, allow_workspace_escape: bool = False) -> dict:
        target = tools.resolve_workspace_path(
            self.workspace_root,
            path,
            allow_workspace_escape=allow_workspace_escape,
        )
        entries = []
        for child in sorted(target.iterdir(), key=lambda item: item.name.lower()):
            entries.append(
                {
                    "name": child.name,
                    "path": str(child.relative_to(self.workspace_root)) if self.workspace_root in child.parents else str(child),
                    "is_dir": child.is_dir(),
                }
            )
        return {"ok": True, "entries": entries}

    def glob(self, pattern: str, path: str = ".", *, max_results: int = 100) -> dict:
        root = tools.resolve_workspace_path(self.workspace_root, path)
        matches = []
        for candidate in root.rglob("*"):
            if len(matches) >= max_results:
                break
            relative = candidate.relative_to(self.workspace_root)
            if fnmatch.fnmatch(str(relative), pattern) or fnmatch.fnmatch(candidate.name, pattern):
                matches.append({"path": str(relative), "is_dir": candidate.is_dir()})
        return {"ok": True, "matches": matches}

    def shell(self, command: str, *, timeout_seconds: int = 30, allow_workspace_escape: bool = False) -> dict:
        return {
            "ok": True,
            "result": tools.shell(
                self.workspace_root,
                command,
                timeout_seconds=timeout_seconds,
                allow_workspace_escape=allow_workspace_escape,
            ).to_dict(),
        }

    def task_create(self, content: str, *, status: str = "pending", task_id: str | None = None) -> dict:
        return create_task([], content, status=status, task_id=task_id)

    def task_update(self, task_id: str, *, content: str | None = None, status: str | None = None) -> dict:
        return update_task([], task_id, content=content, status=status)

    def task_list(self, *, status: str | None = None) -> dict:
        return list_tasks([], status=status)

    def task_get(self, task_id: str) -> dict:
        return get_task([], task_id)

    def task_cancel(self, task_id: str, *, reason: str | None = None) -> dict:
        return cancel_task([], task_id, reason=reason)

    def sandbox_shell(self, command: str, *, timeout_seconds: int = 30, copy_workspace: bool = True) -> dict:
        from .sandbox import run_sandbox_shell

        return run_sandbox_shell(
            self.workspace_root,
            command,
            timeout_seconds=timeout_seconds,
            copy_workspace=copy_workspace,
        )
