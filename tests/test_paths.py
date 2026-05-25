from pathlib import Path

import pytest

from langcode_agent.core.paths import WorkspaceViolation, resolve_workspace_path


def test_resolves_relative_path_inside_workspace(tmp_path: Path) -> None:
    resolved = resolve_workspace_path(tmp_path, "src/app.py")

    assert resolved == tmp_path / "src" / "app.py"


def test_rejects_parent_directory_escape(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceViolation):
        resolve_workspace_path(tmp_path, "../outside.txt")


def test_rejects_absolute_path_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"

    with pytest.raises(WorkspaceViolation):
        resolve_workspace_path(tmp_path, outside)


def test_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)

    with pytest.raises(WorkspaceViolation):
        resolve_workspace_path(tmp_path, "link.txt")
