from pathlib import Path


class WorkspaceViolation(ValueError):
    """Raised when a tool path escapes the configured workspace."""


def resolve_workspace_path(
    workspace_root: str | Path,
    path: str | Path,
    *,
    allow_workspace_escape: bool = False,
) -> Path:
    root = Path(workspace_root).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()

    if not allow_workspace_escape and resolved != root and root not in resolved.parents:
        raise WorkspaceViolation(f"Path escapes workspace: {path}")

    return resolved
