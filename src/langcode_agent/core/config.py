from pathlib import Path
import os
import shlex


def load_env_files(*roots: str | Path) -> None:
    seen: set[Path] = set()
    for root in roots:
        base = Path(root).expanduser().resolve()
        for name in (".env.local", ".env"):
            path = base / name
            if path in seen:
                continue
            seen.add(path)
            if path.exists():
                _load_env_file(path)


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, _parse_env_value(value.strip()))


def _parse_env_value(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = shlex.split(value, comments=False, posix=True)
    except ValueError:
        return value.strip("\"'")
    if len(parsed) == 1:
        return parsed[0]
    return value.strip("\"'")

