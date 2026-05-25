from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
import base64
import json
from pathlib import Path
from typing import Any


DEFAULT_TOOL_RESULT_CHAR_LIMIT = 12000


def compact_tool_result(
    workspace_root: str | Path,
    session_id: str,
    tool_name: str,
    result: dict,
    *,
    max_chars: int = DEFAULT_TOOL_RESULT_CHAR_LIMIT,
) -> dict:
    """Offload very large tool results into workspace artifacts.

    DeepAgents keeps long-horizon contexts healthy by avoiding huge tool
    messages. LangCode follows that behavior here: small results pass through,
    while large JSON payloads are written under `.langcode/artifacts/`.
    """

    result = make_json_safe(result)
    serialized = json.dumps(result, ensure_ascii=False)
    if len(serialized) <= max_chars:
        return result

    root = Path(workspace_root).expanduser().resolve()
    artifact_dir = root / ".langcode" / "artifacts" / _safe_name(session_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact_path = artifact_dir / f"{stamp}-{_safe_name(tool_name)}.json"
    artifact_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    preview = serialized[: max_chars // 3]
    return {
        "ok": bool(result.get("ok", True)),
        "offloaded": True,
        "tool": tool_name,
        "artifact": str(artifact_path.relative_to(root)),
        "summary": f"工具结果过大，已写入 artifact。原始 JSON 长度 {len(serialized)} 字符。",
        "preview": preview,
    }


def make_json_safe(value: Any) -> Any:
    """Convert tool results into values accepted by JSON and model APIs."""

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "type": "bytes",
                "encoding": "base64",
                "data": base64.b64encode(value).decode("ascii"),
            }
    if isinstance(value, bytearray):
        return make_json_safe(bytes(value))
    if isinstance(value, memoryview):
        return make_json_safe(value.tobytes())
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return make_json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, set):
        return [make_json_safe(item) for item in sorted(value, key=str)]
    try:
        json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)
    return value


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    return cleaned.strip("-") or "item"
