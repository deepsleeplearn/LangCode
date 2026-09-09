from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
import base64
import json
import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


_LOGGER = logging.getLogger(__name__)


DEFAULT_TOOL_RESULT_CHAR_LIMIT = 12000
PREVIEW_HEAD_CHARS = 4000
PREVIEW_TAIL_CHARS = 2000

DEFAULT_CONTEXT_MAX_TOKENS = 100000
DEFAULT_CONTEXT_KEEP_RECENT_TOKENS = 24000
CONTEXT_SUMMARY_PREFIX = "[上下文摘要]"
CONTEXT_SUMMARY_ACK = "好的，我已了解以上摘要。"
_SUMMARY_SOURCE_CHAR_LIMIT = 24000
_SUMMARY_MESSAGE_CHAR_LIMIT = 2000
_TOKENS_PER_MESSAGE_OVERHEAD = 4


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

    head_chars = min(PREVIEW_HEAD_CHARS, max_chars * 2 // 3)
    tail_chars = min(PREVIEW_TAIL_CHARS, max_chars - head_chars)
    omitted = max(0, len(serialized) - head_chars - tail_chars)
    preview = f"{serialized[:head_chars]}\n…省略 {omitted} 字符…\n{serialized[-tail_chars:]}"
    return {
        "ok": bool(result.get("ok", True)),
        "offloaded": True,
        "tool": tool_name,
        "artifact": str(artifact_path.relative_to(root)),
        "summary": f"工具结果过大，已写入 artifact。原始 JSON 长度 {len(serialized)} 字符。",
        "preview": preview,
    }


def count_messages_tokens(messages: Any, model_name: str = "") -> int:
    """Approximate the prompt tokens a message list will cost.

    Uses tiktoken (``encoding_for_model`` when the model is known, otherwise
    ``cl100k_base``) over each message's text plus its serialized tool_calls and
    its non-text content parts, with a small fixed per-message overhead for
    role/framing tokens.
    """

    return _TokenCounter(model_name).total(messages)


class _TokenCounter:
    """Per-message token counts, computed once and reused.

    Every caller (budget check, recent-window split, budget enforcement) goes
    through this one helper, so a message is costed identically everywhere. The
    cache also keeps :func:`_enforce_budget` linear: without it the trimming
    loop re-encoded the whole window on every iteration.
    """

    def __init__(self, model_name: str = "") -> None:
        self._encoding = _encoding_for(model_name)
        # id(message) -> (message, tokens); the message is kept alive so a
        # recycled id can never return another message's count.
        self._cache: dict[int, tuple[Any, int]] = {}

    def message_tokens(self, message: Any) -> int:
        key = id(message)
        cached = self._cache.get(key)
        if cached is not None and cached[0] is message:
            return cached[1]
        tokens = self._compute(message)
        self._cache[key] = (message, tokens)
        return tokens

    def total(self, messages: Any) -> int:
        return sum(self.message_tokens(message) for message in list(messages or []))

    def _compute(self, message: Any) -> int:
        total = _TOKENS_PER_MESSAGE_OVERHEAD
        total += _count_text_tokens(message_text(message), self._encoding)
        total += _non_text_content_tokens(message)
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            total += _count_text_tokens(_serialize(tool_calls), self._encoding)
        return total


def _serialize(value: Any) -> str:
    try:
        return json.dumps(make_json_safe(value), ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _non_text_content_tokens(message: Any) -> int:
    """Approximate tokens for content parts :func:`message_text` cannot read.

    An ``image_url`` part carrying a base64 blob has no ``text``/``content``
    string, so it used to cost ~0 tokens and a multimodal history never
    triggered compaction. Four characters per token is the usual rough ratio.
    """

    content = getattr(message, "content", message)
    if not isinstance(content, list):
        return 0
    total = 0
    for item in content:
        if not isinstance(item, dict) or item.get("type") == "text":
            continue
        value = item.get("text") or item.get("content")
        if isinstance(value, str):
            # Already counted by message_text; do not double count.
            continue
        total += len(_serialize(item)) // 4
    return total


def compact_history_if_needed(
    messages: Any,
    *,
    model: Any,
    max_tokens: int | None = None,
    keep_recent_tokens: int | None = None,
) -> tuple[list, bool]:
    """Summarize the older part of a long history so it fits the context budget.

    Returns ``(messages, compacted)``. Leading SystemMessages are always kept.
    The recent window is cut at a HumanMessage boundary, so a tool-calling
    AIMessage is never separated from its ToolMessages. The dropped segment is
    replaced by a HumanMessage/AIMessage summary pair, keeping the transcript
    alternating.

    Compaction is all-or-nothing: if the summary call raises or comes back
    empty, the history is returned untouched with ``False``. Dropping the older
    segment without a summary silently destroyed the conversation.
    """

    history = list(messages or [])
    max_tokens = _budget(max_tokens, "LANGCODE_CONTEXT_MAX_TOKENS", DEFAULT_CONTEXT_MAX_TOKENS)
    keep_recent = _budget(
        keep_recent_tokens,
        "LANGCODE_CONTEXT_KEEP_RECENT_TOKENS",
        DEFAULT_CONTEXT_KEEP_RECENT_TOKENS,
    )
    counter = _TokenCounter(_model_name(model))
    if counter.total(history) <= max_tokens:
        return history, False

    prefix: list = []
    index = 0
    while index < len(history) and isinstance(history[index], SystemMessage):
        prefix.append(history[index])
        index += 1
    body = history[index:]

    split = _recent_window_start(body, keep_recent, counter)
    older, recent = body[:split], body[split:]
    if not older:
        return history, False

    try:
        summary = _summarize_segment(older, model=_unbound_model(model))
    except Exception:
        _LOGGER.warning("上下文压缩失败：摘要模型调用异常，本次保留完整历史", exc_info=True)
        return history, False
    if not summary.strip():
        _LOGGER.warning("上下文压缩失败：摘要为空，本次保留完整历史")
        return history, False

    replacement = [
        HumanMessage(content=f"{CONTEXT_SUMMARY_PREFIX} {summary}"),
        AIMessage(content=CONTEXT_SUMMARY_ACK),
    ]
    return _enforce_budget(prefix, replacement, recent, max_tokens, counter), True


def _unbound_model(model: Any) -> Any:
    """The plain chat model behind a ``bind_tools`` RunnableBinding.

    Summarizing through the tool-bound runnable makes the model answer with a
    tool call instead of prose, so ``message_text`` returns "" and the segment
    used to be dropped with no summary at all.
    """

    bound = getattr(model, "bound", None)
    return model if bound is None else bound


def message_text(message: Any) -> str:
    """Best-effort plain text for a message whose content may be a parts list."""

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


_ENCODING_CACHE: dict[str, Any] = {}


def _encoding_for(model_name: str) -> Any:
    key = str(model_name or "")
    if key in _ENCODING_CACHE:
        return _ENCODING_CACHE[key]
    encoding = None
    try:
        import tiktoken

        if key:
            try:
                encoding = tiktoken.encoding_for_model(key)
            except Exception:
                encoding = None
        if encoding is None:
            encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = None
    _ENCODING_CACHE[key] = encoding
    return encoding


def _count_text_tokens(text: str, encoding: Any) -> int:
    if not text:
        return 0
    if encoding is None:
        return max(1, len(text) // 3)
    try:
        return len(encoding.encode(text, disallowed_special=()))
    except Exception:
        return max(1, len(text) // 3)


def _budget(explicit: int | None, env_name: str, default: int) -> int:
    if explicit is not None:
        try:
            return max(1, int(explicit))
        except (TypeError, ValueError):
            pass
    try:
        return max(1, int(os.getenv(env_name, str(default))))
    except (TypeError, ValueError):
        return default


def _model_name(model: Any) -> str:
    for candidate in (model, getattr(model, "bound", None)):
        if candidate is None:
            continue
        for attribute in ("model_name", "model"):
            value = getattr(candidate, attribute, None)
            if isinstance(value, str) and value:
                return value
    return ""


def _recent_window_start(body: list, keep_recent_tokens: int, counter: "_TokenCounter") -> int:
    """Index where the retained recent window starts, on a HumanMessage boundary.

    Uses the same per-message cost as :func:`count_messages_tokens`; it used to
    count only ``message_text``, so a window full of tool calls was measured
    smaller here than in the budget check that followed.
    """

    if not body:
        return 0
    accumulated = 0
    start = len(body)
    for index in range(len(body) - 1, -1, -1):
        accumulated += counter.message_tokens(body[index])
        start = index
        if accumulated >= keep_recent_tokens:
            break
    # Walk back to the turn boundary so a tool-calling AIMessage keeps its
    # ToolMessages; the window can only grow, never shrink below keep_recent.
    for index in range(start, -1, -1):
        if isinstance(body[index], HumanMessage):
            return index
    return 0


def _summarize_segment(segment: list, *, model: Any) -> str:
    prompt = (
        "以下是一段较早的对话历史，请把它压缩成一份简洁摘要，供后续对话继续使用。\n"
        "必须保留：已确认的事实、已做出的决定、涉及的文件路径与命令、尚未完成的任务与待办、"
        "以及用户明确表达的偏好和约束。不要编造内容，不要输出客套话。\n"
        "请用中文分条列出，总长度不超过 1500 tokens。\n\n"
        f"对话历史：\n{_render_segment(segment)}"
    )
    response = model.invoke([HumanMessage(content=prompt)])
    return message_text(response).strip()


def _render_segment(segment: list) -> str:
    lines: list[str] = []
    for message in segment:
        role = getattr(message, "type", None) or type(message).__name__
        text = message_text(message).strip()
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            names = ", ".join(str(call.get("name", "")) for call in tool_calls if isinstance(call, dict))
            text = f"{text}\n[调用工具: {names}]".strip()
        if not text:
            continue
        if len(text) > _SUMMARY_MESSAGE_CHAR_LIMIT:
            text = f"{text[:_SUMMARY_MESSAGE_CHAR_LIMIT]}…"
        lines.append(f"{role}: {text}")
    rendered = "\n".join(lines)
    if len(rendered) > _SUMMARY_SOURCE_CHAR_LIMIT:
        head = _SUMMARY_SOURCE_CHAR_LIMIT * 2 // 3
        tail = _SUMMARY_SOURCE_CHAR_LIMIT - head
        rendered = f"{rendered[:head]}\n…省略中间内容…\n{rendered[-tail:]}"
    return rendered


def _enforce_budget(
    prefix: list,
    replacement: list,
    recent: list,
    max_tokens: int,
    counter: "_TokenCounter",
) -> list:
    """Trim whole turns off the front of the window until it fits the budget.

    The running total is updated by the tokens actually removed instead of
    recounting the entire window each pass, which made this loop O(n^2).
    """

    window = list(recent)
    total = counter.total([*prefix, *replacement]) + counter.total(window)
    while window and total > max_tokens:
        cut = 1
        while cut < len(window) and not isinstance(window[cut], HumanMessage):
            cut += 1
        if cut >= len(window):
            break
        total -= counter.total(window[:cut])
        window = window[cut:]
    return [*prefix, *replacement, *window]


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
