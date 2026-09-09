from __future__ import annotations

import asyncio
import base64
from concurrent.futures import Future, ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass, field
import gzip
import hashlib
import logging
from pathlib import Path
import argparse
import json
import mimetypes
import os
import platform
import re
import secrets
import subprocess
import sqlite3
import threading
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from sanic import Sanic, response
from sanic.request import Request

from ..runtime.agent import CodeAgent
from ..voice.asr import QwenAsrService, websocket_asr_loop
from ..voice.stream import TtsTurnRegistry, iter_tts_events
from ..runtime.chat import build_openai_model, default_system_prompt, messages_from_json, model_settings_from_env
from ..core.config import load_env_files
from ..core.context_management import compact_history_if_needed, compact_tool_result, make_json_safe
from ..runtime.delegation import run_delegate_agent
from ..runtime.multi_agent import iter_agent_debate_events, run_agent_debate, run_parallel_delegate_agents
from ..runtime.deep_harness import cancel_task, create_task, get_task, list_tasks, update_task
from ..memory.project import handle_local_command, serialize_message
from ..runtime.permissions import ToolCall, remember_shell_permission
from ..storage.job_queue import JobQueue
from ..storage.runtime_state import RuntimeLockTimeout, RuntimeStateStore
from ..storage.session_store import DELETED_REVISION, SessionStore
from ..memory.evolution import reflect_session
from ..voice.tts import TtsService, content_type_for_path
from ..voice.turnsense import TurnSenseService
from .voice_proxy import VoiceWorkerClient


logger = logging.getLogger("langcode.web")

TASK_TOOL_NAMES = {"task_create", "task_update", "task_list", "task_get", "task_cancel"}
INTERNAL_PREVIEW_TOOL_NAMES = TASK_TOOL_NAMES | {"memory", "soul", "self_evolve", "voice_interrupt"}
DEFAULT_WEB_SEARCH_LIMIT = 8
DEFAULT_MAX_TOOL_ROUNDS = 30
DEFAULT_MAX_RESIDENT_SESSIONS = 32
DEFAULT_STREAM_IDLE_TIMEOUT_SEC = 90.0
TOOL_RESULT_PREVIEW_CHARS = 2000
TOOL_ROUND_LIMIT_MOCK_USER = "已达工具调用轮次上限，请基于现有信息直接作答"
STREAM_IDLE_TIMEOUT_ERROR = "模型流在空闲超时内没有返回新内容。"
CANCELLED_TOOL_RESULT_JSON = '{"ok": false, "cancelled": true}'
VOICE_INTERRUPT_SUFFIX = "（此处被用户打断）"
WEB_SEARCH_LIMIT_ERROR = "外部网页搜索次数已达到上限。请基于已获得的搜索结果回答。"
WEB_SEARCH_LIMIT_MOCK_USER = (
    "不要再调用任何工具，不要继续搜索网页。请只基于本轮已经找到的工具结果和对话内容，"
    "直接回答我最初的问题；如果信息不足，也请说明依据和不确定性。"
)


@dataclass
class WebSession:
    id: str
    agent: CodeAgent
    workspace_root: Path
    messages: list[BaseMessage] = field(default_factory=list)
    display_messages: list[BaseMessage] = field(default_factory=list)
    pending: dict | None = None
    model: Any | None = None
    todos: list[dict] = field(default_factory=list)
    store_path: Path | None = None
    last_reflected_count: int = 0
    revision: int = 0


def _load_or_create_api_token(state_dir: Path) -> str:
    """Return the local API token, reusing the one stored under ``state_dir``.

    The token guards every ``/api/*`` route. Generating a fresh one per start
    silently invalidates every browser tab that is already open: the page holds
    the old token in its meta tag and every request comes back 403 with no way
    to tell that a reload is all that is needed. Persisting it (0600, inside the
    gitignored state dir) keeps open tabs working across restarts.

    ``LANGCODE_API_TOKEN`` overrides the file, for setups that inject their own.
    A short or unreadable file is replaced rather than trusted.
    """

    override = (os.getenv("LANGCODE_API_TOKEN") or "").strip()
    if override:
        return override

    token_path = state_dir / "api-token"
    try:
        existing = token_path.read_text(encoding="utf-8").strip()
        if len(existing) >= 32 and existing.isascii():
            return existing
    except (OSError, UnicodeDecodeError):
        pass

    token = secrets.token_urlsafe(32)
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        token_path.write_text(token, encoding="utf-8")
        token_path.chmod(0o600)
    except OSError:
        logger.warning(
            "Could not persist the local API token to %s; open browser tabs will "
            "need a reload after every restart.",
            token_path,
        )
    return token


class WebApp:
    def __init__(self, workspace_root: Path, frontend_dir: Path, *, enable_voice: bool = True) -> None:
        self.workspace_root = self._resolve_workspace(workspace_root)
        load_env_files(Path.cwd(), self.workspace_root)
        self.frontend_dir = frontend_dir
        self.enable_voice = enable_voice
        # Assigned once state_dir is known, so the token can survive a restart.
        self.api_token = ""
        self.sessions: OrderedDict[str, WebSession] = OrderedDict()
        self._sessions_lock = threading.RLock()
        # Item 3: how many in-flight requests currently use each session. LRU
        # eviction must not close an agent out from under a running request -
        # /api/chat (non-streaming) never calls mark_run_started, so
        # has_active_run() alone does not see it.
        self._session_busy: dict[str, int] = {}
        self._deferred_closes: dict[str, WebSession] = {}
        # Item 12: sha1 of the last displayed-history archive written per session,
        # so an unchanged transcript is not rewritten on every save.
        self._display_archive_digests: dict[str, str] = {}
        self.home_workspace_root = self.workspace_root
        self.state_dir = self.home_workspace_root / ".langcode"
        self.api_token = _load_or_create_api_token(self.state_dir)
        self.store = SessionStore(self.state_dir / "web.sqlite")
        runtime_prefix = os.getenv("LANGCODE_REDIS_PREFIX") or (
            "langcode:" + hashlib.sha1(str(self.state_dir).encode("utf-8")).hexdigest()[:12]
        )
        self.runtime_state = RuntimeStateStore(prefix=runtime_prefix)
        self.job_queue = JobQueue(prefix=runtime_prefix)
        self._partial_save_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="partial-history")
        self._partial_save_futures: dict[str, Future] = {}
        self._cancel_displayed_text: dict[str, str] = {}
        # Full-duplex TTS bookkeeping: newest turn per session plus the turns a
        # client cancelled, shared with the voice worker's copy of the endpoint.
        self._tts_turns = TtsTurnRegistry()
        self._configure_workspace_storage()
        # A remote voice worker loads no local models, so it stays available even
        # when this process opts out of local voice models (queue workers, item 34).
        self.voice_worker = self._voice_worker_from_env()
        if enable_voice and self.voice_worker is None:
            self.turnsense = TurnSenseService()
            self.asr = QwenAsrService(turnsense=self.turnsense)
            self.tts = TtsService()
            self.asr.start_preload()
            self.tts.start_preload()
        else:
            self.turnsense = None
            self.asr = None
            self.tts = None

    def close(self) -> None:
        self._partial_save_pool.shutdown(wait=True, cancel_futures=True)
        with self._sessions_lock:
            for session in self.sessions.values():
                session.agent.close()

    def _voice_worker_from_env(self) -> VoiceWorkerClient | None:
        base_url = (os.getenv("LANGCODE_VOICE_WORKER_URL") or "").strip().rstrip("/")
        if not base_url:
            return None
        return VoiceWorkerClient(base_url, timeout_seconds=_float_env("LANGCODE_VOICE_WORKER_TIMEOUT_SEC", 10.0))

    def _configure_workspace_storage(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.store = SessionStore(self.state_dir / "web.sqlite")
        self.store.migrate_json_sessions(
            self.state_dir / "web-sessions",
            self.state_dir / "web-session-metadata.json",
            str(self.home_workspace_root),
        )

    def _resolve_workspace(self, workspace: Path) -> Path:
        candidate = workspace.expanduser().resolve()
        if not candidate.exists():
            raise ValueError(f"Workspace does not exist: {candidate}")
        if not candidate.is_dir():
            raise ValueError(f"Workspace is not a directory: {candidate}")
        return candidate

    def acquire_session_busy(self, session_id: str) -> None:
        """Pin a session in memory for the duration of one request (item 3)."""
        if not session_id:
            return
        with self._sessions_lock:
            self._session_busy[session_id] = self._session_busy.get(session_id, 0) + 1

    def release_session_busy(self, session_id: str) -> None:
        """Drop one request's pin, running any close that was deferred while busy."""
        if not session_id:
            return
        with self._sessions_lock:
            remaining = self._session_busy.get(session_id, 0) - 1
            if remaining > 0:
                self._session_busy[session_id] = remaining
                return
            self._session_busy.pop(session_id, None)
            deferred = self._deferred_closes.pop(session_id, None)
        if deferred is not None:
            self._close_session_agent(session_id, deferred)

    def _close_session_agent(self, session_id: str, session: WebSession) -> None:
        """Close an evicted/dropped session's agent, deferring while it is in use."""
        with self._sessions_lock:
            if self._session_busy.get(session_id, 0) > 0:
                self._deferred_closes[session_id] = session
                return
        close = getattr(session.agent, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Failed to close session agent: %s", session_id)

    def _evict_resident_sessions(self, *, keep: str = "") -> None:
        """Keep at most LANGCODE_MAX_RESIDENT_SESSIONS sessions in memory (LRU)."""
        limit = _max_resident_sessions()
        with self._sessions_lock:
            for session_id in list(self.sessions.keys()):
                if len(self.sessions) <= limit:
                    return
                if (
                    session_id == keep
                    or self._session_busy.get(session_id, 0) > 0
                    or self.runtime_state.has_active_run(session_id)
                ):
                    continue
                session = self.sessions.pop(session_id, None)
                if session is None:
                    continue
                self._close_session_agent(session_id, session)
                self._partial_save_futures.pop(session_id, None)

    def get_session(self, session_id: str) -> WebSession:
        with self._sessions_lock:
            if session_id in self.sessions:
                self.sessions.move_to_end(session_id)
                return self.sessions[session_id]

            stored = self.store.load_session(session_id)
            if stored is None:
                session_workspace = self.workspace_root
                messages = []
                self.store.ensure_session(session_id, str(session_workspace))
            else:
                session_workspace = self._resolve_workspace(Path(stored["workspace"]))
                messages = messages_from_json(stored["messages"])
            state = stored.get("state") if stored else {}
            display_messages = _restore_display_messages(state, session_workspace, session_id, messages)
            session = WebSession(
                id=session_id,
                agent=CodeAgent(session_workspace, checkpoint_path=self.state_dir / "checkpoints.sqlite"),
                workspace_root=session_workspace,
                messages=messages,
                display_messages=display_messages,
                pending=stored.get("pending") if stored else None,
                todos=list((state or {}).get("todos") or []) if stored else [],
                store_path=self.store.path,
                last_reflected_count=int((state or {}).get("last_reflected_count") or 0) if stored else 0,
                revision=int(stored.get("revision") or 0) if stored else 0,
            )
            self.sessions[session_id] = session
            self._evict_resident_sessions(keep=session_id)
            return session

    def refresh_session_from_store(self, session_id: str) -> None:
        """Refresh cached session state from SQLite after cross-process handoff.

        The web app keeps CodeAgent/session objects in memory for local speed,
        but when Redis allows multiple workers to coordinate a session, another
        worker may have saved a newer pending approval or message history. This
        method is called after acquiring the per-session lock so the cached
        object catches up before handling the request.
        """
        if not session_id:
            return
        with self._sessions_lock:
            session = self.sessions.get(session_id)
            if session is None:
                return
            revision = self.store.load_revision(session_id)
            if revision == DELETED_REVISION:
                # Item 4: another worker deleted this session. Keeping the stale
                # cache alive used to resurrect it on the next save, so drop it.
                self.sessions.pop(session_id, None)
                self._partial_save_futures.pop(session_id, None)
                self._close_session_agent(session_id, session)
                return
            if revision is None or revision == session.revision:
                return
            stored = self.store.load_session(session_id)
            if stored is None:
                return
            session_workspace = self._resolve_workspace(Path(stored["workspace"]))
            if session.workspace_root != session_workspace:
                session.agent.close()
                session.agent = CodeAgent(session_workspace, checkpoint_path=self.state_dir / "checkpoints.sqlite")
                session.workspace_root = session_workspace
                session.model = None
            messages = messages_from_json(stored["messages"])
            state = stored.get("state") if stored else {}
            display_messages = _restore_display_messages(state, session_workspace, session_id, messages)
            session.messages = messages
            session.display_messages = display_messages
            session.pending = stored.get("pending")
            session.todos = list((state or {}).get("todos") or [])
            session.last_reflected_count = int((state or {}).get("last_reflected_count") or 0)
            session.revision = int(stored.get("revision") or 0)

    def status(self) -> dict:
        settings = model_settings_from_env()
        voice = self._voice_status()
        return {
            "workspace": str(self.workspace_root),
            "provider": settings.provider,
            "model": settings.model,
            "gateway": os.getenv("LANGCODE_OPENAI_GATEWAY") or "",
            "baseUrl": settings.base_url,
            "hasApiKey": bool(settings.api_key),
            "thinking": (os.getenv("LANGCODE_THINKING") or "").strip().lower() in {"1", "true", "yes", "on", "y"},
            "runtimeState": self.runtime_state.status(),
            "jobQueue": self.job_queue.status(),
            "voiceWorker": voice.get("worker"),
            "asr": voice.get("asr"),
            "turnsense": voice.get("turnsense"),
            "tts": voice.get("tts"),
        }

    def asr_status(self) -> dict:
        if self.voice_worker is not None:
            return self.voice_worker.asr_status()
        if self.asr is None:
            return {"ok": False, "error": "ASR service is not available."}
        return self.asr.status()

    def tts_status(self) -> dict:
        if self.voice_worker is not None:
            return self.voice_worker.tts_status()
        if self.tts is None:
            return {"ok": False, "error": "TTS service is not available."}
        return self.tts.status()

    def tts_voices(self) -> dict:
        if self.voice_worker is not None:
            return self.voice_worker.list_tts_voices()
        if self.tts is None:
            return {"ok": False, "error": "TTS service is not available.", "voices": []}
        return {"ok": True, "voices": self.tts.list_voices()}

    def create_tts_voice(self, payload: dict) -> dict:
        if self.voice_worker is not None:
            return self.voice_worker.create_tts_voice(payload)
        if self.tts is None:
            return {"ok": False, "error": "TTS service is not available."}
        audio = _decode_data_url_or_base64(str(payload.get("audio") or ""))
        profile = self.tts.create_voice_profile(
            name=str(payload.get("name") or ""),
            prompt_text=str(payload.get("promptText") or ""),
            style=str(payload.get("style") or ""),
            wav_bytes=audio,
        )
        return {"ok": True, "voice": profile, "voices": self.tts.list_voices()}

    def tts_voice_preview(self, voice_id: str) -> tuple[bytes, str]:
        voice_id = unquote(voice_id)
        if self.voice_worker is not None:
            return self.voice_worker.tts_voice_preview(voice_id)
        if self.tts is None:
            raise RuntimeError("TTS service is not available.")
        return self.tts.voice_preview(voice_id)

    def tts_speech(self, payload: dict) -> tuple[bytes, str]:
        if self.voice_worker is not None:
            return self.voice_worker.tts_speech(payload)
        if self.tts is None:
            raise RuntimeError("TTS service is not available.")
        return self.tts.synthesize(
            str(payload.get("text") or ""),
            str(payload.get("voiceId") or ""),
        )

    def tts_claim_turn(self, session_id: str, turn_id: str) -> None:
        """Make ``turn_id`` the newest TTS turn of ``session_id``."""
        self._tts_turns.claim(session_id, turn_id)

    def cancel_tts_turn(self, payload: dict) -> dict:
        """Mark one TTS turn cancelled (barge-in / stop), stopping its producer."""
        if not self._tts_turns.cancel(payload.get("sessionId"), payload.get("turnId")):
            return {"ok": False, "error": "Session id and turn id are required"}
        return {"ok": True}

    def is_tts_turn_stale(self, session_id: str, turn_id: str) -> bool:
        """True when this turn was cancelled or superseded by a newer one."""
        return self._tts_turns.is_stale(session_id, turn_id)

    def _voice_status(self) -> dict[str, Any]:
        if self.voice_worker is not None:
            try:
                status = self.voice_worker.status()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                return {
                    "worker": {"enabled": True, "ok": False, "url": self.voice_worker.base_url, "error": error},
                    "asr": {"ok": False, "error": error},
                    "turnsense": {"ok": False, "error": error},
                    "tts": {"ok": False, "error": error},
                }
            return {
                "worker": {"enabled": True, "ok": bool(status.get("ok")), "url": self.voice_worker.base_url},
                "asr": status.get("asr"),
                "turnsense": status.get("turnsense"),
                "tts": status.get("tts"),
            }
        return {
            "worker": {"enabled": False, "ok": True, "url": ""},
            "asr": self.asr.status() if self.asr is not None else {"ok": False},
            "turnsense": self.turnsense.status() if self.turnsense is not None else {"ok": False},
            "tts": self.tts.status() if self.tts is not None else {"ok": False},
        }

    def directories(self, path: str | None = None) -> dict:
        target = Path(path).expanduser() if path else Path.home()
        try:
            current = target.resolve()
        except OSError as exc:
            return {"ok": False, "error": f"Cannot resolve directory: {exc}"}
        if not current.exists() or not current.is_dir():
            return {"ok": False, "error": f"Directory does not exist: {current}"}

        directories = []
        try:
            for child in current.iterdir():
                try:
                    if child.is_dir():
                        directories.append({"name": child.name, "path": str(child.resolve())})
                except OSError:
                    continue
        except PermissionError:
            return {"ok": False, "error": f"Permission denied: {current}"}
        except OSError as exc:
            return {"ok": False, "error": f"Cannot list directory: {exc}"}

        directories.sort(key=lambda item: item["name"].lower())
        parent = current.parent if current.parent != current else None
        return {
            "ok": True,
            "path": str(current),
            "parent": str(parent) if parent else None,
            "home": str(Path.home().resolve()),
            "workspace": str(self.workspace_root),
            "roots": _directory_roots(self.workspace_root),
            "directories": directories,
        }

    def choose_directory(self, payload: dict) -> dict:
        if platform.system() != "Darwin":
            return {"ok": False, "error": "Native directory picker is only available on macOS."}

        start = str(payload.get("start") or self.workspace_root)
        prompt = str(payload.get("prompt") or "Choose workspace directory")
        try:
            start_path = self._resolve_workspace(Path(start))
        except ValueError:
            start_path = self.workspace_root

        script = """
on run argv
  set startFolder to POSIX file (item 1 of argv)
  set promptText to item 2 of argv
  tell application "Finder" to activate
  try
    set chosenFolder to choose folder with prompt promptText default location startFolder
    return POSIX path of chosenFolder
  on error number -128
    return ""
  end try
end run
"""
        try:
            completed = subprocess.run(
                ["osascript", "-e", script, str(start_path), prompt],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": f"Native directory picker failed: {exc}"}

        if completed.returncode != 0:
            return {"ok": False, "error": completed.stderr.strip() or "Native directory picker failed."}

        selected = completed.stdout.strip()
        if not selected:
            return {"ok": True, "cancelled": True, "path": None}
        try:
            selected_path = self._resolve_workspace(Path(selected))
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "cancelled": False, "path": str(selected_path)}

    def set_workspace(self, payload: dict) -> dict:
        raw_workspace = str(payload.get("workspace") or "").strip()
        if not raw_workspace:
            return {"ok": False, "error": "Workspace path is required"}
        try:
            self.workspace_root = self._resolve_workspace(Path(raw_workspace))
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        load_env_files(Path.cwd(), self.workspace_root)
        with self._sessions_lock:
            self.sessions.clear()
        return {"ok": True, **self.status()}

    def set_settings(self, payload: dict) -> dict:
        model = str(payload.get("model") or "").strip()
        provider = str(payload.get("provider") or "").strip().lower()
        gateway = str(payload.get("gateway") or "").strip().lower()
        thinking = bool(payload.get("thinking", False))
        if not model:
            return {"ok": False, "error": "Model is required"}
        if provider:
            os.environ["LANGCODE_PROVIDER"] = provider
        if gateway:
            os.environ["LANGCODE_OPENAI_GATEWAY"] = gateway
        else:
            os.environ.pop("LANGCODE_OPENAI_GATEWAY", None)
        os.environ["LANGCODE_MODEL"] = model
        os.environ["LANGCODE_THINKING"] = "true" if thinking else "false"
        with self._sessions_lock:
            for session in self.sessions.values():
                session.model = None
        return {"ok": True, **self.status()}

    def list_sessions(self) -> dict:
        with self._sessions_lock:
            active = set(self.sessions)
            items = [
                {
                    "id": item["id"],
                    "title": item["title"],
                    "workspace": item["workspace"],
                    "active": item["id"] in active,
                }
                for item in self.store.list_sessions()
            ]
        return {"ok": True, "sessions": items}

    def create_session(self, payload: dict) -> dict:
        session_id = str(payload.get("sessionId") or "").strip()
        raw_workspace = str(payload.get("workspace") or "").strip()
        if not session_id:
            return {"ok": False, "error": "Session id is required"}
        if not raw_workspace:
            return {"ok": False, "error": "Workspace path is required"}
        try:
            session_workspace = self._resolve_workspace(Path(raw_workspace))
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        with self._sessions_lock:
            self.store.ensure_session(session_id, str(session_workspace), title=session_id)
        return self.list_sessions()

    def session_view(self, session_id: str, before: Any = None, limit: Any = None) -> dict:
        """Return the rendered history.

        ``before``/``limit`` are optional (item 18). When both are absent the
        response is byte-for-byte the previous full history so existing clients
        keep working.
        """
        if not session_id:
            return {"ok": False, "error": "Session id is required"}
        session = self.get_session(session_id)
        messages = []
        for message in session.display_messages or session.messages:
            if message.type == "human":
                messages.append({"role": "user", "kind": "message", "content": str(message.content)})
            elif message.type == "ai":
                content = str(message.content or "")
                if content.strip():
                    messages.append({"role": "assistant", "kind": "message", "content": content})
            elif message.type == "tool":
                content = str(message.content)
                diagram = _diagram_message_from_tool_content(content)
                if diagram is not None:
                    messages.append(diagram)
                    continue
                agent_dialogue = _agent_dialogue_message_from_tool_content(content)
                if agent_dialogue is not None:
                    messages.append(agent_dialogue)
                elif _should_show_tool_result_content(content):
                    messages.append({"role": "tool", "kind": "tool_result", "content": content})
        total = len(messages)
        window = _history_window(before, limit)
        if window is None:
            page = messages
            start = 0
        else:
            before_index, page_size = window
            end = total if before_index is None else max(0, min(before_index, total))
            start = max(0, end - page_size)
            page = messages[start:end]
        return {
            "ok": True,
            "sessionId": session_id,
            "title": self.store.load_title(session_id) or session_id,
            "workspace": str(session.workspace_root),
            "messages": page,
            "pendingApproval": session.pending,
            "todos": session.todos,
            "total": total,
            "firstIndex": start,
            "hasMore": start > 0,
        }

    def rename_session(self, payload: dict) -> dict:
        session_id = str(payload.get("sessionId") or "").strip()
        title = str(payload.get("title") or "").strip()
        if not session_id:
            return {"ok": False, "error": "Session id is required"}
        if not title:
            return {"ok": False, "error": "Session name is required"}
        with self._sessions_lock:
            stored = self.store.load_session(session_id)
            workspace = stored["workspace"] if stored else str(self.workspace_root)
            self.store.ensure_session(session_id, workspace)
            self.store.rename_session(session_id, title)
        return self.list_sessions()

    def delete_session(self, payload: dict) -> dict:
        session_id = str(payload.get("sessionId") or "").strip()
        if not session_id:
            return {"ok": False, "error": "Session id is required"}
        with self._sessions_lock:
            session = self.sessions.pop(session_id, None)
            if session is not None:
                session.agent.close()
            self.runtime_state.clear_session(session_id)
            self.store.delete_session(session_id)
            _delete_checkpoint_thread(self.state_dir / "checkpoints.sqlite", session_id)
        return self.list_sessions()

    def clear_session(self, payload: dict) -> dict:
        session_id = str(payload.get("sessionId") or "").strip()
        if not session_id:
            return {"ok": False, "error": "Session id is required"}
        with self._sessions_lock:
            session = self.sessions.pop(session_id, None)
            if session is not None:
                session.agent.close()
            self.runtime_state.clear_session(session_id)
            self.store.clear_session(session_id)
            _delete_checkpoint_thread(self.state_dir / "checkpoints.sqlite", session_id)
        return self.list_sessions()

    def cancel_run(self, payload: dict) -> dict:
        session_id = str(payload.get("sessionId") or "").strip()
        run_id = str(payload.get("runId") or "").strip()
        if not session_id or not run_id:
            return {"ok": False, "error": "Session id and run id are required"}
        displayed = str(
            payload.get("assistantDisplayedText") or payload.get("assistant_displayed_text") or ""
        ).strip()
        if displayed:
            self._cancel_displayed_text[f"{session_id}:{run_id}"] = displayed
        self.runtime_state.cancel_run(session_id, run_id)
        return {"ok": True}

    def _take_cancel_displayed_text(self, session_id: str, run_id: str | None) -> str | None:
        if not run_id:
            return None
        return self._cancel_displayed_text.pop(f"{session_id}:{run_id}", None)

    def _is_run_cancelled(self, session_id: str, run_id: str | None) -> bool:
        if not run_id:
            return False
        return self.runtime_state.is_run_cancelled(session_id, run_id)

    def _forget_cancelled_run(self, session_id: str, run_id: str | None) -> None:
        if not run_id:
            return
        self.runtime_state.forget_cancelled_run(session_id, run_id)

    def chat(self, payload: dict) -> dict:
        session = self.get_session(str(payload.get("sessionId") or "default"))
        message = str(payload.get("message") or "").strip()
        if not message:
            return {"ok": False, "error": "Message is required"}
        if session.pending is not None:
            return {"ok": False, "error": "Resolve the pending tool approval before sending a new message."}

        local_reply = self._handle_local_command(session, message)
        if local_reply is not None:
            self._save_history(session)
            return {
                "ok": True,
                "messages": [{"role": "assistant", "content": local_reply, "kind": "message"}],
                "pendingApproval": None,
            }

        settings = model_settings_from_env()
        if not settings.api_key:
            return {
                "ok": False,
                "error": f"No model API key configured for provider={settings.provider}.",
                "messages": [],
            }

        if session.model is None:
            session.model = build_openai_model().bind_tools(_tool_schemas_for_model())
        _ensure_current_system_message(session)
        _repair_live_tool_history(session)

        human_message = HumanMessage(content=message)
        session.messages.append(human_message)
        session.display_messages.append(human_message)
        _append_voice_interrupt_context(session, payload.get("voiceInterrupt"))
        return self._continue_model(session, extra_messages=_voice_mode_messages(payload.get("voiceMode")))

    def chat_events(self, payload: dict):
        session = self.get_session(str(payload.get("sessionId") or "default"))
        run_id = str(payload.get("runId") or "").strip() or None
        message = str(payload.get("message") or "").strip()
        if not message:
            yield _error_event("Message is required", "internal")
            return
        if session.pending is not None:
            yield _error_event("Resolve the pending tool approval before sending a new message.", "internal")
            return

        local_reply = self._handle_local_command(session, message)
        if local_reply is not None:
            self._save_history(session)
            yield {"type": "delta", "content": local_reply}
            yield {"type": "done", "ok": True}
            return

        settings = model_settings_from_env()
        if not settings.api_key:
            yield _error_event(f"No model API key configured for provider={settings.provider}.", "auth")
            return

        if session.model is None:
            session.model = build_openai_model().bind_tools(_tool_schemas_for_model())
        _ensure_current_system_message(session)
        _repair_live_tool_history(session)

        human_message = HumanMessage(content=message)
        session.messages.append(human_message)
        session.display_messages.append(human_message)
        _append_voice_interrupt_context(session, payload.get("voiceInterrupt"))
        yield from self._continue_model_stream(
            session,
            run_id=run_id,
            extra_messages=_voice_mode_messages(payload.get("voiceMode")),
        )

    def approval_events(self, payload: dict):
        session = self.get_session(str(payload.get("sessionId") or "default"))
        run_id = str(payload.get("runId") or "").strip() or None
        if session.pending is None:
            yield _error_event("No pending approval for this session.", "internal")
            return
        if self._is_run_cancelled(session.id, run_id):
            yield {"type": "done", "ok": False, "cancelled": True}
            return

        approval = dict(payload.get("approval") or {})
        pending = session.pending
        tool_name = pending["toolName"]
        tool_input = dict(pending.get("toolInput") or {})
        yield _progress_event("running", tool_name, tool_input, 1, 1, 0)
        if approval.get("remember") and tool_name == "shell":
            approved_tool_input = approval.get("tool_input") if isinstance(approval.get("tool_input"), dict) else tool_input
            command = str(dict(approved_tool_input).get("command") or "").strip()
            if command:
                remember_shell_permission(session.workspace_root, command, "allow")

        result = session.agent.resume(session.id, approval)
        tool_result = _json_safe(result.get("tool_result", result))
        tool_result = _apply_self_evolve_result(session, tool_name, tool_input, tool_result)
        task_changed = False
        if isinstance(tool_result, dict):
            tool_result = compact_tool_result(session.workspace_root, session.id, tool_name, tool_result)
            tool_result, task_changed = _apply_task_tool_result(session, tool_name, tool_input, tool_result)
        session.pending = None

        session.messages.append(
            ToolMessage(
                content=_tool_result_json(tool_result),
                tool_call_id=pending["toolCallId"],
            )
        )
        session.display_messages.append(
            ToolMessage(
                content=_tool_result_json(tool_result),
                tool_call_id=pending["toolCallId"],
            )
        )
        self._save_history(session)
        if task_changed and isinstance(tool_result, dict) and isinstance(tool_result.get("todos"), list):
            yield {"type": "todos", "todos": session.todos, "summary": tool_result.get("summary", "")}
        yield _progress_event("completed", tool_name, tool_input, 1, 1, 1, tool_result)
        yield from _stream_tool_result_events(tool_name, tool_result)
        if self._is_run_cancelled(session.id, run_id):
            yield {"type": "done", "ok": False, "cancelled": True}
            return
        if session.model is None:
            session.model = build_openai_model().bind_tools(_tool_schemas_for_model())
        yield from self._continue_model_stream(session, run_id=run_id)

    def approve(self, payload: dict) -> dict:
        session = self.get_session(str(payload.get("sessionId") or "default"))
        if session.pending is None:
            return {"ok": False, "error": "No pending approval for this session."}

        approval = dict(payload.get("approval") or {})
        pending = session.pending
        if approval.get("remember") and pending.get("toolName") == "shell":
            tool_input = approval.get("tool_input") if isinstance(approval.get("tool_input"), dict) else pending.get("toolInput", {})
            command = str(dict(tool_input).get("command") or "").strip()
            if command:
                remember_shell_permission(session.workspace_root, command, "allow")
        result = session.agent.resume(session.id, approval)
        tool_result = _json_safe(result.get("tool_result", result))
        tool_result = _apply_self_evolve_result(
            session, pending["toolName"], dict(pending.get("toolInput") or {}), tool_result
        )
        if isinstance(tool_result, dict):
            tool_result = compact_tool_result(session.workspace_root, session.id, pending["toolName"], tool_result)
            tool_result, _task_changed = _apply_task_tool_result(
                session,
                pending["toolName"],
                dict(pending.get("toolInput") or {}),
                tool_result,
            )
        session.pending = None

        session.messages.append(
            ToolMessage(
                content=_tool_result_json(tool_result),
                tool_call_id=pending["toolCallId"],
            )
        )
        session.display_messages.append(
            ToolMessage(
                content=_tool_result_json(tool_result),
                tool_call_id=pending["toolCallId"],
            )
        )
        self._save_history(session)
        if session.model is None:
            session.model = build_openai_model().bind_tools(_tool_schemas_for_model())
        return self._continue_model(session)

    def _continue_model(self, session: WebSession, *, extra_messages: list[BaseMessage] | None = None) -> dict:
        _repair_live_tool_history(session)
        added: list[dict] = []
        plan_written_this_turn = False
        non_plan_tools_completed = 0
        continued_after_plan_only = False
        response_todos: list[dict] | None = None
        web_search_count = 0
        while True:
            web_search_limit_reached = False
            self._compact_session_history(session)
            ai_message = session.model.invoke([*session.messages, *(extra_messages or [])])
            content = _chunk_content_text(ai_message)
            if extra_messages:
                content = _sanitize_voice_mode_output(content)
            tool_calls = list(getattr(ai_message, "tool_calls", None) or [])
            if content != _content_blocks_text(getattr(ai_message, "content", ""), include_thinking=False):
                ai_message = AIMessage(content=content, tool_calls=tool_calls)
            session.messages.append(ai_message)
            session.display_messages.append(ai_message)
            if content.strip():
                added.append({"role": "assistant", "content": content, "kind": "message"})
            if not tool_calls:
                if _should_continue_after_plan_only(
                    session,
                    plan_written_this_turn,
                    non_plan_tools_completed,
                    continued_after_plan_only,
                ):
                    continued_after_plan_only = True
                    session.messages.append(_plan_execution_reminder())
                    added.append({"role": "assistant", "content": "计划已创建，继续执行第一项任务。", "kind": "message"})
                    continue
                self._save_history(session)
                response = {"ok": True, "messages": added, "pendingApproval": None}
                if response_todos is not None:
                    response["todos"] = response_todos
                return response

            for raw_tool_call in tool_calls:
                tool_name = raw_tool_call["name"]
                tool_input = dict(raw_tool_call.get("args") or {})
                tool_input = _prepare_tool_input(tool_name, tool_input, session)
                if _is_external_web_search_tool(tool_name):
                    if web_search_count >= _web_search_limit():
                        tool_result = _web_search_limit_result(tool_input, web_search_count)
                        session.messages.append(
                            ToolMessage(
                                content=_tool_result_json(tool_result),
                                tool_call_id=raw_tool_call.get("id") or tool_name,
                            )
                        )
                        session.display_messages.append(
                            ToolMessage(
                                content=_tool_result_json(tool_result),
                                tool_call_id=raw_tool_call.get("id") or tool_name,
                            )
                        )
                        web_search_limit_reached = True
                        continue
                    web_search_count += 1
                if tool_name == "voice_interrupt":
                    tool_result = _voice_interrupt_tool_result(tool_input)
                    session.messages.append(
                        ToolMessage(
                            content=_tool_result_json(tool_result),
                            tool_call_id=raw_tool_call.get("id") or tool_name,
                        )
                    )
                    continue
                if tool_name == "delegate_agent":
                    tool_result = _json_safe(run_delegate_agent(session.workspace_root, **tool_input))
                    tool_result = compact_tool_result(session.workspace_root, session.id, tool_name, tool_result)
                    tool_event = _tool_result_event(tool_name, tool_result)
                    if tool_event:
                        added.append(tool_event)
                    session.messages.append(
                        ToolMessage(
                            content=_tool_result_json(tool_result),
                            tool_call_id=raw_tool_call.get("id") or tool_name,
                        )
                    )
                    session.display_messages.append(
                        ToolMessage(
                            content=_tool_result_json(tool_result),
                            tool_call_id=raw_tool_call.get("id") or tool_name,
                        )
                    )
                    non_plan_tools_completed += 1
                    continue
                if tool_name == "delegate_agents":
                    tool_result = _json_safe(run_parallel_delegate_agents(session.workspace_root, **tool_input))
                    tool_result = compact_tool_result(session.workspace_root, session.id, tool_name, tool_result)
                    tool_event = _tool_result_event(tool_name, tool_result)
                    if tool_event:
                        added.append(tool_event)
                    session.messages.append(
                        ToolMessage(
                            content=_tool_result_json(tool_result),
                            tool_call_id=raw_tool_call.get("id") or tool_name,
                        )
                    )
                    session.display_messages.append(
                        ToolMessage(
                            content=_tool_result_json(tool_result),
                            tool_call_id=raw_tool_call.get("id") or tool_name,
                        )
                    )
                    non_plan_tools_completed += 1
                    continue
                if tool_name == "agent_debate":
                    tool_result = _json_safe(run_agent_debate(session.workspace_root, **tool_input))
                    tool_event = _tool_result_event(tool_name, tool_result)
                    if tool_event:
                        added.append(tool_event)
                    session.messages.append(
                        ToolMessage(
                            content=_tool_result_json(tool_result),
                            tool_call_id=raw_tool_call.get("id") or tool_name,
                        )
                    )
                    session.display_messages.append(
                        ToolMessage(
                            content=_tool_result_json(tool_result),
                            tool_call_id=raw_tool_call.get("id") or tool_name,
                        )
                    )
                    non_plan_tools_completed += 1
                    continue

                result = session.agent.request_tool(ToolCall(tool_name, tool_input), thread_id=session.id)
                if "__interrupt__" in result:
                    pending = {
                        "kind": "chat_tool",
                        "threadId": session.id,
                        "toolCallId": raw_tool_call.get("id") or tool_name,
                        "toolName": tool_name,
                        "toolInput": tool_input,
                        "payload": result["__interrupt__"][0].value,
                    }
                    session.pending = pending
                    self._save_history(session)
                    response = {"ok": True, "messages": added, "pendingApproval": pending}
                    if response_todos is not None:
                        response["todos"] = response_todos
                    return response

                tool_result = _json_safe(result.get("tool_result", result))
                tool_result = _apply_self_evolve_result(session, tool_name, tool_input, tool_result)
                if isinstance(tool_result, dict):
                    tool_result = compact_tool_result(session.workspace_root, session.id, tool_name, tool_result)
                    tool_result, task_changed = _apply_task_tool_result(session, tool_name, tool_input, tool_result)
                    if task_changed:
                        response_todos = list(session.todos)
                        plan_written_this_turn = True
                    elif not _is_task_tool(tool_name):
                        non_plan_tools_completed += 1
                tool_event = _tool_result_event(tool_name, tool_result)
                if tool_event:
                    added.append(tool_event)
                session.messages.append(
                    ToolMessage(
                        content=_tool_result_json(tool_result),
                        tool_call_id=raw_tool_call.get("id") or tool_name,
                    )
                )
                session.display_messages.append(
                    ToolMessage(
                        content=_tool_result_json(tool_result),
                        tool_call_id=raw_tool_call.get("id") or tool_name,
                    )
                )
            if web_search_limit_reached:
                session.messages.append(_web_search_limit_reminder())

    def _continue_model_stream(
        self,
        session: WebSession,
        *,
        run_id: str | None = None,
        extra_messages: list[BaseMessage] | None = None,
    ):
        try:
            yield from self._continue_model_stream_events(
                session,
                run_id=run_id,
                extra_messages=extra_messages,
            )
        finally:
            # Item A9: never leave a partial-save future behind when the stream is
            # abandoned (client disconnect, cancellation, generator close).
            self._drain_partial_save(session.id)

    def _drain_partial_save(self, session_id: str, *, timeout: float = 10.0) -> None:
        future = self._partial_save_futures.pop(session_id, None)
        if future is None:
            return
        try:
            future.result(timeout=timeout)
        except Exception:
            logger.exception("Partial history save failed for session %s", session_id)

    def _continue_model_stream_events(
        self,
        session: WebSession,
        *,
        run_id: str | None = None,
        extra_messages: list[BaseMessage] | None = None,
    ):
        _repair_live_tool_history(session)
        completed_tools = 0
        plan_written_this_turn = False
        non_plan_tools_completed = 0
        continued_after_plan_only = False
        turn_start_message_count = _failed_turn_start_index(session.messages)
        turn_start_display_count = _failed_turn_start_index(session.display_messages)
        web_search_count = 0
        round_index = 0
        tool_round_limit = _max_tool_rounds()
        tool_round_limit_reached = False
        idle_timeout = _stream_idle_timeout_sec()
        while True:
            round_index += 1
            web_search_limit_reached = False
            if self._is_run_cancelled(session.id, run_id):
                _drop_current_failed_turn(session, turn_start_message_count, turn_start_display_count)
                self._save_history(session)
                yield {"type": "done", "ok": False, "cancelled": True}
                return
            full_chunk = None
            partial_content: list[str] = []
            emitted_content = ""
            raw_thinking_open = False
            voice_output_filter = _VoiceModeOutputFilter() if extra_messages else None
            last_partial_save = 0.0
            last_cancel_check = time.monotonic()
            if self._compact_session_history(session):
                yield _notice_event("context_compacted", "上下文已压缩，较早的对话已折叠为摘要。")
            # Item 24: a threading.Timer flags the stall from outside the loop, so a
            # model that goes silent is caught even while the iterator is blocked
            # (the loop itself never sleeps or polls a clock for this).
            watchdog = _StreamIdleWatchdog(idle_timeout, session.id)
            stream_completed = False
            try:
                stream_iterator = session.model.stream([*session.messages, *(extra_messages or [])])
                # Item 11: hand the iterator to the watchdog so a stall can try to
                # close it instead of waiting for a chunk that never comes.
                watchdog.attach(stream_iterator)
                watchdog.start()
                for chunk in stream_iterator:
                    if watchdog.timed_out:
                        break
                    watchdog.beat()
                    now = time.monotonic()
                    should_check_remote = now - last_cancel_check >= 0.25
                    if self.runtime_state.is_run_cancelled_local(session.id, run_id) or (
                        should_check_remote and self._is_run_cancelled(session.id, run_id)
                    ):
                        if partial_content:
                            _save_cancelled_partial(
                                session,
                                partial_content,
                                self._save_history,
                                displayed_text=self._take_cancel_displayed_text(session.id, run_id),
                            )
                        else:
                            _drop_current_failed_turn(session, turn_start_message_count, turn_start_display_count)
                            self._save_history(session)
                        yield {"type": "done", "ok": False, "cancelled": True}
                        return
                    if should_check_remote:
                        last_cancel_check = now
                    full_chunk = chunk if full_chunk is None else full_chunk + chunk
                    thinking = _chunk_thinking_text(chunk)
                    if thinking:
                        yield {"type": "thinking_delta", "content": thinking}
                    raw_content = _chunk_content_text(chunk, strip_raw_thinking=False)
                    content, raw_thinking, raw_thinking_open = _split_raw_thinking_text(
                        raw_content,
                        in_thinking=raw_thinking_open,
                    )
                    if raw_thinking:
                        yield {"type": "thinking_delta", "content": raw_thinking}
                    if content:
                        content = _novel_stream_content(emitted_content, content)
                        emitted_content += content
                    if content and voice_output_filter is not None:
                        content = voice_output_filter.push(content)
                    if content:
                        partial_content.append(content)
                        if now - last_partial_save >= 3.0:
                            self._save_partial_history(session, "".join(partial_content))
                            last_partial_save = now
                        yield {"type": "delta", "content": content}
                else:
                    # The iterator ran to completion. Disarm the timer right here
                    # so a timeout raised *after* the last chunk cannot turn a
                    # finished answer into a fake stall (item 11): cancel() and
                    # _check() take the same lock, so exactly one of them wins.
                    watchdog.cancel()
                    stream_completed = not watchdog.timed_out
            except Exception as exc:
                if not watchdog.timed_out:
                    if partial_content:
                        self._save_history(session, extra_ai_content="".join(partial_content))
                    else:
                        _drop_current_failed_turn(session, turn_start_message_count, turn_start_display_count)
                        self._save_history(session)
                    yield _exception_error_event(exc, context="model stream")
                    return
                # The watchdog closed the stalled iterator; report the stall below
                # instead of the GeneratorExit fallout it produced.
                logger.debug("Stalled model stream raised while being closed: %r", exc)
            finally:
                watchdog.cancel()

            if watchdog.timed_out and not stream_completed:
                if partial_content:
                    _save_cancelled_partial(session, partial_content, self._save_history)
                else:
                    _drop_current_failed_turn(session, turn_start_message_count, turn_start_display_count)
                    self._save_history(session)
                yield _error_event(
                    f"{STREAM_IDLE_TIMEOUT_ERROR}（{idle_timeout:g} 秒）",
                    "model_timeout",
                    retriable=True,
                )
                return

            if voice_output_filter is not None:
                content = voice_output_filter.flush()
                if content:
                    partial_content.append(content)
                    self._save_history(session, extra_ai_content="".join(partial_content))
                    last_partial_save = time.monotonic()
                    yield {"type": "delta", "content": content}

            if self._is_run_cancelled(session.id, run_id):
                if partial_content:
                    _save_cancelled_partial(
                        session,
                        partial_content,
                        self._save_history,
                        displayed_text=self._take_cancel_displayed_text(session.id, run_id),
                    )
                else:
                    _drop_current_failed_turn(session, turn_start_message_count, turn_start_display_count)
                    self._save_history(session)
                yield {"type": "done", "ok": False, "cancelled": True}
                return
            if full_chunk is None:
                _drop_current_failed_turn(session, turn_start_message_count, turn_start_display_count)
                self._save_history(session)
                yield _error_event("Model returned no response.", "internal")
                return

            usage_event = _usage_event(full_chunk)
            if usage_event is not None:
                yield usage_event

            tool_calls = list(getattr(full_chunk, "tool_calls", None) or [])
            ai_content = "".join(partial_content)
            ai_message = AIMessage(content=ai_content, tool_calls=tool_calls)
            session.messages.append(ai_message)
            session.display_messages.append(ai_message)

            if not tool_calls:
                if _should_continue_after_plan_only(
                    session,
                    plan_written_this_turn,
                    non_plan_tools_completed,
                    continued_after_plan_only,
                ):
                    continued_after_plan_only = True
                    session.messages.append(_plan_execution_reminder())
                    yield {
                        "type": "progress",
                        "status": "summary",
                        "completed": completed_tools,
                        "label": "计划已创建，继续执行第一项任务。",
                    }
                    continue
                self._save_history(session)
                yield {"type": "done", "ok": True}
                return

            if tool_round_limit_reached:
                # The model kept calling tools after being told to stop: close the
                # dangling tool calls and finish the turn with what we already have.
                _answer_unanswered_tool_calls(session, _tool_round_limit_result(round_index))
                self._save_history(session)
                yield _notice_event("tool_round_limit", TOOL_ROUND_LIMIT_MOCK_USER)
                yield {"type": "done", "ok": True}
                return
            if round_index >= tool_round_limit:
                tool_round_limit_reached = True
                _answer_unanswered_tool_calls(session, _tool_round_limit_result(round_index))
                session.messages.append(_tool_round_limit_reminder())
                self._save_history(session)
                yield _notice_event("tool_round_limit", TOOL_ROUND_LIMIT_MOCK_USER)
                continue

            total_tools = len(tool_calls)
            for index, raw_tool_call in enumerate(tool_calls, start=1):
                if self._is_run_cancelled(session.id, run_id):
                    _answer_unanswered_tool_calls(session)
                    self._save_history(session)
                    yield {"type": "done", "ok": False, "cancelled": True}
                    return
                tool_name = raw_tool_call["name"]
                tool_input = dict(raw_tool_call.get("args") or {})
                tool_input = _prepare_tool_input(tool_name, tool_input, session)
                task_changed = False
                yield _progress_event("running", tool_name, tool_input, index, total_tools, completed_tools)
                if _is_external_web_search_tool(tool_name):
                    if web_search_count >= _web_search_limit():
                        tool_result = _web_search_limit_result(tool_input, web_search_count)
                        session.messages.append(
                            ToolMessage(
                                content=_tool_result_json(tool_result),
                                tool_call_id=raw_tool_call.get("id") or tool_name,
                            )
                        )
                        session.display_messages.append(
                            ToolMessage(
                                content=_tool_result_json(tool_result),
                                tool_call_id=raw_tool_call.get("id") or tool_name,
                            )
                        )
                        completed_tools += 1
                        web_search_limit_reached = True
                        yield _progress_event(
                            "completed",
                            tool_name,
                            tool_input,
                            index,
                            total_tools,
                            completed_tools,
                            tool_result,
                        )
                        yield {
                            "type": "progress",
                            "status": "summary",
                            "completed": completed_tools,
                            "label": WEB_SEARCH_LIMIT_ERROR,
                        }
                        continue
                    web_search_count += 1
                if tool_name == "voice_interrupt":
                    tool_result = _voice_interrupt_tool_result(tool_input)
                    session.messages.append(
                        ToolMessage(
                            content=_tool_result_json(tool_result),
                            tool_call_id=raw_tool_call.get("id") or tool_name,
                        )
                    )
                    completed_tools += 1
                    yield _progress_event(
                        "completed",
                        tool_name,
                        tool_input,
                        index,
                        total_tools,
                        completed_tools,
                        tool_result,
                    )
                    continue
                if tool_name == "delegate_agent":
                    tool_result = _json_safe(run_delegate_agent(session.workspace_root, **tool_input))
                    tool_result = compact_tool_result(session.workspace_root, session.id, tool_name, tool_result)
                    session.messages.append(
                        ToolMessage(
                            content=_tool_result_json(tool_result),
                            tool_call_id=raw_tool_call.get("id") or tool_name,
                        )
                    )
                    session.display_messages.append(
                        ToolMessage(
                            content=_tool_result_json(tool_result),
                            tool_call_id=raw_tool_call.get("id") or tool_name,
                        )
                    )
                    completed_tools += 1
                    non_plan_tools_completed += 1
                    yield _progress_event(
                        "completed",
                        tool_name,
                        tool_input,
                        index,
                        total_tools,
                        completed_tools,
                        tool_result,
                    )
                    yield from _stream_tool_result_events(tool_name, tool_result)
                    continue
                if tool_name == "delegate_agents":
                    tool_result = _json_safe(run_parallel_delegate_agents(session.workspace_root, **tool_input))
                    tool_result = compact_tool_result(session.workspace_root, session.id, tool_name, tool_result)
                    session.messages.append(
                        ToolMessage(
                            content=_tool_result_json(tool_result),
                            tool_call_id=raw_tool_call.get("id") or tool_name,
                        )
                    )
                    session.display_messages.append(
                        ToolMessage(
                            content=_tool_result_json(tool_result),
                            tool_call_id=raw_tool_call.get("id") or tool_name,
                        )
                    )
                    completed_tools += 1
                    non_plan_tools_completed += 1
                    yield _progress_event(
                        "completed",
                        tool_name,
                        tool_input,
                        index,
                        total_tools,
                        completed_tools,
                        tool_result,
                    )
                    yield from _stream_tool_result_events(tool_name, tool_result)
                    continue
                if tool_name == "agent_debate":
                    tool_result = None
                    for debate_event in iter_agent_debate_events(session.workspace_root, **tool_input):
                        tool_result = _json_safe(debate_event)
                        tool_event = _tool_result_event(tool_name, tool_result)
                        if tool_event:
                            yield {"type": "tool_result", **tool_event}
                    if tool_result is None:
                        tool_result = {"ok": False, "error": "辩论没有产生任何发言。"}
                    session.messages.append(
                        ToolMessage(
                            content=_tool_result_json(tool_result),
                            tool_call_id=raw_tool_call.get("id") or tool_name,
                        )
                    )
                    session.display_messages.append(
                        ToolMessage(
                            content=_tool_result_json(tool_result),
                            tool_call_id=raw_tool_call.get("id") or tool_name,
                        )
                    )
                    completed_tools += 1
                    non_plan_tools_completed += 1
                    yield _progress_event(
                        "completed",
                        tool_name,
                        tool_input,
                        index,
                        total_tools,
                        completed_tools,
                        tool_result,
                    )
                    continue

                result = session.agent.request_tool(ToolCall(tool_name, tool_input), thread_id=session.id)
                if self._is_run_cancelled(session.id, run_id):
                    _answer_unanswered_tool_calls(session)
                    self._save_history(session)
                    yield {"type": "done", "ok": False, "cancelled": True}
                    return
                if "__interrupt__" in result:
                    pending = {
                        "kind": "chat_tool",
                        "threadId": session.id,
                        "toolCallId": raw_tool_call.get("id") or tool_name,
                        "toolName": tool_name,
                        "toolInput": tool_input,
                        "payload": result["__interrupt__"][0].value,
                    }
                    session.pending = pending
                    self._save_history(session)
                    yield _progress_event("waiting_approval", tool_name, tool_input, index, total_tools, completed_tools)
                    yield {"type": "pending_approval", "ok": True, "pendingApproval": pending}
                    return

                tool_result = _json_safe(result.get("tool_result", result))
                tool_result = _apply_self_evolve_result(session, tool_name, tool_input, tool_result)
                if isinstance(tool_result, dict):
                    tool_result = compact_tool_result(session.workspace_root, session.id, tool_name, tool_result)
                    tool_result, task_changed = _apply_task_tool_result(session, tool_name, tool_input, tool_result)
                    if task_changed:
                        plan_written_this_turn = True
                    elif not _is_task_tool(tool_name):
                        non_plan_tools_completed += 1
                session.messages.append(
                    ToolMessage(
                        content=_tool_result_json(tool_result),
                        tool_call_id=raw_tool_call.get("id") or tool_name,
                    )
                )
                session.display_messages.append(
                    ToolMessage(
                        content=_tool_result_json(tool_result),
                        tool_call_id=raw_tool_call.get("id") or tool_name,
                    )
                )
                if _is_task_tool(tool_name) and task_changed:
                    self._save_history(session)
                    if isinstance(tool_result, dict) and isinstance(tool_result.get("todos"), list):
                        yield {"type": "todos", "todos": session.todos, "summary": tool_result.get("summary", "")}
                completed_tools += 1
                yield _progress_event(
                    "completed",
                    tool_name,
                    tool_input,
                    index,
                    total_tools,
                    completed_tools,
                    tool_result,
                )
                yield from _stream_tool_result_events(tool_name, tool_result)

            if web_search_limit_reached:
                session.messages.append(_web_search_limit_reminder())
            yield {
                "type": "progress",
                "status": "summary",
                "completed": completed_tools,
                "label": f"已完成 {completed_tools} 条工具指令，正在整理结果并决定下一步。",
            }

    def _handle_local_command(self, session: WebSession, message: str) -> str | None:
        archive_path = None
        previous_display = list(session.display_messages or session.messages)
        before_messages = list(session.messages)
        if message.strip() == "/agents":
            session.messages.append(HumanMessage(content=message))
            threads = self.store.list_agent_threads(session.id)
            reply = _agents_command_reply(threads)
            session.messages.append(AIMessage(content=reply))
            session.display_messages.extend(session.messages[len(before_messages) :])
            return reply
        if message.strip().startswith("/compact"):
            stamp = session.id + "-" + str(len(session.messages)).zfill(4)
            archive_path = session.workspace_root / ".langcode" / "compactions" / f"{stamp}.json"
        reply = handle_local_command(session.workspace_root, session.messages, message, archive_path=archive_path)
        if reply is None:
            return None
        if message.strip().startswith("/compact"):
            session.display_messages = previous_display + [HumanMessage(content=message), AIMessage(content=reply)]
        else:
            session.display_messages.extend(session.messages[len(before_messages) :])
        return reply

    def _compact_session_history(self, session: WebSession) -> bool:
        """Summarize old history when it exceeds the context budget (item 36).

        Only ``session.messages`` (what the model sees) is compacted; the
        UI-facing ``display_messages`` keep the full transcript. Persists when
        anything changed. Never raises: a compaction failure must not block the
        turn, so errors are logged and the history is left as-is.
        """
        try:
            compacted, changed = compact_history_if_needed(session.messages, model=session.model)
        except Exception:
            logger.exception("Context compaction failed for session %s", session.id)
            return False
        if not changed:
            return False
        session.messages = list(compacted)
        self._save_history(session)
        return True

    def _write_display_archive(self, session: WebSession, display_serialized: list[dict]) -> str:
        """Store the full displayed history as an artifact file; return its name.

        Returns "" when the file cannot be written, so the caller can fall back to
        the inline copy rather than lose the transcript. The file is rewritten only
        when its content actually changed, which is the common case for a session
        that keeps chatting after a /compact.
        """
        try:
            # The digest must cover only the transcript itself. Hashing the
            # rendered payload would fold ``updatedAt`` into the key and the
            # "unchanged" branch below could then never be taken.
            body = json.dumps(
                {"sessionId": session.id, "messages": display_serialized},
                ensure_ascii=False,
                sort_keys=True,
            )
            digest = hashlib.sha1(body.encode("utf-8")).hexdigest()
            name = _display_archive_name(session.id)
            path = _display_archive_dir(session.workspace_root) / name
            if self._display_archive_digests.get(session.id) == digest and path.exists():
                return name
            payload = json.dumps(
                {"sessionId": session.id, "updatedAt": time.time(), "messages": display_serialized},
                ensure_ascii=False,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(path)
            self._display_archive_digests[session.id] = digest
            return name
        except OSError:
            logger.exception("Could not write the display history archive for session %s", session.id)
            self._display_archive_digests.pop(session.id, None)
            return ""

    def _save_history(self, session: WebSession, *, extra_ai_content: str | None = None) -> None:
        pending_save = self._partial_save_futures.pop(session.id, None)
        if pending_save is not None:
            pending_save.result()
        messages = list(session.messages)
        display_messages = list(session.display_messages or session.messages)
        if extra_ai_content:
            messages.append(AIMessage(content=extra_ai_content))
            display_messages.append(AIMessage(content=extra_ai_content))

        serializable = _serialize_messages(messages)
        display_serialized = _serialize_messages(display_messages)
        # Item 6.4: display_messages normally mirrors messages minus the system
        # prompt; only a /compact makes them genuinely diverge. In the common
        # case store a marker instead of a second full copy of the history.
        display_is_derived = display_serialized == _without_system_messages(serializable)

        archive_name = ""
        if not display_is_derived:
            # Item 12: once a /compact has made the two histories diverge,
            # display_is_derived stays False forever and every later save used to
            # write a second full copy of the transcript into state_json. Park the
            # full displayed history in an artifact file instead and keep only a
            # bounded tail in the row.
            archive_name = self._write_display_archive(session, display_serialized)

        def _state() -> dict:
            state: dict = {
                "todos": session.todos,
                "last_reflected_count": session.last_reflected_count,
            }
            if display_is_derived:
                state["display_same_as_messages"] = True
            elif archive_name:
                state["display_archive"] = archive_name
                state["display_messages_tail"] = display_serialized[-DISPLAY_TAIL_LIMIT:]
            else:
                # The artifact could not be written (read-only workspace, ...):
                # never lose history, fall back to the inline copy.
                state["display_messages"] = display_serialized
            return state

        self.store.save_messages(
            session.id,
            str(session.workspace_root),
            serializable,
            pending=session.pending,
            state=_state(),
        )
        # Item 38: keep checkpoints.sqlite bounded. Only prune once the turn is
        # settled (no pending approval), so an interrupted graph can still resume.
        if session.pending is None:
            prune = getattr(session.agent, "prune_thread", None)
            if callable(prune):
                try:
                    prune(session.id)
                except Exception:
                    logger.exception("Checkpoint pruning failed for session %s", session.id)
        if _auto_reflect_enabled() and extra_ai_content is None and session.pending is None:
            if len(serializable) > session.last_reflected_count and messages and isinstance(messages[-1], AIMessage):
                reflect_session(
                    session.workspace_root,
                    session_id=session.id,
                    messages=serializable,
                    todos=session.todos,
                    apply=True,
                )
                session.last_reflected_count = len(serializable)
                self.store.save_messages(
                    session.id,
                    str(session.workspace_root),
                    serializable,
                    pending=session.pending,
                    state=_state(),
                )
        saved_revision = self.store.load_revision(session.id)
        if saved_revision is not None and saved_revision != DELETED_REVISION:
            session.revision = saved_revision or session.revision

    def _save_partial_history(self, session: WebSession, content: str) -> None:
        previous = self._partial_save_futures.get(session.id)
        if previous is not None and not previous.done():
            return
        future = self._partial_save_pool.submit(
            self.store.upsert_partial,
            session.id,
            len(session.messages),
            "assistant",
            content,
        )
        self._partial_save_futures[session.id] = future

        def forget(completed: Future) -> None:
            if self._partial_save_futures.get(session.id) is completed:
                self._partial_save_futures.pop(session.id, None)

        future.add_done_callback(forget)


def _system_message(workspace_root: Path):
    from langchain_core.messages import SystemMessage

    return SystemMessage(content=default_system_prompt(str(workspace_root)))


def _is_langcode_system_message(message: BaseMessage) -> bool:
    return isinstance(message, SystemMessage) and str(message.content or "").startswith("你是一个谨慎的代码 Agent")


def _ensure_current_system_message(session: WebSession) -> None:
    current = _system_message(session.workspace_root)
    for index, message in enumerate(session.messages):
        if not isinstance(message, SystemMessage):
            continue
        if _is_langcode_system_message(message):
            session.messages[index] = current
        else:
            session.messages.insert(0, current)
        return
    session.messages.insert(0, current)


def _voice_mode_messages(enabled: Any) -> list[BaseMessage]:
    if not enabled:
        return []
    return [
        SystemMessage(
            content=(
                "本轮用户正在使用语音交互。回答必须适合 TTS 直接播报：只用自然中文纯文本短段落；"
                "不要使用 Markdown 表格、emoji、标题符号、复杂列表、代码块或难以朗读的公式。"
                "不要把这些格式要求、字数要求或“知识点可以怎样说明”之类的元说明输出给用户；"
                "第一句话必须直接回答用户问题。"
                "如果需要列举，请用“一、二、三”这样的口语化句子说明。"
            )
        )
    ]


class _VoiceModeOutputFilter:
    def __init__(self) -> None:
        self._buffer = ""
        self._done = False

    def push(self, text: str) -> str:
        if self._done:
            return text
        self._buffer += text
        if not _voice_prefix_ready(self._buffer):
            return ""
        self._done = True
        output = _sanitize_voice_mode_output(self._buffer)
        self._buffer = ""
        return output

    def flush(self) -> str:
        if self._done or not self._buffer:
            return ""
        self._done = True
        output = _sanitize_voice_mode_output(self._buffer)
        self._buffer = ""
        return output


def _voice_prefix_ready(text: str) -> bool:
    value = str(text or "")
    if len(value) >= 80:
        return True
    return any(mark in value for mark in "。！？!?\n")


def _sanitize_voice_mode_output(text: str) -> str:
    value = str(text or "")
    if not value:
        return ""
    value = value.lstrip()
    for _ in range(4):
        stripped = _strip_one_voice_meta_prefix(value)
        if stripped == value:
            break
        value = stripped.lstrip()
    return value


def _strip_one_voice_meta_prefix(text: str) -> str:
    patterns = [
        r"^</?think>\s*",
        r"^(?:思考|推理|分析|元信息)\s*[:：]\s*[^。！？!?\n]{0,180}[。！？!?\n]\s*",
        r"^(?:Meta|Thinking|Reasoning)\s*[:：]\s*[^.。！？!?\n]{0,180}[.。！？!?\n]\s*",
        r"^(?:以下|下面)(?:是|为)?[^:：\n]{0,80}(?:TTS|语音播报|纯文本|口语化|回复|回答)[^:：\n]{0,80}[:：]\s*",
        r"^(?:以下|下面)(?:是|为)?[^。！？!?\n]{0,80}(?:TTS|语音播报|纯文本|口语化|回复|回答)[^。！？!?\n]{0,80}[。！？!?\n]\s*",
        r"^(?:如果)?评估用户[^。！？!?\n]{0,140}(?:断点续传|自然对话|记忆工具|打断|之前我们聊到|上次我们提到)[^。！？!?\n]{0,140}[。！？!?\n]\s*",
        r"^(?:请)?(?:直接)?使用断点续传式自然对话[^。！？!?\n]{0,160}[。！？!?\n]\s*",
        r"^不要每次都用\s*[\"“](?:之前我们聊到|上次我们提到)[\"”][^。！？!?\n]{0,120}[。！？!?\n]\s*",
        r"^回答\s*(?:控制|限制|尽量控制|需要控制|需控制)?\s*在\s*\d+\s*字(?:以内|内)?[。.!！\s]*",
        r"^(?:控制|限制|尽量控制|需要控制|需控制)\s*在\s*(?:\d+|[一二两三四五六七八九十百千万零〇]+)\s*字(?:以内|内)?[。.!！\s]*",
        r"^(?:本轮|这次|以下)?(?:回答|回复)\s*(?:尽量)?(?:简短|口语化|适合\s*TTS\s*播报|适合语音播报)[^。！？!?\n]{0,40}[。！？!?\n]\s*",
        r"^知识点可以[^。！？!?\n]{0,80}[。！？!?\n]\s*",
        r"^(?:好的[，,]\s*)?我(?:会|来)?(?:直接)?(?:用|按)[^。！？!?\n]{0,50}(?:纯文本|短段落|口语化|语音播报|不使用\s*Markdown)[^。！？!?\n]{0,40}[。！？!?\n]\s*",
    ]
    for pattern in patterns:
        next_text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
        if next_text != text:
            return next_text
    return text


def _novel_stream_content(emitted: str, content: str) -> str:
    """Normalize providers that stream cumulative assistant text instead of deltas."""
    if not emitted or not content:
        return content
    if content.startswith(emitted):
        return content[len(emitted) :]
    if emitted.endswith(content):
        return ""
    max_overlap = min(len(emitted), len(content))
    for length in range(max_overlap, 5, -1):
        if emitted[-length:] == content[:length]:
            return content[length:]
    return content


def _web_search_limit() -> int:
    raw = os.getenv("LANGCODE_WEB_SEARCH_LIMIT", str(DEFAULT_WEB_SEARCH_LIMIT))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = DEFAULT_WEB_SEARCH_LIMIT
    return max(1, value)


def _is_external_web_search_tool(tool_name: str) -> bool:
    return tool_name == "web_search"


def _web_search_limit_result(tool_input: dict, count: int) -> dict:
    return {
        "ok": False,
        "error": WEB_SEARCH_LIMIT_ERROR,
        "error_type": "web_search_limit",
        "searches_used": count,
        "blocked_query": str(tool_input.get("query") or ""),
        "instruction": "不要继续调用 web_search；请基于本轮已有搜索结果回答用户问题。",
    }


def _web_search_limit_reminder() -> HumanMessage:
    return HumanMessage(content=WEB_SEARCH_LIMIT_MOCK_USER)


def _serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    return [serialize_message(message) for message in messages]


def _delete_checkpoint_thread(checkpoint_path: Path, thread_id: str) -> int:
    if not thread_id or not checkpoint_path.exists():
        return 0
    with sqlite3.connect(checkpoint_path, timeout=15.0) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        deleted = 0
        conn.execute("BEGIN IMMEDIATE")
        if "writes" in tables:
            deleted += conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,)).rowcount
        if "checkpoints" in tables:
            deleted += conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)).rowcount
        conn.commit()
        return deleted


def _truncate_partial_to_displayed(content: str, displayed_text: str | None) -> str:
    """On a voice barge-in only the displayed prefix was actually heard/seen."""
    text = str(content or "")
    displayed = str(displayed_text or "").strip()
    if not displayed or not text:
        return text
    if not text.startswith(displayed) or len(text) <= len(displayed):
        return text
    return displayed + VOICE_INTERRUPT_SUFFIX


def _save_cancelled_partial(
    session: WebSession,
    partial_content: list[str],
    save_history,
    *,
    displayed_text: str | None = None,
) -> None:
    content = _truncate_partial_to_displayed("".join(partial_content).strip(), displayed_text)
    if content:
        session.messages.append(AIMessage(content=content))
        session.display_messages.append(AIMessage(content=content))
    save_history(session)


def _tool_round_limit_result(rounds: int) -> dict:
    return {
        "ok": False,
        "error": TOOL_ROUND_LIMIT_MOCK_USER,
        "error_type": "tool_round_limit",
        "rounds_used": int(rounds),
        "instruction": TOOL_ROUND_LIMIT_MOCK_USER,
    }


def _answer_unanswered_tool_calls(session: WebSession, tool_result: Any = None) -> int:
    """Close dangling tool_calls so the saved history stays provider-valid.

    Cancelling mid tool phase (or hitting the round limit) otherwise leaves an
    AIMessage(tool_calls=[...]) with no matching ToolMessage, which OpenAI
    compatible providers reject on the next request.
    """
    content = CANCELLED_TOOL_RESULT_JSON if tool_result is None else _tool_result_json(tool_result)
    added = 0
    for messages in (session.messages, session.display_messages):
        if not messages:
            continue
        anchor = None
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, AIMessage) and list(getattr(message, "tool_calls", None) or []):
                anchor = index
                break
            if not isinstance(message, ToolMessage):
                break
        if anchor is None:
            continue
        answered = {
            str(getattr(message, "tool_call_id", ""))
            for message in messages[anchor + 1 :]
            if isinstance(message, ToolMessage)
        }
        for tool_call in getattr(messages[anchor], "tool_calls", None) or []:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = str(tool_call.get("id") or tool_call.get("name") or "tool")
            if tool_call_id in answered:
                continue
            messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
            answered.add(tool_call_id)
            added += 1
    return added


def _truncate_interrupted_assistant_message(session: WebSession, displayed_text: str) -> bool:
    """Trim the partial answer saved by the cancelled run to what the user actually saw."""
    displayed = str(displayed_text or "").strip()
    if not displayed:
        return False
    changed = False
    for messages in (session.messages, session.display_messages):
        if not messages:
            continue
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, HumanMessage):
                continue
            if not isinstance(message, AIMessage) or list(getattr(message, "tool_calls", None) or []):
                break
            content = str(message.content or "")
            truncated = _truncate_partial_to_displayed(content, displayed)
            if truncated != content:
                messages[index] = AIMessage(content=truncated)
                changed = True
            break
    return changed


def _append_voice_interrupt_context(session: WebSession, payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    spoken_text = str(payload.get("spokenText") or payload.get("spoken_text") or "").strip()
    if not spoken_text:
        return
    tool_input = {
        "spoken_text": spoken_text,
        "previous_user_text": str(payload.get("previousUserText") or payload.get("previous_user_text") or "").strip(),
        "assistant_displayed_text": str(
            payload.get("assistantDisplayedText") or payload.get("assistant_displayed_text") or ""
        ).strip(),
    }
    _truncate_interrupted_assistant_message(session, tool_input["assistant_displayed_text"])
    tool_call_id = f"voice_interrupt_{len(session.messages)}"
    session.messages.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "voice_interrupt",
                    "args": tool_input,
                    "id": tool_call_id,
                }
            ],
        )
    )
    session.messages.append(
        ToolMessage(
            content=_tool_result_json(_voice_interrupt_tool_result(tool_input)),
            tool_call_id=tool_call_id,
        )
    )


def _voice_interrupt_tool_result(tool_input: dict) -> dict:
    return {
        "ok": True,
        "event": "voice_interrupt",
        "instruction": "用户在语音播报过程中打断。请结合 previous_user_text、assistant_displayed_text 和 spoken_text 继续回答；不要把这段工具上下文原样复述给用户。",
        "spoken_text": str(tool_input.get("spoken_text") or "").strip(),
        "previous_user_text": str(tool_input.get("previous_user_text") or "").strip(),
        "assistant_displayed_text": str(tool_input.get("assistant_displayed_text") or "").strip(),
    }


def _tool_result_json(tool_result: Any) -> str:
    return json.dumps(_json_safe(tool_result), ensure_ascii=False)


def _json_safe(value: Any) -> Any:
    return make_json_safe(value)


def _chunk_content_text(chunk: Any, *, strip_raw_thinking: bool = True) -> str:
    content = getattr(chunk, "content", "")
    text = _content_blocks_text(content, include_thinking=False)
    return _strip_raw_thinking_text(text) if strip_raw_thinking else text


def _chunk_thinking_text(chunk: Any) -> str:
    parts: list[str] = []
    additional = getattr(chunk, "additional_kwargs", None) or {}
    metadata = getattr(chunk, "response_metadata", None) or {}
    for container in (additional, metadata):
        if not isinstance(container, dict):
            continue
        for key in ("reasoning_content", "reasoning", "reasoning_text", "thinking_content", "thinking"):
            value = container.get(key)
            text = _content_blocks_text(value, include_thinking=True)
            if text:
                parts.append(text)
    content_text = _content_blocks_text(getattr(chunk, "content", ""), include_thinking=True, thinking_only=True)
    if content_text:
        parts.append(content_text)
    return "".join(parts)


def _content_blocks_text(value: Any, *, include_thinking: bool, thinking_only: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return "" if thinking_only else value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                block_type = str(item.get("type") or "").lower()
                is_thinking = any(token in block_type for token in ("reasoning", "thinking", "thought"))
                if thinking_only and not is_thinking:
                    continue
                if is_thinking and not include_thinking:
                    continue
                if not is_thinking and thinking_only:
                    continue
                text = item.get("text") or item.get("content") or item.get("reasoning") or item.get("thinking") or ""
                parts.append(str(text))
            elif not thinking_only:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(value, dict):
        block_type = str(value.get("type") or "").lower()
        is_thinking = any(token in block_type for token in ("reasoning", "thinking", "thought"))
        if thinking_only and not is_thinking:
            return ""
        if is_thinking and not include_thinking:
            return ""
        text = value.get("text") or value.get("content") or value.get("reasoning") or value.get("thinking") or ""
        return str(text)
    return "" if thinking_only else str(value)


def _split_raw_thinking_text(text: str, *, in_thinking: bool = False) -> tuple[str, str, bool]:
    if not text:
        return "", "", in_thinking
    visible_parts: list[str] = []
    thinking_parts: list[str] = []
    index = 0
    lower = text.lower()
    while index < len(text):
        if in_thinking:
            end = lower.find("</think>", index)
            if end == -1:
                thinking_parts.append(text[index:])
                return "".join(visible_parts), "".join(thinking_parts), True
            thinking_parts.append(text[index:end])
            index = end + len("</think>")
            in_thinking = False
            continue

        start = lower.find("<think>", index)
        stray_end = lower.find("</think>", index)
        if stray_end != -1 and (start == -1 or stray_end < start):
            visible_parts.append(text[index:stray_end])
            index = stray_end + len("</think>")
            continue
        if start == -1:
            visible_parts.append(text[index:])
            break
        visible_parts.append(text[index:start])
        index = start + len("<think>")
        in_thinking = True
    return "".join(visible_parts), "".join(thinking_parts), in_thinking


def _strip_raw_thinking_text(text: str) -> str:
    visible, _thinking, _open = _split_raw_thinking_text(text, in_thinking=False)
    return visible


def _repair_live_tool_history(session: WebSession) -> None:
    session.messages = _repair_tool_history(session.messages)


def _repair_tool_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    repaired: list[BaseMessage] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, AIMessage) or not list(getattr(message, "tool_calls", None) or []):
            if not isinstance(message, ToolMessage):
                repaired.append(message)
            index += 1
            continue

        tool_calls = list(getattr(message, "tool_calls", None) or [])
        required_ids = [
            str(tool_call.get("id") or tool_call.get("name") or "tool")
            for tool_call in tool_calls
            if isinstance(tool_call, dict)
        ]
        tool_messages: list[ToolMessage] = []
        cursor = index + 1
        while cursor < len(messages) and isinstance(messages[cursor], ToolMessage):
            tool_messages.append(messages[cursor])
            cursor += 1

        present_ids = {str(getattr(tool_message, "tool_call_id", "")) for tool_message in tool_messages}
        if required_ids and all(tool_call_id in present_ids for tool_call_id in required_ids):
            repaired.append(message)
            repaired.extend(tool_messages)
        else:
            content = str(message.content or "").strip()
            if content:
                repaired.append(AIMessage(content=content))
        index = cursor
    return repaired


def _should_continue_after_plan_only(
    session: WebSession,
    plan_written_this_turn: bool,
    non_plan_tools_completed: int,
    already_continued: bool,
) -> bool:
    return (
        plan_written_this_turn
        and non_plan_tools_completed == 0
        and not already_continued
        and _has_incomplete_todos(session.todos)
    )


def _has_incomplete_todos(todos: list[dict]) -> bool:
    return any(str(item.get("status") or "pending") in {"pending", "in_progress"} for item in todos)


def _plan_execution_reminder() -> SystemMessage:
    return SystemMessage(
        content=(
            "你刚才创建或更新了任务清单，但这一轮还没有执行任何非规划类工具调用。"
            "现在请继续处理第一个待办或正在进行的任务；需要时使用工具查看、编辑、运行或验证。"
            "当某个任务开始执行或完成时，要调用 task_update 更新状态。"
            "除非所有任务都已完成，或者明确遇到阻塞，否则不要给出最终完成答复。"
        )
    )


def _without_system_messages(serialized: list[dict]) -> list[dict]:
    return [item for item in serialized if str(item.get("role") or item.get("type") or "") != "system"]


DISPLAY_TAIL_LIMIT = 200
DISPLAY_ARCHIVE_DIRNAME = "artifacts"


def _display_archive_dir(workspace_root: Path) -> Path:
    return Path(workspace_root) / ".langcode" / DISPLAY_ARCHIVE_DIRNAME


def _display_archive_name(session_id: str) -> str:
    """File name for a session's displayed-history archive.

    The id is slugified and suffixed with a hash so two ids that differ only in
    characters a file name cannot hold never share one archive.
    """
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", str(session_id))[:64] or "session"
    digest = hashlib.sha1(str(session_id).encode("utf-8")).hexdigest()[:8]
    return f"display-{slug}-{digest}.json"


def _read_display_archive(workspace_root: Path, session_id: str, name: str) -> list[BaseMessage] | None:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    path = _display_archive_dir(workspace_root) / name
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    serialized = payload.get("messages") if isinstance(payload, dict) else None
    if not serialized:
        return None
    return messages_from_json(serialized)


def _restore_display_messages(
    state: Any,
    workspace_root: Path,
    session_id: str,
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """Rebuild the user-visible history from ``state_json``.

    Shapes supported, newest first:
      * ``display_archive`` + ``display_messages_tail`` (item 12) - the full
        displayed history lives in ``.langcode/artifacts``, and only a bounded
        tail is kept in the row; the tail is the fallback if the file is gone;
      * ``display_same_as_messages`` marker (item 6.4) - no second copy stored;
      * a full ``display_messages`` copy - what older builds wrote whenever a
        /compact made the displayed history diverge from the model history;
      * none of those (sessions written before state_json existed) - fall back to
        the lossy compaction-archive recovery.
    """
    if isinstance(state, dict):
        archive_name = str(state.get("display_archive") or "")
        if archive_name:
            restored = _read_display_archive(workspace_root, session_id, archive_name)
            if restored:
                return restored
        if state.get("display_messages"):
            return messages_from_json(state.get("display_messages") or [])
        if state.get("display_messages_tail"):
            return messages_from_json(state.get("display_messages_tail") or [])
        if state.get("display_same_as_messages"):
            return [message for message in messages if not isinstance(message, SystemMessage)]
    return _recover_display_messages_from_compaction(workspace_root, session_id, messages)


def _recover_display_messages_from_compaction(
    workspace_root: Path,
    session_id: str,
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    archive_dir = workspace_root / ".langcode" / "compactions"
    archives = sorted(archive_dir.glob(f"{session_id}-*.json")) if archive_dir.exists() else []
    if not archives:
        return list(messages)

    try:
        payload = json.loads(archives[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return list(messages)

    archived_messages = messages_from_json(payload.get("messages") or [])
    recovered = list(archived_messages)
    seen = {_message_signature(message) for message in recovered}
    for message in messages:
        signature = _message_signature(message)
        if signature in seen:
            continue
        recovered.append(message)
        seen.add(signature)
    return recovered


def _drop_current_failed_turn(
    session: WebSession,
    turn_start_message_count: int,
    turn_start_display_count: int,
) -> None:
    session.messages = session.messages[:turn_start_message_count]
    session.display_messages = session.display_messages[:turn_start_display_count]


def _failed_turn_start_index(messages: list[BaseMessage]) -> int:
    if messages and isinstance(messages[-1], HumanMessage):
        return len(messages) - 1
    return len(messages)


def _message_signature(message: BaseMessage) -> tuple:
    return (
        message.type,
        str(message.content),
        getattr(message, "tool_call_id", ""),
        json.dumps(getattr(message, "tool_calls", None) or [], ensure_ascii=False, sort_keys=True),
    )


def _tool_schemas_for_model() -> list[dict]:
    from ..runtime.chat import tool_schemas

    return tool_schemas()


def _progress_event(
    status: str,
    tool_name: str,
    tool_input: dict,
    step: int,
    total: int,
    completed: int,
    result: dict | None = None,
) -> dict:
    return {
        "type": "progress",
        "status": status,
        "toolName": tool_name,
        "target": _progress_target(tool_name, tool_input),
        "step": step,
        "total": total,
        "completed": completed,
        "ok": _progress_ok(result),
    }


def _progress_target(tool_name: str, tool_input: dict) -> str:
    if tool_name in {"read_file", "write_file", "edit_file"}:
        return str(tool_input.get("path") or "")
    if tool_name == "search":
        query = str(tool_input.get("query") or "")
        path = str(tool_input.get("path") or ".")
        return f"{query} @ {path}"
    if tool_name in {"ls", "glob"}:
        return str(tool_input.get("path") or tool_input.get("pattern") or ".")
    if tool_name == "web_search":
        return str(tool_input.get("query") or "")
    if tool_name == "web_fetch":
        return str(tool_input.get("url") or "")
    if tool_name == "memory":
        return str(tool_input.get("target") or "memory")
    if tool_name == "soul":
        return "SOUL.md"
    if tool_name == "self_evolve":
        return str(tool_input.get("action") or "status")
    if tool_name == "cron":
        return str(tool_input.get("name") or tool_input.get("job_id") or tool_input.get("action") or "list")
    if tool_name == "shell":
        return str(tool_input.get("command") or "")
    if tool_name == "sandbox_shell":
        return str(tool_input.get("command") or "")
    if tool_name == "task_create":
        return str(tool_input.get("content") or "")
    if tool_name in {"task_update", "task_get", "task_cancel"}:
        return str(tool_input.get("task_id") or "")
    if tool_name == "task_list":
        return str(tool_input.get("status") or "全部")
    if tool_name == "delegate_agent":
        return str(tool_input.get("task") or tool_input.get("role") or "")
    if tool_name == "delegate_agents":
        agents = tool_input.get("agents")
        if isinstance(agents, list):
            return f"{len(agents)} 个子 Agent"
        return str(tool_input.get("title") or "多视角子 Agent")
    if tool_name == "agent_debate":
        return str(tool_input.get("topic") or tool_input.get("debate_id") or "Agent 辩论")
    if tool_name == "memory":
        return f"{tool_input.get('target', 'memory')}:{tool_input.get('action', 'read')}"
    if tool_name == "session_search":
        return str(tool_input.get("query") or tool_input.get("mode") or "")
    if tool_name == "skill":
        return str(tool_input.get("name") or tool_input.get("action") or "")
    if tool_name == "diagram":
        return str(tool_input.get("title") or tool_input.get("diagram_type") or "")
    return json.dumps(tool_input, ensure_ascii=False)


def _progress_ok(result: dict | None) -> bool | None:
    if result is None or not isinstance(result, dict):
        return None
    return bool(result.get("ok", True))


def _directory_roots(workspace_root: Path) -> list[dict]:
    candidates = [Path("/"), Path.home(), workspace_root]
    roots = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if str(resolved) in seen or not resolved.exists() or not resolved.is_dir():
            continue
        seen.add(str(resolved))
        roots.append({"name": resolved.name or str(resolved), "path": str(resolved)})
    return roots


QUERY_TOKEN_ROUTES = {"/api/asr/stream"}

# Item 1: the loopback names a browser may legitimately use to reach a local
# LangCode. Anything else in the Host header means the request arrived through a
# name that resolves elsewhere - the classic DNS-rebinding setup, where a page on
# http://evil.com re-points that name at 127.0.0.1 and then reads ``/`` to lift
# the API token embedded in index.html.
LOCAL_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def _hostname_of(value: str) -> str:
    """Hostname of a ``Host`` header or origin authority, port and brackets removed."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        hostname = urlparse(raw if "//" in raw else f"//{raw}").hostname
    except ValueError:
        return ""
    return (hostname or "").strip().lower()


def allowed_hostnames() -> set[str]:
    """Loopback names plus the comma-separated ``LANGCODE_ALLOWED_HOSTS`` extras."""
    allowed = {name.strip("[]") for name in LOCAL_HOSTNAMES} | set(LOCAL_HOSTNAMES)
    for item in str(os.getenv("LANGCODE_ALLOWED_HOSTS") or "").split(","):
        name = _hostname_of(item) or item.strip().lower()
        if name:
            allowed.add(name)
    return allowed


def reject_untrusted_host_sync(web_app: WebApp, request: Request):
    """Host/Origin guard for EVERY route, not just ``/api/`` (item 1).

    ``/`` and ``/<path>`` serve an index.html with the API token baked in, so the
    token leaks unless this runs before any handler. Ports are ignored; only the
    hostname is compared. The 403 body deliberately carries no token.
    """
    allowed = allowed_hostnames()
    if _hostname_of(getattr(request, "host", "") or "") not in allowed:
        return response.json({"ok": False, "error": "Rejected Host header"}, status=403)
    origin = request.headers.get("origin")
    if origin:
        # Compare hostnames against the allowlist, never ``netloc != request.host``:
        # that compares the attacker-controlled Host with itself and passes after
        # a rebinding.
        try:
            origin_host = (urlparse(origin).hostname or "").strip().lower()
        except ValueError:
            origin_host = ""
        if origin_host not in allowed:
            return response.json({"ok": False, "error": "Cross-origin request rejected"}, status=403)
    return None


def _tts_uses_queue(web_app: WebApp) -> bool:
    """Offload TTS to the queue only when this process has no local TTS (item 5).

    Queue workers start with ``enable_voice=False`` by default, so routing every
    TTS request to the queue just because Redis is reachable made TTS fail with
    "TTS service is not available." on a server that could have synthesized it
    locally. A remote voice worker is handled before this branch.
    """
    return bool(web_app.job_queue.available and web_app.tts is None)


def _is_api_path(requested: str) -> bool:
    """True when a static path would shadow the ``/api/`` namespace (item 2).

    ``//api/status``, ``/./api/status``, ``/API/status`` and ``/%61pi/status`` all
    slip past a naive ``path.startswith("api/")`` test yet still reach the SPA
    fallback, which answers with the token-bearing index.html.
    """
    raw = unquote(str(requested or "")).replace("\\", "/")
    segments = [segment for segment in raw.split("/") if segment not in ("", ".")]
    return bool(segments) and segments[0].lower() == "api"


def authorize_api_request_sync(web_app: WebApp, request: Request):
    """Auth check for /api/* requests. Returns a 403 response or None when allowed."""
    if not request.path.startswith("/api/"):
        return None
    supplied_token = request.headers.get("x-langcode-token") or ""
    # A token in the query string leaks into browser history and access logs, so
    # it is accepted only for the ASR WebSocket, which cannot send headers.
    if not supplied_token and request.path in QUERY_TOKEN_ROUTES:
        supplied_token = request.args.get("token") or ""
    try:
        authorized = secrets.compare_digest(str(supplied_token).encode("ascii"), web_app.api_token.encode("ascii"))
    except (UnicodeEncodeError, UnicodeDecodeError, TypeError):
        authorized = False
    if not authorized:
        return response.json({"ok": False, "error": "Unauthorized local API request"}, status=403)
    # The Origin check now lives in reject_untrusted_host_sync, which runs for
    # every route (including the token-bearing index.html) instead of only /api/.
    return None


class SessionRequestLock:
    """Serialize one session's requests, locally and across processes.

    Item 3: besides the asyncio lock and the distributed lease, entering also
    pins the session in memory for the whole request. ``/api/chat`` never calls
    ``mark_run_started``, so without the pin a concurrent ``get_session`` could
    LRU-evict and ``close()`` the very session this request is working on, and
    the second live copy would then save over it.
    """

    def __init__(self, web_app: WebApp, session_id: str, local_lock: asyncio.Lock) -> None:
        self.web_app = web_app
        self.session_id = session_id
        self.local_lock = local_lock
        self.lease = None
        self.busy = False

    async def __aenter__(self) -> "SessionRequestLock":
        await self.local_lock.acquire()
        self.web_app.acquire_session_busy(self.session_id)
        self.busy = True
        try:
            self.lease = await asyncio.to_thread(
                self.web_app.runtime_state.acquire_session_lock, self.session_id
            )
            await asyncio.to_thread(self.web_app.refresh_session_from_store, self.session_id)
        except Exception:
            self.busy = False
            self.web_app.release_session_busy(self.session_id)
            self.local_lock.release()
            raise
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        try:
            if self.lease is not None:
                await asyncio.to_thread(self.lease.release)
        finally:
            try:
                if self.busy:
                    self.busy = False
                    self.web_app.release_session_busy(self.session_id)
            finally:
                self.local_lock.release()


DISCONNECT_PRODUCER_WAIT_SEC = 30.0


async def _drain_stream_producer(producer, *, context: str) -> None:
    """Wait for a detached producer, but never block the connection teardown forever."""
    try:
        await asyncio.wait_for(asyncio.shield(producer), timeout=DISCONNECT_PRODUCER_WAIT_SEC)
    except asyncio.TimeoutError:
        logger.warning(
            "%s producer still running %ss after client disconnect; detaching",
            context,
            DISCONNECT_PRODUCER_WAIT_SEC,
        )
    except Exception:
        logger.exception("%s producer failed after client disconnect", context)


def create_sanic_app(web_app: WebApp, *, name: str = "langcode-web") -> Sanic:
    # ``name`` is overridable because Sanic keeps a process-wide app registry and
    # refuses to register two apps under the same name (tests build several).
    sanic_app = Sanic(name)
    sanic_app.config.REQUEST_TIMEOUT = 3600
    sanic_app.config.RESPONSE_TIMEOUT = 3600
    sanic_app.ctx.web_app = web_app
    sanic_app.ctx.session_locks = OrderedDict()
    sanic_app.ctx.session_locks_guard = asyncio.Lock()
    sanic_app.ctx.workspace_lock = asyncio.Lock()
    sanic_app.ctx.gen_pool = ThreadPoolExecutor(
        max_workers=max(1, int(os.getenv("LANGCODE_GENERATION_WORKERS", "16"))),
        thread_name_prefix="generation",
    )

    @sanic_app.middleware("request")
    async def authorize_api_request(request: Request):
        # Host/Origin first (item 1): it covers every route, so a rebound name is
        # refused before any handler can echo the API token back in index.html.
        rejected = reject_untrusted_host_sync(web_app, request)
        if rejected is not None:
            return rejected
        return authorize_api_request_sync(web_app, request)

    async def session_lock(session_id: str) -> SessionRequestLock:
        async with sanic_app.ctx.session_locks_guard:
            locks = sanic_app.ctx.session_locks
            lock = locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
            else:
                locks.pop(session_id, None)
            locks[session_id] = lock
            # Item 25: bound the lock table with the same LRU limit as sessions;
            # only idle locks are evicted, so an in-flight request keeps its lock.
            limit = _max_resident_sessions()
            for candidate in list(locks.keys()):
                if len(locks) <= limit:
                    break
                if candidate == session_id or locks[candidate].locked():
                    continue
                locks.pop(candidate, None)
            return SessionRequestLock(web_app, session_id, lock)

    async def run_json(fn, *args) -> response.HTTPResponse:
        try:
            payload = await asyncio.to_thread(fn, *args)
            return response.json(payload)
        except RuntimeLockTimeout as exc:
            return response.json({"ok": False, "error": str(exc)}, status=409)
        except Exception as exc:
            return response.json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

    @sanic_app.get("/api/status")
    async def status(_request: Request):
        return await run_json(web_app.status)

    @sanic_app.get("/api/asr/status")
    async def asr_status(_request: Request):
        return await run_json(web_app.asr_status)

    @sanic_app.get("/api/tts/status")
    async def tts_status(_request: Request):
        return await run_json(web_app.tts_status)

    @sanic_app.get("/api/tts/voices")
    async def tts_voices(_request: Request):
        return await run_json(web_app.tts_voices)

    @sanic_app.post("/api/tts/voices")
    async def tts_create_voice(request: Request):
        payload = _request_json(request)
        try:
            return response.json(await asyncio.to_thread(web_app.create_tts_voice, payload))
        except Exception as exc:
            return response.json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=400)

    @sanic_app.get("/api/tts/voices/<voice_id:path>/preview")
    async def tts_voice_preview(_request: Request, voice_id: str):
        try:
            audio, content_type = await asyncio.to_thread(web_app.tts_voice_preview, voice_id)
            return response.raw(audio, content_type=content_type, headers={"Cache-Control": "no-cache"})
        except Exception as exc:
            return response.json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=404)

    @sanic_app.post("/api/tts/speech")
    async def tts_speech(request: Request):
        payload = _request_json(request)
        try:
            audio, content_type = await asyncio.to_thread(web_app.tts_speech, payload)
            return response.raw(audio, content_type=content_type, headers={"Cache-Control": "no-cache"})
        except Exception as exc:
            return response.json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

    @sanic_app.post("/api/tts/stream")
    async def tts_stream(request: Request):
        payload = _request_json(request)
        # A new turn supersedes the previous one for this session before any
        # audio is produced, so the older producer stops at its next check even
        # when its own request is still connected (full-duplex barge-in).
        web_app.tts_claim_turn(str(payload.get("sessionId") or ""), str(payload.get("turnId") or ""))

        async def stream(streaming_response):
            if web_app.voice_worker is not None:
                # The payload (sessionId/turnId/segmentIndex included) goes to
                # the worker unchanged; the worker runs the same stop logic and
                # sees this client's disconnect as its own upstream close.
                async for event in web_app.voice_worker.stream_tts(payload):
                    await streaming_response.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
                return
            if _tts_uses_queue(web_app):
                await _stream_queue_job(web_app, streaming_response, "tts_stream", payload, heartbeat=False)
                return
            await _stream_local_tts(web_app, streaming_response, payload)

        return response.ResponseStream(
            stream,
            content_type="application/x-ndjson; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    @sanic_app.post("/api/tts/cancel")
    async def tts_cancel(request: Request):
        payload = _request_json(request)
        result = web_app.cancel_tts_turn(payload)
        if web_app.voice_worker is not None:
            # Best effort: the worker owns the producer in that deployment, but a
            # failed hop must not turn the local cancel into a client error.
            try:
                await web_app.voice_worker.cancel_tts(payload)
            except Exception as exc:
                logger.warning("tts cancel could not reach the voice worker: %s", exc)
        return response.json(result, status=200 if result.get("ok") else 400)

    @sanic_app.websocket("/api/asr/stream")
    async def asr_stream(_request: Request, ws):
        if web_app.voice_worker is not None:
            await web_app.voice_worker.proxy_asr_websocket(ws)
            return
        if web_app.asr is None:
            await ws.send(json.dumps({"type": "error", "error": "ASR service is not available."}, ensure_ascii=False))
            return
        await websocket_asr_loop(ws, web_app.asr)

    @sanic_app.get("/api/directories")
    async def directories(request: Request):
        return await run_json(web_app.directories, request.args.get("path"))

    @sanic_app.get("/api/sessions")
    async def sessions(_request: Request):
        return await run_json(web_app.list_sessions)

    @sanic_app.get("/api/session")
    async def session_view(request: Request):
        session_id = request.args.get("sessionId") or ""
        before = request.args.get("before")
        limit = request.args.get("limit")
        lock = await session_lock(session_id)
        try:
            async with lock:
                return await run_json(web_app.session_view, session_id, before, limit)
        except RuntimeLockTimeout as exc:
            return response.json({"ok": False, "error": str(exc)}, status=409)

    @sanic_app.post("/api/chat")
    async def chat(request: Request):
        payload = _request_json(request)
        session_id = _session_id(payload)
        lock = await session_lock(session_id)
        try:
            async with lock:
                return await run_json(web_app.chat, payload)
        except RuntimeLockTimeout as exc:
            return response.json({"ok": False, "error": str(exc)}, status=409)

    @sanic_app.post("/api/chat-stream")
    async def chat_stream(request: Request):
        payload = _request_json(request)
        session_id = _session_id(payload)
        run_id = str(payload.get("runId") or "").strip() or None
        lock = await session_lock(session_id)

        async def stream(streaming_response):
            if web_app.job_queue.available:
                await _stream_queue_job(web_app, streaming_response, "chat_stream", payload, heartbeat=True)
                return
            try:
                await lock.__aenter__()
            except RuntimeLockTimeout as exc:
                event = _error_event(str(exc), "rate_limit")
                await streaming_response.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
                return
            try:
                await asyncio.to_thread(web_app.runtime_state.mark_run_started, session_id, run_id)
                event_queue: asyncio.Queue[dict | None] = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def produce_events() -> None:
                    try:
                        for event in web_app.chat_events(payload):
                            loop.call_soon_threadsafe(event_queue.put_nowait, event)
                    except Exception as exc:
                        loop.call_soon_threadsafe(event_queue.put_nowait, _exception_error_event(exc))
                    finally:
                        loop.call_soon_threadsafe(event_queue.put_nowait, None)

                producer = asyncio.ensure_future(loop.run_in_executor(sanic_app.ctx.gen_pool, produce_events))
                waiting_count = 0
                try:
                    while True:
                        try:
                            event = await asyncio.wait_for(event_queue.get(), timeout=4.0)
                            waiting_count = 0
                        except asyncio.TimeoutError:
                            waiting_count += 1
                            event = _stream_waiting_event(waiting_count)
                        if event is None:
                            break
                        data = json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
                        await streaming_response.write(data)
                    await producer
                except (asyncio.CancelledError, ConnectionError, BrokenPipeError):
                    await asyncio.to_thread(web_app.runtime_state.cancel_run, session_id, run_id)
                    await _drain_stream_producer(producer, context="chat-stream")
                    raise
            finally:
                try:
                    await asyncio.to_thread(web_app.runtime_state.mark_run_finished, session_id, run_id)
                    web_app._forget_cancelled_run(session_id, run_id)
                finally:
                    await lock.__aexit__(None, None, None)

        return response.ResponseStream(
            stream,
            content_type="application/x-ndjson; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    @sanic_app.post("/api/approval-stream")
    async def approval_stream(request: Request):
        payload = _request_json(request)
        session_id = _session_id(payload)
        run_id = str(payload.get("runId") or "").strip() or None
        lock = await session_lock(session_id)

        async def stream(streaming_response):
            if web_app.job_queue.available:
                await _stream_queue_job(web_app, streaming_response, "approval_stream", payload, heartbeat=True)
                return
            try:
                await lock.__aenter__()
            except RuntimeLockTimeout as exc:
                event = _error_event(str(exc), "rate_limit")
                await streaming_response.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
                return
            try:
                await asyncio.to_thread(web_app.runtime_state.mark_run_started, session_id, run_id)
                event_queue: asyncio.Queue[dict | None] = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def produce_events() -> None:
                    try:
                        for event in web_app.approval_events(payload):
                            loop.call_soon_threadsafe(event_queue.put_nowait, event)
                    except Exception as exc:
                        loop.call_soon_threadsafe(event_queue.put_nowait, _exception_error_event(exc))
                    finally:
                        loop.call_soon_threadsafe(event_queue.put_nowait, None)

                producer = asyncio.ensure_future(loop.run_in_executor(sanic_app.ctx.gen_pool, produce_events))
                waiting_count = 0
                try:
                    while True:
                        try:
                            event = await asyncio.wait_for(event_queue.get(), timeout=4.0)
                            waiting_count = 0
                        except asyncio.TimeoutError:
                            waiting_count += 1
                            event = _stream_waiting_event(waiting_count)
                        if event is None:
                            break
                        data = json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
                        await streaming_response.write(data)
                    await producer
                except (asyncio.CancelledError, ConnectionError, BrokenPipeError):
                    await asyncio.to_thread(web_app.runtime_state.cancel_run, session_id, run_id)
                    await _drain_stream_producer(producer, context="approval-stream")
                    raise
            finally:
                try:
                    await asyncio.to_thread(web_app.runtime_state.mark_run_finished, session_id, run_id)
                    web_app._forget_cancelled_run(session_id, run_id)
                finally:
                    await lock.__aexit__(None, None, None)

        return response.ResponseStream(
            stream,
            content_type="application/x-ndjson; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    @sanic_app.post("/api/session/cancel")
    async def cancel_session_run(request: Request):
        return await run_json(web_app.cancel_run, _request_json(request))

    @sanic_app.post("/api/approval")
    async def approval(request: Request):
        payload = _request_json(request)
        session_id = _session_id(payload)
        lock = await session_lock(session_id)
        try:
            async with lock:
                return await run_json(web_app.approve, payload)
        except RuntimeLockTimeout as exc:
            return response.json({"ok": False, "error": str(exc)}, status=409)

    @sanic_app.post("/api/workspace")
    async def workspace(request: Request):
        async with sanic_app.ctx.workspace_lock:
            async with sanic_app.ctx.session_locks_guard:
                sanic_app.ctx.session_locks.clear()
            return await run_json(web_app.set_workspace, _request_json(request))

    @sanic_app.post("/api/settings")
    async def settings(request: Request):
        return await run_json(web_app.set_settings, _request_json(request))

    @sanic_app.post("/api/session/delete")
    async def delete_session(request: Request):
        payload = _request_json(request)
        session_id = _session_id(payload)
        lock = await session_lock(session_id)
        try:
            async with lock:
                return await run_json(web_app.delete_session, payload)
        except RuntimeLockTimeout as exc:
            return response.json({"ok": False, "error": str(exc)}, status=409)

    @sanic_app.post("/api/session/clear")
    async def clear_session(request: Request):
        payload = _request_json(request)
        session_id = _session_id(payload)
        lock = await session_lock(session_id)
        try:
            async with lock:
                return await run_json(web_app.clear_session, payload)
        except RuntimeLockTimeout as exc:
            return response.json({"ok": False, "error": str(exc)}, status=409)

    @sanic_app.post("/api/session/create")
    async def create_session(request: Request):
        payload = _request_json(request)
        session_id = _session_id(payload)
        lock = await session_lock(session_id)
        try:
            async with lock:
                return await run_json(web_app.create_session, payload)
        except RuntimeLockTimeout as exc:
            return response.json({"ok": False, "error": str(exc)}, status=409)

    @sanic_app.post("/api/session/rename")
    async def rename_session(request: Request):
        payload = _request_json(request)
        session_id = _session_id(payload)
        lock = await session_lock(session_id)
        try:
            async with lock:
                return await run_json(web_app.rename_session, payload)
        except RuntimeLockTimeout as exc:
            return response.json({"ok": False, "error": str(exc)}, status=409)

    @sanic_app.post("/api/directory/native")
    async def native_directory(request: Request):
        return await run_json(web_app.choose_directory, _request_json(request))

    @sanic_app.get("/")
    async def index(request: Request):
        return await _static_response(web_app, "index.html", request)

    @sanic_app.get("/<path:path>")
    async def static_files(request: Request, path: str):
        if _is_api_path(path) or _is_api_path(request.path):
            return response.json({"ok": False, "error": "Not found"}, status=404)
        return await _static_response(web_app, path, request)

    @sanic_app.after_server_stop
    async def close_resources(_app: Sanic):
        _app.ctx.gen_pool.shutdown(wait=True, cancel_futures=True)
        web_app.close()

    return sanic_app


async def _stream_local_tts(web_app: WebApp, streaming_response, payload: dict) -> None:
    """Stream NDJSON TTS events produced by this process's own TtsService.

    Synthesis blocks, so it runs in a thread and reaches the socket through a
    queue. The thread cannot be killed, but it checks ``should_stop`` between
    chunks, so it outlives a barge-in or a disconnect by at most one chunk
    instead of speaking a whole superseded answer into a dead socket.
    """
    text = str(payload.get("text") or "")
    voice_id = str(payload.get("voiceId") or "")
    session_id = str(payload.get("sessionId") or "").strip()
    turn_id = str(payload.get("turnId") or "").strip()
    segment_index = _segment_index(payload)
    if turn_id:
        logger.debug("tts stream turn=%s segment=%d chars=%d", turn_id, segment_index, len(text))
    started_at = time.perf_counter()
    event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    disconnected = threading.Event()

    def should_stop() -> bool:
        return disconnected.is_set() or web_app.is_tts_turn_stale(session_id, turn_id)

    def produce_audio() -> None:
        try:
            if web_app.tts is None:
                raise RuntimeError("TTS service is not available.")
            # Shared producer with voice_worker/worker: audio events, an
            # optional tts_fallback notice, then one terminal done/error.
            for event in iter_tts_events(
                web_app.tts,
                text,
                voice_id=voice_id,
                should_stop=should_stop,
                turn_id=turn_id,
                started_at=started_at,
            ):
                loop.call_soon_threadsafe(event_queue.put_nowait, event)
        except Exception as exc:
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                {"type": "error", "ok": False, "error": f"{type(exc).__name__}: {exc}"},
            )
        finally:
            loop.call_soon_threadsafe(event_queue.put_nowait, None)

    producer = asyncio.create_task(asyncio.to_thread(produce_audio))
    try:
        while True:
            event = await event_queue.get()
            if event is None:
                break
            await streaming_response.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
        await producer
    except (asyncio.CancelledError, ConnectionError, BrokenPipeError):
        disconnected.set()
        await _drain_stream_producer(producer, context="tts-stream")
        raise


def _segment_index(payload: dict) -> int:
    """``segmentIndex`` of a TTS request; 0 when absent or malformed."""
    try:
        return max(0, int(payload.get("segmentIndex") or 0))
    except (TypeError, ValueError):
        return 0


async def _stream_queue_job(
    web_app: WebApp,
    streaming_response,
    kind: str,
    payload: dict,
    *,
    heartbeat: bool,
) -> None:
    try:
        job_id = await asyncio.to_thread(web_app.job_queue.enqueue, kind, payload)
    except Exception as exc:
        event = _exception_error_event(exc, context="queue enqueue")
        await streaming_response.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
        return
    waiting_count = 0
    async for event in _iter_queue_events(web_app, job_id):
        if event is None:
            if heartbeat:
                waiting_count += 1
                event = _stream_waiting_event(waiting_count)
            else:
                continue
        else:
            waiting_count = 0
        if event.get("type") == "queued":
            continue
        await streaming_response.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
        if event.get("type") in {"done", "error"}:
            break


async def _iter_queue_events(web_app: WebApp, job_id: str):
    """Yield job events, preferring a native async xread over a thread per event."""
    if web_app.job_queue.async_events_supported:
        try:
            async for event in web_app.job_queue.aiter_events(job_id, block_ms=4000):
                yield event
            return
        except Exception as exc:
            yield _exception_error_event(exc, context="queue events")
            return
    event_iter = web_app.job_queue.iter_events(job_id, block_ms=4000)
    while True:
        try:
            yield await asyncio.to_thread(next, event_iter)
        except StopIteration:
            return
        except Exception as exc:
            yield _exception_error_event(exc, context="queue events")
            return


def _request_json(request: Request) -> dict:
    return dict(request.json or {})


def _decode_data_url_or_base64(value: str) -> bytes:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("缺少音频数据")
    if "," in raw and raw.split(",", 1)[0].startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        return base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise ValueError("音频数据不是有效 base64") from exc


def _session_id(payload: dict) -> str:
    return str(payload.get("sessionId") or "default")


DEFAULT_HISTORY_PAGE_SIZE = 100
MAX_HISTORY_PAGE_SIZE = 1000


def _history_window(before: Any, limit: Any) -> tuple[int | None, int] | None:
    """Parse ?before=&limit=; ``None`` means "return the full history" (default)."""
    if (before is None or str(before).strip() == "") and (limit is None or str(limit).strip() == ""):
        return None
    before_index: int | None = None
    if before is not None and str(before).strip() != "":
        try:
            before_index = max(0, int(str(before).strip()))
        except (TypeError, ValueError):
            before_index = None
    page_size = DEFAULT_HISTORY_PAGE_SIZE
    if limit is not None and str(limit).strip() != "":
        try:
            page_size = int(str(limit).strip())
        except (TypeError, ValueError):
            page_size = DEFAULT_HISTORY_PAGE_SIZE
    page_size = max(1, min(page_size, MAX_HISTORY_PAGE_SIZE))
    return before_index, page_size


def _stream_waiting_event(waiting_count: int = 1) -> dict:
    """Keep-alive heartbeat for the ndjson stream (never a fake tool progress event)."""
    return {"type": "heartbeat", "waitedSec": int(max(4, waiting_count * 4))}


RETRIABLE_ERROR_CODES = {"rate_limit", "model_timeout", "network"}


_AUTH_ERROR_TYPES = frozenset({"AuthenticationError", "PermissionDeniedError"})
_RATE_LIMIT_ERROR_TYPES = frozenset({"RateLimitError"})
_TIMEOUT_ERROR_TYPES = frozenset(
    {
        "APITimeoutError",
        "Timeout",
        "TimeoutError",
        "TimeoutException",
        "ReadTimeout",
        "WriteTimeout",
        "ConnectTimeout",
        "PoolTimeout",
        "ReadTimeoutError",
        "ConnectTimeoutError",
    }
)
_NETWORK_ERROR_TYPES = frozenset(
    {
        "APIConnectionError",
        "ConnectionError",
        "ConnectError",
        "SSLError",
        "RemoteProtocolError",
        "ProtocolError",
        "NetworkError",
    }
)
# Known API failures that are none of the above. Their message may legitimately
# contain the word "timeout" (a rejected request parameter, say), so the loose
# string fallbacks below must not run for them.
_REQUEST_ERROR_TYPES = frozenset(
    {
        "BadRequestError",
        "UnprocessableEntityError",
        "NotFoundError",
        "ConflictError",
        "InternalServerError",
        "APIStatusError",
        "APIResponseValidationError",
    }
)


def _looks_like_context_overflow(message: str) -> bool:
    return (
        "context length" in message
        or "maximum context" in message
        or "context_length_exceeded" in message
        or "context window" in message
        or ("token" in message and "length" in message)
    )


def _classify_stream_error(exc: BaseException) -> str:
    """Map a stream failure onto a UI error code (item 10).

    The exception *type* decides first; the message is only consulted for types
    we do not recognise. The old substring-first order mislabelled anything whose
    text merely mentioned "timeout" (e.g. ``BadRequestError: invalid parameter
    'timeout'``) as a retriable ``model_timeout``.
    """
    names = {type(exc).__name__}
    for base in type(exc).__mro__:
        names.add(base.__name__)
    message = f"{exc}".lower()

    if names & _AUTH_ERROR_TYPES:
        return "auth"
    if names & _RATE_LIMIT_ERROR_TYPES:
        return "rate_limit"
    # APITimeoutError subclasses APIConnectionError, so timeouts are matched first.
    if names & _TIMEOUT_ERROR_TYPES:
        return "model_timeout"
    if names & _NETWORK_ERROR_TYPES:
        return "network"
    if _looks_like_context_overflow(message):
        return "context_overflow"
    if names & _REQUEST_ERROR_TYPES:
        # A recognised request-level failure: only unambiguous status hints count.
        if "401" in message or "unauthorized" in message or "invalid api key" in message:
            return "auth"
        if "429" in message or "rate limit" in message:
            return "rate_limit"
        return "internal"

    # Unknown exception type (a wrapped provider error, a bare RuntimeError...):
    # fall back to the message.
    if "401" in message or "unauthorized" in message or "invalid api key" in message:
        return "auth"
    if "429" in message or "rate limit" in message:
        return "rate_limit"
    if "timed out" in message or "timeout" in message:
        return "model_timeout"
    if "connection" in message or "network" in message:
        return "network"
    return "internal"


def _error_event(error: str, code: str, *, retriable: bool | None = None) -> dict:
    return {
        "type": "error",
        "ok": False,
        "error": error,
        "code": code,
        "retriable": bool(code in RETRIABLE_ERROR_CODES if retriable is None else retriable),
    }


def _exception_error_event(exc: BaseException, *, context: str = "stream") -> dict:
    logger.exception("LangCode %s failed", context)
    code = _classify_stream_error(exc)
    return _error_event(f"{type(exc).__name__}: {exc}", code)


def _notice_event(kind: str, message: str) -> dict:
    return {"type": "notice", "kind": str(kind or "info"), "message": str(message or "")}


def _usage_event(chunk: Any) -> dict | None:
    usage = getattr(chunk, "usage_metadata", None)
    if not isinstance(usage, dict) or not usage:
        return None
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    if not (input_tokens or output_tokens or total_tokens):
        return None
    return {
        "type": "usage",
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": total_tokens,
    }


def _tool_result_preview_text(tool_result: Any) -> str:
    if isinstance(tool_result, str):
        return tool_result
    if isinstance(tool_result, dict):
        for key in ("content", "text", "output", "stdout", "summary", "result", "preview", "message"):
            value = tool_result.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(_json_safe(tool_result), ensure_ascii=False)
    return json.dumps(_json_safe(tool_result), ensure_ascii=False)


def _tool_success_event(tool_name: str, tool_result: Any) -> dict | None:
    if tool_name in INTERNAL_PREVIEW_TOOL_NAMES:
        return None
    if not _tool_result_succeeded(_json_safe(tool_result)):
        return None
    text = _tool_result_preview_text(_json_safe(tool_result))
    truncated = len(text) > TOOL_RESULT_PREVIEW_CHARS
    return {
        "type": "tool_result",
        "toolName": tool_name,
        "ok": True,
        "preview": text[:TOOL_RESULT_PREVIEW_CHARS],
        "truncated": truncated,
    }


def _stream_tool_result_events(tool_name: str, tool_result: Any):
    """Structured tool_result events for the ndjson stream (successes included)."""
    tool_event = _tool_result_event(tool_name, tool_result)
    if tool_event:
        yield {"type": "tool_result", **tool_event}
        return
    success_event = _tool_success_event(tool_name, tool_result)
    if success_event:
        yield success_event


def _agents_command_reply(threads: list[dict]) -> str:
    lines = [
        "当前可用 Agent 能力：",
        "- 默认：单主 Agent 直接处理。",
        "- 辅助：delegate_agent 单子 Agent 短上下文协助。",
        "- 多视角：delegate_agents 并行运行多个只读子 Agent 后由主 Agent 汇总。",
        "- 辩论/博弈：agent_debate 由 Debate Manager 维护 transcript，A/B/Judge 轮流发言。",
        "",
    ]
    if not threads:
        lines.append("当前会话还没有后台 Agent transcript。")
        return "\n".join(lines)
    lines.append("当前会话后台 Agent transcript：")
    for item in threads:
        participants = item.get("participants") if isinstance(item.get("participants"), list) else []
        names = "、".join(str(participant.get("name") or participant.get("id") or "Agent") for participant in participants)
        lines.append(
            f"- {item.get('id')}：{item.get('title')}（{item.get('kind')}；参与者：{names or '无'}）"
        )
    lines.append("\n打开包含 Agent 对话的消息卡片后，可以点击参与者名称切换指定 Agent 视角。")
    return "\n".join(lines)


def _tool_result_event(tool_name: str, tool_result: Any) -> dict | None:
    tool_result = _json_safe(tool_result)
    if tool_name == "diagram" and isinstance(tool_result, dict) and tool_result.get("ok") is True:
        return {
            "role": "assistant",
            "content": str(tool_result.get("mermaid") or tool_result.get("content") or ""),
            "kind": "diagram",
            "title": str(tool_result.get("title") or "图示"),
            "diagramType": str(tool_result.get("diagram_type") or "flowchart"),
        }
    if isinstance(tool_result, dict) and tool_result.get("kind") == "agent_dialogue" and tool_result.get("ok") is True:
        return _agent_dialogue_event(tool_result)
    if _tool_result_succeeded(tool_result) or _is_internal_tool_result(tool_name):
        return None
    return {
        "role": "tool",
        "content": _tool_result_json(tool_result),
        "kind": "tool_result",
        "toolName": tool_name,
    }


def _tool_result_content_succeeded(content: str) -> bool:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return False
    return _tool_result_succeeded(value)


def _should_show_tool_result_content(content: str) -> bool:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return True
    return not (_tool_result_succeeded(value) or _is_internal_tool_error(value))


def _diagram_message_from_tool_content(content: str) -> dict | None:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or value.get("kind") != "diagram" or value.get("ok") is not True:
        return None
    return {
        "role": "assistant",
        "kind": "diagram",
        "title": str(value.get("title") or "图示"),
        "diagramType": str(value.get("diagram_type") or "flowchart"),
        "content": str(value.get("mermaid") or value.get("content") or ""),
    }


def _agent_dialogue_message_from_tool_content(content: str) -> dict | None:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or value.get("kind") != "agent_dialogue" or value.get("ok") is not True:
        return None
    return _agent_dialogue_event(value)


def _agent_dialogue_event(value: dict) -> dict:
    return {
        "role": "assistant",
        "kind": "agent_dialogue",
        "title": str(value.get("title") or "Agent 协作"),
        "dialogueType": str(value.get("dialogue_type") or value.get("dialogueType") or "agent_dialogue"),
        "threadId": str(value.get("thread_id") or value.get("threadId") or ""),
        "participants": list(value.get("participants") or []),
        "messages": list(value.get("messages") or []),
    }


def _is_internal_tool_result(tool_name: str) -> bool:
    return tool_name in TASK_TOOL_NAMES


def _is_internal_tool_error(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("ok") is True:
        return False
    error = str(value.get("error") or "")
    return value.get("error_type") == "ValueError" and (error.startswith("todo item ") or "任务" in error)


def _tool_result_succeeded(value: Any) -> bool:
    return isinstance(value, dict) and value.get("ok") is True


def _prepare_tool_input(tool_name: str, tool_input: dict, session: WebSession) -> dict:
    if tool_name in {"session_search", "delegate_agents", "agent_debate", "self_evolve"}:
        prepared = dict(tool_input)
        prepared["_current_session_id"] = session.id
        if session.store_path is not None:
            prepared["_session_store_path"] = str(session.store_path)
        if tool_name == "self_evolve":
            # Never put the serialized history into the tool input: it travels
            # through LangGraph state into checkpoints.sqlite. The reflection is
            # re-run in-process from the live session instead (see
            # _reflect_in_process), after the graph applied the approval gate.
            prepared["session_id"] = session.id
            prepared["todos"] = list(session.todos)
        return prepared
    return tool_input


def _is_self_evolve_reflect(tool_name: str, tool_input: dict) -> bool:
    if tool_name != "self_evolve":
        return False
    action = str((tool_input or {}).get("action") or "status").strip().lower()
    return action in {"reflect", "reflect_session"}


def _reflect_in_process(session: WebSession, tool_input: dict) -> dict:
    """Run self_evolve/reflect against the in-memory session history."""
    return _json_safe(
        reflect_session(
            session.workspace_root,
            session_id=session.id,
            messages=_serialize_messages(session.messages),
            todos=list(session.todos),
            apply=bool((tool_input or {}).get("apply", True)),
        )
    )


def _apply_self_evolve_result(session: WebSession, tool_name: str, tool_input: dict, tool_result: Any) -> Any:
    if not _is_self_evolve_reflect(tool_name, tool_input):
        return tool_result
    if isinstance(tool_result, dict) and tool_result.get("ok") is False:
        return tool_result
    try:
        return _reflect_in_process(session, tool_input)
    except Exception as exc:
        logger.exception("self_evolve reflection failed")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "error_type": type(exc).__name__}


def _apply_task_tool_result(session: WebSession, tool_name: str, tool_input: dict, tool_result: Any) -> tuple[Any, bool]:
    if tool_name not in TASK_TOOL_NAMES:
        return tool_result, False
    if isinstance(tool_result, dict) and tool_result.get("ok") is False:
        return tool_result, False
    try:
        if tool_name == "task_create":
            result = create_task(
                session.todos,
                str(tool_input.get("content") or ""),
                status=str(tool_input.get("status") or "pending"),
                task_id=str(tool_input["task_id"]) if tool_input.get("task_id") else None,
            )
        elif tool_name == "task_update":
            result = update_task(
                session.todos,
                str(tool_input.get("task_id") or ""),
                content=str(tool_input["content"]) if tool_input.get("content") is not None else None,
                status=str(tool_input["status"]) if tool_input.get("status") is not None else None,
            )
        elif tool_name == "task_list":
            result = list_tasks(
                session.todos,
                status=str(tool_input["status"]) if tool_input.get("status") else None,
            )
        elif tool_name == "task_get":
            result = get_task(session.todos, str(tool_input.get("task_id") or ""))
        else:
            result = cancel_task(
                session.todos,
                str(tool_input.get("task_id") or ""),
                reason=str(tool_input["reason"]) if tool_input.get("reason") else None,
            )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}, False

    changed = tool_name not in {"task_list", "task_get"}
    if changed and isinstance(result, dict) and isinstance(result.get("todos"), list):
        session.todos = list(result["todos"])
    return result, changed


def _is_task_tool(tool_name: str) -> bool:
    return tool_name in TASK_TOOL_NAMES


def _float_env(name: str, fallback: float) -> float:
    try:
        return float(os.getenv(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback


def _int_env(name: str, fallback: int, *, minimum: int = 1) -> int:
    try:
        value = int(str(os.getenv(name, str(fallback))).strip())
    except (TypeError, ValueError):
        value = fallback
    return max(minimum, value)


def _max_tool_rounds() -> int:
    return _int_env("LANGCODE_MAX_TOOL_ROUNDS", DEFAULT_MAX_TOOL_ROUNDS)


def _max_resident_sessions() -> int:
    return _int_env("LANGCODE_MAX_RESIDENT_SESSIONS", DEFAULT_MAX_RESIDENT_SESSIONS)


def _stream_idle_timeout_sec() -> float:
    value = _float_env("LANGCODE_STREAM_IDLE_TIMEOUT_SEC", DEFAULT_STREAM_IDLE_TIMEOUT_SEC)
    return value if value > 0 else 0.0


class _StreamIdleWatchdog:
    """Flag a model stream that stops producing chunks (item 24).

    A ``threading.Timer`` raises the flag from a background thread, so a stall
    is caught while the streaming iterator is still blocked. ``beat()`` only
    stores a timestamp - the timer reschedules itself for the remaining time
    instead of being recreated per chunk, so a fast stream costs no extra
    threads. The caller turns the flag into a retriable ``model_timeout``.
    """

    __slots__ = ("timeout", "session_id", "timed_out", "_timer", "_lock", "_done", "_last_beat", "_stream")

    def __init__(self, timeout: float, session_id: str = "") -> None:
        self.timeout = float(timeout or 0.0)
        self.session_id = session_id
        self.timed_out = False
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._done = False
        self._last_beat = time.monotonic()
        self._stream: Any | None = None

    def attach(self, stream: Any) -> None:
        """Remember the model iterator so a stall can try to unblock it (item 11).

        ``close()`` on the generator returned by ``model.stream(...)`` raises
        GeneratorExit inside it, which usually frees a consumer parked on the
        next chunk. It is strictly best effort: a generator that is *currently
        executing* refuses to close, and a fully hung socket is only bounded by
        the model client's own ``timeout`` setting.
        """
        with self._lock:
            self._stream = stream

    def _close_stream(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
        if stream is None:
            return
        closer = getattr(stream, "close", None)
        if not callable(closer):
            return
        try:
            closer()
        except Exception:
            logger.debug("Could not close the stalled model stream (session=%s)", self.session_id, exc_info=True)

    def start(self) -> None:
        self._last_beat = time.monotonic()
        self._schedule(self.timeout)

    def beat(self) -> None:
        """Record a chunk. Cheap on purpose: called once per streamed token."""
        self._last_beat = time.monotonic()

    def cancel(self) -> None:
        with self._lock:
            self._done = True
            timer, self._timer = self._timer, None
            self._stream = None
        if timer is not None:
            timer.cancel()

    def _schedule(self, delay: float) -> None:
        with self._lock:
            if self._done or self.timeout <= 0 or self.timed_out:
                return
            timer = threading.Timer(max(0.0, delay), self._check)
            timer.daemon = True
            self._timer = timer
        timer.start()

    def _check(self) -> None:
        remaining = self.timeout - (time.monotonic() - self._last_beat)
        if remaining > 0:
            # A chunk arrived after this timer was armed: wait out the rest.
            self._schedule(remaining)
            return
        with self._lock:
            if self._done:
                return
            self.timed_out = True
        logger.warning(
            "LangCode model stream idle for more than %ss (session=%s)", self.timeout, self.session_id
        )
        # The consumer only notices the flag when a further chunk arrives, which
        # is exactly what a stalled stream never sends - so try to break it.
        self._close_stream()


def _tool_round_limit_reminder() -> HumanMessage:
    return HumanMessage(content=TOOL_ROUND_LIMIT_MOCK_USER)


def _auto_reflect_enabled() -> bool:
    raw = os.getenv("LANGCODE_AUTO_REFLECT", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


GZIP_ASSET_SUFFIXES = {".js", ".mjs", ".cjs", ".css", ".svg", ".json", ".html", ".map", ".txt", ".xml"}
GZIP_MIN_BYTES = 1024
_GZIP_CACHE: dict[tuple[str, int, int], bytes] = {}
_GZIP_CACHE_LOCK = threading.Lock()
_GZIP_CACHE_MAX_ENTRIES = 64
# Item 9: index.html was re-read, token-substituted and gzipped on every single
# request. Cache the rendered HTML plus its gzip body, keyed by file identity and
# token so a rebuilt frontend or a restarted app never serves a stale page.
_INDEX_CACHE: dict[tuple[str, int, int, str], tuple[str, bytes]] = {}
_INDEX_CACHE_LOCK = threading.Lock()


def _rendered_index(path: Path, stat: os.stat_result, token: str) -> tuple[str, bytes]:
    """Return ``(html, gzipped_bytes)`` for index.html, from cache when possible."""
    key = (str(path), stat.st_mtime_ns, stat.st_size, token)
    with _INDEX_CACHE_LOCK:
        cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    html = path.read_text(encoding="utf-8").replace("__LANGCODE_TOKEN__", token)
    body = html.encode("utf-8")
    compressed = gzip.compress(body, 6) if len(body) >= GZIP_MIN_BYTES else b""
    entry = (html, compressed)
    with _INDEX_CACHE_LOCK:
        if len(_INDEX_CACHE) >= _GZIP_CACHE_MAX_ENTRIES:
            _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = entry
    return entry


def _accepts_gzip(request: Request | None) -> bool:
    if request is None:
        return False
    return "gzip" in str(request.headers.get("accept-encoding") or "").lower()


def _is_gzippable_asset(path: Path) -> bool:
    return path.suffix.lower() in GZIP_ASSET_SUFFIXES


def _gzip_cached(path: Path, stat: os.stat_result) -> bytes:
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _GZIP_CACHE_LOCK:
        cached = _GZIP_CACHE.get(key)
    if cached is not None:
        return cached
    compressed = gzip.compress(path.read_bytes(), 6)
    with _GZIP_CACHE_LOCK:
        if len(_GZIP_CACHE) >= _GZIP_CACHE_MAX_ENTRIES:
            _GZIP_CACHE.clear()
        _GZIP_CACHE[key] = compressed
    return compressed


def _apply_static_headers(resp, headers: dict[str, str]):
    """Force exactly one value per header (Sanic's file() adds its own Cache-Control)."""
    for name, value in headers.items():
        try:
            del resp.headers[name]
        except KeyError:
            pass
        resp.headers[name] = value
    return resp


async def _static_response(web_app: WebApp, requested: str, request: Request | None = None):
    frontend_root = web_app.frontend_dir.resolve()
    candidate = (frontend_root / (requested.lstrip("/") or "index.html")).resolve()
    if candidate.is_dir():
        candidate = candidate / "index.html"
    if frontend_root not in candidate.parents and candidate != frontend_root:
        candidate = frontend_root / "index.html"
    if not candidate.exists():
        candidate = frontend_root / "index.html"
    cache_control = (
        "public, max-age=31536000, immutable"
        if candidate.parent.name == "assets" and candidate.name != "index.html"
        else "no-cache"
    )
    stat = candidate.stat()
    etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
    headers = {"Cache-Control": cache_control, "ETag": etag, "Vary": "Accept-Encoding"}
    if request is not None and request.headers.get("if-none-match") == etag:
        return _apply_static_headers(response.empty(status=304), headers)
    wants_gzip = _accepts_gzip(request)
    if candidate.name == "index.html":
        html, compressed = _rendered_index(candidate, stat, web_app.api_token)
        if wants_gzip and compressed:
            resp = response.raw(compressed, content_type="text/html; charset=utf-8")
            return _apply_static_headers(resp, {**headers, "Content-Encoding": "gzip"})
        return _apply_static_headers(response.html(html), headers)
    served = candidate
    encoding = ""
    accepted = request.headers.get("accept-encoding", "") if request is not None else ""
    if "br" in accepted and candidate.with_name(candidate.name + ".br").exists():
        served = candidate.with_name(candidate.name + ".br")
        encoding = "br"
    elif "gzip" in accepted and candidate.with_name(candidate.name + ".gz").exists():
        served = candidate.with_name(candidate.name + ".gz")
        encoding = "gzip"
    mime_type = mimetypes.guess_type(candidate.name)[0]
    if not encoding and wants_gzip and _is_gzippable_asset(candidate) and stat.st_size >= GZIP_MIN_BYTES:
        resp = response.raw(_gzip_cached(candidate, stat), content_type=mime_type or "application/octet-stream")
        return _apply_static_headers(resp, {**headers, "Content-Encoding": "gzip"})
    resp = await response.file(served, mime_type=mime_type)
    if encoding:
        headers = {**headers, "Content-Encoding": encoding}
    return _apply_static_headers(resp, headers)


def _configure_logging() -> None:
    """Make ``langcode.*`` INFO lines visible without touching third-party loggers.

    Sanic only configures its own loggers, so the project's ``logger.info``
    calls (TTS first-audio latency, cancelled turns, ...) were silently dropped
    and only WARNING+ reached stderr through logging's last-resort handler.
    ``LANGCODE_LOG_LEVEL`` overrides the level (e.g. DEBUG for per-segment TTS).
    """
    level_name = (os.getenv("LANGCODE_LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    project_logger = logging.getLogger("langcode")
    project_logger.setLevel(level)
    if not project_logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        project_logger.addHandler(handler)


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Run the LangCode web app")
    parser.add_argument("--workspace", default=".", help="Workspace root")
    parser.add_argument("--host", default="127.0.0.1", help="Host")
    parser.add_argument("--port", type=int, default=8765, help="Port")
    parser.add_argument("--frontend-dir", default="frontend/dist", help="Built frontend directory")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Sanic worker processes. Redis coordinates runtime state when available; keep 1 for local ASR/TTS efficiency.",
    )
    parser.add_argument(
        "--no-voice",
        dest="enable_voice",
        action="store_false",
        default=None,
        help=(
            "Start without local ASR/TTS/turn-detection models. Required on a core-only "
            "install (pip install -e . without the [voice] extra). Equivalent to "
            "LANGCODE_VOICE=0. A remote LANGCODE_VOICE_WORKER_URL still works."
        ),
    )
    args = parser.parse_args(argv)

    # LANGCODE_VOICE mirrors the flag so scripts/start_macos.sh can hand the
    # server the same switch it used to choose the pip extra.
    enable_voice = args.enable_voice
    if enable_voice is None:
        enable_voice = (os.getenv("LANGCODE_VOICE") or "1").strip().lower() not in ("0", "false", "no", "off")

    frontend_dir = Path(args.frontend_dir).expanduser().resolve()
    if not frontend_dir.exists():
        raise SystemExit(f"Frontend build not found: {frontend_dir}. Run npm run build first.")

    app = WebApp(Path(args.workspace), frontend_dir, enable_voice=enable_voice)
    if args.workers > 1 and not app.runtime_state.redis_available:
        raise SystemExit(
            "--workers > 1 requires Redis for cross-process session state, but Redis is unavailable: "
            f"{app.runtime_state.status().get('error') or 'unknown error'}. "
            "Start Redis (or set LANGCODE_REDIS_URL) and retry, or run with --workers 1."
        )
    sanic_app = create_sanic_app(app)
    print(f"LangCode async web app: http://{args.host}:{args.port}")
    print(f"Workspace: {Path(args.workspace).expanduser().resolve()}")
    if not enable_voice:
        print("Voice: disabled (--no-voice / LANGCODE_VOICE=0); no local ASR/TTS models are loaded.")
    elif app.voice_worker is not None:
        print("Voice: remote worker (LANGCODE_VOICE_WORKER_URL); no local ASR/TTS models are loaded.")
    sanic_app.run(
        host=args.host,
        port=args.port,
        workers=args.workers,
        access_log=False,
        single_process=args.workers == 1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
