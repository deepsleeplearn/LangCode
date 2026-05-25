from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import argparse
import json
import os
import platform
import re
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
from ..runtime.chat import build_openai_model, default_system_prompt, messages_from_json, model_settings_from_env
from ..core.config import load_env_files
from ..core.context_management import compact_tool_result, make_json_safe
from ..runtime.delegation import run_delegate_agent
from ..runtime.multi_agent import iter_agent_debate_events, run_agent_debate, run_parallel_delegate_agents
from ..runtime.deep_harness import cancel_task, create_task, get_task, list_tasks, update_task
from ..memory.project import handle_local_command, serialize_message
from ..runtime.permissions import ToolCall, remember_shell_permission
from ..storage.job_queue import JobQueue
from ..storage.runtime_state import RuntimeLockTimeout, RuntimeStateStore
from ..storage.session_store import SessionStore
from ..voice.tts import TtsService, content_type_for_path
from ..voice.turnsense import TurnSenseService
from .voice_proxy import VoiceWorkerClient


TASK_TOOL_NAMES = {"task_create", "task_update", "task_list", "task_get", "task_cancel"}
DEFAULT_WEB_SEARCH_LIMIT = 8
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


class WebApp:
    def __init__(self, workspace_root: Path, frontend_dir: Path) -> None:
        self.workspace_root = self._resolve_workspace(workspace_root)
        load_env_files(Path.cwd(), self.workspace_root)
        self.frontend_dir = frontend_dir
        self.sessions: dict[str, WebSession] = {}
        self._sessions_lock = threading.RLock()
        self.home_workspace_root = self.workspace_root
        self.state_dir = self.home_workspace_root / ".langcode"
        self.store = SessionStore(self.state_dir / "web.sqlite")
        runtime_prefix = os.getenv("LANGCODE_REDIS_PREFIX") or (
            "langcode:" + hashlib.sha1(str(self.state_dir).encode("utf-8")).hexdigest()[:12]
        )
        self.runtime_state = RuntimeStateStore(prefix=runtime_prefix)
        self.job_queue = JobQueue(prefix=runtime_prefix)
        self._configure_workspace_storage()
        self.voice_worker = self._voice_worker_from_env()
        if self.voice_worker is None:
            self.turnsense = TurnSenseService()
            self.asr = QwenAsrService(turnsense=self.turnsense)
            self.tts = TtsService()
            self.asr.start_preload()
            self.tts.start_preload()
        else:
            self.turnsense = None
            self.asr = None
            self.tts = None

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

    def get_session(self, session_id: str) -> WebSession:
        with self._sessions_lock:
            if session_id in self.sessions:
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
            if isinstance(state, dict) and state.get("display_messages"):
                display_messages = messages_from_json(state.get("display_messages") or [])
            else:
                display_messages = _recover_display_messages_from_compaction(session_workspace, session_id, messages)
            session = WebSession(
                id=session_id,
                agent=CodeAgent(session_workspace, checkpoint_path=self.state_dir / "checkpoints.sqlite"),
                workspace_root=session_workspace,
                messages=messages,
                display_messages=display_messages,
                pending=stored.get("pending") if stored else None,
                todos=list((state or {}).get("todos") or []) if stored else [],
                store_path=self.store.path,
            )
            self.sessions[session_id] = session
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
            stored = self.store.load_session(session_id)
            session = self.sessions.get(session_id)
            if stored is None or session is None:
                return
            session_workspace = self._resolve_workspace(Path(stored["workspace"]))
            if session.workspace_root != session_workspace:
                session.agent.close()
                session.agent = CodeAgent(session_workspace, checkpoint_path=self.state_dir / "checkpoints.sqlite")
                session.workspace_root = session_workspace
                session.model = None
            messages = messages_from_json(stored["messages"])
            state = stored.get("state") if stored else {}
            if isinstance(state, dict) and state.get("display_messages"):
                display_messages = messages_from_json(state.get("display_messages") or [])
            else:
                display_messages = _recover_display_messages_from_compaction(session_workspace, session_id, messages)
            session.messages = messages
            session.display_messages = display_messages
            session.pending = stored.get("pending")
            session.todos = list((state or {}).get("todos") or [])

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

    def session_view(self, session_id: str) -> dict:
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
        return {
            "ok": True,
            "sessionId": session_id,
            "title": self.store.load_session(session_id)["title"],
            "workspace": str(session.workspace_root),
            "messages": messages,
            "pendingApproval": session.pending,
            "todos": session.todos,
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
        self.runtime_state.cancel_run(session_id, run_id)
        return {"ok": True}

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
            yield {"type": "error", "ok": False, "error": "Message is required"}
            return
        if session.pending is not None:
            yield {
                "type": "error",
                "ok": False,
                "error": "Resolve the pending tool approval before sending a new message.",
            }
            return

        local_reply = self._handle_local_command(session, message)
        if local_reply is not None:
            self._save_history(session)
            yield {"type": "delta", "content": local_reply}
            yield {"type": "done", "ok": True}
            return

        settings = model_settings_from_env()
        if not settings.api_key:
            yield {
                "type": "error",
                "ok": False,
                "error": f"No model API key configured for provider={settings.provider}.",
            }
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
            yield {"type": "error", "ok": False, "error": "No pending approval for this session."}
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
        tool_event = _tool_result_event(tool_name, tool_result)
        if tool_event:
            yield {"type": "tool_result", **tool_event}
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
        _repair_live_tool_history(session)
        completed_tools = 0
        plan_written_this_turn = False
        non_plan_tools_completed = 0
        continued_after_plan_only = False
        turn_start_message_count = _failed_turn_start_index(session.messages)
        turn_start_display_count = _failed_turn_start_index(session.display_messages)
        web_search_count = 0
        while True:
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
            try:
                for chunk in session.model.stream([*session.messages, *(extra_messages or [])]):
                    if self._is_run_cancelled(session.id, run_id):
                        if partial_content:
                            _save_cancelled_partial(session, partial_content, self._save_history)
                        else:
                            _drop_current_failed_turn(session, turn_start_message_count, turn_start_display_count)
                            self._save_history(session)
                        yield {"type": "done", "ok": False, "cancelled": True}
                        return
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
                        now = time.monotonic()
                        if now - last_partial_save >= 1.0:
                            self._save_history(session, extra_ai_content="".join(partial_content))
                            last_partial_save = now
                        yield {"type": "delta", "content": content}
            except Exception as exc:
                if partial_content:
                    self._save_history(session, extra_ai_content="".join(partial_content))
                else:
                    _drop_current_failed_turn(session, turn_start_message_count, turn_start_display_count)
                    self._save_history(session)
                yield {"type": "error", "ok": False, "error": f"{type(exc).__name__}: {exc}"}
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
                    _save_cancelled_partial(session, partial_content, self._save_history)
                else:
                    _drop_current_failed_turn(session, turn_start_message_count, turn_start_display_count)
                    self._save_history(session)
                yield {"type": "done", "ok": False, "cancelled": True}
                return
            if full_chunk is None:
                _drop_current_failed_turn(session, turn_start_message_count, turn_start_display_count)
                self._save_history(session)
                yield {"type": "error", "ok": False, "error": "Model returned no response."}
                return

            tool_calls = list(getattr(full_chunk, "tool_calls", None) or [])
            ai_message = AIMessage(content="".join(partial_content), tool_calls=tool_calls)
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

            total_tools = len(tool_calls)
            for index, raw_tool_call in enumerate(tool_calls, start=1):
                if self._is_run_cancelled(session.id, run_id):
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
                    tool_event = _tool_result_event(tool_name, tool_result)
                    if tool_event:
                        yield {"type": "tool_result", **tool_event}
                    continue
                if tool_name == "delegate_agents":
                    tool_result = _json_safe(run_parallel_delegate_agents(session.workspace_root, **tool_input))
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
                    tool_event = _tool_result_event(tool_name, tool_result)
                    if tool_event:
                        yield {"type": "tool_result", **tool_event}
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
                tool_event = _tool_result_event(tool_name, tool_result)
                if tool_event:
                    yield {"type": "tool_result", **tool_event}

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

    def _save_history(self, session: WebSession, *, extra_ai_content: str | None = None) -> None:
        messages = list(session.messages)
        display_messages = list(session.display_messages or session.messages)
        if extra_ai_content:
            messages.append(AIMessage(content=extra_ai_content))
            display_messages.append(AIMessage(content=extra_ai_content))

        serializable = _serialize_messages(messages)
        self.store.save_messages(
            session.id,
            str(session.workspace_root),
            serializable,
            pending=session.pending,
            state={"todos": session.todos, "display_messages": _serialize_messages(display_messages)},
        )


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


def _save_cancelled_partial(session: WebSession, partial_content: list[str], save_history) -> None:
    content = "".join(partial_content).strip()
    if content:
        session.messages.append(AIMessage(content=content))
        session.display_messages.append(AIMessage(content=content))
    save_history(session)


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


def create_sanic_app(web_app: WebApp) -> Sanic:
    sanic_app = Sanic("langcode-web")
    sanic_app.config.REQUEST_TIMEOUT = 3600
    sanic_app.config.RESPONSE_TIMEOUT = 3600
    sanic_app.ctx.web_app = web_app
    sanic_app.ctx.session_locks = {}
    sanic_app.ctx.session_locks_guard = asyncio.Lock()
    sanic_app.ctx.workspace_lock = asyncio.Lock()

    class SessionRequestLock:
        def __init__(self, session_id: str, local_lock: asyncio.Lock) -> None:
            self.session_id = session_id
            self.local_lock = local_lock
            self.lease = None

        async def __aenter__(self):
            await self.local_lock.acquire()
            try:
                self.lease = await asyncio.to_thread(web_app.runtime_state.acquire_session_lock, self.session_id)
                await asyncio.to_thread(web_app.refresh_session_from_store, self.session_id)
            except Exception:
                self.local_lock.release()
                raise
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            try:
                if self.lease is not None:
                    await asyncio.to_thread(self.lease.release)
            finally:
                self.local_lock.release()

    async def session_lock(session_id: str) -> SessionRequestLock:
        async with sanic_app.ctx.session_locks_guard:
            lock = sanic_app.ctx.session_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                sanic_app.ctx.session_locks[session_id] = lock
            return SessionRequestLock(session_id, lock)

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
        text = str(payload.get("text") or "")
        voice_id = str(payload.get("voiceId") or "")

        async def stream(streaming_response):
            if web_app.voice_worker is not None:
                async for event in web_app.voice_worker.stream_tts(payload):
                    await streaming_response.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
                return
            if web_app.job_queue.available:
                await _stream_queue_job(web_app, streaming_response, "tts_stream", payload, heartbeat=False)
                return
            event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def produce_audio() -> None:
                try:
                    if web_app.tts is None:
                        raise RuntimeError("TTS service is not available.")
                    for index, (audio, content_type) in enumerate(web_app.tts.synthesize_chunks(text, voice_id=voice_id), start=1):
                        event = {
                            "type": "audio",
                            "index": index,
                            "contentType": content_type,
                            "audio": base64.b64encode(audio).decode("ascii"),
                        }
                        loop.call_soon_threadsafe(event_queue.put_nowait, event)
                    loop.call_soon_threadsafe(event_queue.put_nowait, {"type": "done"})
                except Exception as exc:
                    loop.call_soon_threadsafe(
                        event_queue.put_nowait,
                        {"type": "error", "ok": False, "error": f"{type(exc).__name__}: {exc}"},
                    )
                finally:
                    loop.call_soon_threadsafe(event_queue.put_nowait, None)

            producer = asyncio.create_task(asyncio.to_thread(produce_audio))
            while True:
                event = await event_queue.get()
                if event is None:
                    break
                await streaming_response.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
            await producer

        return response.ResponseStream(
            stream,
            content_type="application/x-ndjson; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

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
        lock = await session_lock(session_id)
        try:
            async with lock:
                return await run_json(web_app.session_view, session_id)
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
                event = {"type": "error", "ok": False, "error": str(exc)}
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
                        loop.call_soon_threadsafe(
                            event_queue.put_nowait,
                            {"type": "error", "ok": False, "error": f"{type(exc).__name__}: {exc}"},
                        )
                    finally:
                        loop.call_soon_threadsafe(event_queue.put_nowait, None)

                producer = asyncio.create_task(asyncio.to_thread(produce_events))
                waiting_count = 0
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
                event = {"type": "error", "ok": False, "error": str(exc)}
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
                        loop.call_soon_threadsafe(
                            event_queue.put_nowait,
                            {"type": "error", "ok": False, "error": f"{type(exc).__name__}: {exc}"},
                        )
                    finally:
                        loop.call_soon_threadsafe(event_queue.put_nowait, None)

                producer = asyncio.create_task(asyncio.to_thread(produce_events))
                waiting_count = 0
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
    async def index(_request: Request):
        return await _static_response(web_app, "index.html")

    @sanic_app.get("/<path:path>")
    async def static_files(_request: Request, path: str):
        if path.startswith("api/"):
            return response.json({"ok": False, "error": "Not found"}, status=404)
        return await _static_response(web_app, path)

    return sanic_app


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
        event = {"type": "error", "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        await streaming_response.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
        return
    event_iter = web_app.job_queue.iter_events(job_id, block_ms=4000)
    waiting_count = 0
    while True:
        try:
            event = await asyncio.to_thread(next, event_iter)
        except Exception as exc:
            event = {"type": "error", "ok": False, "error": f"{type(exc).__name__}: {exc}"}
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


def _stream_waiting_event(waiting_count: int = 1) -> dict:
    elapsed = max(4, waiting_count * 4)
    labels = [
        f"模型响应较慢，已等待约 {elapsed} 秒，连接仍保持中。",
        f"仍在等待模型返回下一段内容，已等待约 {elapsed} 秒。",
        f"长时间无新内容时可以点击停止按钮结束当前轮次，当前已等待约 {elapsed} 秒。",
    ]
    return {
        "type": "progress",
        "status": "running",
        "toolName": "model",
        "completed": 0,
        "label": labels[(waiting_count - 1) % len(labels)],
    }


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
    if tool_name in {"session_search", "delegate_agents", "agent_debate"}:
        prepared = dict(tool_input)
        prepared["_current_session_id"] = session.id
        if session.store_path is not None:
            prepared["_session_store_path"] = str(session.store_path)
        return prepared
    return tool_input


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


async def _static_response(web_app: WebApp, requested: str):
    frontend_root = web_app.frontend_dir.resolve()
    candidate = (frontend_root / (requested.lstrip("/") or "index.html")).resolve()
    if candidate.is_dir():
        candidate = candidate / "index.html"
    if frontend_root not in candidate.parents and candidate != frontend_root:
        candidate = frontend_root / "index.html"
    if not candidate.exists():
        candidate = frontend_root / "index.html"
    return await response.file(candidate)


def main(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)

    frontend_dir = Path(args.frontend_dir).expanduser().resolve()
    if not frontend_dir.exists():
        raise SystemExit(f"Frontend build not found: {frontend_dir}. Run npm run build first.")

    app = WebApp(Path(args.workspace), frontend_dir)
    sanic_app = create_sanic_app(app)
    print(f"LangCode async web app: http://{args.host}:{args.port}")
    print(f"Workspace: {Path(args.workspace).expanduser().resolve()}")
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
