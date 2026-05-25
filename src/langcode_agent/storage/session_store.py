from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Iterable


class SessionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def list_sessions(self) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, COALESCE(NULLIF(title, ''), id) AS title, workspace, updated_at
                FROM sessions
                WHERE deleted_at IS NULL
                ORDER BY updated_at DESC, id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def load_session(self, session_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            session = conn.execute(
                """
                SELECT id, COALESCE(NULLIF(title, ''), id) AS title, workspace, pending_json, state_json
                FROM sessions
                WHERE id = ? AND deleted_at IS NULL
                """,
                (session_id,),
            ).fetchone()
            if session is None:
                return None
            messages = conn.execute(
                """
                SELECT role, content, tool_call_id, tool_calls_json
                FROM messages
                WHERE session_id = ?
                ORDER BY idx
                """,
                (session_id,),
            ).fetchall()
        message_rows = []
        for row in messages:
            message = dict(row)
            tool_calls_json = message.pop("tool_calls_json", None)
            tool_calls = _read_json_value(tool_calls_json)
            if isinstance(tool_calls, list):
                message["tool_calls"] = tool_calls
            message_rows.append(message)
        return {
            "id": session["id"],
            "title": session["title"],
            "workspace": session["workspace"],
            "pending": _read_json_text(session["pending_json"]),
            "state": _read_json_text(session["state_json"]) or {},
            "messages": message_rows,
        }

    def search_messages(self, query: str, *, current_session_id: str = "", limit: int = 8) -> list[dict]:
        term = str(query or "").strip()
        if not term:
            return []
        max_items = max(1, min(int(limit or 8), 20))
        with self._lock, self._connect() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT f.session_id, f.idx, f.role, f.content, s.title, s.workspace
                    FROM message_fts f
                    JOIN sessions s ON s.id = f.session_id
                    WHERE message_fts MATCH ? AND s.deleted_at IS NULL
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (term, max_items),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute(
                    """
                    SELECT m.session_id, m.idx, m.role, m.content, s.title, s.workspace
                    FROM messages m
                    JOIN sessions s ON s.id = m.session_id
                    WHERE m.content LIKE ? AND s.deleted_at IS NULL
                    ORDER BY s.updated_at DESC, m.idx DESC
                    LIMIT ?
                    """,
                    (f"%{term}%", max_items),
                ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "message_id": row["idx"],
                "role": row["role"],
                "title": row["title"],
                "workspace": row["workspace"],
                "current_session": row["session_id"] == current_session_id,
                "snippet": _clip_text(row["content"], 500),
            }
            for row in rows
        ]

    def recent_sessions(self, *, limit: int = 8) -> list[dict]:
        max_items = max(1, min(int(limit or 8), 20))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, COALESCE(NULLIF(title, ''), id) AS title, workspace, updated_at
                FROM sessions
                WHERE deleted_at IS NULL
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (max_items,),
            ).fetchall()
        return [dict(row) for row in rows]

    def messages_around(self, session_id: str, message_id: int, *, before: int = 3, after: int = 3) -> list[dict]:
        lower = max(0, int(message_id) - max(0, int(before)))
        upper = int(message_id) + max(0, int(after))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT idx, role, content
                FROM messages
                WHERE session_id = ? AND idx BETWEEN ? AND ?
                ORDER BY idx
                """,
                (session_id, lower, upper),
            ).fetchall()
        return [{"message_id": row["idx"], "role": row["role"], "content": row["content"]} for row in rows]

    def list_agent_threads(self, session_id: str) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, kind, title, participants_json, state_json, created_at, updated_at
                FROM agent_threads
                WHERE session_id = ?
                ORDER BY updated_at DESC, id DESC
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "kind": row["kind"],
                "title": row["title"],
                "participants": _read_json_value(row["participants_json"]) or [],
                "state": _read_json_text(row["state_json"]) or {},
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def load_agent_thread(self, thread_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            thread = conn.execute(
                """
                SELECT id, session_id, kind, title, participants_json, state_json, created_at, updated_at
                FROM agent_threads
                WHERE id = ?
                """,
                (thread_id,),
            ).fetchone()
            if thread is None:
                return None
            messages = conn.execute(
                """
                SELECT idx, agent_id, agent_name, role, content, round_index, created_at
                FROM agent_messages
                WHERE thread_id = ?
                ORDER BY idx
                """,
                (thread_id,),
            ).fetchall()
        return {
            "id": thread["id"],
            "session_id": thread["session_id"],
            "kind": thread["kind"],
            "title": thread["title"],
            "participants": _read_json_value(thread["participants_json"]) or [],
            "state": _read_json_text(thread["state_json"]) or {},
            "created_at": thread["created_at"],
            "updated_at": thread["updated_at"],
            "messages": [
                {
                    "idx": row["idx"],
                    "agent_id": row["agent_id"],
                    "agent_name": row["agent_name"],
                    "role": row["role"],
                    "content": row["content"],
                    "round": row["round_index"],
                    "created_at": row["created_at"],
                }
                for row in messages
            ],
        }

    def save_agent_dialogue(
        self,
        session_id: str,
        thread_id: str,
        *,
        kind: str,
        title: str,
        participants: list[dict],
        messages: list[dict],
        state: dict | None = None,
    ) -> None:
        now = _utc_now()
        participants_json = json.dumps(participants, ensure_ascii=False)
        state_json = json.dumps(state or {}, ensure_ascii=False)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO agent_threads (
                    id, session_id, kind, title, participants_json, state_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    session_id = excluded.session_id,
                    kind = excluded.kind,
                    title = excluded.title,
                    participants_json = excluded.participants_json,
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (thread_id, session_id, kind, title, participants_json, state_json, now, now),
            )
            conn.execute("DELETE FROM agent_messages WHERE thread_id = ?", (thread_id,))
            conn.executemany(
                """
                INSERT INTO agent_messages (
                    thread_id, idx, agent_id, agent_name, role, content, round_index, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        thread_id,
                        index,
                        str(item.get("agent_id") or item.get("agentId") or "agent"),
                        str(item.get("agent_name") or item.get("agentName") or item.get("agent_id") or "Agent"),
                        str(item.get("role") or "assistant"),
                        str(item.get("content") or ""),
                        int(item.get("round") or item.get("round_index") or 0),
                        now,
                    )
                    for index, item in enumerate(messages)
                ],
            )
            conn.execute(
                "INSERT INTO session_events (session_id, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
                (
                    session_id,
                    "agent_dialogue",
                    json.dumps({"thread_id": thread_id, "kind": kind, "title": title}, ensure_ascii=False),
                    now,
                ),
            )
            conn.commit()

    def ensure_session(self, session_id: str, workspace: str, *, title: str | None = None) -> None:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existed = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ? AND deleted_at IS NULL",
                (session_id,),
            ).fetchone()
            conn.execute(
                """
                INSERT INTO sessions (id, title, workspace, created_at, updated_at, deleted_at)
                VALUES (?, ?, ?, ?, ?, NULL)
                ON CONFLICT(id) DO UPDATE SET
                    workspace = excluded.workspace,
                    updated_at = sessions.updated_at,
                    deleted_at = NULL
                """,
                (session_id, title or session_id, workspace, now, now),
            )
            if existed is None:
                conn.execute(
                    "INSERT INTO session_events (session_id, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
                    (
                        session_id,
                        "create",
                        json.dumps({"workspace": workspace, "title": title or session_id}, ensure_ascii=False),
                        now,
                    ),
                )
            conn.commit()

    def save_messages(
        self,
        session_id: str,
        workspace: str,
        messages: Iterable[dict],
        *,
        title: str | None = None,
        pending: dict | None = None,
        state: dict | None = None,
    ) -> None:
        now = _utc_now()
        pending_json = json.dumps(pending, ensure_ascii=False) if pending else None
        state_json = json.dumps(state, ensure_ascii=False) if state else None
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO sessions (id, title, workspace, created_at, updated_at, deleted_at, pending_json, state_json)
                VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    workspace = excluded.workspace,
                    updated_at = excluded.updated_at,
                    deleted_at = NULL,
                    pending_json = excluded.pending_json,
                    state_json = excluded.state_json
                """,
                (session_id, title or session_id, workspace, now, now, pending_json, state_json),
            )
            materialized = list(messages)
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM message_fts WHERE session_id = ?", (session_id,))
            conn.executemany(
                """
                INSERT INTO messages (session_id, idx, role, content, tool_call_id, tool_calls_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        index,
                        item["role"],
                        item.get("content", ""),
                        item.get("tool_call_id"),
                        json.dumps(item.get("tool_calls"), ensure_ascii=False) if item.get("tool_calls") else None,
                    )
                    for index, item in enumerate(materialized)
                ],
            )
            conn.executemany(
                """
                INSERT INTO message_fts (session_id, idx, role, content)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (session_id, index, item["role"], item.get("content", ""))
                    for index, item in enumerate(materialized)
                    if str(item.get("content", "")).strip()
                ],
            )
            conn.execute(
                "INSERT INTO session_events (session_id, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
                (
                    session_id,
                    "save_messages",
                    json.dumps({"workspace": workspace}, ensure_ascii=False),
                    now,
                ),
            )
            conn.commit()

    def rename_session(self, session_id: str, title: str) -> None:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ? AND deleted_at IS NULL",
                (title, now, session_id),
            )
            conn.execute(
                "INSERT INTO session_events (session_id, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
                (session_id, "rename", json.dumps({"title": title}, ensure_ascii=False), now),
            )
            conn.commit()

    def delete_session(self, session_id: str) -> None:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE sessions SET deleted_at = ?, updated_at = ? WHERE id = ?",
                (now, now, session_id),
            )
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM message_fts WHERE session_id = ?", (session_id,))
            _delete_agent_dialogues(conn, session_id)
            conn.execute(
                "INSERT INTO session_events (session_id, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
                (session_id, "delete", "{}", now),
            )
            conn.commit()

    def clear_session(self, session_id: str) -> None:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                "UPDATE sessions SET updated_at = ?, pending_json = NULL, state_json = NULL WHERE id = ? AND deleted_at IS NULL",
                (now, session_id),
            )
            if updated.rowcount == 0:
                conn.commit()
                return
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM message_fts WHERE session_id = ?", (session_id,))
            _delete_agent_dialogues(conn, session_id)
            conn.execute(
                "INSERT INTO session_events (session_id, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
                (session_id, "clear", "{}", now),
            )
            conn.commit()

    def session_exists(self, session_id: str) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return row is not None

    def migrate_json_sessions(self, sessions_dir: Path, metadata_path: Path, workspace: str) -> None:
        metadata = _read_json_object(metadata_path)
        if not sessions_dir.exists():
            return
        for history_path in sorted(sessions_dir.glob("*.json")):
            session_id = history_path.stem
            if self.session_exists(session_id):
                continue
            try:
                messages = json.loads(history_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            title = session_id
            item = metadata.get(session_id)
            if isinstance(item, dict):
                title = str(item.get("title") or "").strip() or session_id
            self.save_messages(session_id, workspace, messages, title=title)

    def _initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    workspace TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    pending_json TEXT
                    ,
                    state_json TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    session_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_call_id TEXT,
                    tool_calls_json TEXT,
                    PRIMARY KEY (session_id, idx),
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS session_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_threads (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    participants_json TEXT NOT NULL,
                    state_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_messages (
                    thread_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    agent_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    round_index INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (thread_id, idx),
                    FOREIGN KEY (thread_id) REFERENCES agent_threads(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);
                CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, idx);
                CREATE INDEX IF NOT EXISTS idx_session_events_session_id ON session_events(session_id, id);
                CREATE INDEX IF NOT EXISTS idx_agent_threads_session_id ON agent_threads(session_id, updated_at);
                CREATE INDEX IF NOT EXISTS idx_agent_messages_thread_id ON agent_messages(thread_id, idx);
                """
            )
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS message_fts
                    USING fts5(session_id UNINDEXED, idx UNINDEXED, role UNINDEXED, content)
                    """
                )
            except sqlite3.OperationalError:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS message_fts (
                        session_id TEXT NOT NULL,
                        idx INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL
                    )
                    """
                )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "pending_json" not in columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN pending_json TEXT")
            if "state_json" not in columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN state_json TEXT")
            message_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(messages)").fetchall()
            }
            if "tool_calls_json" not in message_columns:
                conn.execute("ALTER TABLE messages ADD COLUMN tool_calls_json TEXT")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_object(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_json_text(text: str | None) -> dict | None:
    data = _read_json_value(text)
    return data if isinstance(data, dict) else None


def _read_json_value(text: str | None) -> object | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _clip_text(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "..."


def _delete_agent_dialogues(conn: sqlite3.Connection, session_id: str) -> None:
    rows = conn.execute("SELECT id FROM agent_threads WHERE session_id = ?", (session_id,)).fetchall()
    thread_ids = [row["id"] for row in rows]
    if not thread_ids:
        return
    conn.executemany("DELETE FROM agent_messages WHERE thread_id = ?", [(thread_id,) for thread_id in thread_ids])
    conn.execute("DELETE FROM agent_threads WHERE session_id = ?", (session_id,))
