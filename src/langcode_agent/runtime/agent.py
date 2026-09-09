import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..core._compat import patch_langchain_debug
from .permissions import ApprovalMode, ToolCall, classify_shell_risk, permission_for_tool
from ..tooling.tools import detect_workspace_escape, execute_tool

patch_langchain_debug()

DEFAULT_CHECKPOINT_KEEP = 20


class AgentState(TypedDict, total=False):
    workspace_root: str
    tool_call: dict
    approval: dict
    workspace_escape: dict
    tool_result: dict


class CodeAgent:
    def __init__(
        self,
        workspace_root: str | Path,
        checkpointer: Any | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self._checkpoint_connection: sqlite3.Connection | None = None
        self._checkpoint_lock = threading.Lock()
        self._checkpoint_blobs_table: str | None | bool = False  # False = not probed yet
        self.checkpointer = checkpointer or self._open_checkpointer(checkpoint_path)
        self.graph = self._build_graph()

    def close(self) -> None:
        if self._checkpoint_connection is not None:
            self._checkpoint_connection.close()
            self._checkpoint_connection = None

    def request_tool(self, tool_call: ToolCall, *, thread_id: str | None = None) -> dict:
        return self.graph.invoke(
            {
                "workspace_root": str(self.workspace_root),
                "tool_call": tool_call.to_dict(),
            },
            config=self._config(thread_id),
        )

    def resume(self, thread_id: str, response: dict) -> dict:
        return self.graph.invoke(
            Command(resume=self._normalize_approval(response)),
            config=self._config(thread_id),
        )

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("review", self._review_tool)
        graph.add_node("execute", self._execute_tool)
        graph.add_edge(START, "review")
        graph.add_edge("review", "execute")
        graph.add_edge("execute", END)
        return graph.compile(checkpointer=self.checkpointer)

    def _review_tool(self, state: AgentState) -> dict:
        tool_call = ToolCall.from_dict(state["tool_call"])
        mode = permission_for_tool(tool_call, workspace_root=state["workspace_root"])
        if mode is ApprovalMode.DENY:
            return {"approval": {"type": "reject", "reason": "该工具不允许执行"}}

        workspace_escape = self._workspace_escape_summary(state["workspace_root"], tool_call)
        if workspace_escape is None and mode is ApprovalMode.ALLOW:
            return {"approval": {"type": "accept"}}

        payload = {
            "tool_name": tool_call.name,
            "tool_input": tool_call.args,
            "risk": workspace_escape or self._risk_summary(tool_call),
            "options": ["accept", "reject", "edit", "feedback"],
        }
        if workspace_escape is not None:
            payload["workspace_escape"] = workspace_escape
        response = interrupt(payload)
        output: dict[str, Any] = {"approval": response}
        if workspace_escape is not None:
            output["workspace_escape"] = workspace_escape
        return output

    def _execute_tool(self, state: AgentState) -> dict:
        tool_call = ToolCall.from_dict(state["tool_call"])
        workspace_error = self._workspace_mismatch_error(state)
        if workspace_error is not None:
            return {"tool_result": workspace_error}

        approval = self._normalize_approval(state.get("approval", {"type": "accept"}))
        response_type = approval["type"]

        if response_type == "reject":
            return {
                "tool_result": {
                    "ok": False,
                    "skipped": True,
                    "reason": approval.get("reason", "用户已拒绝"),
                }
            }
        if response_type == "feedback":
            return {
                "tool_result": {
                    "ok": False,
                    "skipped": True,
                    "feedback": approval.get("feedback") or approval.get("message", ""),
                }
            }
        if response_type == "edit":
            tool_input = dict(approval.get("tool_input", tool_call.args))
        else:
            tool_input = tool_call.args

        actual_workspace_escape = self._workspace_escape_summary(
            state["workspace_root"],
            ToolCall(tool_call.name, tool_input),
        )
        if response_type == "edit" and state.get("workspace_escape") and actual_workspace_escape != state["workspace_escape"]:
            return {
                "tool_result": {
                    "ok": False,
                    "error": (
                        "修改后的工具输入改变了已审批的工作区逃逸目标。"
                        "请把修改后的操作作为新的工具调用重新提交审批。"
                    ),
                }
            }
        allow_workspace_escape = bool(state.get("workspace_escape")) and response_type in {"accept", "edit"}
        if actual_workspace_escape is not None and not allow_workspace_escape:
            return {
                "tool_result": {
                    "ok": False,
                    "error": (
                        "路径逃逸工作区，执行前必须获得明确的人工审批："
                        f"{actual_workspace_escape['reason']}"
                    ),
                }
            }

        try:
            result = execute_tool(
                state["workspace_root"],
                tool_call.name,
                tool_input,
                allow_workspace_escape=allow_workspace_escape,
            )
        except Exception as exc:
            result = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        return {"tool_result": result}

    def _normalize_approval(self, approval: Any) -> dict:
        if not isinstance(approval, dict):
            return {
                "type": "reject",
                "reason": f"审批载荷格式错误：期望对象，实际是 {type(approval).__name__}",
            }

        response_type = approval.get("type")
        if response_type not in {"accept", "reject", "edit", "feedback"}:
            return {
                "type": "reject",
                "reason": f"无效的审批类型：{response_type!r}",
            }

        if response_type == "edit" and not isinstance(approval.get("tool_input"), dict):
            return {
                "type": "reject",
                "reason": "编辑审批必须提供 tool_input 对象",
            }

        return approval

    def _workspace_mismatch_error(self, state: AgentState) -> dict | None:
        state_workspace = Path(state["workspace_root"]).expanduser().resolve()
        if state_workspace == self.workspace_root:
            return None
        return {
            "ok": False,
            "error": (
                "检查点绑定的工作区不匹配："
                f"状态绑定到 {state_workspace}，当前 Agent 绑定到 {self.workspace_root}"
            ),
        }

    def _risk_summary(self, tool_call: ToolCall) -> dict:
        if tool_call.name == "shell":
            return classify_shell_risk(str(tool_call.args.get("command", ""))).__dict__
        if tool_call.name in {"write_file", "edit_file"}:
            return {"dangerous": False, "reason": "文件修改需要审批"}
        return {"dangerous": False, "reason": "未检测到特殊风险"}

    def _workspace_escape_summary(self, workspace_root: str | Path, tool_call: ToolCall) -> dict | None:
        escape = detect_workspace_escape(workspace_root, tool_call.name, tool_call.args)
        if escape is None:
            return None
        return {
            **escape,
            "reason": f"路径逃逸工作区。{escape['reason']}",
        }

    def _config(self, thread_id: str | None) -> dict:
        return {"configurable": {"thread_id": thread_id or str(uuid4())}}

    def _open_checkpointer(self, checkpoint_path: str | Path | None) -> Any:
        if checkpoint_path is None:
            return InMemorySaver()

        path = Path(checkpoint_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, check_same_thread=False)
        # WAL lets the CLI, the web server and the worker read while one writes;
        # busy_timeout replaces the default instant "database is locked" failure.
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        self._checkpoint_connection = connection
        saver = SqliteSaver(connection)
        saver.setup()
        return saver

    def prune_thread(self, thread_id: str, *, keep: int | None = None) -> int:
        """Keep only the newest ``keep`` checkpoints of one thread.

        LangGraph writes a checkpoint per super-step, so a long-lived thread
        grows without bound. ``checkpoint_id`` is a time-ordered UUID, so the
        newest rows sort first under ``ORDER BY checkpoint_id DESC``; everything
        older is deleted together with its ``writes`` rows. Returns the number of
        checkpoints removed (0 for the in-memory checkpointer).

        The connection is shared with ``SqliteSaver``, which serializes its own
        writes on ``SqliteSaver.lock``. Pruning under a *different* lock let a
        concurrent graph step interleave with these DELETEs on the same
        connection, so we take the saver's lock whenever it exposes one.

        langgraph-checkpoint-sqlite 2.0.10 (the installed version) stores only
        ``checkpoints`` and ``writes``; newer layouts add a blob table, so
        ``sqlite_master`` is probed once and its rows pruned too when present.
        """

        connection = self._checkpoint_connection
        if connection is None or not thread_id:
            return 0

        keep_count = _checkpoint_keep(keep)
        with self._checkpoint_write_lock():
            try:
                stale = connection.execute(
                    "SELECT checkpoint_ns, checkpoint_id FROM checkpoints "
                    "WHERE thread_id = ? ORDER BY checkpoint_id DESC",
                    (thread_id,),
                ).fetchall()[keep_count:]
            except sqlite3.Error:
                return 0
            if not stale:
                return 0
            keys = [(thread_id, namespace, checkpoint_id) for namespace, checkpoint_id in stale]
            blobs_table = self._blobs_table(connection)
            try:
                connection.executemany(
                    "DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                    keys,
                )
                if blobs_table:
                    connection.executemany(
                        f"DELETE FROM {blobs_table} "  # noqa: S608 - name comes from sqlite_master, not user input
                        "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                        keys,
                    )
                connection.executemany(
                    "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                    keys,
                )
                connection.commit()
            except sqlite3.Error:
                connection.rollback()
                return 0
        return len(stale)

    def _checkpoint_write_lock(self):
        """The saver's own lock when it has one, so writes stay serialized."""

        lock = getattr(self.checkpointer, "lock", None)
        if lock is not None and hasattr(lock, "__enter__"):
            return lock
        return self._checkpoint_lock

    def _blobs_table(self, connection: sqlite3.Connection) -> str | None:
        """Name of a per-checkpoint blob table, probed once. ``None`` if absent."""

        if self._checkpoint_blobs_table is not False:
            return self._checkpoint_blobs_table  # type: ignore[return-value]
        self._checkpoint_blobs_table = None
        try:
            names = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('blobs', 'checkpoint_blobs')"
                ).fetchall()
            }
            for candidate in ("blobs", "checkpoint_blobs"):
                if candidate not in names:
                    continue
                columns = {
                    str(row[1]) for row in connection.execute(f"PRAGMA table_info({candidate})").fetchall()
                }
                if {"thread_id", "checkpoint_ns", "checkpoint_id"} <= columns:
                    self._checkpoint_blobs_table = candidate
                    break
        except sqlite3.Error:
            self._checkpoint_blobs_table = None
        return self._checkpoint_blobs_table


def _checkpoint_keep(keep: int | None = None) -> int:
    if keep is not None:
        try:
            return max(1, int(keep))
        except (TypeError, ValueError):
            pass
    try:
        return max(1, int(os.getenv("LANGCODE_CHECKPOINT_KEEP", str(DEFAULT_CHECKPOINT_KEEP))))
    except (TypeError, ValueError):
        return DEFAULT_CHECKPOINT_KEEP
