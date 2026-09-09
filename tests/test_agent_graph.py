from pathlib import Path

from langcode_agent.runtime.agent import CodeAgent
from langcode_agent.runtime.permissions import ToolCall


def test_graph_interrupts_before_write_and_resumes_after_accept(tmp_path: Path) -> None:
    agent = CodeAgent(workspace_root=tmp_path)
    thread_id = "approval-test"

    interrupted = agent.request_tool(
        ToolCall("write_file", {"path": "README.md", "content": "approved"}),
        thread_id=thread_id,
    )

    assert "__interrupt__" in interrupted
    assert interrupted["__interrupt__"][0].value["tool_name"] == "write_file"

    resumed = agent.resume(thread_id, {"type": "accept"})

    assert resumed["tool_result"]["ok"] is True
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "approved"


def test_graph_reject_does_not_run_tool(tmp_path: Path) -> None:
    agent = CodeAgent(workspace_root=tmp_path)
    thread_id = "reject-test"

    agent.request_tool(
        ToolCall("write_file", {"path": "README.md", "content": "rejected"}),
        thread_id=thread_id,
    )
    resumed = agent.resume(thread_id, {"type": "reject", "reason": "not now"})

    assert resumed["tool_result"]["ok"] is False
    assert not (tmp_path / "README.md").exists()


def test_graph_feedback_does_not_run_tool(tmp_path: Path) -> None:
    agent = CodeAgent(workspace_root=tmp_path)
    thread_id = "feedback-test"

    agent.request_tool(
        ToolCall("write_file", {"path": "README.md", "content": "needs changes"}),
        thread_id=thread_id,
    )
    resumed = agent.resume(
        thread_id,
        {"type": "feedback", "feedback": "Use a shorter README first."},
    )

    assert resumed["tool_result"]["ok"] is False
    assert resumed["tool_result"]["feedback"] == "Use a shorter README first."
    assert not (tmp_path / "README.md").exists()


def test_graph_malformed_approval_does_not_run_tool(tmp_path: Path) -> None:
    for index, approval in enumerate(
        [{"type": "approve"}, {"type": "ACCEPT"}, {"foo": "bar"}, {}, "accept"]
    ):
        agent = CodeAgent(workspace_root=tmp_path)
        thread_id = f"malformed-{index}"
        target = tmp_path / f"owned-{index}.txt"

        agent.request_tool(
            ToolCall("write_file", {"path": target.name, "content": "owned"}),
            thread_id=thread_id,
        )
        resumed = agent.resume(thread_id, approval)

        assert resumed["tool_result"]["ok"] is False
        assert not target.exists()


def test_graph_auto_allows_low_risk_shell_without_interrupt(tmp_path: Path) -> None:
    agent = CodeAgent(workspace_root=tmp_path)

    result = agent.request_tool(
        ToolCall("shell", {"command": "pwd", "timeout_seconds": 5}),
        thread_id="safe-shell",
    )

    assert "__interrupt__" not in result
    assert result["tool_result"]["ok"] is True
    assert result["tool_result"]["result"]["exit_code"] == 0


def test_graph_still_interrupts_for_risky_shell(tmp_path: Path) -> None:
    agent = CodeAgent(workspace_root=tmp_path)

    result = agent.request_tool(
        ToolCall("shell", {"command": "mkdir output", "timeout_seconds": 5}),
        thread_id="risky-shell",
    )

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["tool_name"] == "shell"
    assert "mutate" in payload["risk"]["reason"]


def test_graph_edit_response_changes_tool_arguments(tmp_path: Path) -> None:
    agent = CodeAgent(workspace_root=tmp_path)
    thread_id = "edit-test"

    agent.request_tool(
        ToolCall("write_file", {"path": "README.md", "content": "original"}),
        thread_id=thread_id,
    )
    resumed = agent.resume(
        thread_id,
        {
            "type": "edit",
            "tool_input": {"path": "README.md", "content": "edited"},
        },
    )

    assert resumed["tool_result"]["ok"] is True
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "edited"


def test_graph_interrupts_before_read_workspace_escape_and_resumes_after_accept(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside content", encoding="utf-8")
    agent = CodeAgent(workspace_root=workspace)
    thread_id = "read-escape"

    interrupted = agent.request_tool(
        ToolCall("read_file", {"path": str(outside)}),
        thread_id=thread_id,
    )

    assert "__interrupt__" in interrupted
    payload = interrupted["__interrupt__"][0].value
    assert payload["tool_name"] == "read_file"
    assert payload["risk"]["dangerous"] is True
    assert "escapes workspace" in payload["risk"]["reason"]

    resumed = agent.resume(thread_id, {"type": "accept"})

    assert resumed["tool_result"]["ok"] is True
    content = resumed["tool_result"]["content"]
    assert content.startswith("1: outside content")
    assert content.rstrip().endswith("[共 1 行，已显示 1-1 行]")


def test_graph_rejects_workspace_escape_without_running_tool(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    agent = CodeAgent(workspace_root=workspace)
    thread_id = "write-escape-reject"

    agent.request_tool(
        ToolCall("write_file", {"path": str(outside), "content": "nope"}),
        thread_id=thread_id,
    )
    resumed = agent.resume(thread_id, {"type": "reject", "reason": "stay in workspace"})

    assert resumed["tool_result"]["ok"] is False
    assert not outside.exists()


def test_graph_edit_cannot_reuse_escape_approval_for_different_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    reviewed = tmp_path / "reviewed.txt"
    different = tmp_path / "different.txt"
    agent = CodeAgent(workspace_root=workspace)
    thread_id = "write-escape-edit-different"

    agent.request_tool(
        ToolCall("write_file", {"path": str(reviewed), "content": "reviewed"}),
        thread_id=thread_id,
    )
    resumed = agent.resume(
        thread_id,
        {"type": "edit", "tool_input": {"path": str(different), "content": "different"}},
    )

    assert resumed["tool_result"]["ok"] is False
    assert "重新提交审批" in resumed["tool_result"]["error"]
    assert not reviewed.exists()
    assert not different.exists()


def test_graph_allows_shell_workspace_escape_after_accept(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    agent = CodeAgent(workspace_root=workspace)
    thread_id = "shell-escape"

    interrupted = agent.request_tool(
        ToolCall("shell", {"command": f"printf escaped > {outside}", "timeout_seconds": 5}),
        thread_id=thread_id,
    )

    assert "__interrupt__" in interrupted
    assert interrupted["__interrupt__"][0].value["workspace_escape"]["dangerous"] is True

    resumed = agent.resume(thread_id, {"type": "accept"})

    assert resumed["tool_result"]["ok"] is True
    assert resumed["tool_result"]["result"]["exit_code"] == 0
    assert outside.read_text(encoding="utf-8") == "escaped"


def test_sqlite_checkpoint_allows_resume_from_new_agent_instance(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / ".langcode" / "checkpoints.sqlite"
    thread_id = "persistent-resume"

    first_agent = CodeAgent(workspace_root=tmp_path, checkpoint_path=checkpoint_path)
    interrupted = first_agent.request_tool(
        ToolCall("write_file", {"path": "README.md", "content": "persisted"}),
        thread_id=thread_id,
    )

    assert "__interrupt__" in interrupted

    second_agent = CodeAgent(workspace_root=tmp_path, checkpoint_path=checkpoint_path)
    resumed = second_agent.resume(thread_id, {"type": "accept"})

    assert resumed["tool_result"]["ok"] is True
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "persisted"


def test_checkpoint_resume_rejects_workspace_mismatch(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    checkpoint_path = root_a / ".langcode" / "checkpoints.sqlite"
    thread_id = "cross-root"

    CodeAgent(workspace_root=root_a, checkpoint_path=checkpoint_path).request_tool(
        ToolCall("write_file", {"path": "where.txt", "content": "old-root"}),
        thread_id=thread_id,
    )
    resumed = CodeAgent(workspace_root=root_b, checkpoint_path=checkpoint_path).resume(
        thread_id,
        {"type": "accept"},
    )

    assert resumed["tool_result"]["ok"] is False
    assert "工作区不匹配" in resumed["tool_result"]["error"]
    assert not (root_a / "where.txt").exists()
    assert not (root_b / "where.txt").exists()


def test_checkpoint_connection_uses_wal_and_busy_timeout(tmp_path: Path) -> None:
    agent = CodeAgent(workspace_root=tmp_path, checkpoint_path=tmp_path / "cp.sqlite")
    try:
        connection = agent._checkpoint_connection
        assert connection is not None
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
    finally:
        agent.close()


def test_prune_thread_keeps_newest_checkpoints_and_their_writes(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    agent = CodeAgent(workspace_root=tmp_path, checkpoint_path=tmp_path / "cp.sqlite")
    try:
        for _ in range(6):
            agent.request_tool(ToolCall("read_file", {"path": "a.txt"}), thread_id="keep-me")
        agent.request_tool(ToolCall("read_file", {"path": "a.txt"}), thread_id="other")

        connection = agent._checkpoint_connection
        before = connection.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'keep-me'"
        ).fetchone()[0]
        other_before = connection.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'other'"
        ).fetchone()[0]
        assert before > 3

        removed = agent.prune_thread("keep-me", keep=3)

        assert removed == before - 3
        assert connection.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'keep-me'"
        ).fetchone()[0] == 3
        assert connection.execute(
            "SELECT COUNT(*) FROM writes w WHERE w.thread_id = 'keep-me' AND NOT EXISTS ("
            "  SELECT 1 FROM checkpoints c WHERE c.thread_id = w.thread_id"
            "  AND c.checkpoint_ns = w.checkpoint_ns AND c.checkpoint_id = w.checkpoint_id)"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'other'"
        ).fetchone()[0] == other_before

        resumed = agent.request_tool(ToolCall("read_file", {"path": "a.txt"}), thread_id="keep-me")
        assert resumed["tool_result"]["ok"] is True
    finally:
        agent.close()


def test_prune_thread_reads_keep_count_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LANGCODE_CHECKPOINT_KEEP", "2")
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    agent = CodeAgent(workspace_root=tmp_path, checkpoint_path=tmp_path / "cp.sqlite")
    try:
        for _ in range(4):
            agent.request_tool(ToolCall("read_file", {"path": "a.txt"}), thread_id="env-keep")

        agent.prune_thread("env-keep")

        assert agent._checkpoint_connection.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'env-keep'"
        ).fetchone()[0] == 2
    finally:
        agent.close()


def test_prune_thread_is_a_noop_for_the_in_memory_checkpointer(tmp_path: Path) -> None:
    agent = CodeAgent(workspace_root=tmp_path)

    assert agent.prune_thread("anything") == 0


def test_prune_thread_serializes_on_the_checkpointer_lock(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    agent = CodeAgent(workspace_root=tmp_path, checkpoint_path=tmp_path / "cp.sqlite")
    try:
        assert agent._checkpoint_write_lock() is agent.checkpointer.lock

        for _ in range(5):
            agent.request_tool(ToolCall("read_file", {"path": "a.txt"}), thread_id="locked")

        # The saver's lock is not reentrant: pruning must not already hold it.
        assert agent.prune_thread("locked", keep=2) > 0
    finally:
        agent.close()


def test_prune_thread_falls_back_to_its_own_lock(tmp_path: Path) -> None:
    agent = CodeAgent(workspace_root=tmp_path, checkpoint_path=tmp_path / "cp.sqlite")
    try:
        agent.checkpointer = object()  # a saver without a `lock` attribute

        assert agent._checkpoint_write_lock() is agent._checkpoint_lock
    finally:
        agent._checkpoint_connection.close()


def test_prune_thread_also_clears_a_blob_table_when_one_exists(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    agent = CodeAgent(workspace_root=tmp_path, checkpoint_path=tmp_path / "cp.sqlite")
    try:
        connection = agent._checkpoint_connection
        # langgraph-checkpoint-sqlite 2.0.10 has no blob table; simulate a
        # newer layout to prove the extra DELETE is wired up.
        connection.execute(
            "CREATE TABLE blobs (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, blob BLOB)"
        )
        for _ in range(5):
            agent.request_tool(ToolCall("read_file", {"path": "a.txt"}), thread_id="blobby")
        rows = connection.execute(
            "SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints WHERE thread_id = 'blobby'"
        ).fetchall()
        connection.executemany("INSERT INTO blobs VALUES (?, ?, ?, x'00')", rows)
        connection.commit()

        agent.prune_thread("blobby", keep=2)

        assert connection.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 2
    finally:
        agent.close()
