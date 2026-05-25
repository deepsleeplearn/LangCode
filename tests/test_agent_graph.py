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
    assert resumed["tool_result"]["content"] == "outside content"


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
