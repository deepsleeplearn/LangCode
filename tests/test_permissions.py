from langcode_agent.runtime.permissions import (
    ApprovalMode,
    ToolCall,
    classify_shell_risk,
    permission_for_tool,
    remember_shell_permission,
)


def test_read_and_search_are_auto_allowed() -> None:
    assert permission_for_tool(ToolCall("read_file", {"path": "README.md"})) is ApprovalMode.ALLOW
    assert permission_for_tool(ToolCall("search", {"query": "TODO"})) is ApprovalMode.ALLOW
    assert permission_for_tool(ToolCall("ls", {"path": "."})) is ApprovalMode.ALLOW
    assert permission_for_tool(ToolCall("glob", {"pattern": "*.py"})) is ApprovalMode.ALLOW
    assert permission_for_tool(ToolCall("task_create", {"content": "规划"})) is ApprovalMode.ALLOW
    assert permission_for_tool(ToolCall("task_update", {"task_id": "1", "status": "completed"})) is ApprovalMode.ALLOW
    assert permission_for_tool(ToolCall("task_list", {})) is ApprovalMode.ALLOW
    assert permission_for_tool(ToolCall("web_search", {"query": "LangGraph docs"})) is ApprovalMode.ALLOW
    assert permission_for_tool(ToolCall("web_fetch", {"url": "https://example.com"})) is ApprovalMode.ALLOW
    assert permission_for_tool(ToolCall("write_todos", {"todos": []})) is ApprovalMode.DENY


def test_write_edit_require_approval_and_safe_shell_is_auto_allowed() -> None:
    assert permission_for_tool(ToolCall("write_file", {"path": "README.md", "content": "x"})) is ApprovalMode.ASK
    assert permission_for_tool(ToolCall("edit_file", {"path": "README.md", "old": "x", "new": "y"})) is ApprovalMode.ASK
    assert permission_for_tool(ToolCall("shell", {"command": "pytest -q"})) is ApprovalMode.ALLOW


def test_risky_shell_commands_still_require_approval() -> None:
    assert permission_for_tool(ToolCall("shell", {"command": "mkdir output"})) is ApprovalMode.ASK
    assert permission_for_tool(ToolCall("shell", {"command": "curl https://example.com"})) is ApprovalMode.ASK
    assert permission_for_tool(ToolCall("shell", {"command": "git push origin main"})) is ApprovalMode.ASK


def test_sandbox_shell_allows_local_commands_but_asks_for_network() -> None:
    assert permission_for_tool(ToolCall("sandbox_shell", {"command": "pytest -q"})) is ApprovalMode.ALLOW
    assert permission_for_tool(ToolCall("sandbox_shell", {"command": "curl https://example.com"})) is ApprovalMode.ASK
    assert permission_for_tool(ToolCall("sandbox_shell", {"command": "python3 -c 'print(1)'"})) is ApprovalMode.ASK


def test_shell_interpreter_eval_requires_approval() -> None:
    assert permission_for_tool(ToolCall("shell", {"command": "python3 -c 'print(1)'"})) is ApprovalMode.ASK
    assert permission_for_tool(ToolCall("shell", {"command": "node -e 'console.log(1)'"})) is ApprovalMode.ASK


def test_shell_permission_rules_can_be_remembered(tmp_path) -> None:
    remember_shell_permission(tmp_path, "mkdir output", "allow")

    assert (
        permission_for_tool(
            ToolCall("shell", {"command": "mkdir output"}),
            workspace_root=tmp_path,
        )
        is ApprovalMode.ALLOW
    )


def test_dangerous_shell_commands_are_flagged() -> None:
    risk = classify_shell_risk("rm -rf /tmp/langcode-demo")

    assert risk.dangerous is True
    assert "rm -rf" in risk.reason


def test_rm_recursive_force_flag_permutations_are_flagged() -> None:
    for command in ["rm -r -f /tmp/x", "rm -Rf /tmp/x", "rm -fR /tmp/x"]:
        risk = classify_shell_risk(command)

        assert risk.dangerous is True
