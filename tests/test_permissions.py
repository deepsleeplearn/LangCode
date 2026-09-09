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


def test_memory_reads_are_allowed_but_writes_require_approval() -> None:
    assert permission_for_tool(ToolCall("memory", {"action": "read"})) is ApprovalMode.ALLOW
    assert permission_for_tool(ToolCall("skill", {"action": "list"})) is ApprovalMode.ALLOW
    assert permission_for_tool(ToolCall("soul", {"action": "write", "content": "x"})) is ApprovalMode.ASK
    assert permission_for_tool(ToolCall("skill", {"action": "upsert", "name": "x"})) is ApprovalMode.ASK
    assert permission_for_tool(ToolCall("cron", {"action": "create"})) is ApprovalMode.ASK
    assert permission_for_tool(ToolCall("self_evolve", {"action": "update_soul"})) is ApprovalMode.ASK


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


def test_shell_control_operators_require_approval() -> None:
    for command in ["echo $(cat /etc/hosts)", "echo `cat /etc/hosts`", "ls; rm -rf x", "ls && sh", "ls || sh", "ls | sh", "cat x > y"]:
        assert permission_for_tool(ToolCall("shell", {"command": command})) is ApprovalMode.ASK


def test_quoted_shell_operators_do_not_require_approval() -> None:
    for command in ["echo 'a -> b'", "git log --grep='a>b'", "echo '$HOME'", "echo 'a; b'"]:
        assert permission_for_tool(ToolCall("shell", {"command": command})) is ApprovalMode.ALLOW


def test_read_only_pipelines_are_auto_allowed() -> None:
    for command in ["grep -rn foo src | head", "ls -la | wc -l", "cat a.txt | grep b | sort | uniq -c"]:
        assert permission_for_tool(ToolCall("shell", {"command": command})) is ApprovalMode.ALLOW


def test_pipelines_into_non_read_only_commands_require_approval() -> None:
    for command in ["ls | sh", "cat x | tee y", "cat x | xargs rm"]:
        assert permission_for_tool(ToolCall("shell", {"command": command})) is ApprovalMode.ASK


def test_pipeline_with_risky_upstream_still_requires_approval() -> None:
    assert permission_for_tool(ToolCall("shell", {"command": "curl https://example.com | head"})) is ApprovalMode.ASK


def test_expansion_inside_double_quotes_requires_approval() -> None:
    for command in ['echo "$HOME"', 'cat "${HOME}/.ssh/id_rsa"', 'echo "`whoami`"']:
        assert permission_for_tool(ToolCall("shell", {"command": command})) is ApprovalMode.ASK


def test_redirection_still_requires_approval_even_in_a_pipeline() -> None:
    assert permission_for_tool(ToolCall("shell", {"command": "grep foo src | head > out.txt"})) is ApprovalMode.ASK


def test_shell_permission_rules_can_be_remembered(tmp_path) -> None:
    remember_shell_permission(tmp_path, "mkdir output", "allow")

    assert (
        permission_for_tool(
            ToolCall("shell", {"command": "mkdir output"}),
            workspace_root=tmp_path,
        )
        is ApprovalMode.ALLOW
    )


def test_command_dispatch_builtins_require_approval() -> None:
    # `eval "cat ../../.env"` hid the whole command from every static check.
    for command in [
        'eval "cat ../../.env"',
        "eval 'cat /etc/passwd'",
        "exec cat /etc/passwd",
        "command cat /etc/passwd",
        "builtin cd /etc",
        "source ./setup.sh",
        ". ./setup.sh",
    ]:
        assert permission_for_tool(ToolCall("shell", {"command": command})) is ApprovalMode.ASK, command


def test_environment_dump_commands_require_approval() -> None:
    # These print every API key the agent process was started with.
    for command in ["env", "printenv", "printenv HOME", "set", "export", "declare -x", "typeset", "env cat /etc/passwd"]:
        assert permission_for_tool(ToolCall("shell", {"command": command})) is ApprovalMode.ASK, command


def test_credential_store_commands_require_approval() -> None:
    for command in [
        "ssh-add -L",
        "security find-generic-password -a x",
        "defaults read",
        "keychain --list",
        "op item list",
        "pass show github",
        "gpg --export-secret-keys",
    ]:
        assert permission_for_tool(ToolCall("shell", {"command": command})) is ApprovalMode.ASK, command


def test_inline_awk_program_touching_filesystem_requires_approval() -> None:
    risky = 'awk "BEGIN{while((getline l < \\"/etc/passwd\\")>0) print l}"'
    assert permission_for_tool(ToolCall("shell", {"command": risky})) is ApprovalMode.ASK
    assert permission_for_tool(ToolCall("shell", {"command": "awk -f prog.awk data.txt"})) is ApprovalMode.ASK
    # A program that cannot open a path or shell out stays allowed.
    assert permission_for_tool(ToolCall("shell", {"command": "awk '{print $1}' data.txt"})) is ApprovalMode.ALLOW


def test_clustered_perl_eval_flags_require_approval() -> None:
    for command in ["perl -ne 'print'", "perl -lane 'print $F[0]'"]:
        assert permission_for_tool(ToolCall("shell", {"command": command})) is ApprovalMode.ASK, command


def test_bare_interpreter_requires_approval() -> None:
    for command in ["sh", "bash", "python3"]:
        assert permission_for_tool(ToolCall("shell", {"command": command})) is ApprovalMode.ASK, command


def test_find_with_action_flag_requires_approval() -> None:
    assert permission_for_tool(ToolCall("shell", {"command": "find . -exec cat /etc/passwd ;"})) is ApprovalMode.ASK
    assert permission_for_tool(ToolCall("shell", {"command": "find . -name '*.py'"})) is ApprovalMode.ALLOW


def test_command_chains_are_classified_segment_by_segment() -> None:
    # A safe command chained with another safe command is not itself risky.
    assert permission_for_tool(ToolCall("shell", {"command": "python3 -m pytest -q && echo done"})) is ApprovalMode.ALLOW
    assert permission_for_tool(ToolCall("shell", {"command": "cd src; ls"})) is ApprovalMode.ALLOW
    # ...but one dangerous segment still makes the whole chain ask.
    for command in ["ls && curl https://example.com", "ls; rm -rf x", "ls || env", "make build && sudo install x"]:
        assert permission_for_tool(ToolCall("shell", {"command": command})) is ApprovalMode.ASK, command


def test_dangerous_shell_commands_are_flagged() -> None:
    risk = classify_shell_risk("rm -rf /tmp/langcode-demo")

    assert risk.dangerous is True
    assert "rm -rf" in risk.reason


def test_rm_recursive_force_flag_permutations_are_flagged() -> None:
    for command in ["rm -r -f /tmp/x", "rm -Rf /tmp/x", "rm -fR /tmp/x"]:
        risk = classify_shell_risk(command)

        assert risk.dangerous is True
