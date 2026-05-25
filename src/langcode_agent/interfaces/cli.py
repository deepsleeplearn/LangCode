from pathlib import Path
import argparse
import json
import os

from ..runtime.agent import CodeAgent
from ..runtime.chat import (
    ChatSession,
    build_openai_model,
    default_system_prompt,
    messages_from_json,
    model_settings_from_env,
)
from ..core.config import load_env_files
from ..core.context_management import make_json_safe
from ..runtime.permissions import ToolCall


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LangGraph code agent CLI")
    parser.add_argument("--workspace", default=".", help="Workspace root")
    parser.add_argument("--session", default="default", help="Session/thread id")
    parser.add_argument("--raw-tools", action="store_true", help="Accept only JSON tool calls")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).expanduser().resolve()
    load_env_files(Path.cwd(), workspace)
    state_dir = workspace / ".langcode"
    state_dir.mkdir(parents=True, exist_ok=True)
    agent = CodeAgent(workspace, checkpoint_path=state_dir / "checkpoints.sqlite")

    print(f"langcode-agent session={args.session} workspace={workspace}")
    if args.raw_tools:
        print("Raw tool mode. Enter JSON tool calls, or 'quit'.")
        return _raw_tool_loop(agent, args.session)

    settings = model_settings_from_env()
    if not settings.api_key:
        print(
            f"No API key configured for provider={settings.provider}. "
            "Natural-language chat may fail; use :tool {...} for local tool testing."
        )
    else:
        print(f"model provider={settings.provider} model={settings.model} base_url={settings.base_url or 'default'}")

    history_path = state_dir / "sessions" / f"{args.session}.json"
    session: ChatSession | None = None
    print("Enter a request, ':tool {json}' for direct tool calls, or 'quit'.")

    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            return 0
        if raw in {"quit", "exit"}:
            return 0
        if not raw:
            continue

        try:
            if raw.startswith(":tool "):
                result = _run_raw_tool(agent, args.session, raw.removeprefix(":tool ").strip())
                print(json.dumps(make_json_safe(result), ensure_ascii=False, indent=2))
                continue

            if not settings.api_key:
                print(
                    f"No API key configured for provider={settings.provider}; "
                    "set ZHIPU_API_KEY or use :tool {...} / --raw-tools for local tool testing."
                )
                continue

            if session is None:
                history = _load_history(history_path)
                session = ChatSession(
                    agent=agent,
                    model=build_openai_model(),
                    approval_callback=_approval_callback,
                    thread_id=args.session,
                    system_prompt=default_system_prompt(str(workspace)),
                    history=history,
                )

            response = session.send(raw)
            _save_history(history_path, session.export_history())
            print(response)
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}")


def _raw_tool_loop(agent: CodeAgent, session_id: str) -> int:
    print("Example: {\"name\":\"read_file\",\"args\":{\"path\":\"README.md\"}}")
    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            return 0
        if raw in {"quit", "exit"}:
            return 0
        if not raw:
            continue
        try:
            print(json.dumps(make_json_safe(_run_raw_tool(agent, session_id, raw)), ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}")


def _run_raw_tool(agent: CodeAgent, session_id: str, raw_json: str) -> dict:
    payload = json.loads(raw_json)
    result = agent.request_tool(ToolCall.from_dict(payload), thread_id=session_id)
    if "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value
        print(json.dumps(make_json_safe(interrupt_value), ensure_ascii=False, indent=2))
        approval = _approval_callback(interrupt_value)
        result = agent.resume(session_id, approval)
    return result.get("tool_result", result)


def _approval_callback(_payload: dict | None = None) -> dict:
    print("approval: accept | reject | feedback | edit")
    action = input("approval> ").strip()
    if action == "accept":
        return {"type": "accept"}
    if action == "reject":
        return {"type": "reject", "reason": input("reason> ").strip()}
    if action == "feedback":
        return {"type": "feedback", "feedback": input("feedback> ").strip()}
    if action == "edit":
        edited = input("tool_input json> ").strip()
        return {"type": "edit", "tool_input": json.loads(edited)}
    return {"type": "reject", "reason": f"Unknown approval action: {action}"}


def _load_history(path: Path):
    if not path.exists():
        return []
    return messages_from_json(json.loads(path.read_text(encoding="utf-8")))


def _save_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
