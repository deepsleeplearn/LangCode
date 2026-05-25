from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


TodoStatus = Literal["pending", "in_progress", "completed", "blocked", "cancelled"]
TASK_STATUSES = {"pending", "in_progress", "completed", "blocked", "cancelled"}


@dataclass(frozen=True)
class TodoItem:
    id: str
    content: str
    status: TodoStatus = "pending"

    def to_dict(self) -> dict:
        return {"id": self.id, "content": self.content, "status": self.status}


def normalize_todos(items: Any) -> list[dict]:
    if not isinstance(items, list):
        raise ValueError("任务清单必须是列表")

    normalized: list[dict] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 个任务项必须是对象")
        content = str(item.get("content") or item.get("task") or "").strip()
        if not content:
            raise ValueError(f"第 {index} 个任务项缺少内容")
        status = str(item.get("status") or "pending").strip()
        if status not in TASK_STATUSES:
            status = "pending"
        item_id = str(item.get("id") or index).strip()
        normalized.append(TodoItem(id=item_id, content=content, status=status).to_dict())
    return normalized


def summarize_todos(todos: list[dict]) -> str:
    if not todos:
        return "计划为空。"
    counts = {"pending": 0, "in_progress": 0, "completed": 0, "blocked": 0, "cancelled": 0}
    for item in todos:
        status = str(item.get("status") or "pending")
        if status in counts:
            counts[status] += 1
    return (
        f"计划已更新：{counts['completed']} 个完成，"
        f"{counts['in_progress']} 个进行中，{counts['pending']} 个待办，"
        f"{counts['blocked']} 个阻塞，{counts['cancelled']} 个取消。"
    )


def create_task(tasks: list[dict], content: str, *, status: str = "pending", task_id: str | None = None) -> dict:
    normalized = normalize_todos(tasks)
    task_content = str(content or "").strip()
    if not task_content:
        raise ValueError("任务内容不能为空")
    next_id = str(task_id or _next_task_id(normalized)).strip()
    if any(str(task.get("id")) == next_id for task in normalized):
        raise ValueError(f"任务 id 已存在：{next_id}")
    task = TodoItem(id=next_id, content=task_content, status=_normalize_status(status)).to_dict()
    updated = _append_with_single_active(normalized, task)
    return {"ok": True, "task": task, "todos": updated, "summary": summarize_todos(updated)}


def update_task(
    tasks: list[dict],
    task_id: str,
    *,
    content: str | None = None,
    status: str | None = None,
) -> dict:
    normalized = normalize_todos(tasks)
    target_id = str(task_id or "").strip()
    if not target_id:
        raise ValueError("task_id 不能为空")
    updated = []
    changed_task = None
    for task in normalized:
        next_task = dict(task)
        if str(task.get("id")) == target_id:
            if content is not None and str(content).strip():
                next_task["content"] = str(content).strip()
            if status is not None:
                next_task["status"] = _normalize_status(status)
            changed_task = next_task
        updated.append(next_task)
    if changed_task is None:
        raise ValueError(f"未找到任务：{target_id}")
    updated = _enforce_single_active(updated, active_id=target_id if changed_task["status"] == "in_progress" else None)
    return {"ok": True, "task": changed_task, "todos": updated, "summary": summarize_todos(updated)}


def list_tasks(tasks: list[dict], *, status: str | None = None) -> dict:
    normalized = normalize_todos(tasks)
    if status:
        expected = _normalize_status(status)
        normalized = [task for task in normalized if task.get("status") == expected]
    return {"ok": True, "tasks": normalized, "todos": normalized, "summary": summarize_todos(normalized)}


def get_task(tasks: list[dict], task_id: str) -> dict:
    normalized = normalize_todos(tasks)
    target_id = str(task_id or "").strip()
    for task in normalized:
        if str(task.get("id")) == target_id:
            return {"ok": True, "task": task, "todos": normalized}
    raise ValueError(f"未找到任务：{target_id}")


def cancel_task(tasks: list[dict], task_id: str, *, reason: str | None = None) -> dict:
    result = update_task(tasks, task_id, status="cancelled")
    if reason:
        result["reason"] = str(reason)
    return result


def deepagents_capability_summary() -> str:
    return (
        "LangCode 已按 DeepAgents 工作流思路启用：任务清单规划、工作区后端、"
        "角色化子 Agent、工具输出压缩、技能和记忆加载、声明式审批策略、沙箱命令。"
    )


def _normalize_status(status: str | None) -> TodoStatus:
    value = str(status or "pending").strip()
    if value in TASK_STATUSES:
        return value  # type: ignore[return-value]
    return "pending"


def _next_task_id(tasks: list[dict]) -> int:
    numeric_ids = []
    for task in tasks:
        try:
            numeric_ids.append(int(str(task.get("id") or "")))
        except ValueError:
            continue
    return (max(numeric_ids) + 1) if numeric_ids else len(tasks) + 1


def _append_with_single_active(tasks: list[dict], task: dict) -> list[dict]:
    if task.get("status") != "in_progress":
        return [*tasks, task]
    updated = []
    for existing in tasks:
        next_task = dict(existing)
        if next_task.get("status") == "in_progress":
            next_task["status"] = "pending"
        updated.append(next_task)
    return [*updated, task]


def _enforce_single_active(tasks: list[dict], *, active_id: str | None) -> list[dict]:
    if active_id is None:
        return [dict(task) for task in tasks]
    updated = []
    for task in tasks:
        next_task = dict(task)
        if str(next_task.get("id")) != active_id and next_task.get("status") == "in_progress":
            next_task["status"] = "pending"
        updated.append(next_task)
    return updated
