from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import threading
from uuid import uuid4


CRON_DIR = ".langcode/cron"
JOBS_FILE = "jobs.json"
_LOCK = threading.RLock()


def cron_tool(workspace_root: str | Path, action: str, tool_input: dict) -> dict:
    root = Path(workspace_root).expanduser().resolve()
    normalized_action = str(action or "list").strip().lower()
    if normalized_action in {"list", "status"}:
        return {"ok": True, "jobs": _load_jobs(root)}
    if normalized_action in {"create", "add"}:
        return _create_job(root, tool_input)
    if normalized_action in {"update", "edit"}:
        return _update_job(root, tool_input)
    if normalized_action in {"delete", "remove"}:
        return _delete_job(root, str(tool_input.get("job_id") or tool_input.get("id") or ""))
    if normalized_action in {"pause", "resume"}:
        return _set_status(root, str(tool_input.get("job_id") or tool_input.get("id") or ""), normalized_action)
    if normalized_action in {"due", "run_due"}:
        return _due_jobs(root)
    if normalized_action == "run":
        return _mark_run(root, str(tool_input.get("job_id") or tool_input.get("id") or ""), str(tool_input.get("result") or ""))
    return {"ok": False, "error": f"未知 cron 操作：{action}"}


def _create_job(root: Path, payload: dict) -> dict:
    name = str(payload.get("name") or "").strip()
    prompt = str(payload.get("prompt") or "").strip()
    schedule = str(payload.get("schedule") or payload.get("rrule") or "").strip()
    if not name:
        return {"ok": False, "error": "创建定时任务必须提供 name"}
    if not prompt:
        return {"ok": False, "error": "创建定时任务必须提供 prompt"}
    if not schedule:
        return {"ok": False, "error": "创建定时任务必须提供 schedule，例如 every 60 minutes 或 daily 09:00"}
    now = _now()
    job = {
        "id": str(payload.get("job_id") or payload.get("id") or uuid4().hex[:12]),
        "name": name,
        "prompt": prompt,
        "schedule": schedule,
        "skills": _string_list(payload.get("skills")),
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "last_run_at": "",
        "next_run_at": _next_run(schedule, now),
        "last_result": "",
    }
    with _LOCK:
        jobs = _load_jobs(root)
        if any(item["id"] == job["id"] for item in jobs):
            return {"ok": False, "error": f"任务已存在：{job['id']}"}
        jobs.append(job)
        _save_jobs(root, jobs)
    return {"ok": True, "job": job}


def _update_job(root: Path, payload: dict) -> dict:
    job_id = str(payload.get("job_id") or payload.get("id") or "").strip()
    if not job_id:
        return {"ok": False, "error": "更新定时任务必须提供 job_id"}
    with _LOCK:
        jobs = _load_jobs(root)
        for job in jobs:
            if job["id"] != job_id:
                continue
            for key in ("name", "prompt", "schedule", "status"):
                if payload.get(key) is not None:
                    job[key] = str(payload[key])
            if payload.get("skills") is not None:
                job["skills"] = _string_list(payload.get("skills"))
            job["updated_at"] = _now()
            if payload.get("schedule") is not None:
                job["next_run_at"] = _next_run(str(job["schedule"]), job["updated_at"])
            _save_jobs(root, jobs)
            return {"ok": True, "job": job}
    return {"ok": False, "error": f"未找到定时任务：{job_id}"}


def _delete_job(root: Path, job_id: str) -> dict:
    if not job_id:
        return {"ok": False, "error": "删除定时任务必须提供 job_id"}
    with _LOCK:
        jobs = _load_jobs(root)
        next_jobs = [job for job in jobs if job["id"] != job_id]
        if len(next_jobs) == len(jobs):
            return {"ok": False, "error": f"未找到定时任务：{job_id}"}
        _save_jobs(root, next_jobs)
    return {"ok": True, "job_id": job_id, "message": "定时任务已删除"}


def _set_status(root: Path, job_id: str, action: str) -> dict:
    return _update_job(root, {"job_id": job_id, "status": "paused" if action == "pause" else "active"})


def _due_jobs(root: Path) -> dict:
    now = datetime.now(timezone.utc)
    due = []
    for job in _load_jobs(root):
        if job.get("status") != "active":
            continue
        next_run = _parse_time(str(job.get("next_run_at") or ""))
        if next_run is None or next_run <= now:
            due.append(job)
    return {"ok": True, "jobs": due}


def _mark_run(root: Path, job_id: str, result: str) -> dict:
    if not job_id:
        return {"ok": False, "error": "运行定时任务必须提供 job_id"}
    with _LOCK:
        jobs = _load_jobs(root)
        for job in jobs:
            if job["id"] != job_id:
                continue
            now = _now()
            job["last_run_at"] = now
            job["last_result"] = result[:2000]
            job["next_run_at"] = _next_run(str(job.get("schedule") or ""), now)
            job["updated_at"] = now
            _save_jobs(root, jobs)
            return {"ok": True, "job": job}
    return {"ok": False, "error": f"未找到定时任务：{job_id}"}


def _jobs_path(root: Path) -> Path:
    path = root / CRON_DIR / JOBS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_jobs(root: Path) -> list[dict]:
    path = _jobs_path(root)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _save_jobs(root: Path, jobs: list[dict]) -> None:
    path = _jobs_path(root)
    path.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")


def _next_run(schedule: str, base_iso: str) -> str:
    base = _parse_time(base_iso) or datetime.now(timezone.utc)
    value = schedule.strip().lower()
    match = re.fullmatch(r"every\s+(\d+)\s+(minute|minutes|hour|hours|day|days)", value)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        if unit.startswith("minute"):
            return (base + timedelta(minutes=amount)).isoformat()
        if unit.startswith("hour"):
            return (base + timedelta(hours=amount)).isoformat()
        return (base + timedelta(days=amount)).isoformat()
    match = re.fullmatch(r"daily\s+(\d{1,2}):(\d{2})", value)
    if match:
        hour = max(0, min(int(match.group(1)), 23))
        minute = max(0, min(int(match.group(2)), 59))
        candidate = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= base:
            candidate += timedelta(days=1)
        return candidate.isoformat()
    return (base + timedelta(hours=24)).isoformat()


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
