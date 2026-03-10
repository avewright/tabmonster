from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def registry_path(root: str | Path = "artifacts") -> Path:
    return Path(root) / "stream_queue_registry.json"


def dashboard_path(root: str | Path = "artifacts") -> Path:
    return Path(root) / "stream_queue_dashboard.json"


def dashboard_csv_path(root: str | Path = "artifacts") -> Path:
    return Path(root) / "stream_queue_dashboard.csv"


def alerts_path(root: str | Path = "artifacts") -> Path:
    return Path(root) / "stream_queue_alerts.txt"


def load_registry(root: str | Path = "artifacts") -> dict[str, object]:
    path = registry_path(root)
    if not path.exists():
        return {"jobs": {}, "updated_at_utc": _utc_now()}
    return json.loads(path.read_text(encoding="utf-8"))


def _refresh_dashboard(payload: dict[str, object], root: str | Path = "artifacts") -> Path:
    jobs = dict(payload.get("jobs", {}))
    by_status: dict[str, int] = {}
    for item in jobs.values():
        status = str(dict(item).get("status", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
    dashboard = {
        "updated_at_utc": payload.get("updated_at_utc", _utc_now()),
        "job_count": len(jobs),
        "by_status": by_status,
        "jobs": sorted(
            [
                {
                    "experiment_name": name,
                    "status": dict(item).get("status"),
                    "current_step": dict(item).get("current_step"),
                    "target_max_steps": dict(item).get("target_max_steps"),
                    "failure_count": dict(item).get("failure_count", 0),
                    "heartbeat_at_utc": dict(item).get("heartbeat_at_utc"),
                    "repo_id": dict(item).get("repo_id"),
                }
                for name, item in jobs.items()
            ],
            key=lambda item: (str(item["status"]), str(item["experiment_name"])),
        ),
    }
    path = dashboard_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    csv_lines = ["experiment_name,status,current_step,target_max_steps,failure_count,heartbeat_at_utc,repo_id"]
    for item in dashboard["jobs"]:
        csv_lines.append(
            ",".join(
                [
                    str(item["experiment_name"]),
                    str(item["status"]),
                    str(item["current_step"]),
                    str(item["target_max_steps"]),
                    str(item["failure_count"]),
                    str(item["heartbeat_at_utc"]),
                    str(item["repo_id"]),
                ]
            )
        )
    dashboard_csv_path(root).write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    alert_lines = [
        f"{item['experiment_name']}: status={item['status']} failures={item['failure_count']} repo={item['repo_id']}"
        for item in dashboard["jobs"]
        if str(item["status"]) in {"failed", "stale", "orphaned"}
    ]
    alerts_path(root).write_text("\n".join(alert_lines) + ("\n" if alert_lines else ""), encoding="utf-8")
    return path


def refresh_dashboard(root: str | Path = "artifacts") -> Path:
    return _refresh_dashboard(load_registry(root), root)


def reconcile_registry(root: str | Path = "artifacts", *, stale_after_seconds: int = 300) -> Path:
    payload = load_registry(root)
    jobs = dict(payload.get("jobs", {}))
    now = datetime.now(timezone.utc)
    for name, item in jobs.items():
        job = dict(item)
        status = str(job.get("status", "unknown"))
        heartbeat_raw = job.get("heartbeat_at_utc")
        heartbeat = datetime.fromisoformat(heartbeat_raw) if heartbeat_raw else None
        if status == "running" and heartbeat is None:
            job["status"] = "orphaned"
            job["last_error"] = job.get("last_error") or "Missing heartbeat for running job."
        elif status == "running" and heartbeat is not None:
            if (now - heartbeat).total_seconds() > stale_after_seconds:
                job["status"] = "stale"
        jobs[name] = job
    payload["jobs"] = jobs
    payload["updated_at_utc"] = _utc_now()
    path = registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _refresh_dashboard(payload, root)
    return path


def update_registry_job(
    experiment_name: str,
    *,
    status: str,
    prepared_dir: str,
    repo_id: str,
    config_name: str | None,
    split: str,
    target_max_steps: int,
    current_step: int,
    heartbeat_at_utc: str | None = None,
    failure_count: int | None = None,
    last_error: str | None = None,
    cooldown_until_utc: str | None = None,
    root: str | Path = "artifacts",
) -> Path:
    payload = load_registry(root)
    jobs = dict(payload.get("jobs", {}))
    now = _utc_now()
    existing = dict(jobs.get(experiment_name, {}))
    jobs[experiment_name] = {
        "experiment_name": experiment_name,
        "status": status,
        "prepared_dir": prepared_dir,
        "repo_id": repo_id,
        "config_name": config_name,
        "split": split,
        "target_max_steps": target_max_steps,
        "current_step": current_step,
        "heartbeat_at_utc": heartbeat_at_utc or now,
        "failure_count": failure_count if failure_count is not None else existing.get("failure_count", 0),
        "last_error": last_error,
        "cooldown_until_utc": cooldown_until_utc,
        "updated_at_utc": now,
        "created_at_utc": existing.get("created_at_utc", now),
    }
    payload["jobs"] = jobs
    payload["updated_at_utc"] = now
    path = registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _refresh_dashboard(payload, root)
    return path
