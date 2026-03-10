from __future__ import annotations

from pathlib import Path
import shutil
import time


def directory_size_bytes(path: str | Path) -> int:
    base = Path(path)
    if not base.exists():
        return 0
    total = 0
    for child in base.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def trim_cache_to_budget(cache_dir: str | Path, max_gb: float) -> dict[str, object]:
    base = Path(cache_dir)
    if not base.exists():
        return {"cache_dir": str(base), "exists": False, "bytes_before": 0, "bytes_after": 0, "removed": []}
    max_bytes = int(max_gb * 1024 * 1024 * 1024)
    entries = [child for child in base.iterdir() if child.is_dir()]
    now = time.time()
    scored_entries: list[tuple[float, Path, int]] = []
    for entry in entries:
        size = directory_size_bytes(entry)
        age_seconds = max(now - entry.stat().st_mtime, 1.0)
        score = age_seconds * max(size, 1)
        scored_entries.append((score, entry, size))
    scored_entries.sort(reverse=True, key=lambda item: item[0])
    removed: list[str] = []
    before = directory_size_bytes(base)
    current = before
    for _, entry, entry_size in scored_entries:
        if current <= max_bytes:
            break
        shutil.rmtree(entry, ignore_errors=True)
        removed.append(str(entry))
        current -= entry_size
    after = directory_size_bytes(base)
    return {
        "cache_dir": str(base),
        "exists": True,
        "bytes_before": before,
        "bytes_after": after,
        "removed": removed,
    }
