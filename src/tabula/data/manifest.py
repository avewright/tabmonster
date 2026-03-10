from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from tabula.data.catalog import KaggleDatasetEntry, get_dataset_entry


@dataclass(frozen=True)
class DatasetManifest:
    id: str
    title: str
    provider: str
    source_type: str
    external_ref: str
    source_url: str
    task_type: str | None
    target_column: str | None
    train_file: str | None
    notes: str


def sanitize_dataset_id(value: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")
    return normalized or "dataset"


def manifest_path(raw_dir: str | Path) -> Path:
    return Path(raw_dir) / "dataset_manifest.json"


def write_manifest(raw_dir: str | Path, manifest: DatasetManifest) -> Path:
    path = manifest_path(raw_dir)
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return path


def load_manifest(raw_dir: str | Path) -> DatasetManifest | None:
    path = manifest_path(raw_dir)
    if not path.exists():
        return None
    return DatasetManifest(**json.loads(path.read_text(encoding="utf-8")))


def kaggle_entry_to_manifest(entry: KaggleDatasetEntry) -> DatasetManifest:
    return DatasetManifest(
        id=entry.id,
        title=entry.title,
        provider="kaggle",
        source_type=entry.source_type,
        external_ref=entry.kaggle_slug,
        source_url=entry.kaggle_url,
        task_type=entry.task_type,
        target_column=entry.target_column,
        train_file=entry.train_file,
        notes=entry.notes,
    )


def resolve_manifest(dataset_id: str, raw_root: str | Path = "data/raw") -> DatasetManifest:
    raw_dir = Path(raw_root) / dataset_id
    manifest = load_manifest(raw_dir)
    if manifest is not None:
        return manifest
    return kaggle_entry_to_manifest(get_dataset_entry(dataset_id))
