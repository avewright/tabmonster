from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class KaggleDatasetEntry:
    id: str
    title: str
    source_type: str
    kaggle_slug: str
    task_type: str
    quality_tier: str
    recommended: bool
    est_rows: int
    target_column: str
    train_file: str
    notes: str

    @property
    def kaggle_url(self) -> str:
        if self.source_type == "competition":
            return f"https://www.kaggle.com/competitions/{self.kaggle_slug}"
        return f"https://www.kaggle.com/datasets/{self.kaggle_slug}"


def default_catalog_path() -> Path:
    return Path(__file__).resolve().parents[3] / "catalogs" / "kaggle_tabular.json"


def load_kaggle_catalog(path: str | Path | None = None) -> list[KaggleDatasetEntry]:
    catalog_path = Path(path) if path else default_catalog_path()
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    return [KaggleDatasetEntry(**entry) for entry in raw["datasets"]]


def get_dataset_entry(dataset_id: str, path: str | Path | None = None) -> KaggleDatasetEntry:
    for entry in load_kaggle_catalog(path):
        if entry.id == dataset_id:
            return entry
    raise KeyError(f"Unknown dataset id: {dataset_id}")


def filter_catalog(
    entries: list[KaggleDatasetEntry],
    quality_tier: str | None = None,
    task_type: str | None = None,
    recommended_only: bool = False,
) -> list[KaggleDatasetEntry]:
    filtered = entries
    if quality_tier:
        filtered = [entry for entry in filtered if entry.quality_tier == quality_tier]
    if task_type:
        filtered = [entry for entry in filtered if entry.task_type == task_type]
    if recommended_only:
        filtered = [entry for entry in filtered if entry.recommended]
    return filtered

