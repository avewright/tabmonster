from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import os
from pathlib import Path

import pandas as pd

from tabula.data.env import load_repo_env_file
from tabula.data.inspection import infer_huggingface_target_metadata, inspect_supervised_frame
from tabula.data.manifest import DatasetManifest, sanitize_dataset_id, write_manifest


@dataclass(frozen=True)
class HuggingFaceDatasetResult:
    repo_id: str
    downloads: int
    likes: int
    last_modified: str
    tags: list[str]

    @property
    def dataset_url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo_id}"


def _load_huggingface_token() -> str | None:
    env_values = load_repo_env_file()
    return (
        env_values.get("HF_TOKEN")
        or env_values.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )


def huggingface_auth_status() -> dict[str, str | bool]:
    token = _load_huggingface_token()
    token_source = "missing"
    if token:
        repo_env = load_repo_env_file()
        token_source = (
            ".env"
            if repo_env.get("HF_TOKEN") or repo_env.get("HUGGINGFACE_HUB_TOKEN")
            else "environment"
        )
    token_hint = f"{token[:4]}...{token[-4:]}" if token and len(token) >= 8 else ""
    return {
        "token_resolved": bool(token),
        "token_source": token_source,
        "token_hint": token_hint,
    }


def search_huggingface_datasets(
    query: str | None = None,
    task_category: str = "tabular-classification",
    limit: int = 20,
    sort: str = "downloads",
) -> list[HuggingFaceDatasetResult]:
    from huggingface_hub import HfApi

    api = HfApi(token=_load_huggingface_token())
    payload = list(
        api.list_datasets(
            filter=f"task_categories:{task_category}",
            search=query,
            sort=sort,
            direction=-1,
            limit=limit,
            expand=["downloads", "likes", "lastModified", "tags"],
        )
    )
    return [
        HuggingFaceDatasetResult(
            repo_id=item.id,
            downloads=int(item.downloads or 0),
            likes=int(item.likes or 0),
            last_modified=item.lastModified or "",
            tags=list(item.tags or []),
        )
        for item in payload
    ]


def fetch_huggingface_dataset(
    repo_id: str,
    output_root: str | Path = "data/raw",
    dataset_id: str | None = None,
    config_name: str | None = None,
    split: str = "train",
    max_rows: int | None = None,
    title: str | None = None,
    task_type: str | None = None,
    target_column: str | None = None,
    notes: str = "",
) -> Path:
    from datasets import load_dataset

    split_spec = split if max_rows is None else f"{split}[:{max_rows}]"
    local_id = dataset_id or sanitize_dataset_id(repo_id)
    raw_dir = Path(output_root) / local_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(repo_id, name=config_name, split=split_spec, token=_load_huggingface_token())
    metadata_target_column, metadata_task_type = infer_huggingface_target_metadata(dataset)
    frame = dataset.to_pandas()
    inspection = inspect_supervised_frame(
        frame,
        preferred_target_column=target_column,
        preferred_task_type=task_type,
        metadata_target_column=metadata_target_column,
        metadata_task_type=metadata_task_type,
    )
    output_path = raw_dir / "train.csv"
    inspection.frame.to_csv(output_path, index=False)

    manifest = DatasetManifest(
        id=local_id,
        title=title or repo_id,
        provider="huggingface",
        source_type="dataset",
        external_ref=repo_id,
        source_url=f"https://huggingface.co/datasets/{repo_id}",
        task_type=inspection.task_type,
        target_column=inspection.target_column,
        train_file=output_path.name,
        notes=notes or f"Hugging Face dataset split={split} config={config_name or ''}".strip(),
    )
    write_manifest(raw_dir, manifest)
    return raw_dir


def bootstrap_huggingface_stream_sample(
    repo_id: str,
    output_root: str | Path = "data/raw",
    dataset_id: str | None = None,
    config_name: str | None = None,
    split: str = "train",
    sample_rows: int = 2048,
    shuffle_buffer_size: int = 10000,
    seed: int = 42,
    title: str | None = None,
    task_type: str | None = None,
    target_column: str | None = None,
    notes: str = "",
) -> Path:
    from datasets import load_dataset

    if sample_rows <= 0:
        raise ValueError("sample_rows must be greater than zero.")

    local_id = dataset_id or sanitize_dataset_id(repo_id)
    raw_dir = Path(output_root) / local_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        repo_id,
        name=config_name,
        split=split,
        streaming=True,
        token=_load_huggingface_token(),
    )
    metadata_target_column, metadata_task_type = infer_huggingface_target_metadata(dataset)
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed, buffer_size=max(shuffle_buffer_size, sample_rows))

    rows = [dict(row) for row in islice(iter(dataset), sample_rows)]
    if not rows:
        raise RuntimeError(f"Hugging Face dataset `{repo_id}` split `{split}` produced no rows while streaming.")

    frame = pd.DataFrame(rows)
    inspection = inspect_supervised_frame(
        frame,
        preferred_target_column=target_column,
        preferred_task_type=task_type,
        metadata_target_column=metadata_target_column,
        metadata_task_type=metadata_task_type,
    )
    output_path = raw_dir / "train.csv"
    inspection.frame.to_csv(output_path, index=False)

    manifest = DatasetManifest(
        id=local_id,
        title=title or repo_id,
        provider="huggingface",
        source_type="dataset",
        external_ref=repo_id,
        source_url=f"https://huggingface.co/datasets/{repo_id}",
        task_type=inspection.task_type,
        target_column=inspection.target_column,
        train_file=output_path.name,
        notes=notes or f"Hugging Face streamed sample split={split} rows={len(inspection.frame)} config={config_name or ''}".strip(),
    )
    write_manifest(raw_dir, manifest)
    return raw_dir
