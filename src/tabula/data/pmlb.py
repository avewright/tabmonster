"""PMLB (Penn Machine Learning Benchmarks) connector.

PMLB (https://epistasislab.github.io/pmlb/) provides ~400 tabular benchmark
datasets served directly from GitHub as TSV files.  No authentication needed.

The datasets are enumerated in a JSON manifest hosted at:
    https://raw.githubusercontent.com/EpistasisLab/pmlb/master/pmlb/all_summary_stats.tsv

Each row has dataset_name, task, n_instances, n_features, n_classes, etc.

Individual datasets are downloaded from:
    https://github.com/EpistasisLab/pmlb/raw/master/datasets/<name>/<name>.tsv.gz

All PMLB datasets have a column called ``target`` that is the prediction target.

Usage
-----
    from tabula.data.pmlb import search_pmlb_datasets, fetch_pmlb_dataset

    # List all classification datasets with < 5000 rows
    results = search_pmlb_datasets(task="classification", max_instances=5000)

    # Download one by name
    raw_dir = fetch_pmlb_dataset("iris", output_root="data/raw")
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO, BytesIO
from pathlib import Path
from typing import Any
import urllib.request
import urllib.error
import csv
import gzip

import pandas as pd

from tabula.data.manifest import DatasetManifest, write_manifest


PMLB_SUMMARY_URL = (
    "https://raw.githubusercontent.com/EpistasisLab/pmlb/master/pmlb/all_summary_stats.tsv"
)
PMLB_DATASET_BASE = (
    "https://github.com/EpistasisLab/pmlb/raw/master/datasets/{name}/{name}.tsv.gz"
)
PMLB_DATASET_BASE_TSV = (
    "https://raw.githubusercontent.com/EpistasisLab/pmlb/master/datasets/{name}/{name}.tsv"
)


@dataclass(frozen=True)
class PMLBDatasetResult:
    name: str
    task: str           # "classification" | "regression"
    n_instances: int
    n_features: int
    n_classes: int
    n_binary_features: int
    endpoint_type: str
    imbalance: float

    @property
    def pmlb_url(self) -> str:
        return f"https://epistasislab.github.io/pmlb/profile/{self.name}.html"

    @property
    def task_type(self) -> str:
        """Map to tabula task_type string."""
        if self.task == "regression":
            return "regression"
        if self.n_classes == 2:
            return "binary"
        return "multiclass"


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _fetch_summary() -> list[PMLBDatasetResult]:
    """Download and parse the PMLB summary TSV."""
    req = urllib.request.Request(PMLB_SUMMARY_URL, headers={"User-Agent": "tabula/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not fetch PMLB summary: {exc}") from exc

    reader = csv.DictReader(StringIO(text), delimiter="\t")
    results: list[PMLBDatasetResult] = []
    for row in reader:
        try:
            results.append(
                PMLBDatasetResult(
                    name=str(row.get("dataset", row.get("Dataset", ""))),
                    task=str(row.get("task", "classification")).lower(),
                    n_instances=int(float(row.get("n_instances", row.get("Instances", 0)) or 0)),
                    n_features=int(float(row.get("n_features", row.get("Features", 0)) or 0)),
                    n_classes=int(float(row.get("n_classes", row.get("Classes", 2)) or 2)),
                    n_binary_features=int(float(row.get("n_binary_features", 0) or 0)),
                    endpoint_type=str(row.get("endpoint_type", "")),
                    imbalance=float(row.get("imbalance", 0) or 0),
                )
            )
        except (ValueError, KeyError):
            continue
    return results


# cached in module-level variable to avoid re-downloading every call
_pmlb_summary_cache: list[PMLBDatasetResult] | None = None


def _get_summary(refresh: bool = False) -> list[PMLBDatasetResult]:
    global _pmlb_summary_cache
    if _pmlb_summary_cache is None or refresh:
        _pmlb_summary_cache = _fetch_summary()
    return _pmlb_summary_cache


# ---------------------------------------------------------------------------
# Public search API
# ---------------------------------------------------------------------------


def search_pmlb_datasets(
    task: str | None = None,
    min_instances: int = 50,
    max_instances: int | None = None,
    min_features: int = 2,
    max_features: int | None = None,
    min_classes: int | None = None,
    max_classes: int | None = None,
    query: str | None = None,
    refresh: bool = False,
) -> list[PMLBDatasetResult]:
    """Search PMLB datasets with optional filters.

    Parameters
    ----------
    task : str, optional
        ``"classification"`` or ``"regression"``.
    min_instances / max_instances : int, optional
        Row count filter.
    min_features / max_features : int, optional
        Feature count filter.
    min_classes / max_classes : int, optional
        Class count filter (ignored for regression).
    query : str, optional
        Substring match against dataset name.
    refresh : bool
        Force re-download of the summary TSV.
    """
    datasets = _get_summary(refresh=refresh)
    if task:
        datasets = [d for d in datasets if d.task.lower() == task.lower()]
    if min_instances:
        datasets = [d for d in datasets if d.n_instances >= min_instances]
    if max_instances is not None:
        datasets = [d for d in datasets if d.n_instances <= max_instances]
    if min_features:
        datasets = [d for d in datasets if d.n_features >= min_features]
    if max_features is not None:
        datasets = [d for d in datasets if d.n_features <= max_features]
    if min_classes is not None:
        datasets = [d for d in datasets if d.n_classes >= min_classes]
    if max_classes is not None:
        datasets = [d for d in datasets if d.n_classes <= max_classes]
    if query:
        q = query.lower()
        datasets = [d for d in datasets if q in d.name.lower()]
    return datasets


def get_pmlb_dataset_info(name: str, refresh: bool = False) -> PMLBDatasetResult | None:
    """Look up metadata for a single PMLB dataset by name."""
    for d in _get_summary(refresh=refresh):
        if d.name == name:
            return d
    return None


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------


def fetch_pmlb_dataset(
    name: str,
    output_root: str | Path = "data/raw",
    local_dataset_id: str | None = None,
    max_rows: int | None = None,
    notes: str = "",
) -> Path:
    """Download a PMLB dataset by name and write raw TSV + manifest.

    PMLB datasets all use ``target`` as the label column.

    Parameters
    ----------
    name : str
        PMLB dataset name (e.g. ``"iris"``, ``"breast_cancer"``).
    output_root : Path
        Root directory for raw dataset storage.
    local_dataset_id : str, optional
        Override local directory name; defaults to ``pmlb_{name}``.
    max_rows : int, optional
        Row cap for large datasets.
    notes : str
        Notes stored in the manifest.

    Returns
    -------
    Path
        Path to the raw directory containing ``train.csv`` and
        ``dataset_manifest.json``.
    """
    if local_dataset_id is None:
        local_dataset_id = f"pmlb_{name}"

    raw_dir = Path(output_root) / local_dataset_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    df = _download_pmlb_df(name=name, max_rows=max_rows)
    csv_path = raw_dir / "train.csv"
    df.to_csv(csv_path, index=False)

    # look up task metadata
    info = get_pmlb_dataset_info(name)
    if info:
        task_type = info.task_type
        n_classes = info.n_classes
        title = name.replace("_", " ").title()
    else:
        task_type = _infer_task_type_from_df(df, "target")
        n_classes = int(df["target"].nunique()) if "target" in df.columns else 2
        title = name.replace("_", " ").title()

    manifest = DatasetManifest(
        id=local_dataset_id,
        title=title,
        provider="pmlb",
        source_type="dataset",
        external_ref=name,
        source_url=f"https://epistasislab.github.io/pmlb/profile/{name}.html",
        task_type=task_type,
        target_column="target",
        train_file="train.csv",
        notes=notes or f"PMLB benchmark dataset: {name}",
    )
    write_manifest(raw_dir, manifest)
    return raw_dir


def _download_pmlb_df(name: str, max_rows: int | None) -> pd.DataFrame:
    """Try gzip TSV, then plain TSV, then pmlb package."""
    # Try gzip URL first
    gz_url = PMLB_DATASET_BASE.format(name=name)
    try:
        req = urllib.request.Request(gz_url, headers={"User-Agent": "tabula/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw_bytes = resp.read()
        with gzip.open(BytesIO(raw_bytes), "rt") as f:
            df = pd.read_csv(f, sep="\t", nrows=max_rows)
        return df
    except Exception:
        pass

    # Try plain TSV
    tsv_url = PMLB_DATASET_BASE_TSV.format(name=name)
    try:
        df = pd.read_csv(tsv_url, sep="\t", nrows=max_rows)
        return df
    except Exception:
        pass

    # Try pmlb package
    try:
        from pmlb import fetch_data  # type: ignore[import]
        df = fetch_data(name)
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)
        return df
    except Exception as exc:
        pass

    raise RuntimeError(
        f"Could not download PMLB dataset {name!r}. "
        "Check the dataset name or install the `pmlb` Python package."
    )


def _infer_task_type_from_df(df: pd.DataFrame, target_column: str) -> str:
    if target_column not in df.columns:
        return "binary"
    series = df[target_column].dropna()
    n_unique = series.nunique()
    if n_unique == 2:
        return "binary"
    try:
        pd.to_numeric(series)
        if n_unique > 20:
            return "regression"
    except (ValueError, TypeError):
        pass
    return "multiclass"


# ---------------------------------------------------------------------------
# Batch fetch helper
# ---------------------------------------------------------------------------


def fetch_pmlb_benchmark_suite(
    task: str = "classification",
    max_datasets: int | None = None,
    min_instances: int = 100,
    max_instances: int = 50000,
    max_features: int = 200,
    output_root: str | Path = "data/raw",
) -> list[Path]:
    """Download a collection of PMLB datasets matching the filter criteria.

    Returns a list of raw directory paths.  Existing directories are skipped
    (idempotent).

    This is the primary entry point for bulk PMLB ingestion.
    """
    datasets = search_pmlb_datasets(
        task=task,
        min_instances=min_instances,
        max_instances=max_instances,
        max_features=max_features,
    )
    if max_datasets is not None:
        datasets = datasets[:max_datasets]

    paths: list[Path] = []
    for info in datasets:
        local_id = f"pmlb_{info.name}"
        raw_dir = Path(output_root) / local_id
        if (raw_dir / "train.csv").exists():
            paths.append(raw_dir)
            continue
        try:
            p = fetch_pmlb_dataset(info.name, output_root=output_root, local_dataset_id=local_id)
            paths.append(p)
        except RuntimeError:
            pass
    return paths
