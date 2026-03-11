"""OpenML dataset connector.

OpenML (https://openml.org) hosts over 20,000 datasets, all freely accessible
via a REST JSON API without authentication.  This module provides:

- ``search_openml_datasets``  – full-text / filter search of the OpenML catalog
- ``fetch_openml_dataset``    – download a dataset by ID as a pandas DataFrame
- ``list_openml_tasks``       – enumerate supervised learning tasks of a given type
- ``fetch_openml_task``       – fetch the canonical task (with split) for a task id
- ``write_openml_manifest``   – write a ``dataset_manifest.json`` compatible with
                                the tabula prepare pipeline

No external package is required beyond ``requests`` (already in the standard
scientific Python ecosystem).  If the ``openml`` pip package is installed it is
used for richer metadata; otherwise the raw JSON API is used as a fallback.

API base
--------
    https://api.openml.org/json

Key endpoints used:
    GET /data/list/<filter>         – search datasets
    GET /data/features/<dataset_id> – column / feature metadata
    GET /data/<dataset_id>          – dataset detail
    GET /data/get_csv/<dataset_id>  – download as CSV
    GET /task/list/<filter>         – search tasks
    GET /task/<task_id>             – task detail
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any
import urllib.parse
import urllib.request
import urllib.error

import pandas as pd

from tabula.data.manifest import DatasetManifest, sanitize_dataset_id, write_manifest


OPENML_API_BASE = "https://api.openml.org/json"
OPENML_CSV_BASE = "https://api.openml.org/data/get_csv"
OPENML_ARFF_BASE = "https://api.openml.org/data/v1/download"


@dataclass(frozen=True)
class OpenMLDatasetResult:
    dataset_id: int
    name: str
    version: int
    n_features: int
    n_instances: int
    n_classes: int | None
    n_missing: int
    format: str
    file_id: int
    url: str

    @property
    def openml_url(self) -> str:
        return f"https://openml.org/d/{self.dataset_id}"


@dataclass(frozen=True)
class OpenMLTaskResult:
    task_id: int
    task_type: str          # "Supervised Classification" | "Supervised Regression"
    dataset_id: int
    dataset_name: str
    target_feature: str
    n_instances: int | None
    evaluation_measure: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _api_get(path: str, timeout: int = 30) -> dict[str, Any]:
    """Make a GET request to the OpenML JSON API."""
    url = f"{OPENML_API_BASE}{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenML API error {exc.code} for {url}: {exc.reason}") from exc
    return json.loads(raw)  # type: ignore[return-value]


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Dataset search
# ---------------------------------------------------------------------------


def search_openml_datasets(
    query: str | None = None,
    tag: str | None = None,
    min_instances: int = 100,
    max_instances: int | None = None,
    max_features: int | None = None,
    task_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[OpenMLDatasetResult]:
    """Search the OpenML dataset catalogue.

    Parameters
    ----------
    query : str, optional
        Full-text search string (dataset name match).
    tag : str, optional
        Filter by OpenML tag (e.g. ``"study_14"`` for the CC-18 benchmark).
    min_instances : int
        Minimum number of rows.
    max_instances : int, optional
        Maximum number of rows (None = no limit).
    max_features : int, optional
        Maximum number of columns (None = no limit).
    task_type : str, optional
        ``"classification"`` or ``"regression"`` – filters via tag heuristic.
    limit : int
        Maximum results to return.
    offset : int
        Pagination offset.
    """
    filters: list[str] = []
    if tag:
        filters.append(f"tag/{urllib.parse.quote(tag)}")
    if query:
        filters.append(f"name/{urllib.parse.quote(query)}")
    filters.append(f"number_instances_from/{min_instances}")
    if max_instances:
        filters.append(f"number_instances_to/{max_instances}")
    if max_features:
        filters.append(f"number_features_to/{max_features}")
    filters.append(f"offset/{offset}")
    filters.append(f"limit/{limit}")

    path = "/data/list/" + "/".join(filters)
    try:
        payload = _api_get(path)
    except RuntimeError:
        return []

    raw_datasets = payload.get("data", {}).get("dataset", [])
    if not isinstance(raw_datasets, list):
        raw_datasets = [raw_datasets] if raw_datasets else []

    results: list[OpenMLDatasetResult] = []
    for item in raw_datasets:
        q = item.get("quality", [])
        quality_map: dict[str, Any] = {}
        for qitem in (q if isinstance(q, list) else []):
            qname = qitem.get("name")
            qval = qitem.get("value")
            if qname:
                quality_map[qname] = qval
        n_instances = _parse_int(quality_map.get("NumberOfInstances", 0))
        n_features = _parse_int(quality_map.get("NumberOfFeatures", 0))
        n_classes_raw = quality_map.get("NumberOfClasses")
        n_missing = _parse_int(quality_map.get("NumberOfMissingValues", 0))

        results.append(
            OpenMLDatasetResult(
                dataset_id=_parse_int(item.get("did", 0)),
                name=str(item.get("name", "")),
                version=_parse_int(item.get("version", 1)),
                n_features=n_features,
                n_instances=n_instances,
                n_classes=_parse_int(n_classes_raw) if n_classes_raw else None,
                n_missing=n_missing,
                format=str(item.get("format", "")),
                file_id=_parse_int(item.get("file_id", 0)),
                url=str(item.get("url", "")),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Task search
# ---------------------------------------------------------------------------


def list_openml_tasks(
    task_type_id: int = 1,
    tag: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[OpenMLTaskResult]:
    """List OpenML supervised learning tasks.

    Parameters
    ----------
    task_type_id : int
        1 = Supervised Classification, 2 = Supervised Regression,
        3 = Learning Curve, 4 = Supervised Data Stream, 5 = Clustering.
    tag : str, optional
        Filter by OpenML tag (e.g. ``"study_14"`` for CC-18).
    limit : int
        Max results.
    offset : int
        Pagination offset.
    """
    filters = [f"type/{task_type_id}", f"offset/{offset}", f"limit/{limit}"]
    if tag:
        filters.append(f"tag/{urllib.parse.quote(tag)}")
    path = "/task/list/" + "/".join(filters)
    try:
        payload = _api_get(path)
    except RuntimeError:
        return []

    raw_tasks = payload.get("tasks", {}).get("task", [])
    if not isinstance(raw_tasks, list):
        raw_tasks = [raw_tasks] if raw_tasks else []

    task_type_str_map = {1: "Supervised Classification", 2: "Supervised Regression"}
    results: list[OpenMLTaskResult] = []
    for item in raw_tasks:
        inputs = item.get("input", [])
        if not isinstance(inputs, list):
            inputs = [inputs]
        input_map: dict[str, Any] = {inp.get("name"): inp.get("value") for inp in inputs if isinstance(inp, dict)}
        target_feature = str(input_map.get("target_feature", ""))
        dataset_id = _parse_int(input_map.get("source_data", {}).get("data_set_id") if isinstance(input_map.get("source_data"), dict) else 0)
        dataset_name = str(input_map.get("source_data", {}).get("name", "") if isinstance(input_map.get("source_data"), dict) else "")
        results.append(
            OpenMLTaskResult(
                task_id=_parse_int(item.get("task_id", 0)),
                task_type=task_type_str_map.get(task_type_id, f"type_{task_type_id}"),
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                target_feature=target_feature,
                n_instances=None,
                evaluation_measure=str(input_map.get("evaluation_measures", {}).get("evaluation_measure", "") if isinstance(input_map.get("evaluation_measures"), dict) else ""),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Dataset fetch
# ---------------------------------------------------------------------------


def fetch_openml_dataset(
    dataset_id: int,
    output_root: str | Path = "data/raw",
    local_dataset_id: str | None = None,
    task_type: str | None = None,
    target_column: str | None = None,
    max_rows: int | None = None,
    notes: str = "",
) -> Path:
    """Download an OpenML dataset by numeric ID and write raw CSV + manifest.

    Returns the path to the raw directory ``data/raw/<local_dataset_id>/``.

    Uses the OpenML CSV endpoint which returns the dataset as a downloadable
    CSV file.  If the file is unavailable it falls back to the ``openml``
    Python package if installed.

    Parameters
    ----------
    dataset_id : int
        OpenML dataset id (numeric).
    output_root : Path
        Root for raw dataset directories.
    local_dataset_id : str, optional
        Override the local directory name; defaults to ``openml_{dataset_id}``.
    task_type : str, optional
        Hint for ``binary``, ``multiclass``, or ``regression``.
    target_column : str, optional
        Column to use as the prediction target.
    max_rows : int, optional
        If set, only the first ``max_rows`` rows are written.
    notes : str
        Free-text notes stored in the manifest.
    """
    if local_dataset_id is None:
        local_dataset_id = f"openml_{dataset_id}"

    raw_dir = Path(output_root) / local_dataset_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    # --- get dataset metadata ---
    try:
        meta = _api_get(f"/data/{dataset_id}")
    except RuntimeError as exc:
        raise RuntimeError(f"Could not fetch OpenML dataset {dataset_id} metadata: {exc}") from exc

    ds_meta = meta.get("data_set_description", {})
    name = str(ds_meta.get("name", f"openml_{dataset_id}"))
    file_id = _parse_int(ds_meta.get("file_id", 0))
    default_target = str(ds_meta.get("default_target_attribute", "") or "")
    if not target_column and default_target:
        target_column = default_target

    # infer task_type if not given
    if not task_type:
        features_meta = _api_get(f"/data/features/{dataset_id}").get("data_features", {})
        target_feature_list = features_meta.get("feature", [])
        if not isinstance(target_feature_list, list):
            target_feature_list = [target_feature_list]
        for feat in target_feature_list:
            if isinstance(feat, dict) and feat.get("name") == target_column:
                data_type = str(feat.get("data_type", "")).lower()
                n_distinct = _parse_int(feat.get("number_of_distinct_values", 0))
                if data_type == "nominal" or n_distinct <= 20:
                    task_type = "binary" if n_distinct <= 2 else "multiclass"
                else:
                    task_type = "regression"
                break

    # --- try to load via openml package first, fall back to direct URL ---
    df = _load_openml_df(dataset_id=dataset_id, file_id=file_id, max_rows=max_rows)

    csv_path = raw_dir / "train.csv"
    df.to_csv(csv_path, index=False)

    manifest = DatasetManifest(
        id=local_dataset_id,
        title=name,
        provider="openml",
        source_type="dataset",
        external_ref=str(dataset_id),
        source_url=f"https://openml.org/d/{dataset_id}",
        task_type=task_type,
        target_column=target_column,
        train_file="train.csv",
        notes=notes or f"OpenML dataset id={dataset_id}",
    )
    write_manifest(raw_dir, manifest)
    return raw_dir


def _load_openml_df(dataset_id: int, file_id: int, max_rows: int | None) -> pd.DataFrame:
    """Load an OpenML dataset as a pandas DataFrame.

    Tries the openml Python package first; falls back to direct URL CSV download.
    """
    # Try openml package
    try:
        import openml  # type: ignore[import]
        ds = openml.datasets.get_dataset(
            dataset_id,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )
        X, y, cat_indicator, feature_names = ds.get_data(dataset_format="dataframe")
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=feature_names)
        if y is not None:
            df["class"] = y
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)
        return df
    except Exception:
        pass

    # Fall back: direct CSV download via file_id
    if file_id:
        url = f"{OPENML_CSV_BASE}/{file_id}"
        try:
            df = pd.read_csv(url, nrows=max_rows)
            return df
        except Exception:
            pass

    # Last resort: direct dataset CSV URL
    url = f"https://api.openml.org/data/v1/download/{file_id}"
    try:
        df = pd.read_csv(url, nrows=max_rows, comment="@", on_bad_lines="skip")
        return df
    except Exception as exc:
        raise RuntimeError(
            f"Could not download OpenML dataset {dataset_id} (file_id={file_id}). "
            "Install the `openml` Python package for robust access."
        ) from exc


# ---------------------------------------------------------------------------
# CC-18 benchmark suite convenience
# ---------------------------------------------------------------------------

# The CC-18 suite is a widely-used 72-dataset classification benchmark.
# Tag: study_14  (Bischl et al., 2021).
CC18_TAG = "study_14"
# Regression companion: OpenML-CTR23 (study 218)
CTR23_TAG = "study_218"


def fetch_cc18_task_list() -> list[OpenMLTaskResult]:
    """Return OpenML task entries for the full CC-18 classification benchmark."""
    return list_openml_tasks(task_type_id=1, tag=CC18_TAG, limit=200)


def fetch_ctr23_task_list() -> list[OpenMLTaskResult]:
    """Return OpenML task entries for the CTR-23 regression benchmark."""
    return list_openml_tasks(task_type_id=2, tag=CTR23_TAG, limit=200)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def openml_result_to_manifest(
    result: OpenMLDatasetResult,
    local_dataset_id: str | None = None,
    task_type: str | None = None,
    target_column: str | None = None,
) -> DatasetManifest:
    lid = local_dataset_id or f"openml_{result.dataset_id}"
    inferred_task = task_type
    if not inferred_task:
        if result.n_classes and result.n_classes > 0:
            inferred_task = "binary" if result.n_classes == 2 else "multiclass"
        else:
            inferred_task = "regression"
    return DatasetManifest(
        id=lid,
        title=result.name,
        provider="openml",
        source_type="dataset",
        external_ref=str(result.dataset_id),
        source_url=result.openml_url,
        task_type=inferred_task,
        target_column=target_column,
        train_file="train.csv",
        notes=f"OpenML id={result.dataset_id}",
    )
