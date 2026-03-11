"""Data utilities for tabular experiments."""

from typing import TYPE_CHECKING

from tabula.data.catalog import KaggleDatasetEntry, filter_catalog, get_dataset_entry, load_kaggle_catalog
from tabula.data.huggingface import (
    HuggingFaceDatasetResult,
    bootstrap_huggingface_stream_sample,
    fetch_huggingface_dataset,
    huggingface_auth_status,
    search_huggingface_datasets,
)
from tabula.data.kaggle import (
    CsvSummary,
    KagglePreparedDataset,
    KaggleSearchResult,
    configure_kaggle_credentials,
    discover_csvs,
    download_dataset,
    download_kaggle_slug,
    ingest_kaggle_dataset,
    kaggle_auth_status,
    search_kaggle_datasets,
)
from tabula.data.episodes import EpisodeBatch, sample_episode_batch
from tabula.data.manifest import DatasetManifest, load_manifest, resolve_manifest, sanitize_dataset_id, write_manifest

# New data-source modules (lazy imports to avoid heavy deps at import time)
def _lazy(module: str, name: str):
    def _get(*args, **kwargs):
        import importlib
        mod = importlib.import_module(module)
        return getattr(mod, name)(*args, **kwargs)
    _get.__name__ = name
    return _get

# OpenML
search_openml_datasets = _lazy("tabula.data.openml", "search_openml_datasets")
fetch_openml_dataset = _lazy("tabula.data.openml", "fetch_openml_dataset")
fetch_cc18_task_list = _lazy("tabula.data.openml", "fetch_cc18_task_list")
fetch_ctr23_task_list = _lazy("tabula.data.openml", "fetch_ctr23_task_list")

# PMLB
search_pmlb_datasets = _lazy("tabula.data.pmlb", "search_pmlb_datasets")
fetch_pmlb_dataset = _lazy("tabula.data.pmlb", "fetch_pmlb_dataset")
fetch_pmlb_benchmark_suite = _lazy("tabula.data.pmlb", "fetch_pmlb_benchmark_suite")

# Synthetic
generate_synthetic_batch = _lazy("tabula.data.synthetic", "generate_synthetic_batch")
sample_random_generator = _lazy("tabula.data.synthetic", "sample_random_generator")

# Time-series
auto_extract_timeseries_features = _lazy("tabula.data.timeseries", "auto_extract_timeseries_features")
detect_temporal_columns = _lazy("tabula.data.timeseries", "detect_temporal_columns")

# Auto-discovery
run_discovery_pass = _lazy("tabula.data.autodiscovery", "run_discovery_pass")

# Stream queue builder
build_auto_queue = _lazy("tabula.data.stream_builder", "build_auto_queue")

if TYPE_CHECKING:
    from tabula.data.prep import PreparedDataset

__all__ = [
    "CsvSummary",
    "DatasetManifest",
    "HuggingFaceDatasetResult",
    "KaggleDatasetEntry",
    "KagglePreparedDataset",
    "KaggleSearchResult",
    "PreparedDataset",
    "auto_extract_timeseries_features",
    "build_auto_queue",
    "configure_kaggle_credentials",
    "detect_temporal_columns",
    "discover_csvs",
    "download_dataset",
    "download_kaggle_slug",
    "ingest_kaggle_dataset",
    "EpisodeBatch",
    "bootstrap_huggingface_stream_sample",
    "fetch_cc18_task_list",
    "fetch_ctr23_task_list",
    "fetch_huggingface_dataset",
    "fetch_openml_dataset",
    "fetch_pmlb_benchmark_suite",
    "fetch_pmlb_dataset",
    "filter_catalog",
    "generate_synthetic_batch",
    "get_dataset_entry",
    "huggingface_auth_status",
    "kaggle_auth_status",
    "load_manifest",
    "load_kaggle_catalog",
    "prepare_dataset",
    "prepared_dataset_to_dict",
    "resolve_manifest",
    "run_discovery_pass",
    "sample_episode_batch",
    "sample_random_generator",
    "sanitize_dataset_id",
    "search_huggingface_datasets",
    "search_kaggle_datasets",
    "search_openml_datasets",
    "search_pmlb_datasets",
    "write_manifest",
]


def prepare_dataset(*args, **kwargs):
    from tabula.data.prep import prepare_dataset as _prepare_dataset

    return _prepare_dataset(*args, **kwargs)


def prepared_dataset_to_dict(*args, **kwargs):
    from tabula.data.prep import prepared_dataset_to_dict as _prepared_dataset_to_dict

    return _prepared_dataset_to_dict(*args, **kwargs)


def __getattr__(name: str):
    if name == "PreparedDataset":
        from tabula.data.prep import PreparedDataset as _PreparedDataset

        return _PreparedDataset
    raise AttributeError(name)
