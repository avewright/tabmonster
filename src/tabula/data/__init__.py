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
    "configure_kaggle_credentials",
    "discover_csvs",
    "download_dataset",
    "download_kaggle_slug",
    "ingest_kaggle_dataset",
    "EpisodeBatch",
    "bootstrap_huggingface_stream_sample",
    "fetch_huggingface_dataset",
    "filter_catalog",
    "get_dataset_entry",
    "huggingface_auth_status",
    "kaggle_auth_status",
    "load_manifest",
    "load_kaggle_catalog",
    "prepare_dataset",
    "prepared_dataset_to_dict",
    "resolve_manifest",
    "sample_episode_batch",
    "sanitize_dataset_id",
    "search_huggingface_datasets",
    "search_kaggle_datasets",
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
