"""Automated dataset discovery pipeline.

Scans multiple data sources (HuggingFace, OpenML, PMLB) for candidate tabular
datasets, validates that each one can be bootstrapped through the schema
builder, and writes a persistent JSON registry so repeated runs only process
new datasets.

Usage
-----
    python -m tabula.data.autodiscovery                \\
        --sources hf openml pmlb                       \\
        --output-root data/raw                         \\
        --registry-file artifacts/discovery_registry.json \\
        --max-new 50

Or programmatically::

    from tabula.data.autodiscovery import run_discovery_pass

    new_paths = run_discovery_pass(
        sources=["hf", "openml", "pmlb"],
        output_root="data/raw",
        max_new=30,
    )
"""

from __future__ import annotations

import json
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryRecord:
    """Persisted metadata about one discovered dataset."""

    dataset_id: str
    source: str          # "hf" | "openml" | "pmlb" | "kaggle"
    external_ref: str    # repo_id, openml id, pmlb name, …
    task_type: str
    n_rows: int
    n_cols: int
    status: str          # "ok" | "schema_fail" | "download_fail" | "skipped"
    raw_dir: str         # relative from workspace root
    notes: str = ""
    discovered_at: float = field(default_factory=time.time)


class DiscoveryRegistry:
    """JSON-backed registry of discovered datasets."""

    def __init__(self, registry_file: str | Path = "artifacts/discovery_registry.json") -> None:
        self.registry_file = Path(registry_file)
        self.records: dict[str, DiscoveryRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_file.exists():
            try:
                raw = json.loads(self.registry_file.read_text())
                for rec in raw:
                    r = DiscoveryRecord(**rec)
                    self.records[r.dataset_id] = r
            except Exception:
                pass

    def save(self) -> None:
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.registry_file.write_text(
            json.dumps([asdict(r) for r in self.records.values()], indent=2)
        )

    def contains(self, dataset_id: str) -> bool:
        return dataset_id in self.records

    def add(self, record: DiscoveryRecord) -> None:
        self.records[record.dataset_id] = record
        self.save()

    def ok_records(self) -> list[DiscoveryRecord]:
        return [r for r in self.records.values() if r.status == "ok"]

    def get_retryable(self, source: str, max_retries: int = 3) -> list[dict]:
        """Return records from *source* with transient failures (download_fail)."""
        retryable = []
        for r in self.records.values():
            if r.source == source and r.status == "download_fail":
                retry_count = int(r.notes.split("retry=")[-1]) if "retry=" in r.notes else 0
                if retry_count < max_retries:
                    retryable.append(asdict(r))
        return retryable

    def update_status(self, dataset_id: str, status: str, notes: str = "") -> None:
        """Update status and notes for an existing record."""
        if dataset_id in self.records:
            rec = self.records[dataset_id]
            old_retry = 0
            if "retry=" in rec.notes:
                try:
                    old_retry = int(rec.notes.split("retry=")[-1])
                except ValueError:
                    pass
            rec.status = status
            if status != "ok":
                rec.notes = f"{notes[:150]} retry={old_retry + 1}"
            else:
                rec.notes = notes
            self.save()

    def __len__(self) -> int:
        return len(self.records)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _validate_raw_dir(raw_dir: Path) -> tuple[bool, str, int, int]:
    """Check that the raw directory has a loadable CSV with enough data.

    Returns (ok, error_msg, n_rows, n_cols).
    """
    csv_candidates = sorted(raw_dir.glob("*.csv"))
    if not csv_candidates:
        return False, "no CSV found", 0, 0
    csv_path = csv_candidates[0]
    try:
        df = pd.read_csv(csv_path, nrows=5000)
        if df.shape[0] < 20 or df.shape[1] < 2:
            return False, f"too small: {df.shape}", 0, 0
        # Check that at least some columns have numeric or low-cardinality data
        usable = 0
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                usable += 1
            elif df[col].nunique() < 200:
                usable += 1
        if usable < 2:
            return False, f"only {usable} usable columns", df.shape[0], df.shape[1]
        return True, "", df.shape[0], df.shape[1]
    except Exception as exc:
        return False, str(exc), 0, 0


# ---------------------------------------------------------------------------
# HuggingFace discovery
# ---------------------------------------------------------------------------


def _discover_hf(
    registry: DiscoveryRegistry,
    output_root: Path,
    limit: int = 40,
    bootstrap_rows: int = 5000,
) -> list[DiscoveryRecord]:
    """Search HF for tabular datasets and bootstrap each."""
    from tabula.data.huggingface import search_huggingface_datasets, fetch_huggingface_dataset  # noqa

    new_records: list[DiscoveryRecord] = []
    # Search multiple task categories to find more datasets
    categories = ["tabular-classification", "tabular-regression"]
    seen_ids: set[str] = set()
    all_results = []
    for cat in categories:
        try:
            results = search_huggingface_datasets(
                task_category=cat,
                limit=limit,
            )
            for r in results:
                if r.repo_id not in seen_ids:
                    seen_ids.add(r.repo_id)
                    all_results.append(r)
        except Exception as exc:
            warnings.warn(f"HF search failed for {cat}: {exc}")

    for res in all_results:
        dataset_id = f"hf_{res.repo_id.replace('/', '_')}"
        if registry.contains(dataset_id):
            continue
        raw_dir = output_root / dataset_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        status = "ok"
        error = ""
        n_rows = n_cols = 0
        try:
            # Use a timeout to avoid blocking on huge datasets
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError("HF dataset download timed out")

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(120)  # 2 minute timeout per dataset
            try:
                fetch_huggingface_dataset(
                    repo_id=res.repo_id,
                    output_root=str(output_root),
                    dataset_id=dataset_id,
                    max_rows=bootstrap_rows,
                )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            ok, error, n_rows, n_cols = _validate_raw_dir(raw_dir)
            status = "ok" if ok else "schema_fail"
        except TimeoutError:
            status = "download_fail"
            error = "download timed out (>120s)"
        except Exception as exc:
            status = "download_fail"
            error = str(exc)[:200]

        rec = DiscoveryRecord(
            dataset_id=dataset_id,
            source="hf",
            external_ref=res.repo_id,
            task_type="unknown",
            n_rows=n_rows,
            n_cols=n_cols,
            status=status,
            raw_dir=str(raw_dir.relative_to(Path.cwd()) if raw_dir.is_relative_to(Path.cwd()) else raw_dir),
            notes=error,
        )
        registry.add(rec)
        new_records.append(rec)
    return new_records


# ---------------------------------------------------------------------------
# OpenML discovery
# ---------------------------------------------------------------------------


def _discover_openml(
    registry: DiscoveryRegistry,
    output_root: Path,
    limit: int = 40,
    max_rows: int = 50000,
) -> list[DiscoveryRecord]:
    """Download CC-18 and CTR-23 benchmark tasks from OpenML."""
    from tabula.data.openml import (  # noqa
        fetch_cc18_task_list,
        fetch_ctr23_task_list,
        fetch_openml_dataset,
    )

    new_records: list[DiscoveryRecord] = []
    task_lists: list[tuple[str, Any]] = []

    try:
        for t in fetch_cc18_task_list():
            task_lists.append(("binary", t))
    except Exception as exc:
        warnings.warn(f"CC18 fetch failed: {exc}")

    try:
        for t in fetch_ctr23_task_list():
            task_lists.append(("regression", t))
    except Exception as exc:
        warnings.warn(f"CTR23 fetch failed: {exc}")

    for task_type, task in task_lists[:limit]:
        dataset_id_str = f"openml_{task.dataset_id}"
        if registry.contains(dataset_id_str):
            continue

        raw_dir = output_root / dataset_id_str
        status = "ok"
        error = ""
        n_rows = n_cols = 0
        try:
            fetch_openml_dataset(
                dataset_id=task.dataset_id,
                output_root=str(output_root),
                local_dataset_id=dataset_id_str,
                task_type=task_type,
                max_rows=max_rows,
            )
            ok, error, n_rows, n_cols = _validate_raw_dir(raw_dir)
            status = "ok" if ok else "schema_fail"
        except Exception as exc:
            status = "download_fail"
            error = str(exc)[:200]
            traceback.print_exc()

        rec = DiscoveryRecord(
            dataset_id=dataset_id_str,
            source="openml",
            external_ref=str(task.dataset_id),
            task_type=task_type,
            n_rows=n_rows,
            n_cols=n_cols,
            status=status,
            raw_dir=str(raw_dir),
            notes=error,
        )
        registry.add(rec)
        new_records.append(rec)
    return new_records


# ---------------------------------------------------------------------------
# PMLB discovery
# ---------------------------------------------------------------------------


def _discover_pmlb(
    registry: DiscoveryRegistry,
    output_root: Path,
    limit: int = 50,
    max_instances: int = 50000,
) -> list[DiscoveryRecord]:
    from tabula.data.pmlb import search_pmlb_datasets, fetch_pmlb_dataset  # noqa

    new_records: list[DiscoveryRecord] = []
    try:
        datasets = search_pmlb_datasets(max_instances=max_instances)
    except Exception as exc:
        warnings.warn(f"PMLB summary fetch failed: {exc}")
        return new_records

    for info in datasets[:limit]:
        local_id = f"pmlb_{info.name}"
        if registry.contains(local_id):
            continue
        raw_dir = output_root / local_id
        status = "ok"
        error = ""
        n_rows = n_cols = 0
        try:
            fetch_pmlb_dataset(info.name, output_root=str(output_root), local_dataset_id=local_id)
            ok, error, n_rows, n_cols = _validate_raw_dir(raw_dir)
            status = "ok" if ok else "schema_fail"
        except Exception as exc:
            status = "download_fail"
            error = str(exc)[:200]

        rec = DiscoveryRecord(
            dataset_id=local_id,
            source="pmlb",
            external_ref=info.name,
            task_type=info.task_type,
            n_rows=n_rows,
            n_cols=n_cols,
            status=status,
            raw_dir=str(raw_dir),
            notes=error,
        )
        registry.add(rec)
        new_records.append(rec)
    return new_records


# ---------------------------------------------------------------------------
# Top-level discovery pass
# ---------------------------------------------------------------------------


def run_discovery_pass(
    sources: list[str] | None = None,
    output_root: str | Path = "data/raw",
    registry_file: str | Path = "artifacts/discovery_registry.json",
    max_new: int | None = None,
    hf_limit: int = 40,
    openml_limit: int = 40,
    pmlb_limit: int = 50,
    openml_max_rows: int = 50_000,
    pmlb_max_instances: int = 50_000,
) -> list[DiscoveryRecord]:
    """Run a full discovery pass over the requested sources.

    Already-seen datasets (tracked in ``registry_file``) are skipped so
    repeated runs only process genuinely new datasets.

    Parameters
    ----------
    sources : list[str]
        Any combination of ``"hf"``, ``"openml"``, ``"pmlb"``.
        Defaults to all three.
    output_root : Path
        Root directory for raw dataset downloads.
    registry_file : Path
        Persistent JSON registry path.
    max_new : int, optional
        Stop after this many *new* validated datasets.

    Returns
    -------
    list[DiscoveryRecord]
        All newly added records (ok + failed) from this pass.
    """
    if sources is None:
        sources = ["hf", "openml", "pmlb"]

    output_root = Path(output_root)
    registry = DiscoveryRegistry(registry_file)
    all_new: list[DiscoveryRecord] = []

    for source in sources:
        if max_new is not None and sum(r.status == "ok" for r in all_new) >= max_new:
            break

        if source == "hf":
            new = _discover_hf(registry, output_root, limit=hf_limit)
        elif source == "openml":
            new = _discover_openml(
                registry, output_root, limit=openml_limit, max_rows=openml_max_rows
            )
        elif source == "pmlb":
            new = _discover_pmlb(
                registry, output_root, limit=pmlb_limit, max_instances=pmlb_max_instances
            )
        else:
            warnings.warn(f"Unknown source {source!r}")
            new = []

        all_new.extend(new)

    ok_count = sum(r.status == "ok" for r in all_new)
    fail_count = len(all_new) - ok_count
    print(f"Discovery pass complete. New ok={ok_count}, failed={fail_count}, registry size={len(registry)}")
    return all_new


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-discover tabular datasets")
    parser.add_argument(
        "--sources", nargs="+", default=["hf", "openml", "pmlb"],
        choices=["hf", "openml", "pmlb"],
    )
    parser.add_argument("--output-root", default="data/raw")
    parser.add_argument("--registry-file", default="artifacts/discovery_registry.json")
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--hf-limit", type=int, default=40)
    parser.add_argument("--openml-limit", type=int, default=40)
    parser.add_argument("--pmlb-limit", type=int, default=50)
    args = parser.parse_args()

    run_discovery_pass(
        sources=args.sources,
        output_root=args.output_root,
        registry_file=args.registry_file,
        max_new=args.max_new,
        hf_limit=args.hf_limit,
        openml_limit=args.openml_limit,
        pmlb_limit=args.pmlb_limit,
    )
