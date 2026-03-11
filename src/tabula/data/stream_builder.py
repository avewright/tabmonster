"""RAM-budgeted stream queue builder.

Builds a ``curriculum_queue.json``-compatible JSON file by scanning available
prepared datasets, estimating their per-batch RAM footprints and scheduling
them within a user-specified memory budget.

Key design goals
----------------
* Round-robin task types (binary / multiclass / regression) for diversity.
* Inverse-size weighting so large datasets don't crowd out small ones.
* Optionally interleave synthetic episodes derived by the ``synthetic``
  module.
* Idempotent: running twice with the same arguments produces the same queue.

Usage
-----
    from tabula.data.stream_builder import StreamQueueBuilder

    builder = (
        StreamQueueBuilder(ram_budget_gb=8)
        .add_from_prepared_dir("data/processed")
        .add_from_catalog("catalogs/kaggle_tabular.json", prepared_root="data/processed")
        .add_from_hf_catalog("catalogs/hf_tabular.json", prepared_root="data/processed")
        .add_synthetic(n_datasets=20)
    )
    queue = builder.build()
    builder.save("queues/auto_8gb.json")

CLI::

    tabula data build-queue \\
        --ram-budget-gb 8 \\
        --prepared-dir data/processed \\
        --output queues/auto_8gb.json
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# RAM estimation
# ---------------------------------------------------------------------------


# Empirical safety multiplier: pandas + torch overhead
_RAM_SAFETY_FACTOR = 1.5
# Bytes per numeric float32 cell
_BYTES_PER_FLOAT = 4
# Categorical: treat as 2-byte int on average
_BYTES_PER_CAT = 2


def estimate_dataset_ram_mb(
    n_rows: int,
    n_features: int,
    batch_size: int = 256,
    n_numeric: int | None = None,
    n_categorical: int | None = None,
) -> float:
    """Estimate the peak RAM in MB needed to hold one batch of this dataset.

    We estimate for *one batch* (not the full dataset) because training is
    streaming.  The scheduler uses this to decide how many datasets can fit
    concurrently in a DataLoader worker pool.

    Parameters
    ----------
    n_rows : int
        Total rows (used to determine max batch size relative to dataset).
    n_features : int
        Total feature count.
    batch_size : int
        Batch size (training hyperparameter).
    n_numeric, n_categorical : int, optional
        If provided, a more accurate estimate is made.  Otherwise we assume
        50/50 split.
    """
    eff_batch = min(batch_size, n_rows)
    if n_numeric is None:
        n_numeric = n_features // 2
    if n_categorical is None:
        n_categorical = n_features - n_numeric

    raw_bytes = (
        eff_batch * n_numeric * _BYTES_PER_FLOAT
        + eff_batch * n_categorical * _BYTES_PER_CAT
    )
    # Full dataset in memory (streaming epoch still loads parquet/CSV once per cycle)
    full_dataset_bytes = n_rows * n_features * _BYTES_PER_FLOAT
    peak_bytes = max(raw_bytes, full_dataset_bytes * 0.15)  # 15% in flight at once
    return (peak_bytes * _RAM_SAFETY_FACTOR) / (1024 ** 2)


# ---------------------------------------------------------------------------
# Queue entry
# ---------------------------------------------------------------------------


@dataclass
class QueueEntry:
    """One entry in the generated stream queue."""

    dataset_id: str
    prepared_dir: str
    task_type: str          # "binary" | "multiclass" | "regression"
    source: str             # "kaggle" | "hf" | "openml" | "pmlb" | "synthetic"
    steps_per_cycle: int
    est_ram_mb: float
    priority: float = 1.0   # higher = appears more often
    notes: str = ""

    def to_curriculum_entry(self) -> dict[str, Any]:
        """Convert to the CurriculumEntry schema expected by the training engine."""
        return {
            "dataset_id": self.dataset_id,
            "prepared_dir": self.prepared_dir,
            "task_type": self.task_type,
            "steps_per_cycle": self.steps_per_cycle,
            "max_total_steps": None,
        }


# ---------------------------------------------------------------------------
# Builder class
# ---------------------------------------------------------------------------


class StreamQueueBuilder:
    """Fluent builder that assembles a priority-weighted, RAM-budgeted queue."""

    def __init__(
        self,
        ram_budget_gb: float = 8.0,
        batch_size: int = 256,
        default_steps_per_cycle: int = 100,
        rng_seed: int = 42,
    ) -> None:
        self.ram_budget_mb = ram_budget_gb * 1024
        self.batch_size = batch_size
        self.default_steps_per_cycle = default_steps_per_cycle
        self._rng = random.Random(rng_seed)
        self._entries: list[QueueEntry] = []

    # ------------------------------------------------------------------
    # Adds
    # ------------------------------------------------------------------

    def add_entry(self, entry: QueueEntry) -> "StreamQueueBuilder":
        self._entries.append(entry)
        return self

    def add_from_prepared_dir(
        self,
        prepared_root: str | Path,
        source: str = "local",
    ) -> "StreamQueueBuilder":
        """Scan a directory of prepared datasets and add each one."""
        prepared_root = Path(prepared_root)
        if not prepared_root.exists():
            return self
        for dataset_dir in sorted(prepared_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            info = _read_prepared_info(dataset_dir)
            if info is None:
                continue
            entry = _make_entry(
                dataset_id=dataset_dir.name,
                prepared_dir=str(dataset_dir),
                info=info,
                source=source,
                default_steps=self.default_steps_per_cycle,
                batch_size=self.batch_size,
            )
            if entry:
                self._entries.append(entry)
        return self

    def add_from_catalog(
        self,
        catalog_path: str | Path,
        prepared_root: str | Path = "data/processed",
    ) -> "StreamQueueBuilder":
        """Add Kaggle-catalog entries whose prepared directories exist."""
        catalog_path = Path(catalog_path)
        prepared_root = Path(prepared_root)
        if not catalog_path.exists():
            return self
        try:
            entries = json.loads(catalog_path.read_text())
        except Exception:
            return self
        for e in entries:
            did = e.get("id") or e.get("dataset_id") or ""
            if not did:
                continue
            prepared_dir = prepared_root / did
            if not prepared_dir.exists():
                continue
            info = _read_prepared_info(prepared_dir)
            if info is None:
                # Infer from catalog entry
                info = {
                    "task_type": e.get("task_type", "binary"),
                    "n_rows": e.get("est_rows", 1000),
                    "n_features": 20,
                }
            entry = _make_entry(
                dataset_id=did,
                prepared_dir=str(prepared_dir),
                info=info,
                source="kaggle",
                default_steps=self.default_steps_per_cycle,
                batch_size=self.batch_size,
            )
            if entry:
                self._entries.append(entry)
        return self

    def add_from_hf_catalog(
        self,
        catalog_path: str | Path,
        prepared_root: str | Path = "data/processed",
    ) -> "StreamQueueBuilder":
        """Add HF catalog entries whose prepared directories exist."""
        catalog_path = Path(catalog_path)
        prepared_root = Path(prepared_root)
        if not catalog_path.exists():
            return self
        try:
            entries = json.loads(catalog_path.read_text())
        except Exception:
            return self
        for e in entries:
            if e.get("skip"):
                continue
            repo_id = e.get("repo_id") or e.get("dataset_id") or ""
            local_id = e.get("dataset_id") or repo_id.replace("/", "_")
            prepared_dir = prepared_root / local_id
            if not prepared_dir.exists():
                continue
            info = _read_prepared_info(prepared_dir)
            if info is None:
                info = {
                    "task_type": e.get("task_type", "binary"),
                    "n_rows": e.get("est_rows", 1000),
                    "n_features": 20,
                }
            entry = _make_entry(
                dataset_id=local_id,
                prepared_dir=str(prepared_dir),
                info=info,
                source="hf",
                default_steps=self.default_steps_per_cycle,
                batch_size=self.batch_size,
            )
            if entry:
                self._entries.append(entry)
        return self

    def add_from_discovery_registry(
        self,
        registry_file: str | Path = "artifacts/discovery_registry.json",
        prepared_root: str | Path = "data/processed",
    ) -> "StreamQueueBuilder":
        """Add entries from an autodiscovery registry."""
        registry_file = Path(registry_file)
        if not registry_file.exists():
            return self
        try:
            records = json.loads(registry_file.read_text())
        except Exception:
            return self
        prepared_root = Path(prepared_root)
        for rec in records:
            if rec.get("status") != "ok":
                continue
            did = rec.get("dataset_id", "")
            prepared_dir = prepared_root / did
            if not prepared_dir.exists():
                continue
            info = _read_prepared_info(prepared_dir)
            if info is None:
                info = {
                    "task_type": rec.get("task_type", "binary"),
                    "n_rows": rec.get("n_rows", 1000),
                    "n_features": rec.get("n_cols", 20),
                }
            entry = _make_entry(
                dataset_id=did,
                prepared_dir=str(prepared_dir),
                info=info,
                source=rec.get("source", "unknown"),
                default_steps=self.default_steps_per_cycle,
                batch_size=self.batch_size,
            )
            if entry:
                self._entries.append(entry)
        return self

    def add_synthetic(
        self,
        n_datasets: int = 20,
        prepared_root: str | Path = "data/processed",
        seed: int = 0,
        write_to_disk: bool = True,
    ) -> "StreamQueueBuilder":
        """Generate synthetic datasets and add them to the queue.

        If ``write_to_disk`` is True, each generated dataset is saved as a CSV
        and prepared so the training engine can load it exactly like a real
        dataset.
        """
        from tabula.data.synthetic import generate_synthetic_batch  # noqa: PLC0415
        from tabula.data.prep import prepare_dataset  # noqa: PLC0415

        prepared_root = Path(prepared_root)
        batches = generate_synthetic_batch(n_datasets=n_datasets, seed=seed)

        for df, meta in batches:
            local_id = f"synthetic_{meta.generator_type}_{abs(hash(meta.generator_type + str(seed + meta.n_samples)))}"
            prepared_dir = prepared_root / local_id
            if prepared_dir.exists():
                info = _read_prepared_info(prepared_dir)
            else:
                if write_to_disk:
                    raw_dir = Path("data/raw") / local_id
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    df.to_csv(raw_dir / "train.csv", index=False)
                    try:
                        prepare_dataset(
                            raw_dir=str(raw_dir),
                            target_column=meta.target_column,
                            output_dir=str(prepared_dir),
                            task_type=meta.task_type,
                        )
                    except Exception:
                        continue
                info = {
                    "task_type": meta.task_type,
                    "n_rows": meta.n_samples,
                    "n_features": meta.n_features,
                }

            entry = QueueEntry(
                dataset_id=local_id,
                prepared_dir=str(prepared_dir),
                task_type=meta.task_type,
                source="synthetic",
                steps_per_cycle=self.default_steps_per_cycle,
                est_ram_mb=estimate_dataset_ram_mb(
                    meta.n_samples, meta.n_features, self.batch_size
                ),
                priority=0.5,  # lower weight than real data
                notes=f"synthetic:{meta.generator_type}",
            )
            self._entries.append(entry)
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, enforce_budget: bool = True) -> list[QueueEntry]:
        """Return RAM-budgeted, diversity-weighted queue entries.

        Removes exact duplicates, then applies inverse-size weighting.
        If ``enforce_budget`` is True, entries whose cumulative RAM footprint
        would exceed the budget are dropped (keeping the most diverse
        selection).
        """
        seen: set[str] = set()
        unique: list[QueueEntry] = []
        for e in self._entries:
            if e.dataset_id not in seen:
                seen.add(e.dataset_id)
                unique.append(e)

        # Apply inverse-size priority adjustment
        for entry in unique:
            # Smaller datasets get boosted so they cycle more
            size_factor = max(1.0, math.log10(max(entry.est_ram_mb, 1.0)))
            entry.priority = entry.priority / size_factor

        # Sort by task_type for round-robin, then by priority descending
        task_order = ["binary", "multiclass", "regression"]
        by_task: dict[str, list[QueueEntry]] = {t: [] for t in task_order}
        other: list[QueueEntry] = []
        for e in unique:
            bucket = by_task.get(e.task_type)
            if bucket is not None:
                bucket.append(e)
            else:
                other.append(e)
        for bucket in by_task.values():
            bucket.sort(key=lambda e: e.priority, reverse=True)

        # Round-robin merge
        ordered: list[QueueEntry] = []
        while any(by_task.values()):
            for t in task_order:
                if by_task[t]:
                    ordered.append(by_task[t].pop(0))
        ordered.extend(other)

        if not enforce_budget:
            return ordered

        # Greedy RAM budget enforcement
        selected: list[QueueEntry] = []
        cumulative_ram = 0.0
        for entry in ordered:
            if cumulative_ram + entry.est_ram_mb <= self.ram_budget_mb:
                selected.append(entry)
                cumulative_ram += entry.est_ram_mb
        return selected

    def save(
        self,
        output_path: str | Path = "queues/auto_tabular.json",
        enforce_budget: bool = True,
        format: str = "curriculum",  # "curriculum" | "raw"
    ) -> Path:
        """Build queue and write to a JSON file.

        Parameters
        ----------
        output_path : Path
            Destination JSON path.
        format : str
            ``"curriculum"`` writes the CurriculumEntry format expected by the
            training engine.  ``"raw"`` writes the full QueueEntry objects.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        queue = self.build(enforce_budget=enforce_budget)
        if format == "curriculum":
            data = [e.to_curriculum_entry() for e in queue]
        else:
            data = [asdict(e) for e in queue]
        output_path.write_text(json.dumps(data, indent=2))
        print(
            f"Saved {len(queue)} entries to {output_path}  "
            f"(est. total RAM: {sum(e.est_ram_mb for e in queue):.0f} MB)"
        )
        return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_prepared_info(prepared_dir: Path) -> dict[str, Any] | None:
    """Try to read schema/manifest metadata from a prepared dir."""
    # Try schema.json
    for name in ("schema.json", "dataset_manifest.json", "manifest.json"):
        p = prepared_dir / name
        if p.exists():
            try:
                d = json.loads(p.read_text())
                return d
            except Exception:
                pass
    # Try to infer from train.parquet or train.csv size
    for name in ("train.parquet", "train.csv"):
        p = prepared_dir / name
        if p.exists():
            try:
                size_mb = p.stat().st_size / (1024 ** 2)
                return {"task_type": "binary", "n_rows": int(size_mb * 1000), "n_features": 20}
            except Exception:
                pass
    return None


def _make_entry(
    dataset_id: str,
    prepared_dir: str,
    info: dict[str, Any],
    source: str,
    default_steps: int,
    batch_size: int,
) -> QueueEntry | None:
    task_type = (
        info.get("task_type")
        or info.get("task")
        or "binary"
    )
    # Normalise task_type values
    task_type = task_type.lower()
    if task_type not in ("binary", "multiclass", "regression"):
        if "class" in task_type:
            task_type = "binary"
        elif "regress" in task_type:
            task_type = "regression"
        else:
            task_type = "binary"

    n_rows = int(info.get("n_rows") or info.get("n_instances") or 1000)
    n_features = int(info.get("n_features") or info.get("n_cols") or 20)

    est_ram = estimate_dataset_ram_mb(n_rows, n_features, batch_size)
    return QueueEntry(
        dataset_id=dataset_id,
        prepared_dir=prepared_dir,
        task_type=task_type,
        source=source,
        steps_per_cycle=default_steps,
        est_ram_mb=est_ram,
        priority=1.0,
    )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def build_auto_queue(
    ram_budget_gb: float = 8.0,
    prepared_root: str | Path = "data/processed",
    kaggle_catalog: str | Path = "catalogs/kaggle_tabular.json",
    hf_catalog: str | Path = "catalogs/hf_tabular.json",
    discovery_registry: str | Path = "artifacts/discovery_registry.json",
    include_synthetic: bool = True,
    n_synthetic: int = 20,
    output_path: str | Path = "queues/auto_tabular.json",
) -> Path:
    """One-shot helper: build and save a RAM-budgeted queue from all sources."""
    builder = StreamQueueBuilder(ram_budget_gb=ram_budget_gb)
    builder.add_from_prepared_dir(prepared_root)
    builder.add_from_catalog(kaggle_catalog, prepared_root=prepared_root)
    builder.add_from_hf_catalog(hf_catalog, prepared_root=prepared_root)
    builder.add_from_discovery_registry(discovery_registry, prepared_root=prepared_root)
    if include_synthetic:
        builder.add_synthetic(n_datasets=n_synthetic, prepared_root=prepared_root, write_to_disk=True)
    return builder.save(output_path, format="curriculum")
