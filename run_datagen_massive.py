#!/usr/bin/env python3
"""Massive parallel synthetic data generation — targeting 1B+ rows.

Generates synthetic tabular datasets in parallel across all CPUs, saving
them as parquet files in a local corpus directory. Each shard is a parquet
file containing multiple datasets concatenated with a _dataset_id column.

Architecture:
  - ProcessPoolExecutor across all available CPUs
  - Each worker generates one dataset, returns (df, metadata)
  - Main thread collects into shards of ~1M rows, writes parquet
  - Runs forever until target row count or Ctrl+C

Hardware: 48 CPUs, 247 GB RAM, RTX A4500 (20 GB VRAM)

Usage:
    python run_datagen_massive.py [--target-rows 1000000000] [--output-dir corpus/pretrain]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, '.')

from tabula.data.synthetic import (
    TreePriorGenerator,
    GaussianMixtureGenerator,
    PolynomialGenerator,
    SCMGenerator,
    RegressionSyntheticGenerator,
    TimeSeriesSyntheticGenerator,
    MixedTypeGenerator,
    SyntheticDatasetMeta,
    sample_random_generator,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CPUS = os.cpu_count() or 4
GENERATION_WORKERS = max(1, NUM_CPUS - 2)  # Leave 2 for IO/main

# Dataset size distribution — biased toward larger datasets for row count
SAMPLE_SIZES = [
    5_000, 10_000, 10_000, 20_000, 20_000, 50_000, 50_000,
    100_000, 100_000, 100_000, 200_000, 200_000, 500_000,
]

# Feature count range
MIN_FEATURES = 4
MAX_FEATURES = 64

# Shard size
ROWS_PER_SHARD = 2_000_000  # ~2M rows per parquet file

# Methods to use
METHODS = [
    "TreePrior", "SCM", "GaussianMixture", "Polynomial",
    "Regression", "MixedType_TreePrior", "MixedType_SCM",
    "MixedType_GaussianMixture",
]

# Task type distribution
TASK_TYPES = ["binary", "binary", "multiclass", "multiclass", "regression"]

# Domain vocabulary for realistic column naming
DOMAIN_VOCAB = {
    "finance": ["income", "age", "debt_ratio", "credit_score", "loan_amount",
                "balance", "expenses", "interest_rate", "assets", "liabilities",
                "revenue", "margin", "tax_rate", "net_worth", "monthly_payment"],
    "health": ["bmi", "age", "blood_pressure", "cholesterol", "glucose",
               "heart_rate", "hemoglobin", "white_cell_count", "temperature_c",
               "weight_kg", "height_cm", "systolic_bp", "diastolic_bp", "pulse_rate",
               "oxygen_saturation"],
    "ecommerce": ["price", "quantity", "discount", "return_rate", "rating",
                  "reviews", "shipping_days", "revenue", "cart_size",
                  "session_minutes", "clicks", "conversion", "page_views",
                  "bounce_rate", "avg_order_value"],
    "iot": ["temperature", "humidity", "pressure", "vibration", "voltage",
            "current", "uptime_hours", "error_rate", "latency_ms",
            "cpu_pct", "mem_pct", "disk_iops", "fan_rpm", "power_watts",
            "signal_strength_dbm"],
    "science": ["wavelength", "intensity", "mass", "velocity", "concentration",
                "ph", "reaction_time", "yield_pct", "purity", "entropy",
                "molar_mass", "density_gcm3", "viscosity_pa_s", "absorbance",
                "peak_area"],
}
DOMAINS = list(DOMAIN_VOCAB.keys())


def _build_generator(method: str, n_samples: int, n_features: int,
                     n_classes: int, task_type: str, rng: np.random.Generator):
    """Build a generator instance for the given method."""
    if method == "TreePrior":
        return TreePriorGenerator(n_samples=n_samples, n_features=n_features, n_classes=n_classes)
    elif method == "SCM":
        return SCMGenerator(n_samples=n_samples, n_features=n_features, n_classes=n_classes)
    elif method == "GaussianMixture":
        return GaussianMixtureGenerator(
            n_samples=n_samples, n_features=n_features, n_classes=n_classes,
            n_components=rng.integers(3, 12),
        )
    elif method == "Polynomial":
        return PolynomialGenerator(
            n_samples=n_samples, n_features=n_features, n_classes=n_classes,
            degree=int(rng.choice([2, 3, 4])),
        )
    elif method == "Regression":
        return RegressionSyntheticGenerator(
            n_samples=n_samples, n_features=n_features,
            noise_std=float(rng.uniform(0.05, 0.5)),
        )
    elif method.startswith("MixedType_"):
        base_method = method.replace("MixedType_", "")
        base_gen = _build_generator(base_method, n_samples, n_features, n_classes, task_type, rng)
        return MixedTypeGenerator(base_generator=base_gen)
    else:
        return TreePriorGenerator(n_samples=n_samples, n_features=n_features, n_classes=n_classes)


def _rename_columns_domain(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Rename feature columns with domain-specific names."""
    domain = rng.choice(DOMAINS)
    vocab = list(DOMAIN_VOCAB[domain])
    rng.shuffle(vocab)
    feature_cols = [c for c in df.columns if c not in ("target",)]
    n = len(feature_cols)

    # Use vocab names, pad with generic if needed
    names = vocab[:n]
    if len(names) < n:
        for i in range(len(names), n):
            names.append(f"feat_{i}")

    mapping = {old: names[i] for i, old in enumerate(feature_cols)}
    return df.rename(columns=mapping)


def generate_one_dataset(seed: int) -> dict | None:
    """Generate a single synthetic dataset. Called in worker processes."""
    try:
        rng = np.random.default_rng(seed)

        method = rng.choice(METHODS)
        n_samples = int(rng.choice(SAMPLE_SIZES))
        n_features = int(rng.integers(MIN_FEATURES, MAX_FEATURES + 1))
        task_type = rng.choice(TASK_TYPES)
        n_classes = 2 if task_type == "binary" else int(rng.integers(3, 10)) if task_type == "multiclass" else 0

        gen = _build_generator(method, n_samples, n_features, n_classes, task_type, rng)
        df, meta = gen.generate(seed=int(rng.integers(0, 2**31)))

        if df is None or len(df) < 100:
            return None

        # Rename to domain columns ~70% of the time
        if rng.random() < 0.7:
            df = _rename_columns_domain(df, rng)

        # Add metadata
        dataset_id = f"synth_{seed}_{method}_{n_samples}"

        return {
            "df": df,
            "dataset_id": dataset_id,
            "method": method,
            "task_type": task_type,
            "n_rows": len(df),
            "n_features": len(df.columns) - 1,  # exclude target
            "seed": seed,
        }

    except Exception as e:
        return None


def write_shard(dfs: list[pd.DataFrame], dataset_ids: list[str],
                shard_idx: int, output_dir: Path) -> tuple[int, str]:
    """Write a shard of datasets as a parquet file.
    
    Each dataset may have different columns, so we normalize them all to
    a common schema: feature_0..feature_N (numeric), target, _dataset_id, _task_type.
    Categorical columns are label-encoded to integers.
    """
    frames = []
    for df, did in zip(dfs, dataset_ids):
        # Separate target from features
        if "target" not in df.columns:
            continue
        target = df["target"].copy()
        feature_cols = [c for c in df.columns if c != "target"]
        
        # Normalize feature names to generic indexed names
        renamed = {}
        for i, col in enumerate(feature_cols):
            renamed[col] = f"feature_{i}"
        df_norm = df[feature_cols].rename(columns=renamed).copy()
        
        # Convert categorical/object columns to numeric (label encode)
        for col in df_norm.columns:
            if df_norm[col].dtype == object or str(df_norm[col].dtype) == "category":
                df_norm[col] = pd.Categorical(df_norm[col]).codes.astype(np.float32)
        
        # Ensure all columns are numeric
        for col in df_norm.columns:
            df_norm[col] = pd.to_numeric(df_norm[col], errors="coerce").astype(np.float32)
        
        # Handle target - label encode if categorical
        if target.dtype == object or str(target.dtype) == "category":
            target = pd.Categorical(target).codes.astype(np.float32)
        else:
            target = pd.to_numeric(target, errors="coerce").astype(np.float32)
        
        df_norm["target"] = target.values
        df_norm["_dataset_id"] = did
        df_norm["_n_features"] = len(feature_cols)
        
        frames.append(df_norm)

    if not frames:
        return 0, ""

    # Pad to same number of feature columns using NaN
    max_features = max(len([c for c in f.columns if c.startswith("feature_")]) for f in frames)
    for i, f in enumerate(frames):
        existing = len([c for c in f.columns if c.startswith("feature_")])
        for j in range(existing, max_features):
            frames[i][f"feature_{j}"] = np.float32(np.nan)

    combined = pd.concat(frames, ignore_index=True)
    n_rows = len(combined)

    shard_path = output_dir / f"shard_{shard_idx:06d}.parquet"
    combined.to_parquet(shard_path, engine="pyarrow", compression="snappy")

    return n_rows, str(shard_path)


def main():
    parser = argparse.ArgumentParser(description="Massive parallel synthetic data generation")
    parser.add_argument("--target-rows", type=int, default=1_000_000_000, help="Target total rows")
    parser.add_argument("--output-dir", type=str, default="corpus/pretrain", help="Output directory")
    parser.add_argument("--workers", type=int, default=GENERATION_WORKERS, help="Number of workers")
    parser.add_argument("--batch-size", type=int, default=100, help="Datasets per batch")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "metadata"
    meta_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("MASSIVE SYNTHETIC DATA GENERATION")
    print(f"Target: {args.target_rows:,} rows")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.workers}")
    print(f"Batch size: {args.batch_size} datasets")
    print(f"CPUs: {NUM_CPUS}")
    print("=" * 70)

    total_rows = 0
    total_datasets = 0
    shard_idx = 0
    batch_num = 0
    master_rng = np.random.default_rng(42)

    # Check for existing shards to resume
    existing_shards = sorted(output_dir.glob("shard_*.parquet"))
    if existing_shards:
        shard_idx = len(existing_shards)
        # Count existing rows
        for sp in existing_shards:
            pf = pq.read_metadata(sp)
            total_rows += pf.num_rows
        print(f"Resuming: found {shard_idx} existing shards, {total_rows:,} rows")
        # Advance RNG to avoid duplicates
        for _ in range(shard_idx * args.batch_size):
            master_rng.integers(0, 2**31)

    start_time = time.time()
    log_path = output_dir / "generation_log.jsonl"

    try:
        while total_rows < args.target_rows:
            batch_start = time.time()
            batch_num += 1

            # Generate seeds for this batch
            seeds = [int(master_rng.integers(0, 2**63)) for _ in range(args.batch_size)]

            # Parallel generation
            results = []
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(generate_one_dataset, s): s for s in seeds}
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=120)  # 2 min timeout per dataset
                        if result is not None:
                            results.append(result)
                    except Exception:
                        pass

            if not results:
                print(f"Batch {batch_num}: no valid datasets generated, skipping")
                continue

            # Collect into shard
            shard_dfs = []
            shard_ids = []
            shard_rows = 0
            batch_metas = []

            for r in results:
                shard_dfs.append(r["df"])
                shard_ids.append(r["dataset_id"])
                shard_rows += r["n_rows"]
                batch_metas.append({
                    "dataset_id": r["dataset_id"],
                    "method": r["method"],
                    "task_type": r["task_type"],
                    "n_rows": r["n_rows"],
                    "n_features": r["n_features"],
                })

                # Write shard when it gets big enough
                if shard_rows >= ROWS_PER_SHARD:
                    n_written, shard_path = write_shard(shard_dfs, shard_ids, shard_idx, output_dir)
                    total_rows += n_written
                    total_datasets += len(shard_dfs)
                    shard_idx += 1

                    # Save metadata
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "shard_idx": shard_idx - 1,
                            "shard_path": shard_path,
                            "n_rows": n_written,
                            "n_datasets": len(shard_dfs),
                            "datasets": batch_metas[:len(shard_dfs)],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }) + "\n")

                    shard_dfs = []
                    shard_ids = []
                    shard_rows = 0
                    batch_metas = []
                    gc.collect()

            # Write remaining data in buffer
            if shard_dfs:
                n_written, shard_path = write_shard(shard_dfs, shard_ids, shard_idx, output_dir)
                total_rows += n_written
                total_datasets += len(shard_dfs)
                shard_idx += 1

                with open(log_path, "a") as f:
                    f.write(json.dumps({
                        "shard_idx": shard_idx - 1,
                        "shard_path": shard_path,
                        "n_rows": n_written,
                        "n_datasets": len(shard_dfs),
                        "datasets": batch_metas,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }) + "\n")
                gc.collect()

            batch_duration = time.time() - batch_start
            elapsed = time.time() - start_time
            rows_per_sec = total_rows / max(elapsed, 1)
            remaining = (args.target_rows - total_rows) / max(rows_per_sec, 1)

            print(
                f"Batch {batch_num}: +{sum(r['n_rows'] for r in results):,} rows | "
                f"Total: {total_rows:,}/{args.target_rows:,} ({100*total_rows/args.target_rows:.1f}%) | "
                f"Shards: {shard_idx} | Datasets: {total_datasets} | "
                f"Rate: {rows_per_sec:,.0f} rows/s | "
                f"ETA: {remaining/3600:.1f}h | "
                f"Batch: {batch_duration:.1f}s"
            )

            # Disk space check
            import shutil
            disk = shutil.disk_usage(str(output_dir))
            if disk.free < 5e9:  # Less than 5GB free
                print("WARNING: Low disk space! Pausing generation.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"GENERATION COMPLETE")
    print(f"Total rows: {total_rows:,}")
    print(f"Total datasets: {total_datasets}")
    print(f"Total shards: {shard_idx}")
    print(f"Elapsed: {elapsed/3600:.2f} hours")
    print(f"Rate: {total_rows/max(elapsed,1):,.0f} rows/s")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Write summary
    summary = {
        "total_rows": total_rows,
        "total_datasets": total_datasets,
        "total_shards": shard_idx,
        "elapsed_seconds": elapsed,
        "target_rows": args.target_rows,
        "completed": total_rows >= args.target_rows,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
