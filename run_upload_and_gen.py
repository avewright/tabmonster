#!/usr/bin/env python3
"""Upload corpus/pretrain_v2 to HuggingFace and generate more data.

1. Create dataset card (README.md)
2. Upload existing 135 shards (~272M rows, 41GB) 
3. Resume generating more data, uploading each shard as it completes
"""
import gc
import json
import os
import shutil
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "avewright/tabula-pretraining-corpus-v2"
CORPUS_DIR = Path("corpus/pretrain_v2/data")
LOG_FILE = Path("corpus/pretrain_v2/generation_log.jsonl")


# ── Utility ──────────────────────────────────────────────────────────────────

def corpus_stats():
    """Read generation log and compute summary stats."""
    if not LOG_FILE.exists():
        return {}
    with open(LOG_FILE) as f:
        logs = [json.loads(l) for l in f if l.strip()]
    if not logs:
        return {}
    
    total_rows = sum(l["rows"] for l in logs)
    total_datasets = sum(l["datasets"] for l in logs)
    total_fails = sum(l["gate_fails"] for l in logs)
    total_errors = sum(l.get("errors", 0) for l in logs)
    utils = [l["mean_utility_auc"] for l in logs if l.get("mean_utility_auc", 0) > 0]
    
    gens, doms, tasks = Counter(), Counter(), Counter()
    for l in logs:
        for k, v in l["generators"].items():
            gens[k] += v
        for k, v in l["domains"].items():
            doms[k] += v
        for k, v in l["task_types"].items():
            tasks[k] += v
    
    return {
        "shards": len(logs),
        "total_rows": total_rows,
        "total_datasets": total_datasets,
        "gate_fail_rate": total_fails / max(total_datasets + total_fails, 1),
        "total_errors": total_errors,
        "mean_utility_auc": sum(utils) / max(len(utils), 1),
        "generators": dict(gens.most_common()),
        "tasks": dict(tasks.most_common()),
        "domains": dict(doms.most_common()),
    }


def make_dataset_card(stats: dict) -> str:
    """Generate a HuggingFace dataset card README.md."""
    rows = f"{stats.get('total_rows', 0):,}"
    datasets = f"{stats.get('total_datasets', 0):,}"
    shards = stats.get("shards", 0)
    utility = f"{stats.get('mean_utility_auc', 0):.3f}"
    
    generators = stats.get("generators", {})
    gen_lines = "\n".join(f"| {k} | {v:,} |" for k, v in generators.items())
    
    domains = stats.get("domains", {})
    domain_lines = "\n".join(f"| {k} | {v:,} |" for k, v in domains.items())
    
    tasks = stats.get("tasks", {})
    task_lines = "\n".join(f"| {k} | {v:,} |" for k, v in tasks.items())
    
    return f"""---
language:
- en
license: apache-2.0
task_categories:
- tabular-classification
- tabular-regression
tags:
- tabular
- synthetic
- pretraining
- in-context-learning
size_categories:
- 100M<n<1B
---

# Tabula Pretraining Corpus v2

A large-scale synthetic tabular dataset for pretraining transformer-based in-context learning models for tabular data (similar to TabPFN).

## Overview

| Metric | Value |
|--------|-------|
| Total rows | {rows} |
| Total datasets | {datasets} |
| Shards | {shards} |
| Mean utility AUC | {utility} |
| Format | Parquet (float32) |

## Schema

Each shard is a Parquet file with a fixed-width schema:

- **feat_0** through **feat_63**: Float32 feature columns. Unused slots are NaN.
- **target**: Float32 target variable (classification label or regression target).
- **_source_meta**: JSON string with dataset metadata including:
  - `generator`: Which synthetic generator produced this dataset
  - `task_type`: "binary", "multiclass", or "regression"
  - `n_features`: Number of active features (rest are NaN-padded)
  - `n_classes`: Number of target classes
  - `n_samples`: Number of rows in the original dataset
  - `domain`: Semantic domain (finance, health, etc.)
  - `feature_names`: Original domain-specific column names

## Generators

| Generator | Datasets |
|-----------|----------|
{gen_lines}

## Task Types

| Type | Datasets |
|------|----------|
{task_lines}

## Domains

| Domain | Datasets |
|--------|----------|
{domain_lines}

## Quality Gates

Every generated dataset passes quality gates before inclusion:
- **No constant columns** — all features must vary
- **No all-null columns**
- **Minority class fraction ≥ 5%** for classification
- **Duplicate row fraction ≤ 30%**
- **RF utility AUC ≥ 0.55** — a Random Forest must achieve above-chance cross-validated AUC

Gate failure rate: {stats.get('gate_fail_rate', 0)*100:.1f}%

## Data Augmentation

- **Missingness injection**: ~30% of datasets have random missing values injected
- **Concept drift**: ~20% of datasets have feature distribution shifts

## Usage

```python
from datasets import load_dataset

ds = load_dataset("avewright/tabula-pretraining-corpus-v2", split="train", streaming=True)

for batch in ds.iter(batch_size=512):
    features = batch["feat_0"]  # access individual features
    target = batch["target"]
    meta = batch["_source_meta"]  # JSON metadata string
```

## License

Apache 2.0
"""


# ── Upload ───────────────────────────────────────────────────────────────────

def upload_existing_shards():
    """Upload all existing parquet shards to HuggingFace."""
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    
    # Get list of already-uploaded files
    try:
        existing_files = set()
        for item in api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo="data"):
            existing_files.add(item.rpath)
    except Exception:
        existing_files = set()
    
    # Get local shards
    local_shards = sorted(CORPUS_DIR.glob("train-*.parquet"))
    print(f"Local shards: {len(local_shards)}")
    print(f"Already uploaded: {len(existing_files)}")
    
    to_upload = []
    for shard in local_shards:
        remote_path = f"data/{shard.name}"
        if remote_path not in existing_files:
            to_upload.append(shard)
    
    print(f"Shards to upload: {len(to_upload)}")
    
    if not to_upload:
        print("All shards already uploaded!")
        return
    
    # Upload README first
    stats = corpus_stats()
    card = make_dataset_card(stats)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Update dataset card",
    )
    print("Dataset card uploaded.")
    
    # Upload shards in batches of 5 (to avoid timeout on large uploads)
    BATCH_SIZE = 5
    for i in range(0, len(to_upload), BATCH_SIZE):
        batch = to_upload[i:i + BATCH_SIZE]
        batch_files = [(str(s), f"data/{s.name}") for s in batch]
        
        start = time.time()
        try:
            api.upload_folder(
                folder_path=str(CORPUS_DIR),
                repo_id=REPO_ID,
                repo_type="dataset",
                path_in_repo="data",
                allow_patterns=[s.name for s in batch],
                commit_message=f"Add shards {batch[0].name} - {batch[-1].name}",
            )
            elapsed = time.time() - start
            total_mb = sum(s.stat().st_size for s in batch) / 1e6
            print(f"  Uploaded {len(batch)} shards ({total_mb:.0f} MB) in {elapsed:.0f}s "
                  f"[{i+len(batch)}/{len(to_upload)}]")
        except Exception as e:
            print(f"  FAILED batch starting {batch[0].name}: {e}")
            # Try one at a time
            for shard in batch:
                try:
                    api.upload_file(
                        path_or_fileobj=str(shard),
                        path_in_repo=f"data/{shard.name}",
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        commit_message=f"Add {shard.name}",
                    )
                    print(f"    Uploaded {shard.name} individually")
                except Exception as e2:
                    print(f"    FAILED {shard.name}: {e2}")


def upload_single_shard(shard_path: Path):
    """Upload a single newly-generated shard."""
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    
    try:
        api.upload_file(
            path_or_fileobj=str(shard_path),
            path_in_repo=f"data/{shard_path.name}",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Add {shard_path.name}",
        )
        print(f"  Uploaded {shard_path.name} to HF")
        return True
    except Exception as e:
        print(f"  Failed to upload {shard_path.name}: {e}")
        return False


# ── Data Generation (from run_datagen_hf.py) ────────────────────────────────
# Import the generation infrastructure from existing script

def generate_more_data():
    """Continue generating data and uploading to HF."""
    # Import generation functions from existing script
    # We need to check disk space and generate if there's room
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    print(f"\nDisk: {free_gb:.1f} GB free")
    
    if free_gb < 5:
        print("Not enough disk space for more generation. Upload-only mode.")
        return
    
    # Import and run the generation pipeline
    print("Starting data generation pipeline...")
    
    # Use the existing run_datagen_hf.py infrastructure
    import importlib.util
    spec = importlib.util.spec_from_file_location("datagen", "run_datagen_hf.py")
    datagen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(datagen)
    
    # Override disk limit to leave room
    datagen.MAX_DISK_GB = 45  # tighter limit since we're also uploading
    
    # Run the main generation loop
    datagen.main()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("TABULA CORPUS UPLOAD + GENERATION")
    print(f"Repo: {REPO_ID}")
    print(f"Local corpus: {CORPUS_DIR}")
    print("=" * 70)
    
    # Phase 1: Upload existing data
    print("\n--- Phase 1: Upload existing shards to HuggingFace ---")
    upload_existing_shards()
    
    # Phase 2: Continue generating more data (if disk allows)
    print("\n--- Phase 2: Generate more data and upload ---")
    generate_more_data()


if __name__ == "__main__":
    main()
