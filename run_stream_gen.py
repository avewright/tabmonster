#!/usr/bin/env python3
"""Generate synthetic data and stream-upload to HuggingFace.

Each shard is generated → quality-checked → uploaded to HF → deleted locally.
This allows unlimited generation without filling disk.

Continues from where the previous run left off (reads generation_log.jsonl).
"""
import gc
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, ".")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "avewright/tabula-pretraining-corpus-v2"

# Import the generation infrastructure from run_datagen_hf.py
import importlib.util
spec = importlib.util.spec_from_file_location("datagen", "run_datagen_hf.py")
datagen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(datagen)


def get_next_shard_idx():
    """Determine next shard index from HF repo."""
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    try:
        files = list(api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo="data"))
        parquet_files = [f for f in files if f.path.endswith(".parquet")]
        if not parquet_files:
            return 0
        # Extract max index from filenames like data/train-00134.parquet
        indices = []
        for f in parquet_files:
            name = f.path.split("/")[-1]   # train-00134.parquet
            idx_str = name.replace("train-", "").replace(".parquet", "")
            try:
                indices.append(int(idx_str))
            except ValueError:
                pass
        return max(indices) + 1 if indices else 0
    except Exception as e:
        print(f"Warning: could not check HF repo: {e}")
        return 0


def upload_shard(shard_path: Path) -> bool:
    """Upload one shard to HF and return success status."""
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
        return True
    except Exception as e:
        print(f"  Upload failed: {e}")
        return False


def update_readme(stats: dict):
    """Update HF dataset card with latest stats."""
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    
    rows = f"{stats.get('total_rows', 0):,}"
    shards = stats.get("shards", 0)
    
    card = f"""---
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

Large-scale synthetic tabular data for pretraining transformer-based in-context learning models.

**{rows} rows** across **{shards} shards**.

## Schema

Fixed-width Parquet with 66 columns:
- `feat_0` through `feat_63`: Float32 features (unused slots are NaN)
- `target`: Float32 target variable
- `_source_meta`: JSON metadata (generator, task_type, domain, feature_names, etc.)

## Generators
TreePrior, SCM, GaussianMixture, Polynomial, Regression

## Quality Gates
- No constant/all-null columns
- Minority class ≥ 5%
- Duplicate rows ≤ 30%
- RF utility AUC ≥ 0.55

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{REPO_ID}", split="train", streaming=True)
for batch in ds.iter(batch_size=512):
    features = [batch[f"feat_{{i}}"] for i in range(64)]
    targets = batch["target"]
```
"""
    try:
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Update card: {rows} rows, {shards} shards",
        )
    except Exception as e:
        print(f"  README update failed: {e}")


def main():
    print("=" * 70)
    print("STREAMING DATA GENERATION → HUGGINGFACE")
    print(f"Repo: {REPO_ID}")
    print(f"Workers: {datagen.NUM_WORKERS}")
    print("=" * 70)
    
    # Determine starting shard index
    start_idx = get_next_shard_idx()
    print(f"Next shard index: {start_idx}")
    
    # Read existing log for total row count
    log_path = datagen.LOG_PATH
    total_rows = 0
    total_shards = start_idx
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    total_rows += entry.get("rows", 0)
    print(f"Previous rows: {total_rows:,}")
    
    datagen.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    global_start = time.time()
    shard_idx = start_idx
    shards_this_session = 0
    
    while True:
        # Check disk space (need at least 2GB for generation + upload buffer)
        total_disk, used_disk, free_disk = shutil.disk_usage("/")
        free_gb = free_disk / (1024**3)
        if free_gb < 2:
            print(f"Low disk ({free_gb:.1f} GB free). Cleaning up...")
            # Delete any leftover local shards
            for old in datagen.CORPUS_DIR.glob("train-*.parquet"):
                old.unlink()
            free_gb = shutil.disk_usage("/")[2] / (1024**3)
            if free_gb < 2:
                print("Still not enough disk. Exiting.")
                break
        
        base_seed = shard_idx * 1_000_000
        elapsed = time.time() - global_start
        rate = total_rows / max(elapsed, 1)
        
        print(f"\n{'='*60}")
        print(f"SHARD {shard_idx:05d} | Total: {total_rows:,} rows | "
              f"Session: {shards_this_session} shards | {rate:,.0f} rows/s avg")
        
        try:
            # Generate shard
            gen_start = time.time()
            path, n_rows, stats = datagen.build_shard(shard_idx, base_seed)
            gen_time = time.time() - gen_start
            
            print(f"  Generated {n_rows:,} rows in {gen_time:.0f}s "
                  f"({n_rows/gen_time:,.0f} rows/s)")
            print(f"  Generators: {stats.get('generators', {})}")
            print(f"  Quality: mean_utility={stats.get('mean_utility_auc', 0):.3f} | "
                  f"gate_fails={stats.get('gate_fails', 0)} | errors={stats.get('errors', 0)}")
            
            # Upload to HF
            upload_start = time.time()
            success = upload_shard(path)
            upload_time = time.time() - upload_start
            
            if success:
                size_mb = path.stat().st_size / 1e6
                print(f"  Uploaded {size_mb:.0f} MB in {upload_time:.0f}s "
                      f"({size_mb/max(upload_time,1):.0f} MB/s)")
                # Delete local copy
                path.unlink()
                print(f"  Deleted local copy (freed {size_mb:.0f} MB)")
            else:
                print(f"  Upload failed — keeping local copy at {path}")
            
            # Log
            total_rows += n_rows
            total_shards = shard_idx + 1
            shards_this_session += 1
            
            log_entry = {
                "shard_idx": shard_idx,
                "n_rows": n_rows,
                "rows": n_rows,
                "datasets": stats.get("datasets", 0),
                "gate_fails": stats.get("gate_fails", 0),
                "errors": stats.get("errors", 0),
                "size_mb": round(stats.get("size_mb", 0), 1),
                "generators": stats.get("generators", {}),
                "task_types": stats.get("task_types", {}),
                "domains": stats.get("domains", {}),
                "mean_utility_auc": round(stats.get("mean_utility_auc", 0), 4),
                "gen_time_s": round(gen_time, 1),
                "upload_time_s": round(upload_time, 1),
                "total_rows": total_rows,
                "corpus_shards": total_shards,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            # Update README every 10 shards
            if shards_this_session % 10 == 0:
                update_readme({"total_rows": total_rows, "shards": total_shards})
                print(f"  Updated HF README: {total_rows:,} rows, {total_shards} shards")
            
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            # Try to clean up any partial shard
            partial = datagen.CORPUS_DIR / f"train-{shard_idx:05d}.parquet"
            if partial.exists():
                partial.unlink()
        
        shard_idx += 1
        gc.collect()
    
    # Final stats
    elapsed = time.time() - global_start
    print(f"\n{'='*70}")
    print(f"SESSION COMPLETE")
    print(f"Generated: {shards_this_session} shards this session")
    print(f"Total corpus: {total_rows:,} rows in {total_shards} shards")
    print(f"Elapsed: {elapsed/3600:.1f} hours ({total_rows/max(elapsed,1):,.0f} rows/s)")
    print(f"{'='*70}")
    
    # Final README update
    update_readme({"total_rows": total_rows, "shards": total_shards})


if __name__ == "__main__":
    main()
