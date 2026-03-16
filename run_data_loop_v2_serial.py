#!/usr/bin/env python3
"""Continuous data discovery → preparation → corpus building → HF upload loop.

Discovers real tabular datasets from HuggingFace, OpenML, and PMLB via the
existing autodiscovery pipeline, prepares each through the leakage-safe
prep pipeline (schema.json, feature_transforms.json, train/val/test splits),
encodes into the fixed-width parquet pretraining format, quality-gates,
assembles shards, and pushes to HuggingFace.

This runs forever:
  1. Discover real datasets (HF, OpenML, PMLB)
  2. Prepare each through tabula.data.prep.prepare_dataset()
  3. Encode prepared train split to fixed-width parquet (feat_0..feat_63, target, _source_meta)
  4. Quality-gate each dataset
  5. Accumulate into ~2M row shards
  6. Upload each shard to HF and delete local copy
  7. Mix in synthetic data to fill shard gaps
  8. Repeat

Corpus format (matches avewright/tabula-pretraining-corpus-v2):
  feat_0..feat_63 : Float32 feature columns (unused = NaN)
  target          : Float32 target variable
  _source_meta    : JSON string with dataset metadata
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from tabula.data.env import load_repo_env_file

# ── Load token from .env (project convention) ────────────────────────
_env = load_repo_env_file()
HF_TOKEN = (
    _env.get("HF_TOKEN")
    or _env.get("HUGGINGFACE_HUB_TOKEN")
    or os.environ.get("HF_TOKEN")
    or ""
)

# ── Config ────────────────────────────────────────────────────────────
HF_REPO = "avewright/tabula-pretraining-corpus-v2"
CORPUS_DIR = Path("corpus/real_data/data")
LOG_PATH = Path("corpus/real_data/processing_log.jsonl")
PREPARED_ROOT = Path("data/processed_loop")
RAW_ROOT = Path("data/raw_loop")
REGISTRY_FILE = Path("artifacts/loop_discovery_registry.json")

MAX_FEATURES = 64
FEAT_COLS = [f"feat_{i}" for i in range(MAX_FEATURES)]
ROWS_PER_SHARD = 2_000_000
MAX_DISK_GB = 45
MAX_ROWS_PER_DATASET = 200_000
MIN_ROWS = 50


def log_entry(entry: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def disk_used_gb() -> float:
    total, used, free = shutil.disk_usage("/")
    return used / (1024**3)


# ── Shard indexing from HF repo ───────────────────────────────────────

def get_next_shard_idx() -> int:
    """Determine the next shard index by inspecting the HF repo."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN or None)
        files = list(api.list_repo_tree(HF_REPO, repo_type="dataset", path_in_repo="data"))
        max_idx = -1
        for f in files:
            name = f.path.split("/")[-1]
            if name.startswith("train-") and name.endswith(".parquet"):
                try:
                    idx = int(name.replace("train-", "").replace(".parquet", ""))
                    max_idx = max(max_idx, idx)
                except ValueError:
                    pass
        return max_idx + 1
    except Exception as e:
        print(f"  [WARN] Could not read HF repo: {e}")
        return 300  # Safe default past existing synthetic shards


# ── Discovery (uses existing autodiscovery pipeline) ──────────────────

def discover_new_datasets(max_new: int = 30) -> list:
    """Run the autodiscovery pipeline to find new real datasets.
    
    Sources tried in order: PMLB (fast), OpenML (fast), HF (slow, large downloads).
    """
    from tabula.data.autodiscovery import run_discovery_pass
    try:
        records = run_discovery_pass(
            sources=["pmlb", "openml", "hf"],
            output_root=str(RAW_ROOT),
            registry_file=str(REGISTRY_FILE),
            max_new=max_new,
            hf_limit=20,
            openml_limit=60,
            pmlb_limit=80,
        )
        return [r for r in records if r.status == "ok"]
    except Exception as e:
        print(f"  [WARN] Discovery pass failed: {e}")
        traceback.print_exc()
        return []


# ── Preparation (uses existing prep pipeline) ─────────────────────────

def prepare_one_dataset(dataset_id: str) -> dict | None:
    """Run prepare_dataset and return info dict, or None on failure."""
    from tabula.data.prep import prepare_dataset
    try:
        result = prepare_dataset(
            dataset_id=dataset_id,
            raw_root=str(RAW_ROOT),
            processed_root=str(PREPARED_ROOT),
            seed=42,
            max_rows=MAX_ROWS_PER_DATASET,
            drop_identifier_columns=True,
            feature_engineering=True,
        )
        return {
            "dataset_id": result.dataset_id,
            "processed_dir": result.processed_dir,
            "train_rows": result.train_rows,
            "val_rows": result.val_rows,
            "test_rows": result.test_rows,
            "target_column": result.target_column,
            "numeric_columns": result.numeric_columns,
            "categorical_columns": result.categorical_columns,
        }
    except Exception as e:
        print(f"    [WARN] Prepare failed for {dataset_id}: {e}")
        return None


# ── Encode prepared dataset to fixed-width parquet format ─────────────

def encode_prepared_to_fixed_width(prep_info: dict) -> pd.DataFrame | None:
    """Read a prepared dataset's train.csv and encode to fixed-width format.

    Uses the schema.json and feature_transforms.json already saved by prep.
    """
    processed_dir = Path(prep_info["processed_dir"])
    train_path = processed_dir / "train.csv"
    schema_path = processed_dir / "schema.json"
    card_path = processed_dir / "dataset_card.json"

    if not train_path.exists():
        print(f"    No train.csv at {train_path}")
        return None

    try:
        df = pd.read_csv(train_path)
    except Exception as e:
        print(f"    Failed to read train.csv: {e}")
        return None

    if len(df) < MIN_ROWS:
        print(f"    Too few rows: {len(df)}")
        return None

    target_col = prep_info["target_column"]
    if target_col not in df.columns:
        print(f"    Target column '{target_col}' not in {list(df.columns)[:10]}")
        return None

    # Read schema for metadata
    schema = {}
    if schema_path.exists():
        with open(schema_path) as f:
            schema = json.load(f)

    card = {}
    if card_path.exists():
        with open(card_path) as f:
            card = json.load(f)

    # Separate target
    target = df[target_col].copy()
    feature_cols = [c for c in df.columns if c != target_col]

    # Encode all features to float32
    encoded_cols = []
    feature_names = []

    for col in feature_cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            encoded_cols.append(series.astype(np.float32).values)
            feature_names.append(col)
        elif series.dtype == object or hasattr(series.dtype, "categories"):
            # Categorical → label encode
            n_unique = series.nunique()
            if n_unique > 200 and n_unique > len(series) * 0.5:
                continue  # Skip high-cardinality free-text
            cats = series.astype(str).fillna("_missing_")
            codes = pd.Categorical(cats).codes.astype(np.float32)
            encoded_cols.append(codes)
            feature_names.append(col)
        else:
            try:
                numeric = pd.to_numeric(series, errors="coerce")
                if numeric.notna().sum() > len(series) * 0.3:
                    encoded_cols.append(numeric.astype(np.float32).values)
                    feature_names.append(col)
            except Exception:
                pass

    if len(encoded_cols) < 2:
        print(f"    Only {len(encoded_cols)} encodable features")
        return None

    actual_n = min(len(encoded_cols), MAX_FEATURES)
    encoded_cols = encoded_cols[:actual_n]
    feature_names = feature_names[:actual_n]

    # Encode target
    task_type = card.get("task_type", _infer_task_type(target))
    if task_type in ("binary", "multiclass"):
        target_cats = target.astype(str).fillna("_missing_")
        target_encoded = pd.Categorical(target_cats).codes.astype(np.float32)
    else:
        target_encoded = pd.to_numeric(target, errors="coerce").astype(np.float32).values

    valid_mask = ~np.isnan(target_encoded)
    if valid_mask.sum() < MIN_ROWS:
        print(f"    Only {valid_mask.sum()} valid target rows")
        return None

    n_rows = int(valid_mask.sum())
    target_encoded = target_encoded[valid_mask]

    padded = np.full((n_rows, MAX_FEATURES), np.nan, dtype=np.float32)
    for i, col_vals in enumerate(encoded_cols):
        padded[:, i] = col_vals[valid_mask]

    out = pd.DataFrame(padded, columns=FEAT_COLS)
    out["target"] = target_encoded

    n_classes = int(np.unique(target_encoded[~np.isnan(target_encoded)]).shape[0])

    source_meta = {
        "generator": "real_data",
        "task_type": task_type,
        "n_features": actual_n,
        "n_classes": n_classes,
        "n_samples": n_rows,
        "domain": "real",
        "feature_names": feature_names,
        "source_repo": card.get("external_ref", prep_info["dataset_id"]),
        "dataset_id": prep_info["dataset_id"],
        "seed": 42,
        "method": "real_data_prepared",
        "missingness_rate": float(np.isnan(padded[:, :actual_n]).mean()),
        "concept_drift": False,
        "utility_auc": 0.0,
    }
    out["_source_meta"] = json.dumps(source_meta)
    return out


def _infer_task_type(series: pd.Series) -> str:
    n_unique = series.nunique()
    if n_unique <= 2:
        return "binary"
    elif n_unique <= 20 and n_unique < len(series) * 0.05:
        return "multiclass"
    return "regression"


# ── Quality gate ──────────────────────────────────────────────────────

def quality_gate(df: pd.DataFrame, meta: dict) -> tuple[bool, str]:
    n = len(df)
    if n < MIN_ROWS:
        return False, f"too few rows: {n}"

    actual_n = meta["n_features"]
    feat_arr = df[FEAT_COLS[:actual_n]].values

    for i in range(actual_n):
        col = feat_arr[:, i]
        non_nan = col[~np.isnan(col)]
        if len(non_nan) <= 1 or np.nanstd(non_nan) < 1e-10:
            return False, f"constant/empty col {i}"

    target = df["target"].values
    if meta["task_type"] in ("binary", "multiclass"):
        uniq = np.unique(target[~np.isnan(target)])
        if len(uniq) < 2:
            return False, "target < 2 classes"
        _, counts = np.unique(target[~np.isnan(target)], return_counts=True)
        minority = counts.min() / counts.sum()
        if minority < 0.01:
            return False, f"minority too small: {minority:.4f}"

    return True, "ok"


# ── RF utility ────────────────────────────────────────────────────────

def compute_utility(df: pd.DataFrame, meta: dict) -> float:
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        actual_n = meta["n_features"]
        X = df[FEAT_COLS[:actual_n]].values
        y = df["target"].values

        if len(X) > 5000:
            idx = np.random.default_rng(42).choice(len(X), 5000, replace=False)
            X, y = X[idx], y[idx]

        X = np.nan_to_num(X, nan=0.0)

        if meta["task_type"] == "regression":
            clf = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=0, n_jobs=2)
            scores = cross_val_score(clf, X, y, cv=3, scoring="r2", n_jobs=1)
        else:
            clf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=0, n_jobs=2)
            try:
                scores = cross_val_score(clf, X, y, cv=3, scoring="roc_auc_ovr_weighted", n_jobs=1)
            except Exception:
                scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy", n_jobs=1)
        return float(np.mean(scores))
    except Exception:
        return 0.0


# ── Upload shard to HF ───────────────────────────────────────────────

def upload_shard(shard_path: Path) -> bool:
    if not HF_TOKEN:
        print("  [WARN] No HF_TOKEN, skipping upload")
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=str(shard_path),
            path_in_repo=f"data/{shard_path.name}",
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"Add real-data shard {shard_path.name}",
        )
        size_mb = shard_path.stat().st_size / (1024 * 1024)
        print(f"  Uploaded {shard_path.name} ({size_mb:.0f} MB) in {time.time()-t0:.0f}s")
        shard_path.unlink()
        return True
    except Exception as e:
        print(f"  [WARN] Upload failed: {e}")
        return False


# ── Save shard ────────────────────────────────────────────────────────

def save_shard(collected: list[pd.DataFrame], shard_idx: int) -> Path:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CORPUS_DIR / f"train-{shard_idx:05d}.parquet"
    combined = pd.concat(collected, ignore_index=True)
    combined.to_parquet(out_path, index=False, engine="pyarrow")
    del combined
    gc.collect()
    return out_path


# ── Synthetic data filler ─────────────────────────────────────────────

def generate_synthetic_batch(n_datasets: int = 30, seed_base: int = 0) -> list[pd.DataFrame]:
    """Generate synthetic datasets in fixed-width format to fill shard gaps."""
    try:
        from tabula.data.synthetic import (
            TreePriorGenerator,
            GaussianMixtureGenerator,
            PolynomialGenerator,
            SCMGenerator,
            RegressionSyntheticGenerator,
        )
    except ImportError:
        return []

    generators = [
        ("TreePrior", TreePriorGenerator),
        ("GaussianMixture", GaussianMixtureGenerator),
        ("Polynomial", PolynomialGenerator),
        ("SCM", SCMGenerator),
        ("Regression", RegressionSyntheticGenerator),
    ]
    DOMAINS = ["finance", "health", "ecommerce", "iot", "hr",
               "science", "logistics", "education", "manufacturing",
               "environment", "telecom"]

    collected = []
    rng = np.random.default_rng(seed_base)

    for i in range(n_datasets):
        try:
            seed = seed_base + i
            n_features = int(rng.integers(4, MAX_FEATURES + 1))
            n_samples = int(rng.integers(500, 50_000))
            task_type = str(rng.choice(["binary", "binary", "multiclass", "regression"]))
            n_classes = 2 if task_type == "binary" else int(rng.integers(3, 10)) if task_type == "multiclass" else 2

            gen_name, gen_cls = generators[i % len(generators)]
            if gen_name == "Regression":
                gen = gen_cls(n_samples=n_samples, n_features=n_features)
            else:
                gen = gen_cls(n_samples=n_samples, n_features=n_features, n_classes=n_classes)

            df, meta = gen.generate(seed=seed)
            feature_cols = [c for c in df.columns if c != "target"]
            actual_n = min(len(feature_cols), MAX_FEATURES)
            feat_vals = df[feature_cols[:actual_n]].values.astype(np.float32)
            target_vals = df["target"].values.astype(np.float32)
            n_rows = len(df)

            if rng.random() < 0.3:
                rate = float(rng.uniform(0.02, 0.15))
                mask = rng.random(size=feat_vals.shape) < rate
                feat_vals[mask] = np.nan

            padded = np.full((n_rows, MAX_FEATURES), np.nan, dtype=np.float32)
            padded[:, :actual_n] = feat_vals

            out = pd.DataFrame(padded, columns=FEAT_COLS)
            out["target"] = target_vals
            out["_source_meta"] = json.dumps({
                "generator": gen_name,
                "task_type": meta.task_type,
                "n_features": actual_n,
                "n_classes": meta.n_classes,
                "n_samples": n_rows,
                "domain": str(rng.choice(DOMAINS)),
                "feature_names": [f"feat_{j}" for j in range(actual_n)],
                "seed": seed,
                "method": gen_name,
                "missingness_rate": 0.0,
                "concept_drift": False,
                "utility_auc": 0.0,
            })
            collected.append(out)
        except Exception:
            continue

    return collected


# ── Clean up disk ─────────────────────────────────────────────────────

def cleanup_raw_data():
    """Remove raw downloads to save disk space."""
    for d in [RAW_ROOT, PREPARED_ROOT]:
        if d.exists():
            total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            if total > 2 * 1024**3:  # >2 GB
                print(f"  Cleaning {d} ({total/1e9:.1f} GB)...")
                shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)


# ── Main Loop ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CONTINUOUS DATA LOOP: DISCOVER → PREPARE → ENCODE → UPLOAD")
    print(f"HF Repo: {HF_REPO}")
    print(f"HF Token: {'set (write)' if HF_TOKEN else 'NOT SET'}")
    print(f"Max features: {MAX_FEATURES}")
    print(f"Rows per shard: {ROWS_PER_SHARD:,}")
    print(f"Disk: {disk_used_gb():.1f} GB used")
    print("=" * 70)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)

    shard_idx = get_next_shard_idx()
    print(f"Starting shard index: {shard_idx}")

    total_real_ok = 0
    total_real_fail = 0
    total_rows = 0
    total_shards = 0
    global_start = time.time()
    synthetic_seed = 20_000_000

    # Accumulate datasets for current shard
    shard_buf: list[pd.DataFrame] = []
    shard_rows = 0

    round_num = 0
    while True:
        round_num += 1
        elapsed_min = (time.time() - global_start) / 60
        print(f"\n{'='*70}")
        print(f"ROUND {round_num} | real_ok={total_real_ok} fail={total_real_fail} | "
              f"rows={total_rows:,} | shards={total_shards} | "
              f"disk={disk_used_gb():.1f}GB | elapsed={elapsed_min:.0f}min")

        # Disk check
        if disk_used_gb() >= MAX_DISK_GB:
            print("Disk pressure — uploading pending shards and cleaning up...")
            for f in sorted(CORPUS_DIR.glob("train-*.parquet")):
                upload_shard(f)
            cleanup_raw_data()
            if disk_used_gb() >= MAX_DISK_GB:
                print("Still over limit. Waiting 60s...")
                time.sleep(60)
                continue

        # ── Phase 1: Discover real datasets ───────────────────────
        print("\n--- Phase 1: Discovery ---")
        ok_records = discover_new_datasets(max_new=20)
        print(f"  Discovered {len(ok_records)} new validated datasets")

        # ── Phase 2: Prepare and encode each ──────────────────────
        if ok_records:
            print("\n--- Phase 2: Prepare & Encode ---")
            for rec in ok_records:
                print(f"\n  {rec.dataset_id} ({rec.source}, {rec.n_rows}r x {rec.n_cols}c)")

                # Prepare through the standard pipeline
                prep_info = prepare_one_dataset(rec.dataset_id)
                if prep_info is None:
                    total_real_fail += 1
                    log_entry({"type": "dataset", "id": rec.dataset_id,
                               "status": "prep_fail", "source": rec.source})
                    continue

                print(f"    Prepared: {prep_info['train_rows']} train rows, "
                      f"{len(prep_info['numeric_columns'])} num, "
                      f"{len(prep_info['categorical_columns'])} cat")

                # Encode to fixed-width parquet format
                encoded_df = encode_prepared_to_fixed_width(prep_info)
                if encoded_df is None:
                    total_real_fail += 1
                    log_entry({"type": "dataset", "id": rec.dataset_id,
                               "status": "encode_fail", "source": rec.source})
                    continue

                # Quality gate
                meta = json.loads(encoded_df["_source_meta"].iloc[0])
                passed, reason = quality_gate(encoded_df, meta)
                if not passed:
                    total_real_fail += 1
                    print(f"    Gate fail: {reason}")
                    log_entry({"type": "dataset", "id": rec.dataset_id,
                               "status": "gate_fail", "reason": reason})
                    continue

                # Utility check
                utility = compute_utility(encoded_df, meta)
                meta["utility_auc"] = utility
                encoded_df["_source_meta"] = json.dumps(meta)

                total_real_ok += 1
                shard_buf.append(encoded_df)
                shard_rows += len(encoded_df)
                total_rows += len(encoded_df)

                print(f"    OK: {len(encoded_df):,} rows, {meta['n_features']} feats, "
                      f"{meta['task_type']}, utility={utility:.3f}")
                log_entry({"type": "dataset", "id": rec.dataset_id,
                           "status": "ok", "source": rec.source,
                           "rows": len(encoded_df), "features": meta["n_features"],
                           "task_type": meta["task_type"], "utility": utility})

                # Flush shard if full
                if shard_rows >= ROWS_PER_SHARD:
                    path = save_shard(shard_buf, shard_idx)
                    size_mb = path.stat().st_size / (1024 * 1024)
                    print(f"\n  >>> SHARD {shard_idx:05d}: {shard_rows:,} rows, "
                          f"{len(shard_buf)} datasets, {size_mb:.0f} MB")
                    log_entry({"type": "shard", "idx": shard_idx,
                               "rows": shard_rows, "datasets": len(shard_buf),
                               "size_mb": round(size_mb, 1)})
                    upload_shard(path)
                    total_shards += 1
                    shard_idx += 1
                    shard_buf = []
                    shard_rows = 0
                    gc.collect()
                    cleanup_raw_data()

        # ── Phase 3: Fill with synthetic data ─────────────────────
        if shard_rows > 0 and shard_rows < ROWS_PER_SHARD:
            remaining = ROWS_PER_SHARD - shard_rows
            n_synth = max(5, min(50, remaining // 10_000))
            print(f"\n--- Phase 3: Filling with ~{n_synth} synthetic datasets "
                  f"({remaining:,} rows needed) ---")
            synth = generate_synthetic_batch(n_synth, synthetic_seed)
            synthetic_seed += n_synth
            for df in synth:
                shard_buf.append(df)
                shard_rows += len(df)
                total_rows += len(df)

            if shard_rows >= ROWS_PER_SHARD:
                path = save_shard(shard_buf, shard_idx)
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"\n  >>> SHARD {shard_idx:05d}: {shard_rows:,} rows, "
                      f"{len(shard_buf)} datasets, {size_mb:.0f} MB")
                log_entry({"type": "shard", "idx": shard_idx,
                           "rows": shard_rows, "datasets": len(shard_buf),
                           "size_mb": round(size_mb, 1)})
                upload_shard(path)
                total_shards += 1
                shard_idx += 1
                shard_buf = []
                shard_rows = 0
                gc.collect()

        # If no new datasets found and shard is empty, generate pure synthetic
        if not ok_records and shard_rows == 0:
            print("\n--- No new real data. Generating full synthetic shard ---")
            synth = generate_synthetic_batch(50, synthetic_seed)
            synthetic_seed += 50
            for df in synth:
                shard_buf.append(df)
                shard_rows += len(df)
                total_rows += len(df)

        print(f"\n  Round {round_num} done. Buf: {shard_rows:,}/{ROWS_PER_SHARD:,} rows, "
              f"{len(shard_buf)} datasets")

        # Brief cooldown between rounds
        time.sleep(2)


if __name__ == "__main__":
    main()
