#!/usr/bin/env python3
"""Continuous real-data discovery → processing → HF upload loop.

Discovers real tabular datasets from HuggingFace, processes them into the
fixed-width parquet pretraining format (feat_0..feat_63, target, _source_meta),
performs quality gates, assembles shards, and pushes to HuggingFace.

Format (matching existing corpus at avewright/tabula-pretraining-corpus-v2):
  - feat_0 through feat_63: Float32 feature columns. Unused slots are NaN.
  - target: Float32 target variable.
  - _source_meta: JSON string with dataset metadata.

This script loops forever. It:
  1. Searches HF for tabular datasets (classification + regression)
  2. Downloads each, encodes features to float32
  3. Quality-gates each dataset
  4. Accumulates into ~2M row shards
  5. Uploads each shard to HF and deletes local copy
  6. Tracks everything in a log file
"""
from __future__ import annotations

import gc
import io
import json
import os
import shutil
import sys
import time
import traceback
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from itertools import islice

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

# ── Config ────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "avewright/tabula-pretraining-corpus-v2"
CORPUS_DIR = Path("corpus/real_data/data")
LOG_PATH = Path("corpus/real_data/processing_log.jsonl")
PROCESSED_PATH = Path("corpus/real_data/processed_datasets.json")
MAX_FEATURES = 64
FEAT_COLS = [f"feat_{i}" for i in range(MAX_FEATURES)]
ROWS_PER_SHARD = 2_000_000
MAX_DISK_GB = 45
MAX_ROWS_PER_DATASET = 200_000  # Cap extremely large datasets
MIN_ROWS = 50  # Skip tiny datasets
MIN_FEATURES = 2  # Need at least 2 features

# Search categories for discovering datasets
SEARCH_CATEGORIES = [
    "task_categories:tabular-classification",
    "task_categories:tabular-regression",
]

# Known good search terms for finding tabular data
SEARCH_TERMS = [
    None,  # no filter, just sort by downloads
    "tabular",
    "classification",
    "regression",
    "census",
    "fraud",
    "churn",
    "credit",
    "housing",
    "medical",
    "health",
    "finance",
    "insurance",
    "marketing",
    "customer",
    "income",
    "loan",
    "diabetes",
    "heart",
    "cancer",
    "wine",
    "iris",
    "titanic",
    "weather",
    "salary",
    "stock",
    "energy",
    "covid",
    "bank",
    "employee",
    "car",
    "price prediction",
    "default",
    "click",
    "recommendation",
    "air quality",
    "retail",
    "supply chain",
]

# Datasets to skip (too large, not tabular, broken, etc.)
SKIP_DATASETS = {
    "avewright/tabula-pretraining-corpus-v2",  # our own repo
    "laion/laion2B-meta",  # too large, image metadata
    "ccdv/pubmed-summarization",  # text
    "datajuicer/OAG-challenge-paperwithcode",  # text
    "imodels/reviews",  # text
    "alexodavies/cleantablib",  # meta-dataset
    "torchgeo/CropClimateX",  # geo tiles
    "FraDra/GEO_satellite_maneuvers",  # special format
    "phanerozoic/qiskit-calibration-drift",  # quantum computing
}

# Common target column names by priority
TARGET_NAMES = [
    "target", "label", "class", "y", "outcome", "response", "output",
    "labels", "classes", "is_fraud", "fraud", "churn", "default",
    "income", "price", "salary", "survived", "diagnosis", "quality",
    "rating", "score", "Exited", "Churn", "Attrition", "Category",
    "species", "variety", "type", "status",
]


def load_processed_datasets() -> set[str]:
    """Load set of already processed dataset repo_ids."""
    if PROCESSED_PATH.exists():
        with open(PROCESSED_PATH) as f:
            return set(json.load(f))
    return set()


def save_processed_dataset(repo_id: str):
    """Mark a dataset as processed."""
    processed = load_processed_datasets()
    processed.add(repo_id)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_PATH, "w") as f:
        json.dump(sorted(processed), f, indent=2)


def log_entry(entry: dict):
    """Append a log entry."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def disk_used_gb() -> float:
    total, used, free = shutil.disk_usage("/")
    return used / (1024**3)


# ── Dataset Discovery ─────────────────────────────────────────────────

def discover_datasets(processed: set[str], batch_size: int = 30) -> list[dict]:
    """Search HuggingFace for tabular datasets we haven't processed yet."""
    from huggingface_hub import HfApi
    api = HfApi()

    candidates = {}

    for category in SEARCH_CATEGORIES:
        for term in SEARCH_TERMS:
            try:
                results = list(api.list_datasets(
                    filter=category,
                    search=term,
                    sort="downloads",
                    limit=batch_size,
                ))
                for r in results:
                    if r.id not in processed and r.id not in SKIP_DATASETS:
                        if r.id not in candidates:
                            candidates[r.id] = {
                                "repo_id": r.id,
                                "downloads": r.downloads or 0,
                                "tags": list(r.tags or []),
                            }
            except Exception as e:
                print(f"  [WARN] Search failed ({category}, {term}): {e}")
                continue

    # Sort by downloads (most popular first = most likely to be good)
    result = sorted(candidates.values(), key=lambda x: -x["downloads"])
    return result


# ── Guess Target Column ──────────────────────────────────────────────

def guess_target_column(df: pd.DataFrame) -> str | None:
    """Guess the target column from a DataFrame."""
    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in cols}

    # Check known target names
    for name in TARGET_NAMES:
        if name in cols:
            return name
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]

    # Check for columns containing target-like words
    for col in cols:
        cl = col.lower()
        for kw in ["target", "label", "class", "output", "y_"]:
            if kw in cl:
                return col

    # Last column is often the target
    if len(cols) > 2:
        last = cols[-1]
        # Only if it's not clearly an ID or feature
        if not any(kw in last.lower() for kw in ["id", "index", "name", "date", "time"]):
            return last

    return None


def infer_task_type(series: pd.Series) -> str:
    """Infer whether a target column is binary, multiclass, or regression."""
    n_unique = series.nunique()
    if n_unique <= 2:
        return "binary"
    elif n_unique <= 20 and n_unique < len(series) * 0.05:
        return "multiclass"
    else:
        return "regression"


# ── Encode Dataset to Fixed-Width Format ──────────────────────────────

def encode_dataset_to_fixed_width(
    df: pd.DataFrame,
    target_col: str,
    repo_id: str,
) -> tuple[pd.DataFrame | None, dict]:
    """Convert a real dataset to the fixed-width parquet format.

    Returns (fixed_width_df, metadata) or (None, error_info).
    """
    # Separate target
    if target_col not in df.columns:
        return None, {"error": f"target '{target_col}' not in columns"}

    target = df[target_col].copy()
    features = df.drop(columns=[target_col])

    # Drop columns that are clearly identifiers
    drop_cols = []
    for col in features.columns:
        cl = col.lower()
        if any(kw in cl for kw in ["unnamed:", "index", "_id"]):
            # Only drop if it looks like an ID (monotonic or unique)
            if features[col].nunique() > len(features) * 0.9:
                drop_cols.append(col)
    if drop_cols:
        features = features.drop(columns=drop_cols)

    # Drop columns with all NaN
    features = features.dropna(axis=1, how="all")

    # Drop constant columns
    nunique = features.nunique()
    features = features.loc[:, nunique > 1]

    if len(features.columns) < MIN_FEATURES:
        return None, {"error": f"only {len(features.columns)} features after cleanup"}

    # Encode features to float32
    encoded_cols = []
    feature_names = []
    feature_types = []

    for col in features.columns:
        series = features[col]

        # Try numeric first
        if pd.api.types.is_numeric_dtype(series):
            vals = series.astype(np.float32).values
            encoded_cols.append(vals)
            feature_names.append(col)
            feature_types.append("numeric")
            continue

        # Try to convert to numeric
        try:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().sum() > len(series) * 0.5:
                encoded_cols.append(numeric.astype(np.float32).values)
                feature_names.append(col)
                feature_types.append("numeric_coerced")
                continue
        except Exception:
            pass

        # Boolean
        if pd.api.types.is_bool_dtype(series):
            encoded_cols.append(series.astype(np.float32).values)
            feature_names.append(col)
            feature_types.append("boolean")
            continue

        # Categorical / string → label encode
        if series.dtype == object or pd.api.types.is_categorical_dtype(series):
            # Skip high-cardinality text columns (likely free text)
            n_unique = series.nunique()
            if n_unique > 100 and n_unique > len(series) * 0.5:
                continue  # Skip, likely free text or IDs

            # Label encode
            cats = series.astype(str).fillna("_missing_")
            codes = pd.Categorical(cats).codes.astype(np.float32)
            encoded_cols.append(codes)
            feature_names.append(col)
            feature_types.append("categorical")
            continue

        # Datetime → extract numeric features
        if pd.api.types.is_datetime64_any_dtype(series):
            try:
                ts = series.astype(np.int64) / 1e9  # Unix timestamp
                encoded_cols.append(ts.astype(np.float32).values)
                feature_names.append(f"{col}_timestamp")
                feature_types.append("datetime")
            except Exception:
                pass
            continue

        # Skip other types
        continue

    if len(encoded_cols) < MIN_FEATURES:
        return None, {"error": f"only {len(encoded_cols)} encodable features"}

    # Limit to MAX_FEATURES
    actual_n = min(len(encoded_cols), MAX_FEATURES)
    encoded_cols = encoded_cols[:actual_n]
    feature_names = feature_names[:actual_n]
    feature_types = feature_types[:actual_n]

    # Encode target
    task_type = infer_task_type(target)
    if task_type in ("binary", "multiclass"):
        # Label encode target
        target_cats = target.astype(str).fillna("_missing_")
        target_encoded = pd.Categorical(target_cats).codes.astype(np.float32)
    else:
        # Regression: try numeric
        target_encoded = pd.to_numeric(target, errors="coerce").astype(np.float32).values

    # Drop rows where target is NaN
    valid_mask = ~np.isnan(target_encoded)
    if valid_mask.sum() < MIN_ROWS:
        return None, {"error": f"only {valid_mask.sum()} valid target rows"}

    n_rows = int(valid_mask.sum())
    target_encoded = target_encoded[valid_mask]

    # Build fixed-width array
    padded = np.full((n_rows, MAX_FEATURES), np.nan, dtype=np.float32)
    for i, col_vals in enumerate(encoded_cols):
        padded[:, i] = col_vals[valid_mask]

    # Build DataFrame
    out = pd.DataFrame(padded, columns=FEAT_COLS)
    out["target"] = target_encoded

    # Count classes
    n_classes = int(np.unique(target_encoded[~np.isnan(target_encoded)]).shape[0])

    meta = {
        "generator": "real_data",
        "task_type": task_type,
        "n_features": actual_n,
        "n_classes": n_classes,
        "n_samples": n_rows,
        "domain": "real",
        "feature_names": feature_names,
        "feature_types": feature_types,
        "source_repo": repo_id,
        "seed": 0,
        "method": "real_data_hf",
        "missingness_rate": float(np.isnan(padded[:, :actual_n]).mean()),
        "concept_drift": False,
        "utility_auc": 0.0,
    }
    out["_source_meta"] = json.dumps(meta)

    return out, meta


# ── Quality Gates ─────────────────────────────────────────────────────

def quality_gate(df: pd.DataFrame, meta: dict) -> tuple[bool, str]:
    """Check if processed dataset passes quality gates."""
    n = len(df)
    if n < MIN_ROWS:
        return False, f"too few rows: {n}"

    actual_n = meta["n_features"]
    feat_arr = df[FEAT_COLS[:actual_n]].values

    # Check for constant columns
    for i in range(actual_n):
        col = feat_arr[:, i]
        non_nan = col[~np.isnan(col)]
        if len(non_nan) <= 1:
            return False, f"constant or empty col {i}"
        if np.nanstd(non_nan) < 1e-10:
            return False, f"near-constant col {i}"

    # Target checks
    target = df["target"].values
    task_type = meta["task_type"]
    if task_type in ("binary", "multiclass"):
        uniq = np.unique(target[~np.isnan(target)])
        if len(uniq) < 2:
            return False, "target has < 2 classes"
        _, counts = np.unique(target[~np.isnan(target)], return_counts=True)
        minority = counts.min() / counts.sum()
        if minority < 0.01:
            return False, f"minority class too small: {minority:.4f}"

    # Duplicate check
    n_dupes = n - len(df.drop(columns=["_source_meta"]).drop_duplicates())
    dupe_rate = n_dupes / n
    if dupe_rate > 0.5:
        return False, f"too many duplicates: {dupe_rate:.2%}"

    return True, "ok"


# ── RF Utility Check ─────────────────────────────────────────────────

def compute_utility(df: pd.DataFrame, meta: dict) -> float:
    """Quick RF utility check on a sample."""
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        actual_n = meta["n_features"]
        X = df[FEAT_COLS[:actual_n]].values
        y = df["target"].values

        # Sample for speed
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


# ── Download & Process One Dataset ────────────────────────────────────

def process_one_dataset(repo_id: str) -> tuple[pd.DataFrame | None, dict]:
    """Download and process one HuggingFace dataset.

    Returns (fixed_width_df, info_dict) or (None, error_dict).
    """
    from datasets import load_dataset

    info = {"repo_id": repo_id, "status": "unknown"}
    start = time.time()

    try:
        # Try streaming first for large datasets
        try:
            ds = load_dataset(repo_id, split="train", streaming=True, trust_remote_code=True)
            # Sample up to MAX_ROWS_PER_DATASET
            rows = []
            for row in islice(iter(ds), MAX_ROWS_PER_DATASET):
                rows.append(dict(row))
            if not rows:
                info["status"] = "empty"
                info["error"] = "no rows from streaming"
                return None, info
            df = pd.DataFrame(rows)
        except Exception as stream_err:
            # Fall back to full load with row limit
            try:
                split_spec = f"train[:{MAX_ROWS_PER_DATASET}]"
                ds = load_dataset(repo_id, split=split_spec, trust_remote_code=True)
                df = ds.to_pandas()
            except Exception:
                # Try without split specification
                try:
                    ds = load_dataset(repo_id, trust_remote_code=True)
                    # Pick the first available split
                    if hasattr(ds, "keys"):
                        split_name = list(ds.keys())[0]
                        df = ds[split_name].to_pandas()
                    else:
                        df = ds.to_pandas()
                except Exception as e:
                    info["status"] = "download_failed"
                    info["error"] = str(e)[:200]
                    return None, info

        # Cap rows
        if len(df) > MAX_ROWS_PER_DATASET:
            df = df.sample(n=MAX_ROWS_PER_DATASET, random_state=42)

        info["raw_rows"] = len(df)
        info["raw_cols"] = len(df.columns)
        info["columns"] = list(df.columns)[:20]

        # Must have at least a few columns, a few rows
        if len(df) < MIN_ROWS:
            info["status"] = "too_small"
            info["error"] = f"only {len(df)} rows"
            return None, info

        if len(df.columns) < 3:
            info["status"] = "too_few_cols"
            info["error"] = f"only {len(df.columns)} columns"
            return None, info

        # Guess target
        target_col = guess_target_column(df)
        if target_col is None:
            info["status"] = "no_target"
            info["error"] = "could not identify target column"
            return None, info

        info["target_col"] = target_col

        # Encode to fixed-width format
        result_df, meta = encode_dataset_to_fixed_width(df, target_col, repo_id)
        if result_df is None:
            info["status"] = "encode_failed"
            info["error"] = meta.get("error", "unknown encode error")
            return None, info

        # Quality gate
        passed, reason = quality_gate(result_df, meta)
        if not passed:
            info["status"] = "gate_fail"
            info["error"] = reason
            return None, info

        # Utility check
        utility = compute_utility(result_df, meta)
        meta["utility_auc"] = utility
        # Update the _source_meta in the DataFrame
        result_df["_source_meta"] = json.dumps(meta)

        info["status"] = "ok"
        info["n_rows"] = len(result_df)
        info["n_features"] = meta["n_features"]
        info["task_type"] = meta["task_type"]
        info["n_classes"] = meta["n_classes"]
        info["utility_auc"] = utility
        info["duration_s"] = round(time.time() - start, 1)

        return result_df, info

    except Exception as e:
        info["status"] = "error"
        info["error"] = traceback.format_exc()[-300:]
        return None, info


# ── Shard Builder ─────────────────────────────────────────────────────

def get_next_shard_idx() -> int:
    """Determine the next shard index from the HF repo."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
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
    except Exception:
        # Fall back to local files
        existing = sorted(CORPUS_DIR.glob("train-*.parquet"))
        if existing:
            last = existing[-1].stem
            return int(last.replace("train-", "")) + 1
        return 300  # Start after synthetic shards (240 exist)


def save_shard(collected: list[pd.DataFrame], shard_idx: int) -> tuple[Path, dict]:
    """Save collected DataFrames as a shard."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CORPUS_DIR / f"train-{shard_idx:05d}.parquet"

    combined = pd.concat(collected, ignore_index=True)
    combined.to_parquet(out_path, index=False, engine="pyarrow")

    stats = {
        "shard_idx": shard_idx,
        "rows": len(combined),
        "datasets": len(collected),
        "size_mb": round(out_path.stat().st_size / (1024 * 1024), 1),
    }

    del combined
    gc.collect()
    return out_path, stats


def upload_shard(shard_path: Path) -> bool:
    """Upload a shard to HuggingFace and delete local copy."""
    if not HF_TOKEN:
        print("  [WARN] No HF_TOKEN set, skipping upload. Set HF_TOKEN to enable.")
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        upload_start = time.time()
        api.upload_file(
            path_or_fileobj=str(shard_path),
            path_in_repo=f"data/{shard_path.name}",
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"Add real-data shard {shard_path.name}",
        )
        elapsed = time.time() - upload_start
        size_mb = shard_path.stat().st_size / (1024 * 1024)
        print(f"  Uploaded {shard_path.name} ({size_mb:.0f} MB) in {elapsed:.0f}s")

        # Delete local copy to save disk
        shard_path.unlink()
        print(f"  Deleted local copy {shard_path.name}")
        return True

    except Exception as e:
        print(f"  [WARN] Upload failed: {e}")
        return False


# ── Also generate synthetic data to fill between real datasets ────────

def generate_synthetic_batch(n_datasets: int = 20, seed_base: int = 0) -> list[pd.DataFrame]:
    """Generate a batch of synthetic datasets in fixed-width format."""
    try:
        from tabula.data.synthetic import (
            TreePriorGenerator,
            GaussianMixtureGenerator,
            PolynomialGenerator,
            SCMGenerator,
            RegressionSyntheticGenerator,
        )
    except ImportError:
        print("  [WARN] Synthetic generators not available")
        return []

    generators = [
        ("TreePrior", TreePriorGenerator),
        ("GaussianMixture", GaussianMixtureGenerator),
        ("Polynomial", PolynomialGenerator),
        ("SCM", SCMGenerator),
        ("Regression", RegressionSyntheticGenerator),
    ]

    DOMAINS = list({
        "finance", "health", "ecommerce", "iot", "hr",
        "science", "logistics", "education", "manufacturing",
        "environment", "telecom",
    })

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
            elif gen_name in ("GaussianMixture",):
                gen = gen_cls(n_samples=n_samples, n_features=n_features, n_classes=n_classes)
            else:
                gen = gen_cls(n_samples=n_samples, n_features=n_features, n_classes=n_classes)

            df, meta = gen.generate(seed=seed)

            # Convert to fixed-width
            feature_cols = [c for c in df.columns if c != "target"]
            actual_n = min(len(feature_cols), MAX_FEATURES)
            feat_vals = df[feature_cols[:actual_n]].values.astype(np.float32)
            target_vals = df["target"].values.astype(np.float32)
            n_rows = len(df)

            # Missingness
            if rng.random() < 0.3:
                rate = float(rng.uniform(0.02, 0.15))
                mask = rng.random(size=feat_vals.shape) < rate
                feat_vals[mask] = np.nan

            padded = np.full((n_rows, MAX_FEATURES), np.nan, dtype=np.float32)
            padded[:, :actual_n] = feat_vals

            domain = str(rng.choice(DOMAINS))

            out = pd.DataFrame(padded, columns=FEAT_COLS)
            out["target"] = target_vals
            source_meta = {
                "generator": gen_name,
                "task_type": meta.task_type,
                "n_features": actual_n,
                "n_classes": meta.n_classes,
                "n_samples": n_rows,
                "domain": domain,
                "feature_names": [f"feat_{j}" for j in range(actual_n)],
                "seed": seed,
                "method": gen_name,
                "missingness_rate": 0.0,
                "concept_drift": False,
                "utility_auc": 0.0,
            }
            out["_source_meta"] = json.dumps(source_meta)
            collected.append(out)

        except Exception as e:
            continue

    return collected


# ── Main Loop ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CONTINUOUS REAL DATA DISCOVERY → PROCESSING → HF UPLOAD LOOP")
    print(f"HF Repo: {HF_REPO}")
    print(f"Local corpus: {CORPUS_DIR}")
    print(f"HF Token: {'set' if HF_TOKEN else 'NOT SET (will skip uploads)'}")
    print(f"Max features: {MAX_FEATURES}")
    print(f"Rows per shard: {ROWS_PER_SHARD:,}")
    print(f"Disk: {disk_used_gb():.1f} GB used")
    print("=" * 70)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    processed = load_processed_datasets()
    print(f"Already processed: {len(processed)} datasets")

    shard_idx = get_next_shard_idx()
    print(f"Starting shard index: {shard_idx}")

    total_datasets_ok = 0
    total_datasets_fail = 0
    total_rows = 0
    total_shards = 0
    global_start = time.time()
    synthetic_seed_counter = 10_000_000  # High seed to avoid collision

    # Accumulator for current shard
    shard_collected = []
    shard_rows = 0

    round_num = 0
    while True:
        round_num += 1
        print(f"\n{'='*70}")
        print(f"DISCOVERY ROUND {round_num} | "
              f"Datasets: {total_datasets_ok} ok, {total_datasets_fail} fail | "
              f"Rows: {total_rows:,} | Shards: {total_shards}")
        print(f"Disk: {disk_used_gb():.1f} GB | "
              f"Elapsed: {(time.time()-global_start)/60:.1f} min")

        if disk_used_gb() >= MAX_DISK_GB:
            print("Disk limit reached, pausing to clear space...")
            # Try uploading any pending local shards
            for f in sorted(CORPUS_DIR.glob("train-*.parquet")):
                upload_shard(f)
            if disk_used_gb() >= MAX_DISK_GB:
                print("Still over limit after upload attempts. Waiting 60s...")
                time.sleep(60)
                continue

        # ── Phase 1: Discover datasets ────────────────────────────
        print("\n--- Discovering datasets ---")
        processed = load_processed_datasets()
        candidates = discover_datasets(processed, batch_size=50)
        print(f"Found {len(candidates)} new candidate datasets")

        if not candidates:
            print("No new candidates found. Generating synthetic data...")
            # Generate synthetic to fill shard
            synth = generate_synthetic_batch(30, synthetic_seed_counter)
            synthetic_seed_counter += 30
            for df in synth:
                shard_collected.append(df)
                shard_rows += len(df)
                total_rows += len(df)

            # Check if shard is full
            if shard_rows >= ROWS_PER_SHARD:
                path, stats = save_shard(shard_collected, shard_idx)
                print(f"\n  SHARD {shard_idx:05d}: {stats['rows']:,} rows, "
                      f"{stats['size_mb']:.0f} MB, {stats['datasets']} datasets")
                log_entry({"type": "shard", **stats})
                upload_shard(path)
                total_shards += 1
                shard_idx += 1
                shard_collected = []
                shard_rows = 0

            time.sleep(5)
            continue

        # ── Phase 2: Process each candidate ───────────────────────
        for cand in candidates[:20]:  # Process up to 20 per round
            repo_id = cand["repo_id"]
            print(f"\n  Processing: {repo_id} (dl={cand['downloads']})")

            result_df, info = process_one_dataset(repo_id)

            if result_df is not None:
                print(f"    OK: {info['n_rows']:,} rows, {info['n_features']} features, "
                      f"{info['task_type']}, utility={info.get('utility_auc', 0):.3f}")
                shard_collected.append(result_df)
                shard_rows += len(result_df)
                total_rows += len(result_df)
                total_datasets_ok += 1
            else:
                print(f"    SKIP: {info.get('status', '?')} - {info.get('error', '?')[:80]}")
                total_datasets_fail += 1

            # Mark as processed regardless of outcome
            save_processed_dataset(repo_id)
            log_entry({"type": "dataset", **info})

            # Check if shard is full
            if shard_rows >= ROWS_PER_SHARD:
                path, stats = save_shard(shard_collected, shard_idx)
                print(f"\n  SHARD {shard_idx:05d}: {stats['rows']:,} rows, "
                      f"{stats['size_mb']:.0f} MB, {stats['datasets']} datasets")
                log_entry({"type": "shard", **stats})
                upload_shard(path)
                total_shards += 1
                shard_idx += 1
                shard_collected = []
                shard_rows = 0
                gc.collect()

            # Brief pause to be nice to HF API
            time.sleep(1)

        # ── Phase 3: Fill remainder with synthetic if needed ──────
        if shard_rows > 0 and shard_rows < ROWS_PER_SHARD:
            remaining = ROWS_PER_SHARD - shard_rows
            n_synth = max(1, remaining // 10_000)
            print(f"\n  Filling shard with ~{n_synth} synthetic datasets ({remaining:,} rows needed)")
            synth = generate_synthetic_batch(min(n_synth, 50), synthetic_seed_counter)
            synthetic_seed_counter += len(synth)
            for df in synth:
                shard_collected.append(df)
                shard_rows += len(df)
                total_rows += len(df)

            if shard_rows >= ROWS_PER_SHARD:
                path, stats = save_shard(shard_collected, shard_idx)
                print(f"\n  SHARD {shard_idx:05d}: {stats['rows']:,} rows, "
                      f"{stats['size_mb']:.0f} MB, {stats['datasets']} datasets")
                log_entry({"type": "shard", **stats})
                upload_shard(path)
                total_shards += 1
                shard_idx += 1
                shard_collected = []
                shard_rows = 0
                gc.collect()

        print(f"\n  Round {round_num} complete. "
              f"Running total: {total_datasets_ok} datasets, {total_rows:,} rows, "
              f"{total_shards} shards uploaded")


if __name__ == "__main__":
    main()
