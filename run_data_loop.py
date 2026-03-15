#!/usr/bin/env python3
"""Parallel continuous data loop: discover → prepare → encode → upload.

Maximises CPU utilisation by running discovery, preparation, encoding,
quality‑gating, and utility scoring across multiple processes/threads.

Architecture
------------
- Discovery:  ThreadPoolExecutor (3 threads, one per source: PMLB, OpenML, HF)
- Prep+Encode: ProcessPoolExecutor (N_WORKERS workers)
- Synthetic:   ProcessPoolExecutor (parallel batch generation)
- HF downloads: Subprocess‑based timeout (120 s) instead of SIGALRM
- OpenML:      Expanded search — CC18 + CTR23 + general catalogue queries
"""
from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
import warnings
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    Future,
)
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from tabula.data.env import load_repo_env_file

# ── Load token ────────────────────────────────────────────────────────
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
MAX_DISK_GB = 35
MAX_ROWS_PER_DATASET = 200_000
MIN_ROWS = 50
HF_DOWNLOAD_TIMEOUT = 60  # seconds per dataset
HF_PARALLEL_FETCHES = 4  # concurrent HF downloads
N_WORKERS = min(mp.cpu_count(), 12)  # cap at 12 for prep workers
N_SYNTH_WORKERS = min(mp.cpu_count(), 8)

# ── Logging ───────────────────────────────────────────────────────────

def log_entry(entry: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def disk_used_gb() -> float:
    total, used, free = shutil.disk_usage("/")
    return used / (1024**3)


# ── Shard index from HF ──────────────────────────────────────────────

def get_next_shard_idx() -> int:
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
        return 300


# ======================================================================
# DISCOVERY – parallel over sources
# ======================================================================

def _discover_pmlb_safe(registry_file: str, output_root: str,
                        limit: int, max_instances: int) -> list[dict]:
    """Thread‑safe PMLB discovery (returns dicts, not dataclass)."""
    from tabula.data.pmlb import search_pmlb_datasets, fetch_pmlb_dataset
    from tabula.data.autodiscovery import DiscoveryRegistry, DiscoveryRecord, _validate_raw_dir

    registry = DiscoveryRegistry(registry_file)
    output = Path(output_root)
    records = []
    try:
        datasets = search_pmlb_datasets(max_instances=max_instances)
    except Exception as exc:
        print(f"  [WARN] PMLB summary failed: {exc}")
        return records

    for info in datasets[:limit]:
        local_id = f"pmlb_{info.name}"
        if registry.contains(local_id):
            continue
        raw_dir = output / local_id
        status, error, n_rows, n_cols = "ok", "", 0, 0
        try:
            fetch_pmlb_dataset(info.name, output_root=output_root, local_dataset_id=local_id)
            ok, error, n_rows, n_cols = _validate_raw_dir(raw_dir)
            status = "ok" if ok else "schema_fail"
        except Exception as exc:
            status, error = "download_fail", str(exc)[:200]

        rec = DiscoveryRecord(
            dataset_id=local_id, source="pmlb", external_ref=info.name,
            task_type=info.task_type, n_rows=n_rows, n_cols=n_cols,
            status=status, raw_dir=str(raw_dir), notes=error,
        )
        registry.add(rec)
        records.append(rec.__dict__)
    return records


def _discover_openml_safe(registry_file: str, output_root: str,
                          limit: int, max_rows: int) -> list[dict]:
    """Thread‑safe OpenML discovery — CC18 + CTR23 + bulk catalogue pagination."""
    from tabula.data.openml import (
        fetch_cc18_task_list, fetch_ctr23_task_list,
        fetch_openml_dataset,
    )
    from tabula.data.autodiscovery import DiscoveryRegistry, DiscoveryRecord, _validate_raw_dir
    import urllib.request

    registry = DiscoveryRegistry(registry_file)
    output = Path(output_root)
    records: list[dict] = []

    # --- Benchmark tasks (CC18 + CTR23) ---
    task_list: list[tuple[str, Any]] = []
    try:
        for t in fetch_cc18_task_list():
            task_list.append(("binary", t))
    except Exception as exc:
        print(f"  [WARN] CC18 failed: {exc}")
    try:
        for t in fetch_ctr23_task_list():
            task_list.append(("regression", t))
    except Exception as exc:
        print(f"  [WARN] CTR23 failed: {exc}")

    for task_type, task in task_list[:limit]:
        did = f"openml_{task.dataset_id}"
        if registry.contains(did):
            continue
        raw_dir = output / did
        status, error, n_rows, n_cols = "ok", "", 0, 0
        try:
            fetch_openml_dataset(
                dataset_id=task.dataset_id, output_root=output_root,
                local_dataset_id=did, task_type=task_type, max_rows=max_rows,
            )
            ok, error, n_rows, n_cols = _validate_raw_dir(raw_dir)
            status = "ok" if ok else "schema_fail"
        except Exception as exc:
            status, error = "download_fail", str(exc)[:200]

        rec = DiscoveryRecord(
            dataset_id=did, source="openml", external_ref=str(task.dataset_id),
            task_type=task_type, n_rows=n_rows, n_cols=n_cols,
            status=status, raw_dir=str(raw_dir), notes=error,
        )
        registry.add(rec)
        records.append(rec.__dict__)

    # --- Bulk catalogue pagination (find MORE datasets beyond benchmarks) ---
    # API quality filters are broken, so paginate and filter client-side
    extra_count = 0
    # Start offset from a tracking file to avoid re-scanning same pages
    offset_file = Path("artifacts/openml_offset.txt")
    offset = 0
    if offset_file.exists():
        try:
            offset = int(offset_file.read_text().strip())
        except Exception:
            pass

    batch_size = 100  # datasets per API page
    max_pages = 20    # scan up to 20 pages per round

    for page in range(max_pages):
        if extra_count >= limit:
            break
        api_url = f"https://www.openml.org/api/v1/json/data/list/limit/{batch_size}/offset/{offset}"
        try:
            req = urllib.request.urlopen(api_url, timeout=15)
            payload = json.loads(req.read())
            items = payload.get("data", {}).get("dataset", [])
            if not items:
                offset = 0  # wrap around
                break
        except Exception:
            break

        for item in items:
            if extra_count >= limit:
                break
            dataset_id = int(item.get("did", 0))
            if dataset_id == 0:
                continue
            did = f"openml_{dataset_id}"
            if registry.contains(did):
                continue

            # Client-side filter: check qualities
            qmap = {}
            for qi in (item.get("quality", []) if isinstance(item.get("quality"), list) else []):
                qmap[qi.get("name", "")] = qi.get("value", "")
            n_inst = int(float(qmap.get("NumberOfInstances", 0)))
            n_feat = int(float(qmap.get("NumberOfFeatures", 0)))
            if n_inst < 100 or n_inst > max_rows or n_feat < 2 or n_feat > 500:
                continue

            # Try to fetch
            raw_dir = output / did
            status, error, n_rows, n_cols = "ok", "", 0, 0
            try:
                fetch_openml_dataset(
                    dataset_id=dataset_id, output_root=output_root,
                    local_dataset_id=did, max_rows=max_rows,
                )
                ok, error, n_rows, n_cols = _validate_raw_dir(raw_dir)
                status = "ok" if ok else "schema_fail"
            except Exception as exc:
                status, error = "download_fail", str(exc)[:200]

            rec = DiscoveryRecord(
                dataset_id=did, source="openml", external_ref=str(dataset_id),
                task_type="unknown", n_rows=n_rows, n_cols=n_cols,
                status=status, raw_dir=str(raw_dir), notes=error,
            )
            registry.add(rec)
            records.append(rec.__dict__)
            extra_count += 1

        offset += batch_size

    # Save offset for next round
    offset_file.parent.mkdir(parents=True, exist_ok=True)
    offset_file.write_text(str(offset))

    return records


def _hf_fetch_with_timeout(repo_id: str, output_root: str,
                           dataset_id: str, max_rows: int,
                           timeout: int = HF_DOWNLOAD_TIMEOUT) -> bool:
    """Fetch a single HF dataset in a subprocess with a hard timeout."""
    def _worker(repo_id, output_root, dataset_id, max_rows):
        sys.path.insert(0, ".")
        from tabula.data.huggingface import fetch_huggingface_dataset
        fetch_huggingface_dataset(
            repo_id=repo_id, output_root=output_root,
            dataset_id=dataset_id, max_rows=max_rows,
        )

    proc = mp.Process(target=_worker, args=(repo_id, output_root, dataset_id, max_rows))
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        return False  # timed out
    return proc.exitcode == 0


def _fetch_one_hf(args: tuple) -> dict:
    """Fetch + validate one HF dataset. Runs in a thread."""
    repo_id, output_root, did, bootstrap_rows = args
    from tabula.data.autodiscovery import DiscoveryRecord, _validate_raw_dir
    raw_dir = Path(output_root) / did
    raw_dir.mkdir(parents=True, exist_ok=True)
    status, error, n_rows, n_cols = "ok", "", 0, 0
    try:
        ok_fetch = _hf_fetch_with_timeout(repo_id, output_root, did, bootstrap_rows)
        if not ok_fetch:
            status, error = "download_fail", "download timed out or crashed"
        else:
            ok, error, n_rows, n_cols = _validate_raw_dir(raw_dir)
            status = "ok" if ok else "schema_fail"
    except Exception as exc:
        status, error = "download_fail", str(exc)[:200]
    return DiscoveryRecord(
        dataset_id=did, source="hf", external_ref=repo_id,
        task_type="unknown", n_rows=n_rows, n_cols=n_cols,
        status=status, raw_dir=str(raw_dir), notes=error,
    ).__dict__


def _discover_hf_safe(registry_file: str, output_root: str,
                      limit: int, bootstrap_rows: int) -> list[dict]:
    """Thread‑safe HF discovery with parallel subprocess‑based timeout."""
    from tabula.data.huggingface import search_huggingface_datasets
    from tabula.data.autodiscovery import DiscoveryRegistry, DiscoveryRecord

    registry = DiscoveryRegistry(registry_file)
    records: list[dict] = []

    categories = ["tabular-classification", "tabular-regression"]
    seen: set[str] = set()
    all_results = []
    for cat in categories:
        try:
            results = search_huggingface_datasets(task_category=cat, limit=limit)
            for r in results:
                if r.repo_id not in seen:
                    seen.add(r.repo_id)
                    all_results.append(r)
        except Exception as exc:
            print(f"  [WARN] HF search failed for {cat}: {exc}")

    # Build list of candidates not yet in registry
    candidates = []
    for res in all_results:
        did = f"hf_{res.repo_id.replace('/', '_')}"
        if registry.contains(did):
            continue
        candidates.append((res.repo_id, output_root, did, bootstrap_rows))

    if not candidates:
        return records

    # Fetch HF datasets in parallel (HF_PARALLEL_FETCHES at a time)
    with ThreadPoolExecutor(max_workers=HF_PARALLEL_FETCHES,
                            thread_name_prefix="hf_fetch") as pool:
        for rec_dict in pool.map(_fetch_one_hf, candidates):
            rec = DiscoveryRecord(**rec_dict)
            registry.add(rec)
            records.append(rec_dict)
    return records


def discover_all_parallel() -> list[dict]:
    """Run PMLB, OpenML, HF discovery concurrently in threads.
    
    If HF takes more than 300s, proceed without it.
    """
    registry_file = str(REGISTRY_FILE)
    output_root = str(RAW_ROOT)
    all_records: list[dict] = []

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="disc") as pool:
        futures: dict[Future, str] = {}
        futures[pool.submit(
            _discover_pmlb_safe, registry_file, output_root, 500, 100_000
        )] = "pmlb"
        futures[pool.submit(
            _discover_openml_safe, registry_file, output_root, 200, 50_000
        )] = "openml"
        futures[pool.submit(
            _discover_hf_safe, registry_file, output_root, 30, 5000
        )] = "hf"

        for fut in as_completed(futures, timeout=600):
            source = futures[fut]
            try:
                recs = fut.result(timeout=10)
                ok_count = sum(1 for r in recs if r.get("status") == "ok")
                print(f"  {source}: {ok_count} ok / {len(recs)} total")
                all_records.extend(recs)
            except Exception as exc:
                print(f"  [WARN] {source} discovery failed: {exc}")

    ok_recs = [r for r in all_records if r.get("status") == "ok"]
    print(f"  Discovery total: {len(ok_recs)} ok / {len(all_records)} scanned")
    return ok_recs


# ======================================================================
# PREPARATION + ENCODING – one function per dataset, run in a pool
# ======================================================================

def _process_one_dataset(dataset_id: str, raw_root: str,
                         processed_root: str) -> dict | None:
    """Prepare → encode → quality‑gate → utility for a single dataset.

    Runs in a worker process. Returns a dict with the encoded DataFrame
    serialised to a temp parquet file, or None on failure.
    """
    warnings.filterwarnings("ignore")
    sys.path.insert(0, ".")

    from tabula.data.prep import prepare_dataset

    # ── Prepare ───────────────────────────────────────────────
    try:
        prep = prepare_dataset(
            dataset_id=dataset_id,
            raw_root=raw_root,
            processed_root=processed_root,
            seed=42,
            max_rows=MAX_ROWS_PER_DATASET,
            drop_identifier_columns=True,
            feature_engineering=True,
        )
    except Exception as exc:
        return {"dataset_id": dataset_id, "status": "prep_fail",
                "error": str(exc)[:200]}

    # ── Read train.csv ────────────────────────────────────────
    proc_dir = Path(prep.processed_dir)
    train_path = proc_dir / "train.csv"
    schema_path = proc_dir / "schema.json"
    card_path = proc_dir / "dataset_card.json"
    if not train_path.exists():
        return {"dataset_id": dataset_id, "status": "encode_fail",
                "error": "no train.csv"}
    try:
        df = pd.read_csv(train_path)
    except Exception as exc:
        return {"dataset_id": dataset_id, "status": "encode_fail",
                "error": str(exc)[:200]}

    if len(df) < MIN_ROWS:
        return {"dataset_id": dataset_id, "status": "encode_fail",
                "error": f"too few rows: {len(df)}"}

    target_col = prep.target_column
    if target_col not in df.columns:
        return {"dataset_id": dataset_id, "status": "encode_fail",
                "error": f"target '{target_col}' missing"}

    # Load metadata
    schema = {}
    if schema_path.exists():
        with open(schema_path) as f:
            schema = json.load(f)
    card = {}
    if card_path.exists():
        with open(card_path) as f:
            card = json.load(f)

    # ── Encode features ───────────────────────────────────────
    target = df[target_col].copy()
    feature_cols_list = [c for c in df.columns if c != target_col]

    encoded_cols: list[np.ndarray] = []
    feature_names: list[str] = []
    for col in feature_cols_list:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            encoded_cols.append(series.astype(np.float32).values)
            feature_names.append(col)
        elif series.dtype == object or hasattr(series.dtype, "categories"):
            n_unique = series.nunique()
            if n_unique > 200 and n_unique > len(series) * 0.5:
                continue
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
        return {"dataset_id": dataset_id, "status": "encode_fail",
                "error": f"only {len(encoded_cols)} features"}

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
        return {"dataset_id": dataset_id, "status": "encode_fail",
                "error": f"only {valid_mask.sum()} valid targets"}

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
        "source_repo": card.get("external_ref", dataset_id),
        "dataset_id": dataset_id,
        "seed": 42,
        "method": "real_data_prepared",
        "missingness_rate": float(np.isnan(padded[:, :actual_n]).mean()),
        "concept_drift": False,
        "utility_auc": 0.0,
    }

    # ── Quality gate ──────────────────────────────────────────
    feat_arr = padded[:, :actual_n]
    for i in range(actual_n):
        col = feat_arr[:, i]
        non_nan = col[~np.isnan(col)]
        if len(non_nan) <= 1 or np.nanstd(non_nan) < 1e-10:
            return {"dataset_id": dataset_id, "status": "gate_fail",
                    "error": f"constant/empty col {i}"}

    if task_type in ("binary", "multiclass"):
        uniq = np.unique(target_encoded[~np.isnan(target_encoded)])
        if len(uniq) < 2:
            return {"dataset_id": dataset_id, "status": "gate_fail",
                    "error": "target < 2 classes"}
        _, counts = np.unique(target_encoded[~np.isnan(target_encoded)], return_counts=True)
        minority = counts.min() / counts.sum()
        if minority < 0.01:
            return {"dataset_id": dataset_id, "status": "gate_fail",
                    "error": f"minority too small: {minority:.4f}"}

    # ── Utility (RF cross‑val) ────────────────────────────────
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        X = padded[:, :actual_n].copy()
        y = target_encoded.copy()
        if len(X) > 5000:
            idx = np.random.default_rng(42).choice(len(X), 5000, replace=False)
            X, y = X[idx], y[idx]
        X = np.nan_to_num(X, nan=0.0)
        if task_type == "regression":
            clf = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=0, n_jobs=1)
            scores = cross_val_score(clf, X, y, cv=3, scoring="r2", n_jobs=1)
        else:
            clf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=0, n_jobs=1)
            try:
                scores = cross_val_score(clf, X, y, cv=3, scoring="roc_auc_ovr_weighted", n_jobs=1)
            except Exception:
                scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy", n_jobs=1)
        utility = float(np.mean(scores))
    except Exception:
        utility = 0.0

    source_meta["utility_auc"] = utility
    out["_source_meta"] = json.dumps(source_meta)

    # Save to temp parquet so we can move it across process boundary
    tmp_dir = Path("corpus/real_data/tmp_encoded")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{dataset_id}.parquet"
    out.to_parquet(str(tmp_path), index=False, engine="pyarrow")
    del out, padded
    gc.collect()

    return {
        "dataset_id": dataset_id,
        "status": "ok",
        "rows": n_rows,
        "features": actual_n,
        "task_type": task_type,
        "utility": utility,
        "tmp_path": str(tmp_path),
        "train_rows": prep.train_rows,
        "num_cols": len(prep.numeric_columns),
        "cat_cols": len(prep.categorical_columns),
    }


def _infer_task_type(series: pd.Series) -> str:
    n_unique = series.nunique()
    if n_unique <= 2:
        return "binary"
    elif n_unique <= 20 and n_unique < len(series) * 0.05:
        return "multiclass"
    return "regression"


# ======================================================================
# SYNTHETIC DATA – parallel generation
# ======================================================================

def _generate_synth_chunk(args: tuple) -> list[str]:
    """Generate a chunk of synthetic datasets in a worker process.

    Returns list of temp parquet file paths.
    """
    chunk_datasets, seed_base = args
    sys.path.insert(0, ".")
    warnings.filterwarnings("ignore")

    try:
        from tabula.data.synthetic import (
            TreePriorGenerator, GaussianMixtureGenerator,
            PolynomialGenerator, SCMGenerator,
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
    DOMAINS = [
        "finance", "health", "ecommerce", "iot", "hr", "science",
        "logistics", "education", "manufacturing", "environment", "telecom",
    ]

    paths: list[str] = []
    rng = np.random.default_rng(seed_base)
    tmp_dir = Path("corpus/real_data/tmp_synth")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i in range(chunk_datasets):
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
            p = tmp_dir / f"synth_{seed}.parquet"
            out.to_parquet(str(p), index=False, engine="pyarrow")
            paths.append(str(p))
            del out, padded
        except Exception:
            continue
    return paths


def generate_synthetic_parallel(n_datasets: int, seed_base: int) -> list[str]:
    """Generate synthetic datasets across multiple processes.
    Returns list of temp parquet file paths.
    """
    n_workers = min(N_SYNTH_WORKERS, n_datasets)
    chunk_size = max(1, n_datasets // n_workers)
    chunks = []
    for i in range(n_workers):
        start = i * chunk_size
        count = chunk_size if i < n_workers - 1 else (n_datasets - start)
        if count <= 0:
            break
        chunks.append((count, seed_base + start))

    all_paths: list[str] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(_generate_synth_chunk, chunks, timeout=300):
            all_paths.extend(result)
    return all_paths


# ======================================================================
# UPLOAD
# ======================================================================

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


def save_shard(parquet_paths: list[str], shard_idx: int) -> tuple[Path | None, int]:
    """Concatenate temp parquet files into a single shard parquet.
    Returns (shard_path, n_rows).
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    dfs = []
    for p in parquet_paths:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception:
            pass
    if not dfs:
        return None, 0
    combined = pd.concat(dfs, ignore_index=True)
    n_rows = len(combined)
    out_path = CORPUS_DIR / f"train-{shard_idx:05d}.parquet"
    combined.to_parquet(out_path, index=False, engine="pyarrow")
    del combined, dfs
    gc.collect()
    # Clean up temp files
    for p in parquet_paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass
    return out_path, n_rows


def cleanup_raw_data():
    for d in [RAW_ROOT, PREPARED_ROOT]:
        if d.exists():
            total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            if total > 2 * 1024**3:
                print(f"  Cleaning {d} ({total/1e9:.1f} GB)...")
                shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
    # Clean temp dirs
    for d in [Path("corpus/real_data/tmp_encoded"), Path("corpus/real_data/tmp_synth")]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
    # Clean HF cache (datasets lib + hub cache can grow huge)
    for d in [Path.home() / ".cache/huggingface/datasets",
              Path.home() / ".cache/huggingface/xet"]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    # Clean hub dataset caches
    hub_dir = Path.home() / ".cache/huggingface/hub"
    if hub_dir.exists():
        for p in hub_dir.glob("datasets--*"):
            shutil.rmtree(p, ignore_errors=True)


# ======================================================================
# MAIN LOOP
# ======================================================================

def main():
    print("=" * 70)
    print("PARALLEL DATA LOOP: DISCOVER -> PREPARE -> ENCODE -> UPLOAD")
    print(f"  HF Repo:     {HF_REPO}")
    print(f"  HF Token:    {'set (write)' if HF_TOKEN else 'NOT SET'}")
    print(f"  Workers:     {N_WORKERS} prep | {N_SYNTH_WORKERS} synth")
    print(f"  CPUs:        {mp.cpu_count()}")
    print(f"  Shard size:  {ROWS_PER_SHARD:,} rows")
    print(f"  Disk:        {disk_used_gb():.1f} GB used")
    print("=" * 70)

    for d in [CORPUS_DIR, RAW_ROOT, PREPARED_ROOT, REGISTRY_FILE.parent,
              Path("corpus/real_data/tmp_encoded"),
              Path("corpus/real_data/tmp_synth")]:
        d.mkdir(parents=True, exist_ok=True)

    shard_idx = get_next_shard_idx()
    print(f"Starting shard index: {shard_idx}")

    total_real_ok = 0
    total_real_fail = 0
    total_rows = 0
    total_shards = 0
    global_start = time.time()
    synthetic_seed = 30_000_000

    shard_paths: list[str] = []  # temp parquet paths for pending shard
    shard_rows = 0
    shard_datasets = 0

    round_num = 0
    while True:
        round_num += 1
        elapsed_min = (time.time() - global_start) / 60
        print(f"\n{'='*70}")
        print(f"ROUND {round_num} | ok={total_real_ok} fail={total_real_fail} | "
              f"rows={total_rows:,} | shards={total_shards} | "
              f"disk={disk_used_gb():.1f}GB | elapsed={elapsed_min:.0f}min")

        # Disk check
        if disk_used_gb() >= MAX_DISK_GB:
            print("Disk pressure -- cleaning up...")
            for f in sorted(CORPUS_DIR.glob("train-*.parquet")):
                upload_shard(f)
            cleanup_raw_data()
            if disk_used_gb() >= MAX_DISK_GB:
                time.sleep(60)
                continue

        # ── Phase 1: Parallel discovery ───────────────────────
        print("\n--- Phase 1: Parallel Discovery (3 sources) ---")
        t0 = time.time()
        ok_records = discover_all_parallel()
        print(f"  Discovery took {time.time()-t0:.0f}s -- {len(ok_records)} new datasets")

        # ── Phase 2: Parallel prep + encode ───────────────────
        if ok_records:
            print(f"\n--- Phase 2: Parallel Prep ({N_WORKERS} workers, "
                  f"{len(ok_records)} datasets) ---")
            t0 = time.time()

            dataset_ids = [r["dataset_id"] for r in ok_records]

            with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                futures_map: dict[Future, str] = {}
                for did in dataset_ids:
                    f = pool.submit(
                        _process_one_dataset, did,
                        str(RAW_ROOT), str(PREPARED_ROOT),
                    )
                    futures_map[f] = did

                for fut in as_completed(futures_map):
                    did = futures_map[fut]
                    try:
                        result = fut.result(timeout=300)
                    except Exception as exc:
                        result = {"dataset_id": did, "status": "crash",
                                  "error": str(exc)[:200]}

                    if result is None:
                        total_real_fail += 1
                        continue

                    status = result.get("status", "unknown")
                    if status == "ok":
                        total_real_ok += 1
                        n = result["rows"]
                        shard_paths.append(result["tmp_path"])
                        shard_rows += n
                        shard_datasets += 1
                        total_rows += n
                        print(f"  OK: {did} -- {n:,} rows, {result['features']} feats, "
                              f"{result['task_type']}, u={result['utility']:.3f}")
                        log_entry({
                            "type": "dataset", "id": did, "status": "ok",
                            "rows": n, "features": result["features"],
                            "task_type": result["task_type"],
                            "utility": result["utility"],
                        })
                    else:
                        total_real_fail += 1
                        err = result.get("error", "")
                        print(f"  FAIL: {did} -- {status}: {err[:80]}")
                        log_entry({
                            "type": "dataset", "id": did,
                            "status": status, "error": err,
                        })

                    # Flush shard if buffer full
                    if shard_rows >= ROWS_PER_SHARD:
                        path, n = save_shard(shard_paths, shard_idx)
                        if path:
                            size_mb = path.stat().st_size / (1024 * 1024)
                            print(f"\n  >>> SHARD {shard_idx:05d}: {n:,} rows, "
                                  f"{shard_datasets} datasets, {size_mb:.0f} MB")
                            log_entry({"type": "shard", "idx": shard_idx,
                                       "rows": n, "datasets": shard_datasets,
                                       "size_mb": round(size_mb, 1)})
                            upload_shard(path)
                            total_shards += 1
                            shard_idx += 1
                        shard_paths = []
                        shard_rows = 0
                        shard_datasets = 0
                        gc.collect()
                        cleanup_raw_data()

            print(f"  Prep phase took {time.time()-t0:.0f}s")

        # ── Phase 3: Parallel synthetic fill ──────────────────
        if shard_rows > 0 and shard_rows < ROWS_PER_SHARD:
            remaining = ROWS_PER_SHARD - shard_rows
            n_synth = max(10, min(80, remaining // 10_000))
            print(f"\n--- Phase 3: Parallel Synthetic ({n_synth} datasets, "
                  f"{remaining:,} rows needed, {N_SYNTH_WORKERS} workers) ---")
            t0 = time.time()
            synth_paths = generate_synthetic_parallel(n_synth, synthetic_seed)
            synthetic_seed += n_synth

            for p in synth_paths:
                try:
                    df = pd.read_parquet(p)
                    shard_rows += len(df)
                    shard_datasets += 1
                    total_rows += len(df)
                    shard_paths.append(p)
                    del df
                except Exception:
                    pass

            print(f"  Synthetic: {len(synth_paths)} datasets in {time.time()-t0:.0f}s "
                  f"-- buf now {shard_rows:,} rows")

            if shard_rows >= ROWS_PER_SHARD:
                path, n = save_shard(shard_paths, shard_idx)
                if path:
                    size_mb = path.stat().st_size / (1024 * 1024)
                    print(f"\n  >>> SHARD {shard_idx:05d}: {n:,} rows, "
                          f"{shard_datasets} datasets, {size_mb:.0f} MB")
                    log_entry({"type": "shard", "idx": shard_idx,
                               "rows": n, "datasets": shard_datasets,
                               "size_mb": round(size_mb, 1)})
                    upload_shard(path)
                    total_shards += 1
                    shard_idx += 1
                shard_paths = []
                shard_rows = 0
                shard_datasets = 0
                gc.collect()

        # If nothing new and shard is empty, pure synthetic shard
        if not ok_records and shard_rows == 0:
            print("\n--- No new real data -> generating full synthetic shard ---")
            t0 = time.time()
            n_synth = 80
            synth_paths = generate_synthetic_parallel(n_synth, synthetic_seed)
            synthetic_seed += n_synth
            for p in synth_paths:
                try:
                    df = pd.read_parquet(p)
                    shard_rows += len(df)
                    shard_datasets += 1
                    total_rows += len(df)
                    shard_paths.append(p)
                    del df
                except Exception:
                    pass
            print(f"  Generated {len(synth_paths)} synthetic datasets in "
                  f"{time.time()-t0:.0f}s -- {shard_rows:,} rows")

        print(f"\n  Round {round_num} done. Buf: {shard_rows:,}/{ROWS_PER_SHARD:,} rows, "
              f"{shard_datasets} datasets")

        time.sleep(2)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
