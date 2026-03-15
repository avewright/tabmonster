#!/usr/bin/env python3
"""Online pretraining: generate synthetic data on-the-fly, zero disk usage.

Instead of writing 40GB to disk, we generate batches in background worker
threads and feed them to the GPU immediately. This gives us effectively
unlimited data — target is 1B+ rows seen during training.

Architecture:
  - N CPU worker threads generate synthetic datasets in parallel
  - Results go into a bounded queue (backpressure if GPU is slow)
  - Main thread pulls batches from queue, trains on GPU
  - Each batch is a fresh synthetic dataset — never see same data twice

Hardware: 48 CPUs, 247 GB RAM, RTX A4500 (20 GB VRAM)
Target: 200K steps × 512 batch = 102M rows (increase MAX_STEPS for more)
"""
from __future__ import annotations

import gc
import json
import math
import os
import queue
import random
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, '.')

import warnings
import numpy as np
import torch
from torch import nn

warnings.filterwarnings('ignore', category=RuntimeWarning)

from tabula.config import (
    ExperimentConfig, ModelConfig, TrainingConfig,
    TaskConfig, DataConfig, EpisodeConfig,
)
from tabula.data.datasets import TabularBatch
from tabula.data.synthetic import (
    TreePriorGenerator,
    GaussianMixtureGenerator,
    PolynomialGenerator,
    SCMGenerator,
    MixedTypeGenerator,
)
from tabula.models.transformer import TabularTransformer
from tabula.utils import set_seed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_FEATURES = 32
BATCH_SIZE = 512
MAX_STEPS = 200_000      # 200K steps × 512 = 102M rows
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 2000
LR = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda"
SEED = 42

# Online generation settings
NUM_GEN_WORKERS = 24      # Background threads for data generation
QUEUE_MAXSIZE = 128       # Max batches buffered (backpressure)
DATASETS_PER_BATCH = 8    # Mix N datasets per batch for diversity

# Dataset generation parameters
MIN_SAMPLES = 256         # Min rows per synthetic dataset 
MAX_SAMPLES = 4096        # Max rows per synthetic dataset
MIN_FEATURES = 4
MAX_GEN_FEATURES = 48

METHODS = [
    "TreePrior", "SCM", "GaussianMixture", "Polynomial",
    "MixedType_TreePrior", "MixedType_SCM",
]


# ---------------------------------------------------------------------------
# Online batch generator (runs in background threads)
# ---------------------------------------------------------------------------
def _generate_one_dataset(seed: int, max_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Generate one synthetic dataset and return (features, masks, targets).
    
    Returns arrays with shape (n_samples, max_features), (n_samples, max_features), (n_samples,).
    Features beyond the real feature count are zero-filled, with masks=False.
    """
    rng = np.random.default_rng(seed)
    
    n_features = rng.integers(MIN_FEATURES, MAX_GEN_FEATURES + 1)
    n_samples = rng.integers(MIN_SAMPLES, MAX_SAMPLES + 1)
    method = rng.choice(METHODS)
    
    # Task type
    task_choices = ["binary", "binary", "multiclass", "multiclass", "regression"]
    task_type = rng.choice(task_choices)
    n_classes = 2 if task_type == "binary" else (rng.integers(3, 8) if task_type == "multiclass" else 2)
    
    try:
        if method == "TreePrior":
            gen = TreePriorGenerator(n_samples=n_samples, n_features=n_features, n_classes=n_classes)
        elif method == "SCM":
            gen = SCMGenerator(n_samples=n_samples, n_features=n_features, n_classes=n_classes)
        elif method == "GaussianMixture":
            gen = GaussianMixtureGenerator(
                n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                n_components=int(rng.integers(3, 10)),
                label_strategy=rng.choice(["linear", "quadratic", "tree"]),
            )
        elif method == "Polynomial":
            gen = PolynomialGenerator(
                n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                degree=int(rng.choice([2, 3, 4])),
            )
        elif method.startswith("MixedType_"):
            base_method = method.replace("MixedType_", "")
            if base_method == "TreePrior":
                base = TreePriorGenerator(n_samples=n_samples, n_features=max(2, n_features - 4), n_classes=n_classes)
            elif base_method == "SCM":
                base = SCMGenerator(n_samples=n_samples, n_features=max(2, n_features - 4), n_classes=n_classes)
            else:
                base = GaussianMixtureGenerator(n_samples=n_samples, n_features=max(2, n_features - 4), n_classes=n_classes)
            gen = MixedTypeGenerator(base, n_categorical=min(4, n_features // 3))
        else:
            gen = TreePriorGenerator(n_samples=n_samples, n_features=n_features, n_classes=n_classes)
        
        df, meta = gen.generate(seed=seed)
    except Exception:
        return None
    
    # Extract numeric columns only (skip categorical for pretraining speed)
    target_col = meta.target_name
    feature_cols = [c for c in df.columns if c != target_col]
    
    # Convert to numeric, drop non-numeric columns  
    numeric_data = []
    for col in feature_cols:
        try:
            vals = pd.to_numeric(df[col], errors='coerce').values.astype(np.float32)
            numeric_data.append(vals)
        except Exception:
            pass
    
    if len(numeric_data) == 0:
        return None
    
    actual_features = min(len(numeric_data), max_features)
    raw = np.column_stack(numeric_data[:actual_features])
    
    # Normalize per-column (z-score) to reduce loss scale variance
    if raw.size == 0:
        return None
    means = np.nanmean(raw, axis=0, keepdims=True)
    stds = np.nanstd(raw, axis=0, keepdims=True)
    stds = np.where(stds < 1e-8, 1.0, stds)
    means = np.nan_to_num(means, nan=0.0)
    raw = (raw - means) / stds
    raw = np.nan_to_num(raw, nan=0.0).astype(np.float32)
    
    # Pad to max_features
    n_rows = raw.shape[0]
    features = np.zeros((n_rows, max_features), dtype=np.float32)
    features[:, :actual_features] = raw
    
    masks = np.zeros((n_rows, max_features), dtype=np.float32)
    masks[:, :actual_features] = 1.0
    
    # Normalize targets too
    targets = df[target_col].values.astype(np.float32)
    if task_type == "regression":
        t_mean = np.nanmean(targets)
        t_std = np.nanstd(targets)
        if t_std > 1e-8:
            targets = (targets - t_mean) / t_std
    targets = np.nan_to_num(targets, nan=0.0)
    
    return features, masks, targets


# Need pandas for the generator
import pandas as pd


def _batch_producer(batch_queue: queue.Queue, stop_event: threading.Event,
                    max_features: int, batch_size: int):
    """Background thread that generates batches and puts them in the queue."""
    thread_rng = random.Random()
    thread_rng.seed(threading.get_ident())
    
    while not stop_event.is_set():
        try:
            # Generate several small datasets and combine into one batch
            all_features = []
            all_masks = []
            all_targets = []
            total_rows = 0
            
            while total_rows < batch_size * 2:  # Generate 2x batch for variety
                seed = thread_rng.randint(0, 2**31)
                result = _generate_one_dataset(seed, max_features)
                if result is not None:
                    f, m, t = result
                    all_features.append(f)
                    all_masks.append(m)
                    all_targets.append(t)
                    total_rows += len(f)
            
            if not all_features:
                continue
            
            # Concatenate and sample a batch
            features = np.concatenate(all_features, axis=0)
            masks = np.concatenate(all_masks, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            
            # Random sample
            n = len(features)
            if n >= batch_size:
                idx = np.random.choice(n, size=batch_size, replace=False)
            else:
                idx = np.random.choice(n, size=batch_size, replace=True)
            
            feat_batch = torch.tensor(features[idx], dtype=torch.float32)
            mask_batch = torch.tensor(masks[idx], dtype=torch.bool)
            target_batch = torch.tensor(targets[idx], dtype=torch.float32)
            
            batch_queue.put((feat_batch, mask_batch, target_batch), timeout=5.0)
        except queue.Full:
            continue
        except Exception as e:
            if not stop_event.is_set():
                print(f"[gen worker] Error: {e}", file=sys.stderr)
            continue


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)
    device = torch.device(DEVICE)
    
    print("=" * 70)
    print("ONLINE PRETRAINING — UNLIMITED DATA, ZERO DISK")
    print(f"Max features: {MAX_FEATURES}")
    print(f"Batch size: {BATCH_SIZE}, Max steps: {MAX_STEPS:,}")
    print(f"Target rows: {MAX_STEPS * BATCH_SIZE:,} ({MAX_STEPS * BATCH_SIZE / 1e9:.2f}B)")
    print(f"Gen workers: {NUM_GEN_WORKERS}, Queue size: {QUEUE_MAXSIZE}")
    print(f"LR: {LR}, WD: {WEIGHT_DECAY}")
    print("=" * 70)
    
    # Build model config
    config = ExperimentConfig(
        experiment_name="pretrain_online_v1",
        artifacts_root="artifacts",
        seed=SEED,
        task=TaskConfig(mode="pretrain", problem_type="regression", target_column="target"),
        data=DataConfig(
            dataset_type="synthetic",
            batch_size=BATCH_SIZE,
            standardize_numeric=False,
        ),
        model=ModelConfig(
            d_model=256,
            n_heads=8,
            n_layers=8,
            d_ff=512,
            dropout=0.1,
            feature_token_dropout=0.05,
            norm="rmsnorm",
            ffn_activation="swiglu",
            numeric_embedding="periodic",
            numeric_periodic_features=16,
            schema_encoder="hash",
            text_encoder="custom",
            pooling="cls",
        ),
        training=TrainingConfig(
            device=DEVICE,
            max_steps=MAX_STEPS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            grad_clip_norm=1.0,
            log_interval=LOG_INTERVAL,
            amp=True,
            amp_dtype="float16",
        ),
        episode=EpisodeConfig(enabled=False),
    )
    
    # Build model
    num_numeric = MAX_FEATURES
    num_categorical = 0
    num_text = 0
    output_dim = 1
    
    model = TabularTransformer(config, num_numeric, num_categorical, num_text, output_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    
    # LR schedule: warmup + cosine
    warmup_steps = 2000
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(MAX_STEPS - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Start background data generation
    batch_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    stop_event = threading.Event()
    
    workers = []
    for _ in range(NUM_GEN_WORKERS):
        t = threading.Thread(
            target=_batch_producer,
            args=(batch_queue, stop_event, MAX_FEATURES, BATCH_SIZE),
            daemon=True,
        )
        t.start()
        workers.append(t)
    
    print(f"Started {NUM_GEN_WORKERS} background data generators")
    
    # Wait for initial data
    print("Waiting for first batch...")
    time.sleep(3)
    
    output_dir = Path("artifacts/pretrain_online_v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.train()
    global_step = 0
    rows_seen = 0
    recent_losses = []
    best_loss = float("inf")
    start_time = time.time()
    queue_waits = 0
    
    print("\nStarting training...")
    
    try:
        while global_step < MAX_STEPS:
            # Get batch from queue (block up to 30s)
            try:
                features, masks, targets = batch_queue.get(timeout=30.0)
            except queue.Empty:
                queue_waits += 1
                print(f"  [WARNING] Queue empty for 30s (waits={queue_waits})")
                continue
            
            # Build TabularBatch on device
            bs = features.shape[0]
            empty_2d_long = torch.zeros(bs, 0, dtype=torch.long, device=device)
            empty_3d_long = torch.zeros(bs, 0, 0, dtype=torch.long, device=device)
            empty_2d_bool = torch.zeros(bs, 0, dtype=torch.bool, device=device)
            
            batch = TabularBatch(
                x_num=features.to(device),
                x_cat=empty_2d_long,
                x_text_token_ids=empty_3d_long,
                x_text_values=[],
                x_num_mask=masks.to(device),
                x_cat_mask=empty_2d_bool,
                x_text_mask=empty_2d_bool,
                num_schema_texts=None,
                cat_schema_texts=None,
                text_schema_texts=None,
                num_name_token_ids=None,
                cat_name_token_ids=None,
                text_name_token_ids=None,
                num_profile_vectors=None,
                cat_profile_vectors=None,
                text_profile_vectors=None,
                y=targets.to(device),
            )
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch)
                loss = nn.functional.mse_loss(logits.squeeze(-1), batch.y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            
            global_step += 1
            rows_seen += bs
            recent_losses.append(float(loss.item()))
            if len(recent_losses) > 200:
                recent_losses.pop(0)
            
            if global_step % LOG_INTERVAL == 0:
                avg_loss = np.mean(recent_losses)
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                rows_per_sec = rows_seen / max(elapsed, 1)
                qsize = batch_queue.qsize()
                print(
                    f"step={global_step:,}/{MAX_STEPS:,} "
                    f"loss={avg_loss:.4f} "
                    f"lr={lr_now:.2e} "
                    f"rows={rows_seen:,} "
                    f"rate={rows_per_sec:,.0f} rows/s "
                    f"queue={qsize}/{QUEUE_MAXSIZE}"
                )
            
            if global_step % CHECKPOINT_INTERVAL == 0:
                avg_loss = np.mean(recent_losses)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "global_step": global_step,
                        "rows_seen": rows_seen,
                        "best_loss": best_loss,
                        "num_numeric": num_numeric,
                        "num_categorical": num_categorical,
                        "num_text": num_text,
                        "output_dim": output_dim,
                    }, output_dir / "best.pt")
                    print(f"  -> Saved best checkpoint (loss={best_loss:.4f})")
                
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "rows_seen": rows_seen,
                }, output_dir / "latest.pt")
    
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        stop_event.set()
        for t in workers:
            t.join(timeout=5)
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PRETRAINING COMPLETE")
    print(f"Steps: {global_step:,}, Rows: {rows_seen:,} ({rows_seen/1e9:.2f}B)")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Elapsed: {elapsed/60:.1f} min ({elapsed/3600:.1f} hr)")
    print(f"Rate: {rows_seen/elapsed:,.0f} rows/s")
    print(f"Checkpoint: {output_dir / 'best.pt'}")
    print("=" * 70)
    
    (output_dir / "summary.json").write_text(json.dumps({
        "global_step": global_step,
        "rows_seen": rows_seen,
        "best_loss": best_loss,
        "elapsed_seconds": elapsed,
        "model_params": n_params,
    }, indent=2))


if __name__ == "__main__":
    main()
