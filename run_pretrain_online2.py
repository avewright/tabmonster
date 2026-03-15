#!/usr/bin/env python3
"""Online pretraining v2: multiprocess generation, zero disk usage.

Uses ProcessPoolExecutor for true parallel CPU generation (no GIL).
Targets 200K+ steps = 102M+ rows seen. Each batch is freshly generated
synthetic data — effectively infinite, never repeats.

Architecture:
  - ProcessPoolExecutor generates datasets across all CPUs
  - Completed futures are assembled into batches
  - GPU trains on each batch immediately
  - Features normalized per-dataset (z-score) before training

Hardware: 48 CPUs, 247 GB RAM, RTX A4500 (20 GB VRAM)
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from pathlib import Path

warnings.filterwarnings('ignore', category=RuntimeWarning)

sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import torch
from torch import nn

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
MAX_STEPS = 200_000      # 200K steps × 512 = 102M rows (increase for more)
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 2000
LR = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda"
SEED = 42

# Process pool settings
NUM_WORKERS = 32          # Parallel dataset generators
PREFETCH_DATASETS = 128   # Keep this many datasets in flight

# Dataset sizes — generate many small datasets for diversity
MIN_SAMPLES_PER_DS = 128
MAX_SAMPLES_PER_DS = 2048
MIN_FEATURES = 4
MAX_GEN_FEATURES = 48

METHODS = [
    "TreePrior", "SCM", "GaussianMixture", "Polynomial",
]


# ---------------------------------------------------------------------------
# Worker function (runs in separate process - no GIL)
# ---------------------------------------------------------------------------
def generate_batch(args: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Generate a full training batch from multiple synthetic datasets.
    
    Runs in a child process. Produces exactly batch_size rows by 
    concatenating several small datasets — amortizes IPC overhead.
    """
    seed, batch_size = args
    rng = np.random.default_rng(seed)
    
    all_features = []
    all_masks = []
    all_targets = []
    total_rows = 0
    attempt = 0
    
    while total_rows < batch_size and attempt < 50:
        attempt += 1
        ds_seed = int(rng.integers(0, 2**31))
        ds_rng = np.random.default_rng(ds_seed)
        
        n_features = int(ds_rng.integers(MIN_FEATURES, MAX_GEN_FEATURES + 1))
        need = batch_size - total_rows
        n_samples = int(ds_rng.integers(max(MIN_SAMPLES_PER_DS, min(need, MIN_SAMPLES_PER_DS)),
                                         min(MAX_SAMPLES_PER_DS, max(need + 64, MAX_SAMPLES_PER_DS)) + 1))
        method = str(ds_rng.choice(METHODS))
        
        task_types = ["binary", "binary", "multiclass", "regression"]
        task_type = str(ds_rng.choice(task_types))
        n_classes = 2 if task_type == "binary" else (int(ds_rng.integers(3, 8)) if task_type == "multiclass" else 2)
        
        try:
            if method == "TreePrior":
                gen = TreePriorGenerator(n_samples=n_samples, n_features=n_features, n_classes=n_classes)
            elif method == "SCM":
                gen = SCMGenerator(n_samples=n_samples, n_features=n_features, n_classes=n_classes)
            elif method == "GaussianMixture":
                gen = GaussianMixtureGenerator(
                    n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                    n_components=int(ds_rng.integers(3, 10)),
                    label_strategy=str(ds_rng.choice(["linear", "quadratic", "tree"])),
                )
            elif method == "Polynomial":
                gen = PolynomialGenerator(
                    n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                    degree=int(ds_rng.choice([2, 3, 4])),
                )
            else:
                continue
            
            df, meta = gen.generate(seed=ds_seed)
        except Exception:
            continue
        
        target_col = meta.target_name
        feature_cols = [c for c in df.columns if c != target_col]
        actual_features = min(len(feature_cols), MAX_FEATURES)
        if actual_features < 2:
            continue
        
        raw = df[feature_cols[:actual_features]].values.astype(np.float32)
        means = np.nanmean(raw, axis=0, keepdims=True)
        stds = np.nanstd(raw, axis=0, keepdims=True)
        stds = np.where(stds < 1e-8, 1.0, stds)
        means = np.nan_to_num(means, nan=0.0)
        raw = (raw - means) / stds
        raw = np.nan_to_num(raw, nan=0.0).astype(np.float32)
        
        n_rows = raw.shape[0]
        features = np.zeros((n_rows, MAX_FEATURES), dtype=np.float32)
        features[:, :actual_features] = raw
        masks = np.zeros((n_rows, MAX_FEATURES), dtype=np.float32)
        masks[:, :actual_features] = 1.0
        
        targets = df[target_col].values.astype(np.float32)
        if task_type == "regression":
            t_mean = np.nanmean(targets)
            t_std = np.nanstd(targets)
            if t_std > 1e-8:
                targets = (targets - t_mean) / t_std
        targets = np.nan_to_num(targets, nan=0.0)
        
        all_features.append(features)
        all_masks.append(masks)
        all_targets.append(targets)
        total_rows += n_rows
    
    if total_rows < batch_size // 2:
        return None
    
    feat = np.concatenate(all_features, axis=0)[:batch_size]
    mask = np.concatenate(all_masks, axis=0)[:batch_size]
    tgt = np.concatenate(all_targets, axis=0)[:batch_size]
    
    # Shuffle so datasets are interleaved
    idx = rng.permutation(len(feat))
    return feat[idx], mask[idx], tgt[idx]


# ---------------------------------------------------------------------------
# Batch assembler: collects completed futures into training batches
# ---------------------------------------------------------------------------
class OnlineBatchAssembler:
    """Manages a ProcessPoolExecutor that produces full batches in parallel."""
    
    def __init__(self, num_workers: int, prefetch: int, batch_size: int):
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.executor = ProcessPoolExecutor(max_workers=num_workers)
        self.pending: list[Future] = []
        self.seed_counter = 0
        
        # Submit initial wave — each future produces one full batch
        self._submit_wave(prefetch)
    
    def _submit_wave(self, count: int):
        for _ in range(count):
            self.seed_counter += 1
            future = self.executor.submit(generate_batch, (self.seed_counter + 1_000_000, self.batch_size))
            self.pending.append(future)
    
    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get one training batch. Blocks until a batch is ready."""
        while True:
            for i, f in enumerate(self.pending):
                if f.done():
                    # Remove from pending and submit replacement
                    self.pending.pop(i)
                    self._submit_wave(1)
                    
                    try:
                        result = f.result()
                        if result is not None:
                            feats, masks, targets = result
                            return (
                                torch.from_numpy(feats),
                                torch.from_numpy(masks).bool(),
                                torch.from_numpy(targets),
                            )
                    except Exception:
                        pass
                    # If result was None or errored, try next
                    continue
            
            time.sleep(0.005)  # Brief sleep if nothing ready
    
    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)
    device = torch.device(DEVICE)
    
    print("=" * 70)
    print("ONLINE PRETRAINING v2 — MULTIPROCESS, UNLIMITED DATA")
    print(f"Max features: {MAX_FEATURES}")
    print(f"Batch size: {BATCH_SIZE}, Max steps: {MAX_STEPS:,}")
    print(f"Target rows: {MAX_STEPS * BATCH_SIZE:,} ({MAX_STEPS * BATCH_SIZE / 1e9:.2f}B)")
    print(f"Workers: {NUM_WORKERS}, Prefetch: {PREFETCH_DATASETS}")
    print(f"LR: {LR}, WD: {WEIGHT_DECAY}")
    print("=" * 70)
    
    # Build model config
    config = ExperimentConfig(
        experiment_name="pretrain_online_v2",
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
    
    num_numeric = MAX_FEATURES
    num_categorical = 0
    num_text = 0
    output_dim = 1
    
    model = TabularTransformer(config, num_numeric, num_categorical, num_text, output_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    
    warmup_steps = 2000
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(MAX_STEPS - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Start batch assembler
    assembler = OnlineBatchAssembler(NUM_WORKERS, PREFETCH_DATASETS, BATCH_SIZE)
    print(f"Started {NUM_WORKERS} process pool workers with {PREFETCH_DATASETS} prefetch")
    
    # Wait for initial data to populate
    print("Waiting for initial data generation...")
    time.sleep(5)
    
    output_dir = Path("artifacts/pretrain_online_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.train()
    global_step = 0
    rows_seen = 0
    recent_losses = []
    best_loss = float("inf")
    start_time = time.time()
    
    print("\nStarting training...")
    
    try:
        while global_step < MAX_STEPS:
            features, masks, targets = assembler.get_batch()
            
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
                pending = len(assembler.pending)
                print(
                    f"step={global_step:,}/{MAX_STEPS:,} "
                    f"loss={avg_loss:.4f} "
                    f"lr={lr_now:.2e} "
                    f"rows={rows_seen:,} "
                    f"rate={rows_per_sec:,.0f} rows/s "
                    f"pending={pending}"
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
        assembler.shutdown()
    
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
