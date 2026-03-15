#!/usr/bin/env python3
"""Pretrain on local parquet corpus (253M rows).

Reads shards from corpus/pretrain/, samples random chunks of rows from 
random datasets, feeds them to the TabularTransformer for multi-task
pretraining (binary/multiclass/regression mixed).

The model learns general-purpose tabular representations by training on 
thousands of diverse synthetic tasks. After pretraining, we finetune on
Adult Census to see if transfer helps.

Usage:
    python run_pretrain.py
"""
from __future__ import annotations

import gc
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from tabula.config import ExperimentConfig, ModelConfig, TrainingConfig, TaskConfig, DataConfig, EpisodeConfig
from tabula.data.datasets import TabularBatch
from tabula.models.transformer import TabularTransformer
from tabula.evaluation.metrics import compute_metrics
from tabula.utils import set_seed


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CORPUS_DIR = Path("corpus/pretrain")
MAX_FEATURES = 32   # Fixed feature dimension for pretraining (most datasets have <32 features)
BATCH_SIZE = 512
MAX_STEPS = 50_000
VAL_INTERVAL = 1000
LOG_INTERVAL = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda"
SEED = 42


# ---------------------------------------------------------------------------
# Fast in-memory corpus loader
# ---------------------------------------------------------------------------
class CorpusLoader:
    """Loads random shards into a memory pool and yields random batches.
    
    Strategy:
    - Keep a pool of ~5 shards in memory (~10M rows) 
    - Sample random batches from pool
    - Periodically swap in new shards
    """
    
    def __init__(self, corpus_dir: Path, max_features: int, batch_size: int,
                 pool_shards: int = 3):
        self.corpus_dir = corpus_dir
        self.max_features = max_features
        self.batch_size = batch_size
        self.pool_shards = pool_shards
        self.shard_paths = sorted(corpus_dir.glob("shard_*.parquet"))
        if not self.shard_paths:
            raise ValueError(f"No shards found in {corpus_dir}")
        
        self.rng = np.random.default_rng(42)
        self._pool_features = None
        self._pool_masks = None
        self._pool_targets = None
        self._pool_size = 0
        self._batches_from_pool = 0
        self._max_batches_per_pool = 500  # Refresh pool every N batches
        
    def _load_shard(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a shard and extract features/targets/masks."""
        df = pd.read_parquet(path, columns=[f"feature_{i}" for i in range(self.max_features)] + ["target", "_n_features"])
        
        feature_cols = [f"feature_{i}" for i in range(self.max_features)]
        features = df[feature_cols].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)
        
        targets = df["target"].values.astype(np.float32)
        targets = np.nan_to_num(targets, nan=0.0)
        
        n_features = df["_n_features"].values.astype(np.int32)
        masks = np.zeros((len(df), self.max_features), dtype=np.float32)
        for i, nf in enumerate(n_features):
            masks[i, :min(nf, self.max_features)] = 1.0
        
        return features, masks, targets
    
    def _refresh_pool(self):
        """Load new random shards into memory pool."""
        selected = self.rng.choice(self.shard_paths, size=min(self.pool_shards, len(self.shard_paths)), replace=False)
        
        all_features = []
        all_masks = []
        all_targets = []
        
        for spath in selected:
            try:
                f, m, t = self._load_shard(spath)
                all_features.append(f)
                all_masks.append(m)
                all_targets.append(t)
            except Exception as e:
                print(f"Warning: failed to load {spath}: {e}")
        
        if not all_features:
            raise RuntimeError("Could not load any shards")
        
        self._pool_features = np.concatenate(all_features, axis=0)
        self._pool_masks = np.concatenate(all_masks, axis=0)
        self._pool_targets = np.concatenate(all_targets, axis=0)
        self._pool_size = len(self._pool_features)
        self._batches_from_pool = 0
        
        # Shuffle the pool
        perm = self.rng.permutation(self._pool_size)
        self._pool_features = self._pool_features[perm]
        self._pool_masks = self._pool_masks[perm]
        self._pool_targets = self._pool_targets[perm]
    
    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a random batch from the pool."""
        if self._pool_features is None or self._batches_from_pool >= self._max_batches_per_pool:
            self._refresh_pool()
        
        indices = self.rng.choice(self._pool_size, size=self.batch_size, replace=False)
        self._batches_from_pool += 1
        
        features = torch.tensor(self._pool_features[indices], dtype=torch.float32)
        masks = torch.tensor(self._pool_masks[indices], dtype=torch.bool)
        targets = torch.tensor(self._pool_targets[indices], dtype=torch.float32)
        
        return features, masks, targets


def collate_pretrain(batch):
    """Custom collate for pretraining batches."""
    features, masks, targets, n_classes_arr = zip(*batch)
    
    x_num = torch.tensor(np.array(features), dtype=torch.float32)
    x_num_mask = torch.tensor(np.array(masks), dtype=torch.bool)
    y = torch.tensor(np.array(targets), dtype=torch.float32)
    n_classes = torch.tensor(np.array(n_classes_arr), dtype=torch.long)
    
    batch_size = x_num.shape[0]
    n_features = x_num.shape[1]
    
    # Build a minimal TabularBatch (no categoricals/text for pretraining)
    empty_2d_long = torch.zeros(batch_size, 0, dtype=torch.long)
    empty_3d_long = torch.zeros(batch_size, 0, 0, dtype=torch.long)
    empty_2d_bool = torch.zeros(batch_size, 0, dtype=torch.bool)
    
    # Dummy metadata tensors — use None to skip metadata path in model
    # When these are None, the model skips the metadata embedding addition
    
    return TabularBatch(
        x_num=x_num,
        x_cat=empty_2d_long,
        x_text_token_ids=empty_3d_long,
        x_text_values=[],
        x_num_mask=x_num_mask,
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
        y=y,
    ), n_classes


# ---------------------------------------------------------------------------
# Multi-task loss: handle mixed classification + regression in same batch
# ---------------------------------------------------------------------------
def compute_multitask_loss(logits, y, n_classes):
    """Mixed loss: classification tasks use CE, regression tasks use MSE."""
    # All tasks use the same output head — for pretraining we use 
    # a simple MSE on the raw target value. This works because:
    # 1. Classification targets are integers (0, 1, 2...) 
    # 2. MSE can approximate classification as regression
    # 3. It's simpler than switching loss per-sample
    return nn.functional.mse_loss(logits.squeeze(-1), y)


# ---------------------------------------------------------------------------
# Build validation set from Adult Census (our target benchmark)
# ---------------------------------------------------------------------------
def build_val_loader():
    """Build Adult Census val loader for evaluating pretrained representations."""
    from run_experiment import make_adult_config
    from tabula.data.datasets import build_dataloaders
    
    config = make_adult_config("pretrain_eval", 
                                numeric_embedding="periodic",
                                numeric_periodic_features=16,
                                amp=True)
    _, val_loader, num_numeric, num_categorical, num_text, output_dim = build_dataloaders(config)
    return val_loader, num_numeric, num_categorical, num_text, output_dim


# ---------------------------------------------------------------------------
# Main pretraining loop
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)
    device = torch.device(DEVICE)
    
    print("=" * 70)
    print("PRETRAINING ON LOCAL CORPUS")
    print(f"Corpus: {CORPUS_DIR}")
    n_shards = len(list(CORPUS_DIR.glob("shard_*.parquet")))
    print(f"Shards: {n_shards}")
    print(f"Max features: {MAX_FEATURES}")
    print(f"Batch size: {BATCH_SIZE}, Max steps: {MAX_STEPS}")
    print(f"LR: {LR}, WD: {WEIGHT_DECAY}")
    print("=" * 70)
    
    # Build pretrain config
    config = ExperimentConfig(
        experiment_name="pretrain_corpus_v1",
        artifacts_root="artifacts",
        seed=SEED,
        task=TaskConfig(mode="pretrain", problem_type="regression", target_column="target"),
        data=DataConfig(
            dataset_type="synthetic",
            batch_size=BATCH_SIZE,
            standardize_numeric=False,  # data is already generated
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
            schema_encoder="hash",  # No pretrained schema encoder for pretraining
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
    output_dim = 1  # regression-style pretraining
    
    model = TabularTransformer(config, num_numeric, num_categorical, num_text, output_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    
    # LR schedule: warmup + cosine
    warmup_steps = 1000
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(MAX_STEPS - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Build corpus loader (in-memory pool sampling)
    corpus = CorpusLoader(CORPUS_DIR, max_features=MAX_FEATURES, batch_size=BATCH_SIZE, pool_shards=1)
    
    # Training loop
    output_dir = Path("artifacts/pretrain_corpus_v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.train()
    global_step = 0
    rows_seen = 0
    recent_losses = []
    best_loss = float("inf")
    start_time = time.time()
    
    print("\nStarting training...")
    
    while global_step < MAX_STEPS:
        features, masks, targets = corpus.get_batch()
        
        # Build TabularBatch on device
        batch_size_actual = features.shape[0]
        empty_2d_long = torch.zeros(batch_size_actual, 0, dtype=torch.long, device=device)
        empty_3d_long = torch.zeros(batch_size_actual, 0, 0, dtype=torch.long, device=device)
        empty_2d_bool = torch.zeros(batch_size_actual, 0, dtype=torch.bool, device=device)
        
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
        rows_seen += batch.y.shape[0]
        recent_losses.append(float(loss.item()))
        if len(recent_losses) > 200:
            recent_losses.pop(0)
        
        if global_step % LOG_INTERVAL == 0:
            avg_loss = np.mean(recent_losses)
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            rows_per_sec = rows_seen / max(elapsed, 1)
            print(
                f"step={global_step}/{MAX_STEPS} "
                f"loss={avg_loss:.4f} "
                f"lr={lr_now:.2e} "
                f"rows={rows_seen:,} "
                f"rate={rows_per_sec:,.0f} rows/s"
            )
        
        if global_step % VAL_INTERVAL == 0:
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
            
            # Save latest
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "rows_seen": rows_seen,
            }, output_dir / "latest.pt")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PRETRAINING COMPLETE")
    print(f"Steps: {global_step}, Rows: {rows_seen:,}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print(f"Checkpoint: {output_dir / 'best.pt'}")
    print("=" * 70)
    
    # Save summary
    (output_dir / "summary.json").write_text(json.dumps({
        "global_step": global_step,
        "rows_seen": rows_seen,
        "best_loss": best_loss,
        "elapsed_seconds": elapsed,
        "model_params": n_params,
    }, indent=2))


if __name__ == "__main__":
    main()
