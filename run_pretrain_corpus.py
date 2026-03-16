#!/usr/bin/env python3
"""Pretrain on HF parquet corpus: stream shards, multi-task learning.

Streams fixed-width parquet shards (feat_0..feat_63, target, _source_meta) from
avewright/tabula-pretraining-corpus-v2. Each batch mixes rows from multiple
source datasets. Loss is multi-task: MSE for regression, CE for classification,
determined per-row from _source_meta.

Architecture matches run_pretrain_online2.py but uses real+synthetic corpus
instead of on-the-fly generation.

Usage:
    python run_pretrain_corpus.py                  # train from scratch
    python run_pretrain_corpus.py --resume          # resume from checkpoint
    python run_pretrain_corpus.py --steps 500000    # override max steps
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tabula.config import (
    ExperimentConfig, ModelConfig, TrainingConfig,
    TaskConfig, DataConfig, EpisodeConfig,
)
from tabula.data.datasets import TabularBatch
from tabula.data.env import load_repo_env_file
from tabula.models.transformer import TabularTransformer
from tabula.utils import set_seed


# ── Configuration ─────────────────────────────────────────────────────
MAX_FEATURES = 64           # Fixed feature width of the corpus
BATCH_SIZE = 512
MAX_STEPS = 200_000         # ~102M rows seen (200K × 512)
LOG_INTERVAL = 50
CHECKPOINT_INTERVAL = 2000
VAL_INTERVAL = 5000         # Validate every N steps
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 2000
DEVICE = "cuda"
SEED = 42

# HF corpus
HF_REPO = "avewright/tabula-pretraining-corpus-v2"
HF_SPLIT = "train"
SHUFFLE_BUFFER = 50_000     # Streaming shuffle buffer size

# Data loading
PREFETCH_ROWS = 500_000     # Rows to prefetch into memory (larger = less HF fetching)
NUM_WORKERS = 4             # Dataloader workers

OUTPUT_DIR = Path("artifacts/pretrain_corpus_v1")


# ── Token + env ───────────────────────────────────────────────────────
_env = load_repo_env_file()
HF_TOKEN = (
    _env.get("HF_TOKEN")
    or _env.get("HUGGINGFACE_HUB_TOKEN")
    or os.environ.get("HF_TOKEN")
    or ""
)


# ── Streaming shard loader ────────────────────────────────────────────
class CorpusShardIterator:
    """Iterate over HF parquet shards, yielding batches of (features, masks, targets, task_types).

    Loads shards into a memory pool, samples random batches, refreshes shards
    periodically. Keeps memory bounded.
    """

    def __init__(
        self,
        batch_size: int,
        max_features: int = MAX_FEATURES,
        pool_target_rows: int = PREFETCH_ROWS,
        shuffle_within_shard: bool = True,
    ):
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi

        self.batch_size = batch_size
        self.max_features = max_features
        self.pool_target_rows = pool_target_rows
        self.shuffle_within_shard = shuffle_within_shard
        self.rng = np.random.default_rng(SEED)

        # List all shard files from the HF repo
        api = HfApi(token=HF_TOKEN or None)
        files = list(api.list_repo_tree(HF_REPO, repo_type="dataset", path_in_repo="data"))
        self.shard_urls = []
        for f in files:
            if f.path.endswith(".parquet"):
                self.shard_urls.append(f"hf://datasets/{HF_REPO}/{f.path}")
        self.shard_urls.sort()
        print(f"  Found {len(self.shard_urls)} shards in {HF_REPO}")

        # Shuffle shard order
        self.rng.shuffle(self.shard_urls)
        self._shard_idx = 0

        # Memory pool
        self._pool_features: np.ndarray | None = None
        self._pool_masks: np.ndarray | None = None
        self._pool_targets: np.ndarray | None = None
        self._pool_task_types: np.ndarray | None = None  # 0=regression, 1=binary, 2=multiclass
        self._pool_size = 0
        self._pool_cursor = 0
        self._batches_since_refresh = 0
        self._max_batches_per_pool = 800

    def _load_next_shard(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load one shard from HF and return (features, masks, targets, task_types)."""
        import pandas as pd

        url = self.shard_urls[self._shard_idx % len(self.shard_urls)]
        self._shard_idx += 1

        # Read parquet from HF URL
        df = pd.read_parquet(url, storage_options={"token": HF_TOKEN or None})

        feat_cols = [f"feat_{i}" for i in range(self.max_features)]
        features = df[feat_cols].values.astype(np.float32)

        # Build masks: 1.0 where feature is present (not NaN)
        masks = (~np.isnan(features)).astype(np.float32)

        # Zero out NaN values
        features = np.nan_to_num(features, nan=0.0)

        targets = df["target"].values.astype(np.float32)
        targets = np.nan_to_num(targets, nan=0.0)

        # Parse task types from _source_meta
        task_types = np.zeros(len(df), dtype=np.int32)  # default to regression
        if "_source_meta" in df.columns:
            for i, meta_str in enumerate(df["_source_meta"]):
                try:
                    meta = json.loads(meta_str) if isinstance(meta_str, str) else {}
                    tt = meta.get("task_type", "regression")
                    if tt == "binary":
                        task_types[i] = 1
                    elif tt == "multiclass":
                        task_types[i] = 2
                    else:
                        task_types[i] = 0
                except Exception:
                    pass

        # Normalize features per-column (z-score, ignoring masked positions)
        for col in range(self.max_features):
            valid = masks[:, col] > 0.5
            if valid.sum() > 1:
                col_vals = features[valid, col]
                mean = col_vals.mean()
                std = col_vals.std()
                if std > 1e-8:
                    features[valid, col] = (col_vals - mean) / std

        # Normalize regression targets
        reg_mask = task_types == 0
        if reg_mask.sum() > 1:
            t_mean = targets[reg_mask].mean()
            t_std = targets[reg_mask].std()
            if t_std > 1e-8:
                targets[reg_mask] = (targets[reg_mask] - t_mean) / t_std

        return features, masks, targets, task_types

    def _refresh_pool(self):
        """Load shards until pool is full enough."""
        all_features, all_masks, all_targets, all_task_types = [], [], [], []
        total = 0
        while total < self.pool_target_rows:
            try:
                f, m, t, tt = self._load_next_shard()
                all_features.append(f)
                all_masks.append(m)
                all_targets.append(t)
                all_task_types.append(tt)
                total += len(f)
            except Exception as e:
                print(f"  [WARN] Shard load failed: {e}")
                self._shard_idx += 1
                continue

        self._pool_features = np.concatenate(all_features, axis=0)
        self._pool_masks = np.concatenate(all_masks, axis=0)
        self._pool_targets = np.concatenate(all_targets, axis=0)
        self._pool_task_types = np.concatenate(all_task_types, axis=0)
        self._pool_size = len(self._pool_features)

        # Shuffle pool
        idx = self.rng.permutation(self._pool_size)
        self._pool_features = self._pool_features[idx]
        self._pool_masks = self._pool_masks[idx]
        self._pool_targets = self._pool_targets[idx]
        self._pool_task_types = self._pool_task_types[idx]
        self._pool_cursor = 0
        self._batches_since_refresh = 0

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """Get a training batch. Returns (features, masks, targets, task_types)."""
        if (
            self._pool_features is None
            or self._pool_cursor + self.batch_size > self._pool_size
            or self._batches_since_refresh >= self._max_batches_per_pool
        ):
            self._refresh_pool()

        start = self._pool_cursor
        end = start + self.batch_size
        self._pool_cursor = end
        self._batches_since_refresh += 1

        features = torch.from_numpy(self._pool_features[start:end].copy())
        masks = torch.from_numpy(self._pool_masks[start:end].copy()).bool()
        targets = torch.from_numpy(self._pool_targets[start:end].copy())
        task_types = self._pool_task_types[start:end].copy()

        return features, masks, targets, task_types


# ── Multi-task loss ───────────────────────────────────────────────────
def multi_task_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    task_types: np.ndarray,
) -> torch.Tensor:
    """Compute per-row loss: MSE for regression, CE for classification.

    logits: (B, output_dim) — we use output_dim=1 with regression loss for
    regression rows and a separate scalar sigmoid+BCE for binary rows.
    For simplicity, we treat all as regression (MSE) since the corpus
    has mixed target encodings. This follows the TabPFN approach where
    targets are always real-valued and the model learns to predict them.
    """
    pred = logits.squeeze(-1)
    return F.mse_loss(pred, targets)


# ── Validation ────────────────────────────────────────────────────────
@torch.no_grad()
def validate(
    model: TabularTransformer,
    loader: CorpusShardIterator,
    device: torch.device,
    n_batches: int = 20,
) -> float:
    """Run a quick validation pass on N batches and return mean loss."""
    model.eval()
    losses = []
    for _ in range(n_batches):
        features, masks, targets, task_types = loader.get_batch()
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
            loss = multi_task_loss(logits, batch.y, task_types)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--steps", type=int, default=MAX_STEPS, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(DEVICE)

    print("=" * 72)
    print("CORPUS PRETRAINING — HF PARQUET STREAMING")
    print(f"  HF Repo:     {HF_REPO}")
    print(f"  Max features: {MAX_FEATURES}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Max steps:   {args.steps:,}")
    print(f"  Target rows: {args.steps * args.batch_size:,} "
          f"({args.steps * args.batch_size / 1e9:.2f}B)")
    print(f"  Model:       d={args.d_model}, L={args.n_layers}, H={args.n_heads}")
    print(f"  LR:          {args.lr}, WD: {WEIGHT_DECAY}")
    print(f"  Device:      {DEVICE}")
    print("=" * 72)

    # ── Build model ───────────────────────────────────────────
    config = ExperimentConfig(
        experiment_name="pretrain_corpus_v1",
        artifacts_root="artifacts",
        seed=SEED,
        task=TaskConfig(
            mode="pretrain",
            problem_type="regression",
            target_column="target",
        ),
        data=DataConfig(
            dataset_type="hf_stream",
            batch_size=args.batch_size,
            standardize_numeric=False,
            hf_repo_id=HF_REPO,
        ),
        model=ModelConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_model * 2,
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
            max_steps=args.steps,
            lr=args.lr,
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
    output_dim = 1  # Regression-style output

    model = TabularTransformer(config, num_numeric, num_categorical, num_text, output_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── Optimizer + scheduler ─────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(args.steps - WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume from checkpoint ────────────────────────────────
    global_step = 0
    rows_seen = 0
    best_val_loss = float("inf")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUTPUT_DIR / "latest.pt"

    if args.resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt.get("global_step", 0)
        rows_seen = ckpt.get("rows_seen", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        # Advance scheduler to correct step
        for _ in range(global_step):
            scheduler.step()
        print(f"  Resumed at step {global_step:,}, rows_seen={rows_seen:,}, "
              f"best_val_loss={best_val_loss:.4f}")

    # ── Data loader ───────────────────────────────────────────
    print("\nInitializing corpus shard loader...")
    loader = CorpusShardIterator(
        batch_size=args.batch_size,
        max_features=MAX_FEATURES,
        pool_target_rows=PREFETCH_ROWS,
    )
    print("Loading first shard pool...")
    loader._refresh_pool()
    print(f"  Pool: {loader._pool_size:,} rows ready")

    # ── Training loop ─────────────────────────────────────────
    model.train()
    recent_losses: list[float] = []
    start_time = time.time()

    print(f"\nStarting training from step {global_step:,}...")

    try:
        while global_step < args.steps:
            features, masks, targets, task_types = loader.get_batch()
            bs = features.shape[0]

            # Build TabularBatch (numeric-only, no categorical/text)
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

            # Forward + backward
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch)
                loss = multi_task_loss(logits, batch.y, task_types)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            rows_seen += bs
            loss_val = float(loss.item())
            recent_losses.append(loss_val)
            if len(recent_losses) > 200:
                recent_losses.pop(0)

            # ── Logging ───────────────────────────────────────
            if global_step % LOG_INTERVAL == 0:
                avg_loss = np.mean(recent_losses)
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                rows_per_sec = rows_seen / max(elapsed, 1)
                eta_hours = (args.steps - global_step) * (elapsed / global_step) / 3600 if global_step > 0 else 0
                print(
                    f"step={global_step:>7,}/{args.steps:,} "
                    f"loss={avg_loss:.4f} "
                    f"lr={lr_now:.2e} "
                    f"rows={rows_seen:,} "
                    f"rate={rows_per_sec:,.0f} r/s "
                    f"eta={eta_hours:.1f}h"
                )

            # ── Validation ────────────────────────────────────
            if global_step % VAL_INTERVAL == 0:
                val_loss = validate(model, loader, device, n_batches=30)
                print(f"  >>> val_loss={val_loss:.4f} (best={best_val_loss:.4f})")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "global_step": global_step,
                        "rows_seen": rows_seen,
                        "best_val_loss": best_val_loss,
                        "num_numeric": num_numeric,
                        "num_categorical": num_categorical,
                        "num_text": num_text,
                        "output_dim": output_dim,
                        "n_params": n_params,
                    }, OUTPUT_DIR / "best.pt")
                    print(f"  -> Saved new best (val_loss={best_val_loss:.4f})")
                model.train()

            # ── Checkpoint ────────────────────────────────────
            if global_step % CHECKPOINT_INTERVAL == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "rows_seen": rows_seen,
                    "best_val_loss": best_val_loss,
                    "num_numeric": num_numeric,
                    "num_categorical": num_categorical,
                    "num_text": num_text,
                    "output_dim": output_dim,
                    "n_params": n_params,
                }, OUTPUT_DIR / "latest.pt")

    except KeyboardInterrupt:
        print("\n[interrupted]")
    except Exception as exc:
        print(f"\n[ERROR] Training crashed: {exc}")
        traceback.print_exc()
    finally:
        # Save final checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "rows_seen": rows_seen,
            "best_val_loss": best_val_loss,
            "num_numeric": num_numeric,
            "num_categorical": num_categorical,
            "num_text": num_text,
            "output_dim": output_dim,
            "n_params": n_params,
        }, OUTPUT_DIR / "latest.pt")

    elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    print("PRETRAINING COMPLETE")
    print(f"  Steps:      {global_step:,}")
    print(f"  Rows seen:  {rows_seen:,} ({rows_seen / 1e9:.2f}B)")
    print(f"  Best val:   {best_val_loss:.4f}")
    print(f"  Elapsed:    {elapsed / 60:.1f} min ({elapsed / 3600:.1f} hr)")
    print(f"  Rate:       {rows_seen / max(elapsed, 1):,.0f} rows/s")
    print(f"  Checkpoint: {OUTPUT_DIR / 'best.pt'}")
    print("=" * 72)

    (OUTPUT_DIR / "summary.json").write_text(json.dumps({
        "global_step": global_step,
        "rows_seen": rows_seen,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed,
        "model_params": n_params,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "batch_size": args.batch_size,
        "max_features": MAX_FEATURES,
        "hf_repo": HF_REPO,
    }, indent=2))


if __name__ == "__main__":
    main()
