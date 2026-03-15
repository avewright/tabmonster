#!/usr/bin/env python3
"""Finetune from pretrained corpus checkpoint on Adult Census.

Transfer the 8-layer transformer backbone from pretraining, reinitialize 
input/output heads for Adult Census schema (6 numeric + 8 categorical features).
"""
import sys
sys.path.insert(0, '.')
import copy
import gc
import time
import traceback

import numpy as np
import torch
from pathlib import Path

from run_experiment import (
    make_adult_config,
    log_experiment,
    get_best_val_loss,
    update_best,
)
from tabula.data.datasets import build_dataloaders
from tabula.models.transformer import TabularTransformer
from tabula.training.trunk import load_trunk_weights
from tabula.training.engine import (
    _move_batch, _make_criterion, _run_epoch, _compute_loss,
    _autocast_dtype, _use_amp, _build_scheduler,
)
from tabula.evaluation.metrics import compute_metrics
from tabula.utils import set_seed


CHECKPOINT_PATH = "artifacts/pretrain_corpus_v1/best.pt"


def run_transfer_experiment(exp_id, desc, config, checkpoint_path, swa_start, total_epochs,
                           freeze_backbone_epochs=0):
    """Finetune from a pretrained checkpoint with optional backbone freezing."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"DESC: {desc}")
    print(f"{'='*60}\n")

    start_time = time.time()
    torch.cuda.empty_cache()
    gc.collect()

    set_seed(config.seed)
    device = torch.device(config.training.device)
    train_loader, val_loader, num_numeric, num_categorical, num_text, output_dim = build_dataloaders(config)
    effective_output_dim = 1 if config.task.problem_type in ("binary", "regression") else output_dim

    # Build model with same architecture as pretrained (d256, 8L, 8H)
    model = TabularTransformer(config, num_numeric, num_categorical, num_text, effective_output_dim).to(device)
    
    # Transfer trunk weights
    summary = load_trunk_weights(model, checkpoint_path, device=str(device), verbose=True)
    print(f"  Transferred: {summary['transferred_count']} params")
    print(f"  Skipped (shape): {summary['skipped_shape_count']}")
    print(f"  Skipped (missing): {summary['skipped_missing_count']}")

    criterion = _make_criterion(config.task.problem_type, effective_output_dim)
    
    # Separate LR for backbone vs head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in ['blocks.', 'norm.']):
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config.training.lr * 0.1},  # 10x lower for backbone
        {'params': head_params, 'lr': config.training.lr},
    ], weight_decay=config.training.weight_decay)
    
    scaler = torch.amp.GradScaler("cuda", enabled=_use_amp(config, device))
    scheduler = _build_scheduler(config, optimizer)

    state_dicts = []
    best_val_loss = float("inf")

    for epoch in range(1, total_epochs + 1):
        # Optional: freeze backbone for first N epochs
        if epoch <= freeze_backbone_epochs:
            for p in backbone_params:
                p.requires_grad = False
        elif epoch == freeze_backbone_epochs + 1:
            for p in backbone_params:
                p.requires_grad = True

        model.train()
        train_losses = []
        for batch in train_loader:
            batch = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=_autocast_dtype(config), enabled=_use_amp(config, device)):
                logits = model(batch)
                loss = _compute_loss(config.task.problem_type, effective_output_dim, criterion, logits, batch.y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(loss.item()))

        if scheduler is not None:
            scheduler.step()

        # Validate
        val_result = _run_epoch(model, val_loader, device, criterion, None, config, effective_output_dim, None)
        train_loss = np.mean(train_losses)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_result.loss:.4f} val_metrics={val_result.metrics}")

        if epoch >= swa_start:
            state_dicts.append(copy.deepcopy(model.state_dict()))
        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss

    if not state_dicts:
        state_dicts.append(model.state_dict())

    # SWA averaging
    avg_state_dict = {}
    for key in state_dicts[0]:
        avg_state_dict[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)
    model.load_state_dict(avg_state_dict)

    # Final evaluation
    model.eval()
    all_y, all_logits = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = _move_batch(batch, device)
            logits = model(batch)
            all_y.append(batch.y.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_y)
    logits_arr = np.concatenate(all_logits)
    swa_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(logits_arr.reshape(-1)), torch.tensor(y_true.astype(np.float32))
    ).item())
    metrics = compute_metrics(config.task.problem_type, y_true, logits_arr)
    duration = time.time() - start_time

    print(f"\nSWA results: val_loss={swa_loss:.6f} accuracy={metrics.get('accuracy'):.4f} roc_auc={metrics.get('roc_auc'):.4f}")
    log_experiment(exp_id, desc, 'success', swa_loss, metrics.get('accuracy'), metrics.get('roc_auc'), total_epochs, duration)
    prev_best = get_best_val_loss()
    if swa_loss < prev_best:
        update_best(exp_id, swa_loss, metrics.get('roc_auc'))
        print(f"\n*** NEW BEST: val_loss={swa_loss:.6f} (prev={prev_best:.6f}) ***\n")
    else:
        print(f"\nNo improvement: val_loss={swa_loss:.6f} vs best={prev_best:.6f}\n")

    torch.cuda.empty_cache()
    gc.collect()
    return {"status": "success", "val_loss": swa_loss, "accuracy": metrics.get('accuracy'), "roc_auc": metrics.get('roc_auc'), "duration": duration}


def build_transfer_experiments():
    experiments = []
    base = 50  # Start from 50 to not conflict with wave 2

    # Exp 50: Transfer d256 8L 8H, finetune all, SWA
    def exp_050():
        config = make_adult_config(
            "exp_050_transfer_d256_swa",
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            d_model=256, n_heads=8, n_layers=8, d_ff=512,
            batch_size=512,
            lr=2e-4,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        config.model.schema_encoder = "hash"
        return "exp_050_transfer_d256_swa", "Transfer d256 8L finetune-all lr=2e-4 SWA from 25", config, 25, 40, 0

    # Exp 51: Transfer with frozen backbone for 5 epochs then unfreeze
    def exp_051():
        config = make_adult_config(
            "exp_051_transfer_freeze5_swa",
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            d_model=256, n_heads=8, n_layers=8, d_ff=512,
            batch_size=512,
            lr=3e-4,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        config.model.schema_encoder = "hash"
        return "exp_051_transfer_freeze5_swa", "Transfer freeze-5ep then unfreeze lr=3e-4 SWA from 25", config, 25, 40, 5

    # Exp 52: Transfer with lower LR for backbone (10x diff already in code)
    def exp_052():
        config = make_adult_config(
            "exp_052_transfer_lowlr_swa",
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            d_model=256, n_heads=8, n_layers=8, d_ff=512,
            batch_size=512,
            lr=5e-4,  # Head LR=5e-4, backbone LR=5e-5
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        config.model.schema_encoder = "hash"
        return "exp_052_transfer_lowlr_swa", "Transfer high head-lr=5e-4 backbone-lr=5e-5 SWA from 25", config, 25, 40, 0

    # Exp 53: Transfer with longer training (60 epochs)
    def exp_053():
        config = make_adult_config(
            "exp_053_transfer_long60_swa",
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            d_model=256, n_heads=8, n_layers=8, d_ff=512,
            batch_size=512,
            lr=1e-4,
            max_epochs=60,
            early_stopping_patience=999,
            amp=True,
        )
        config.model.schema_encoder = "hash"
        return "exp_053_transfer_long60_swa", "Transfer lr=1e-4 60ep SWA from 35", config, 35, 60, 0

    # Exp 54: Transfer with cosine warmup
    def exp_054():
        config = make_adult_config(
            "exp_054_transfer_cosine_swa",
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            d_model=256, n_heads=8, n_layers=8, d_ff=512,
            batch_size=512,
            lr=2e-4,
            scheduler='cosine_warmup',
            warmup_epochs=3,
            lr_min=1e-6,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        config.model.schema_encoder = "hash"
        return "exp_054_transfer_cosine_swa", "Transfer cosine-warmup lr=2e-4 SWA from 25", config, 25, 40, 0

    experiments.extend([exp_050, exp_051, exp_052, exp_053, exp_054])
    return experiments


if __name__ == "__main__":
    print("=" * 60)
    print("TRANSFER LEARNING EXPERIMENTS")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Current best val_loss: {get_best_val_loss():.6f}")
    print("=" * 60)

    if not Path(CHECKPOINT_PATH).exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Wait for pretraining to finish or update the path")
        sys.exit(1)

    experiments = build_transfer_experiments()
    for i, exp_fn in enumerate(experiments):
        exp_id, desc, config, swa_start, total_epochs, freeze_epochs = exp_fn()
        print(f"\n[{i+1}/{len(experiments)}] Starting {exp_id}...")
        try:
            result = run_transfer_experiment(
                exp_id, desc, config, CHECKPOINT_PATH,
                swa_start, total_epochs, freeze_epochs
            )
            print(f"  -> val_loss={result.get('val_loss')}, roc_auc={result.get('roc_auc')}, duration={result.get('duration', 0):.1f}s")
        except Exception as e:
            print(f"  -> CRASH: {e}")
            traceback.print_exc()
            log_experiment(exp_id, desc, 'crash', None, None, None, None, 0, notes=str(e)[:200])
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 60)
    print(f"TRANSFER COMPLETE. Best val_loss: {get_best_val_loss():.6f}")
    print("=" * 60)
