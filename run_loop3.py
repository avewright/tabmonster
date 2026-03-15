#!/usr/bin/env python3
"""Wave 3: Creative approaches to break the 0.3015 ceiling.

Focuses on:
- Label smoothing (regularization via soft targets)
- Snapshot ensembles (cosine restarts, average predictions at minima)
- Attention pooling instead of CLS
- Larger SWA windows
- Multi-seed averaging (same config, different seeds)
"""
import sys
sys.path.insert(0, '.')
import copy
import gc
import math
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
from run_loop import run_swa_experiment
from tabula.data.datasets import build_dataloaders
from tabula.models.transformer import TabularTransformer
from tabula.training.engine import (
    _move_batch, _make_criterion, _run_epoch, _compute_loss,
    _autocast_dtype, _use_amp, _build_scheduler,
)
from tabula.evaluation.metrics import compute_metrics
from tabula.utils import set_seed


def run_snapshot_ensemble(exp_id, desc, config, n_cycles=5, epochs_per_cycle=8):
    """Train with cosine annealing restarts. At each minimum, save a snapshot.
    Final prediction = average of all snapshot predictions."""
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

    model = TabularTransformer(config, num_numeric, num_categorical, num_text, effective_output_dim).to(device)
    criterion = _make_criterion(config.task.problem_type, effective_output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=_use_amp(config, device))

    snapshots = []  # List of state_dicts at cycle minima
    total_epochs = n_cycles * epochs_per_cycle

    for epoch in range(1, total_epochs + 1):
        # Cosine annealing within cycle
        cycle_epoch = (epoch - 1) % epochs_per_cycle
        lr = config.training.lr * 0.5 * (1 + math.cos(math.pi * cycle_epoch / epochs_per_cycle))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

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

        val_result = _run_epoch(model, val_loader, device, criterion, None, config, effective_output_dim, None)
        print(f"epoch={epoch}/{total_epochs} lr={lr:.2e} train_loss={np.mean(train_losses):.4f} val_loss={val_result.loss:.4f}")

        # Save snapshot at end of each cycle (minimum LR)
        if epoch % epochs_per_cycle == 0:
            snapshots.append(copy.deepcopy(model.state_dict()))
            print(f"  -> Snapshot {len(snapshots)} saved (cycle end)")

    # Ensemble: average predictions from all snapshots
    all_snapshot_logits = []
    for i, sd in enumerate(snapshots):
        model.load_state_dict(sd)
        model.eval()
        logits_list = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                logits = model(batch)
                logits_list.append(logits.cpu().numpy())
        all_snapshot_logits.append(np.concatenate(logits_list))

    # Average logits across snapshots
    ensemble_logits = np.mean(all_snapshot_logits, axis=0)
    
    # Get true labels
    all_y = []
    for batch in val_loader:
        all_y.append(batch.y.numpy())
    y_true = np.concatenate(all_y)

    ensemble_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(ensemble_logits.reshape(-1)), torch.tensor(y_true.astype(np.float32))
    ).item())
    metrics = compute_metrics(config.task.problem_type, y_true, ensemble_logits)
    duration = time.time() - start_time

    print(f"\nSnapshot ensemble ({len(snapshots)} snapshots): val_loss={ensemble_loss:.6f} accuracy={metrics.get('accuracy'):.4f} roc_auc={metrics.get('roc_auc'):.4f}")
    log_experiment(exp_id, desc, 'success', ensemble_loss, metrics.get('accuracy'), metrics.get('roc_auc'), total_epochs, duration)
    prev_best = get_best_val_loss()
    if ensemble_loss < prev_best:
        update_best(exp_id, ensemble_loss, metrics.get('roc_auc'))
        print(f"\n*** NEW BEST: val_loss={ensemble_loss:.6f} (prev={prev_best:.6f}) ***\n")
    else:
        print(f"\nNo improvement: val_loss={ensemble_loss:.6f} vs best={prev_best:.6f}\n")

    torch.cuda.empty_cache()
    gc.collect()
    return {"status": "success", "val_loss": ensemble_loss, "accuracy": metrics.get('accuracy'), "roc_auc": metrics.get('roc_auc'), "duration": duration}


def run_multiseed_ensemble(exp_id, desc, base_config, seeds, swa_start, total_epochs):
    """Train same config with different seeds, average predictions."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"DESC: {desc}")
    print(f"{'='*60}\n")

    start_time = time.time()
    device = torch.device(base_config.training.device)

    # Build data once
    train_loader, val_loader, num_numeric, num_categorical, num_text, output_dim = build_dataloaders(base_config)
    effective_output_dim = 1 if base_config.task.problem_type in ("binary", "regression") else output_dim
    criterion = _make_criterion(base_config.task.problem_type, effective_output_dim)

    all_seed_logits = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed_idx+1}/{len(seeds)} (seed={seed}) ---")
        torch.cuda.empty_cache()
        gc.collect()
        set_seed(seed)

        model = TabularTransformer(base_config, num_numeric, num_categorical, num_text, effective_output_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_config.training.lr, weight_decay=base_config.training.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=_use_amp(base_config, device))
        scheduler = _build_scheduler(base_config, optimizer)

        state_dicts = []
        for epoch in range(1, total_epochs + 1):
            train_result = _run_epoch(model, train_loader, device, criterion, optimizer, base_config, effective_output_dim, scaler)
            val_result = _run_epoch(model, val_loader, device, criterion, None, base_config, effective_output_dim, None)
            if scheduler is not None:
                scheduler.step()
            if epoch >= swa_start:
                state_dicts.append(copy.deepcopy(model.state_dict()))
            if epoch % 10 == 0:
                print(f"  epoch={epoch} val_loss={val_result.loss:.4f}")

        # SWA average
        if state_dicts:
            avg_sd = {}
            for key in state_dicts[0]:
                avg_sd[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)
            model.load_state_dict(avg_sd)

        # Collect predictions
        model.eval()
        logits_list = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                logits = model(batch)
                logits_list.append(logits.cpu().numpy())
        all_seed_logits.append(np.concatenate(logits_list))
        del model, optimizer, scaler, scheduler
        torch.cuda.empty_cache()

    # Average logits across seeds
    ensemble_logits = np.mean(all_seed_logits, axis=0)

    all_y = []
    for batch in val_loader:
        all_y.append(batch.y.numpy())
    y_true = np.concatenate(all_y)

    ensemble_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(ensemble_logits.reshape(-1)), torch.tensor(y_true.astype(np.float32))
    ).item())
    metrics = compute_metrics(base_config.task.problem_type, y_true, ensemble_logits)
    duration = time.time() - start_time

    print(f"\nMulti-seed ensemble ({len(seeds)} seeds): val_loss={ensemble_loss:.6f} accuracy={metrics.get('accuracy'):.4f} roc_auc={metrics.get('roc_auc'):.4f}")
    log_experiment(exp_id, desc, 'success', ensemble_loss, metrics.get('accuracy'), metrics.get('roc_auc'), total_epochs, duration)
    prev_best = get_best_val_loss()
    if ensemble_loss < prev_best:
        update_best(exp_id, ensemble_loss, metrics.get('roc_auc'))
        print(f"\n*** NEW BEST: val_loss={ensemble_loss:.6f} (prev={prev_best:.6f}) ***\n")
    else:
        print(f"\nNo improvement: val_loss={ensemble_loss:.6f} vs best={prev_best:.6f}\n")

    torch.cuda.empty_cache()
    gc.collect()
    return {"status": "success", "val_loss": ensemble_loss, "accuracy": metrics.get('accuracy'), "roc_auc": metrics.get('roc_auc'), "duration": duration}


def build_wave3_experiments():
    experiments = []
    base = 60

    # --- Exp 60: Label smoothing 0.05 + batch=512 + SWA ---
    def exp_060():
        eid = f"exp_{base:03d}_labelsmooth005_batch512_swa"
        desc = "Label smoothing=0.05 batch=512 periodic k=16 lr=2e-4 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            batch_size=512,
            lr=2e-4,
            label_smoothing=0.05,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_060)

    # --- Exp 61: Label smoothing 0.1 + batch=512 + SWA ---
    def exp_061():
        eid = f"exp_{base+1:03d}_labelsmooth01_batch512_swa"
        desc = "Label smoothing=0.1 batch=512 periodic k=16 lr=2e-4 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            batch_size=512,
            lr=2e-4,
            label_smoothing=0.1,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_061)

    # --- Exp 62: Snapshot ensemble 5 cycles × 8 epochs ---
    def exp_062():
        eid = f"exp_{base+2:03d}_snapshot_5x8"
        desc = "Snapshot ensemble: 5 cycles × 8 epochs, average predictions"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            batch_size=512,
            lr=2e-4,
            max_epochs=40,  # Not directly used (snapshot manages epochs)
            early_stopping_patience=999,
            amp=True,
        )
        return "snapshot", eid, desc, config, 5, 8
    experiments.append(exp_062)

    # --- Exp 63: Attention pooling + batch=512 + SWA ---
    def exp_063():
        eid = f"exp_{base+3:03d}_attnpool_batch512_swa"
        desc = "Attention pooling batch=512 periodic k=16 lr=2e-4 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            batch_size=512,
            lr=2e-4,
            pooling='attention',
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_063)

    # --- Exp 64: Multi-seed ensemble (5 seeds) of best config ---
    def exp_064():
        eid = f"exp_{base+4:03d}_multiseed5_best"
        desc = "Multi-seed ensemble: 5 seeds of best config (batch=512 periodic k=16)"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            batch_size=512,
            lr=2e-4,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "multiseed", eid, desc, config, [42, 123, 456, 789, 1337]
    experiments.append(exp_064)

    # --- Exp 65: Wider SWA window (from epoch 15 of 40) ---
    def exp_065():
        eid = f"exp_{base+5:03d}_swa_wide_15_batch512"
        desc = "Wider SWA from epoch 15 batch=512 periodic k=16 lr=2e-4, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            batch_size=512,
            lr=2e-4,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 15, 40
    experiments.append(exp_065)

    # --- Exp 66: Hash encoder + batch=512 + SWA (does schema matter?) ---
    def exp_066():
        eid = f"exp_{base+6:03d}_hash_batch512_swa"
        desc = "Hash schema encoder batch=512 periodic k=16 lr=2e-4 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            batch_size=512,
            lr=2e-4,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        config.model.schema_encoder = "hash"
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_066)

    # --- Exp 67: Feature token dropout=0.1 + batch=512 + SWA ---
    def exp_067():
        eid = f"exp_{base+7:03d}_ftdrop01_batch512_swa"
        desc = "Feature token dropout=0.1 batch=512 periodic k=16 lr=2e-4 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            feature_token_dropout=0.1,
            batch_size=512,
            lr=2e-4,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_067)

    # --- Exp 68: Snapshot ensemble 8 cycles × 5 epochs ---
    def exp_068():
        eid = f"exp_{base+8:03d}_snapshot_8x5"
        desc = "Snapshot ensemble: 8 cycles × 5 epochs, average predictions"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            batch_size=512,
            lr=2e-4,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "snapshot", eid, desc, config, 8, 5
    experiments.append(exp_068)

    # --- Exp 69: Label smoothing 0.05 + wd=3e-4 + batch=512 + SWA ---
    def exp_069():
        eid = f"exp_{base+9:03d}_ls005_wd3e4_batch512_swa"
        desc = "Label smooth=0.05 wd=3e-4 batch=512 periodic k=16 lr=2e-4 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            batch_size=512,
            lr=2e-4,
            weight_decay=3e-4,
            label_smoothing=0.05,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_069)

    return experiments


if __name__ == "__main__":
    print("=" * 60)
    print("WAVE 3 EXPERIMENT LOOP — CREATIVE APPROACHES")
    print(f"Current best val_loss: {get_best_val_loss():.6f}")
    print("=" * 60)

    experiments = build_wave3_experiments()

    for i, exp_fn in enumerate(experiments):
        spec = exp_fn()
        mode = spec[0]
        exp_id = spec[1]
        desc = spec[2]
        config = spec[3]

        print(f"\n[{i+1}/{len(experiments)}] Starting {exp_id}...")
        try:
            if mode == "snapshot":
                n_cycles = spec[4]
                epochs_per_cycle = spec[5]
                result = run_snapshot_ensemble(exp_id, desc, config, n_cycles, epochs_per_cycle)
            elif mode == "multiseed":
                seeds = spec[4]
                result = run_multiseed_ensemble(exp_id, desc, config, seeds, 25, 40)
            elif mode == "swa":
                swa_start = spec[4]
                total_epochs = spec[5]
                result = run_swa_experiment(exp_id, desc, config, swa_start, total_epochs)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            print(f"  -> val_loss={result.get('val_loss')}, roc_auc={result.get('roc_auc')}, duration={result.get('duration', 0):.1f}s")

        except Exception as e:
            print(f"  -> CRASH: {e}")
            traceback.print_exc()
            log_experiment(exp_id, desc, 'crash', None, None, None, None, 0, notes=str(e)[:200])

        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 60)
    print(f"WAVE 3 COMPLETE. Best val_loss: {get_best_val_loss():.6f}")
    print("=" * 60)
