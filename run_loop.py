#!/usr/bin/env python3
"""Autonomous experiment loop for Tabula.

Runs experiments one by one, logs results, keeps improvements, discards failures.
Designed to be run unattended for hours.
"""
import sys
sys.path.insert(0, '.')
import copy
import gc
import json
import time
import traceback

import numpy as np
import torch
from pathlib import Path

from run_experiment import (
    make_adult_config,
    run_single_experiment,
    log_experiment,
    get_best_val_loss,
    update_best,
    EXPERIMENTS_LOG,
)
from tabula.data.datasets import build_dataloaders
from tabula.models.transformer import TabularTransformer
from tabula.training.engine import (
    _move_batch, _make_criterion, _run_epoch, _compute_loss,
    _autocast_dtype, _use_amp, EpochResult, _build_scheduler,
)
from tabula.evaluation.metrics import compute_metrics
from tabula.utils import set_seed


def get_next_exp_id():
    """Get the next experiment ID based on existing log."""
    if not EXPERIMENTS_LOG.exists():
        return "exp_019"
    with open(EXPERIMENTS_LOG, 'r') as f:
        lines = f.readlines()
    max_num = 18  # We know we have up to exp_018
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if parts:
            eid = parts[0]
            # Extract number from exp_NNN_xxx
            for part in eid.split('_'):
                try:
                    n = int(part)
                    max_num = max(max_num, n)
                except ValueError:
                    continue
    return f"exp_{max_num + 1:03d}"


def run_swa_experiment(exp_id, desc, config, swa_start, total_epochs):
    """Train with SWA (Stochastic Weight Averaging)."""
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
    criterion = _make_criterion(config.task.problem_type, effective_output_dim, getattr(config.training, 'label_smoothing', 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=_use_amp(config, device))
    scheduler = _build_scheduler(config, optimizer)

    state_dicts = []
    best_val_loss = float("inf")

    for epoch in range(1, total_epochs + 1):
        train_result = _run_epoch(model, train_loader, device, criterion, optimizer, config, effective_output_dim, scaler)
        val_result = _run_epoch(model, val_loader, device, criterion, None, config, effective_output_dim, None)
        if scheduler is not None:
            scheduler.step()
        print(f"epoch={epoch} train_loss={train_result.loss:.4f} val_loss={val_result.loss:.4f} val_metrics={val_result.metrics}")

        if epoch >= swa_start:
            state_dicts.append(copy.deepcopy(model.state_dict()))

        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss

    if not state_dicts:
        # Fallback: no SWA averaging
        state_dicts.append(model.state_dict())

    print(f"\nAveraging {len(state_dicts)} state dicts from epochs {swa_start}-{total_epochs}...")
    avg_state_dict = {}
    for key in state_dicts[0]:
        avg_state_dict[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)
    model.load_state_dict(avg_state_dict)

    # Evaluate SWA model
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

    if config.task.problem_type == 'binary':
        swa_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(logits_arr.reshape(-1)), torch.tensor(y_true.astype(np.float32))
        ).item())
    else:
        swa_loss = best_val_loss

    metrics = compute_metrics(config.task.problem_type, y_true, logits_arr)
    accuracy = metrics.get('accuracy')
    roc_auc = metrics.get('roc_auc')
    duration = time.time() - start_time

    print(f"\nSWA results: val_loss={swa_loss:.6f} accuracy={accuracy:.4f} roc_auc={roc_auc:.4f}")

    log_experiment(exp_id, desc, 'success', swa_loss, accuracy, roc_auc, total_epochs, duration)
    prev_best = get_best_val_loss()
    if swa_loss < prev_best:
        update_best(exp_id, swa_loss, roc_auc)
        print(f"\n*** NEW BEST: val_loss={swa_loss:.6f} (prev={prev_best:.6f}) ***\n")
    else:
        print(f"\nNo improvement: val_loss={swa_loss:.6f} vs best={prev_best:.6f}\n")

    torch.cuda.empty_cache()
    gc.collect()
    return {"status": "success", "val_loss": swa_loss, "accuracy": accuracy, "roc_auc": roc_auc, "duration": duration}


def run_standard_experiment(exp_id, desc, config):
    """Run a standard training experiment (no SWA)."""
    torch.cuda.empty_cache()
    gc.collect()
    result = run_single_experiment(exp_id, desc, config)
    torch.cuda.empty_cache()
    gc.collect()
    return result


# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

def build_experiments():
    """Build a queue of experiment configurations to try."""
    experiments = []
    base_id_num = 19  # Starting ID

    # --- Exp 19: Piecewise Linear Encoding (PLE) approach via higher periodic k=32 ---
    def exp_019():
        eid = f"exp_{base_id_num:03d}_periodic32_swa"
        desc = "Periodic k=32 + SWA from epoch 25, lr=1e-4, 40 epochs"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=32,
            lr=1e-4,
            max_epochs=40,
            early_stopping_patience=999,
            d_model=192, n_heads=6, n_layers=6, d_ff=384,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_019)

    # --- Exp 20: Deeper model 10L with periodic k=16 + SWA ---
    def exp_020():
        eid = f"exp_{base_id_num+1:03d}_deep10_periodic16_swa"
        desc = "Deep 10L + periodic k=16 lr=1e-4 SWA from 20, 35ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            n_layers=10,
            lr=1e-4,
            max_epochs=35,
            early_stopping_patience=999,
            d_model=192, n_heads=6, d_ff=384,
            amp=True,
        )
        return "swa", eid, desc, config, 20, 35
    experiments.append(exp_020)

    # --- Exp 21: Wider d_model=256 with periodic k=16 + SWA ---
    def exp_021():
        eid = f"exp_{base_id_num+2:03d}_wide256_periodic16_swa"
        desc = "Wide d256 8H + periodic k=16 lr=8e-5 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            d_model=256, n_heads=8, n_layers=6, d_ff=512,
            lr=8e-5,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_021)

    # --- Exp 22: Cosine warmup + SWA combo ---
    def exp_022():
        eid = f"exp_{base_id_num+3:03d}_cosine_warmup_swa"
        desc = "Periodic k=16 cosine_warmup lr=2e-4 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            lr=2e-4,
            scheduler='cosine_warmup',
            warmup_epochs=3,
            lr_min=1e-6,
            max_epochs=40,
            early_stopping_patience=999,
            d_model=192, n_heads=6, n_layers=6, d_ff=384,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_022)

    # --- Exp 23: Higher dropout + periodic k=16 + SWA ---
    def exp_023():
        eid = f"exp_{base_id_num+4:03d}_dropout02_periodic16_swa"
        desc = "Periodic k=16 dropout=0.2 ftd=0.1 lr=1e-4 SWA from 20, 35ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            dropout=0.2,
            feature_token_dropout=0.1,
            lr=1e-4,
            max_epochs=35,
            early_stopping_patience=999,
            d_model=192, n_heads=6, n_layers=6, d_ff=384,
            amp=True,
        )
        return "swa", eid, desc, config, 20, 35
    experiments.append(exp_023)

    # --- Exp 24: Lower lr with more epochs + SWA ---
    def exp_024():
        eid = f"exp_{base_id_num+5:03d}_lr5e5_periodic16_swa"
        desc = "Periodic k=16 lr=5e-5 SWA from 30, 50ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            lr=5e-5,
            max_epochs=50,
            early_stopping_patience=999,
            d_model=192, n_heads=6, n_layers=6, d_ff=384,
            amp=True,
        )
        return "swa", eid, desc, config, 30, 50
    experiments.append(exp_024)

    # --- Exp 25: Weight decay tuning with SWA ---
    def exp_025():
        eid = f"exp_{base_id_num+6:03d}_wd3e4_periodic16_swa"
        desc = "Periodic k=16 lr=1e-4 wd=3e-4 SWA from 20, 35ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            lr=1e-4,
            weight_decay=3e-4,
            max_epochs=35,
            early_stopping_patience=999,
            d_model=192, n_heads=6, n_layers=6, d_ff=384,
            amp=True,
        )
        return "swa", eid, desc, config, 20, 35
    experiments.append(exp_025)

    # --- Exp 26: Batch size 512 with periodic k=16 + SWA ---
    def exp_026():
        eid = f"exp_{base_id_num+7:03d}_batch512_periodic16_swa"
        desc = "Periodic k=16 batch=512 lr=2e-4 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            batch_size=512,
            lr=2e-4,
            max_epochs=40,
            early_stopping_patience=999,
            d_model=192, n_heads=6, n_layers=6, d_ff=384,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_026)

    # --- Exp 27: Wider FFN=768 + periodic k=16 + SWA ---
    def exp_027():
        eid = f"exp_{base_id_num+8:03d}_ff768_periodic16_swa"
        desc = "Periodic k=16 d_ff=768 lr=1e-4 SWA from 20, 35ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            d_ff=768,
            lr=1e-4,
            max_epochs=35,
            early_stopping_patience=999,
            d_model=192, n_heads=6, n_layers=6,
            amp=True,
        )
        return "swa", eid, desc, config, 20, 35
    experiments.append(exp_027)

    # --- Exp 28: Mean pooling + periodic k=16 + SWA ---
    def exp_028():
        eid = f"exp_{base_id_num+9:03d}_meanpool_periodic16_swa"
        desc = "Mean pooling + periodic k=16 lr=1e-4 SWA from 20, 35ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            pooling='mean',
            lr=1e-4,
            max_epochs=35,
            early_stopping_patience=999,
            d_model=192, n_heads=6, n_layers=6, d_ff=384,
            amp=True,
        )
        return "swa", eid, desc, config, 20, 35
    experiments.append(exp_028)

    # --- Exp 29: Cosine annealing (no warmup) + periodic k=16 + SWA ---
    def exp_029():
        eid = f"exp_{base_id_num+10:03d}_cosine_periodic16_swa"
        desc = "Periodic k=16 cosine lr=1.5e-4 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            lr=1.5e-4,
            scheduler='cosine',
            lr_min=1e-6,
            max_epochs=40,
            early_stopping_patience=999,
            d_model=192, n_heads=6, n_layers=6, d_ff=384,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_029)

    # --- Exp 30: Grad accumulation 4 + larger effective batch + SWA ---
    def exp_030():
        eid = f"exp_{base_id_num+11:03d}_gradaccum4_periodic16_swa"
        desc = "Periodic k=16 grad_accum=4 lr=3e-4 SWA from 20, 35ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            gradient_accumulation_steps=4,
            lr=3e-4,
            max_epochs=35,
            early_stopping_patience=999,
            d_model=192, n_heads=6, n_layers=6, d_ff=384,
            amp=True,
        )
        return "swa", eid, desc, config, 20, 35
    experiments.append(exp_030)

    return experiments


if __name__ == "__main__":
    print("=" * 60)
    print("AUTONOMOUS EXPERIMENT LOOP")
    print(f"Current best val_loss: {get_best_val_loss():.6f}")
    print("=" * 60)

    experiments = build_experiments()

    for i, exp_fn in enumerate(experiments):
        spec = exp_fn()
        mode = spec[0]
        exp_id = spec[1]
        desc = spec[2]
        config = spec[3]

        print(f"\n[{i+1}/{len(experiments)}] Starting {exp_id}...")
        try:
            if mode == "swa":
                swa_start = spec[4]
                total_epochs = spec[5]
                result = run_swa_experiment(exp_id, desc, config, swa_start, total_epochs)
            else:
                result = run_standard_experiment(exp_id, desc, config)

            if result.get("status") == "success":
                print(f"  -> val_loss={result.get('val_loss', 'N/A')}, roc_auc={result.get('roc_auc', 'N/A')}, duration={result.get('duration', 0):.1f}s")
            else:
                print(f"  -> FAILED: {result.get('error', 'unknown')}")

        except Exception as e:
            print(f"  -> CRASH: {e}")
            traceback.print_exc()
            duration = 0
            log_experiment(exp_id, desc, 'crash', None, None, None, None, duration, notes=str(e)[:200])

        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 60)
    print(f"ALL EXPERIMENTS COMPLETE. Best val_loss: {get_best_val_loss():.6f}")
    print("=" * 60)
