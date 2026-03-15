#!/usr/bin/env python3
"""Wave 2 experiments: more aggressive architectural changes + transfer learning.

Builds on Wave 1 findings:
  - Best: batch=512, periodic k=16, lr=2e-4, SWA (val_loss=0.3015)
  - SWA consistently helps
  - Wider (d256) and higher weight decay are promising

Wave 2 explores:
  - Transfer from pretrained backbone
  - Bigger models with more data
  - Mixup data augmentation
  - Different optimizers 
  - Ensemble of top-3 models
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
    log_experiment,
    get_best_val_loss,
    update_best,
    EXPERIMENTS_LOG,
)
from run_loop import run_swa_experiment, run_standard_experiment
from tabula.data.datasets import build_dataloaders
from tabula.models.transformer import TabularTransformer
from tabula.training.engine import (
    _move_batch, _make_criterion, _run_epoch, _compute_loss,
    _autocast_dtype, _use_amp, EpochResult, _build_scheduler,
    _save_checkpoint,
)
from tabula.evaluation.metrics import compute_metrics
from tabula.utils import set_seed


def run_mixup_experiment(exp_id, desc, config, swa_start, total_epochs, mixup_alpha=0.2):
    """Train with Mixup data augmentation + SWA."""
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
    scheduler = _build_scheduler(config, optimizer)

    state_dicts = []
    best_val_loss = float("inf")

    for epoch in range(1, total_epochs + 1):
        # Train with Mixup
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = _move_batch(batch, device)
            
            # Mixup: interpolate between random pairs
            lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
            batch_size = batch.y.shape[0]
            perm = torch.randperm(batch_size, device=device)
            
            with torch.autocast(device_type=device.type, dtype=_autocast_dtype(config), enabled=_use_amp(config, device)):
                # Mix features
                mixed_x_num = lam * batch.x_num + (1 - lam) * batch.x_num[perm]
                from tabula.data.datasets import TabularBatch
                mixed_batch = TabularBatch(
                    x_num=mixed_x_num,
                    x_cat=batch.x_cat,  # Don't mix categoricals
                    x_text_token_ids=batch.x_text_token_ids,
                    x_text_values=batch.x_text_values,
                    x_num_mask=batch.x_num_mask,
                    x_cat_mask=batch.x_cat_mask,
                    x_text_mask=batch.x_text_mask,
                    num_schema_texts=batch.num_schema_texts,
                    cat_schema_texts=batch.cat_schema_texts,
                    text_schema_texts=batch.text_schema_texts,
                    num_name_token_ids=batch.num_name_token_ids,
                    cat_name_token_ids=batch.cat_name_token_ids,
                    text_name_token_ids=batch.text_name_token_ids,
                    num_profile_vectors=batch.num_profile_vectors,
                    cat_profile_vectors=batch.cat_profile_vectors,
                    text_profile_vectors=batch.text_profile_vectors,
                    y=batch.y,  # Will use mixed loss below
                )
                logits = model(mixed_batch)
                loss_a = _compute_loss(config.task.problem_type, effective_output_dim, criterion, logits, batch.y)
                loss_b = _compute_loss(config.task.problem_type, effective_output_dim, criterion, logits, batch.y[perm])
                loss = lam * loss_a + (1 - lam) * loss_b
            
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

    # Evaluate
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


# ============================================================================
# EXPERIMENT DEFINITIONS — WAVE 2
# ============================================================================

def build_wave2_experiments():
    experiments = []
    base = 31

    # --- Exp 31: Best config (batch=512) but with wider d256 + SWA ---
    def exp_031():
        eid = f"exp_{base:03d}_batch512_wide256_swa"
        desc = "Batch=512 + wide d256 8H periodic k=16 lr=2e-4 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            d_model=256, n_heads=8, d_ff=512,
            batch_size=512,
            lr=2e-4,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_031)

    # --- Exp 32: Mixup + best config + SWA ---
    def exp_032():
        eid = f"exp_{base+1:03d}_mixup_batch512_swa"
        desc = "Mixup alpha=0.2 + batch=512 periodic k=16 lr=2e-4 SWA from 25, 40ep"
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
        return "mixup", eid, desc, config, 25, 40
    experiments.append(exp_032)

    # --- Exp 33: Best config with wd=3e-4 (from exp_025 insight) ---
    def exp_033():
        eid = f"exp_{base+2:03d}_batch512_wd3e4_swa"
        desc = "Batch=512 wd=3e-4 periodic k=16 lr=2e-4 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            batch_size=512,
            lr=2e-4,
            weight_decay=3e-4,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_033)

    # --- Exp 34: Very deep 12L with batch=512 + SWA ---
    def exp_034():
        eid = f"exp_{base+3:03d}_deep12_batch512_swa"
        desc = "Deep 12L batch=512 periodic k=16 lr=1.5e-4 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            n_layers=12,
            batch_size=512,
            lr=1.5e-4,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_034)

    # --- Exp 35: Periodic k=24 with batch=512 + SWA ---
    def exp_035():
        eid = f"exp_{base+4:03d}_periodic24_batch512_swa"
        desc = "Periodic k=24 batch=512 lr=2e-4 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic', 
            numeric_periodic_features=24,
            batch_size=512,
            lr=2e-4,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_035)

    # --- Exp 36: Cosine warmup + batch=512 + higher LR ---
    def exp_036():
        eid = f"exp_{base+5:03d}_cosine_batch512_lr3e4_swa"
        desc = "Cosine warmup batch=512 lr=3e-4 periodic k=16 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            batch_size=512,
            lr=3e-4,
            scheduler='cosine_warmup',
            warmup_epochs=3,
            lr_min=1e-6,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_036)

    # --- Exp 37: Wide d256 + wd=3e-4 + batch=512 ---
    def exp_037():
        eid = f"exp_{base+6:03d}_wide256_wd3e4_batch512_swa"
        desc = "Wide d256 wd=3e-4 batch=512 lr=1.5e-4 periodic k=16 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            d_model=256, n_heads=8, d_ff=512,
            batch_size=512,
            lr=1.5e-4,
            weight_decay=3e-4,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_037)

    # --- Exp 38: Batch=1024 (large effective batch) + SWA ---
    def exp_038():
        eid = f"exp_{base+7:03d}_batch1024_swa"
        desc = "Batch=1024 periodic k=16 lr=3e-4 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            batch_size=1024,
            lr=3e-4,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_038)

    # --- Exp 39: Best combo: d256 + batch=512 + periodic k=32 + SWA ---
    def exp_039():
        eid = f"exp_{base+8:03d}_d256_batch512_k32_swa"
        desc = "Wide d256 batch=512 periodic k=32 lr=1.5e-4 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=32,
            d_model=256, n_heads=8, d_ff=512,
            batch_size=512,
            lr=1.5e-4,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_039)

    # --- Exp 40: Deep 8L + d256 + batch=512 + SWA ---
    def exp_040():
        eid = f"exp_{base+9:03d}_deep8_d256_batch512_swa"
        desc = "Deep 8L d256 8H batch=512 periodic k=16 lr=1e-4 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            d_model=256, n_heads=8, n_layers=8, d_ff=512,
            batch_size=512,
            lr=1e-4,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_040)

    # --- Exp 41: Dropout=0.15 + wd=2e-4 + batch=512 + SWA ---
    def exp_041():
        eid = f"exp_{base+10:03d}_dropout015_wd2e4_batch512_swa"
        desc = "Dropout=0.15 wd=2e-4 batch=512 lr=2e-4 periodic k=16 SWA from 25, 40ep"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            dropout=0.15,
            weight_decay=2e-4,
            batch_size=512,
            lr=2e-4,
            max_epochs=40,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 25, 40
    experiments.append(exp_041)

    # --- Exp 42: Extended training 60ep + SWA from 35 ---
    def exp_042():
        eid = f"exp_{base+11:03d}_long60_batch512_swa"
        desc = "Long 60ep batch=512 lr=2e-4 periodic k=16 SWA from 35"
        config = make_adult_config(
            eid,
            numeric_embedding='periodic',
            numeric_periodic_features=16,
            batch_size=512,
            lr=2e-4,
            max_epochs=60,
            early_stopping_patience=999,
            amp=True,
        )
        return "swa", eid, desc, config, 35, 60
    experiments.append(exp_042)

    return experiments


if __name__ == "__main__":
    print("=" * 60)
    print("WAVE 2 EXPERIMENT LOOP")
    print(f"Current best val_loss: {get_best_val_loss():.6f}")
    print("=" * 60)

    experiments = build_wave2_experiments()

    for i, exp_fn in enumerate(experiments):
        spec = exp_fn()
        mode = spec[0]
        exp_id = spec[1]
        desc = spec[2]
        config = spec[3]

        print(f"\n[{i+1}/{len(experiments)}] Starting {exp_id}...")
        try:
            if mode == "mixup":
                swa_start = spec[4]
                total_epochs = spec[5]
                result = run_mixup_experiment(exp_id, desc, config, swa_start, total_epochs)
            elif mode == "swa":
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
            log_experiment(exp_id, desc, 'crash', None, None, None, None, 0, notes=str(e)[:200])

        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 60)
    print(f"WAVE 2 COMPLETE. Best val_loss: {get_best_val_loss():.6f}")
    print("=" * 60)
