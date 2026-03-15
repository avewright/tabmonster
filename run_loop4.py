#!/usr/bin/env python3
"""Wave 4: Ensembles and combined best approaches.

Key insight from 34+ experiments: we're at a ceiling around 0.3015.
Individual models vary by ~±0.002 due to seed variance.

Strategy:
- 10-seed ensemble of proven best config 
- Combined best regularizations (wd + ft_dropout + wider SWA)
- Diverse architecture ensemble (top configs averaged)
- SGD with momentum (potentially flatter minima)
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
from run_loop import run_swa_experiment
from tabula.data.datasets import build_dataloaders
from tabula.models.transformer import TabularTransformer
from tabula.training.engine import (
    _move_batch, _make_criterion, _run_epoch, _compute_loss,
    _autocast_dtype, _use_amp, _build_scheduler,
)
from tabula.evaluation.metrics import compute_metrics
from tabula.utils import set_seed


def train_one_model_return_logits(config, train_loader, val_loader, num_numeric, num_categorical, num_text, effective_output_dim, seed, swa_start=25, total_epochs=40):
    """Train one model and return its final logits on val set."""
    device = torch.device(config.training.device)
    set_seed(seed)

    model = TabularTransformer(config, num_numeric, num_categorical, num_text, effective_output_dim).to(device)
    criterion = _make_criterion(config.task.problem_type, effective_output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=_use_amp(config, device))
    scheduler = _build_scheduler(config, optimizer)

    state_dicts = []
    for epoch in range(1, total_epochs + 1):
        train_result = _run_epoch(model, train_loader, device, criterion, optimizer, config, effective_output_dim, scaler)
        if scheduler is not None:
            scheduler.step()
        if epoch >= swa_start:
            state_dicts.append(copy.deepcopy(model.state_dict()))

    # SWA
    if state_dicts:
        avg_sd = {}
        for key in state_dicts[0]:
            avg_sd[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)
        model.load_state_dict(avg_sd)

    model.eval()
    logits_list = []
    with torch.no_grad():
        for batch in val_loader:
            batch = _move_batch(batch, device)
            logits = model(batch)
            logits_list.append(logits.cpu().numpy())

    del model, optimizer, scaler, scheduler, state_dicts
    torch.cuda.empty_cache()
    gc.collect()
    return np.concatenate(logits_list)


def run_large_ensemble(exp_id, desc, configs_and_seeds, swa_start=25, total_epochs=40):
    """Train multiple models with different configs/seeds, average predictions."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_id}")
    print(f"DESC: {desc}")
    print(f"Num models: {len(configs_and_seeds)}")
    print(f"{'='*60}\n")

    start_time = time.time()
    torch.cuda.empty_cache()
    gc.collect()

    # Build data from the first config (all configs same data)
    first_config = configs_and_seeds[0][0]
    train_loader, val_loader, num_numeric, num_categorical, num_text, output_dim = build_dataloaders(first_config)
    effective_output_dim = 1 if first_config.task.problem_type in ("binary", "regression") else output_dim

    all_logits = []
    for i, (config, seed) in enumerate(configs_and_seeds):
        print(f"  Training model {i+1}/{len(configs_and_seeds)} (seed={seed})...")
        logits = train_one_model_return_logits(
            config, train_loader, val_loader,
            num_numeric, num_categorical, num_text, effective_output_dim,
            seed, swa_start, total_epochs
        )
        all_logits.append(logits)

    # Average logits
    ensemble_logits = np.mean(all_logits, axis=0)

    all_y = []
    for batch in val_loader:
        all_y.append(batch.y.numpy())
    y_true = np.concatenate(all_y)

    ensemble_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(ensemble_logits.reshape(-1)), torch.tensor(y_true.astype(np.float32))
    ).item())
    metrics = compute_metrics(first_config.task.problem_type, y_true, ensemble_logits)
    duration = time.time() - start_time

    print(f"\nEnsemble ({len(configs_and_seeds)} models): val_loss={ensemble_loss:.6f} accuracy={metrics.get('accuracy'):.4f} roc_auc={metrics.get('roc_auc'):.4f}")
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


if __name__ == "__main__":
    print("=" * 60)
    print("WAVE 4: ENSEMBLES & COMBINED APPROACHES")
    print(f"Current best val_loss: {get_best_val_loss():.6f}")
    print("=" * 60)

    experiments = []

    # --- Exp 70: Combined best: wd=3e-4 + ft_dropout=0.1 + wide SWA + batch=512 ---
    def exp_070():
        eid = "exp_070_combined_best_swa"
        desc = "Combined: wd=3e-4 ft_drop=0.1 batch=512 periodic k=16 lr=2e-4 SWA from 15, 40ep"
        config = make_adult_config(eid, numeric_embedding='periodic', numeric_periodic_features=16,
                                   batch_size=512, lr=2e-4, weight_decay=3e-4, feature_token_dropout=0.1,
                                   max_epochs=40, early_stopping_patience=999, amp=True)
        return "swa", eid, desc, config, 15, 40
    experiments.append(exp_070)

    # --- Exp 71: 10-seed ensemble of best config ---
    def exp_071():
        eid = "exp_071_ensemble10_best"
        desc = "10-seed ensemble: best config (batch=512 periodic k=16 lr=2e-4 SWA)"
        base_config = make_adult_config(eid, numeric_embedding='periodic', numeric_periodic_features=16,
                                        batch_size=512, lr=2e-4, max_epochs=40, early_stopping_patience=999, amp=True)
        seeds = [42, 123, 456, 789, 1337, 2024, 7777, 9999, 31415, 27182]
        configs_and_seeds = [(base_config, s) for s in seeds]
        return "ensemble", eid, desc, configs_and_seeds
    experiments.append(exp_071)

    # --- Exp 72: Diverse architecture ensemble (top 5 configs from wave 1-3) ---
    def exp_072():
        eid = "exp_072_diverse_ensemble5"
        desc = "Diverse ensemble: 5 diff architectures from best configs"
        configs_and_seeds = []
        
        # Config 1: Original best (d192, 6L, 6H, batch=512)
        c1 = make_adult_config("div_1", numeric_embedding='periodic', numeric_periodic_features=16,
                               batch_size=512, lr=2e-4, max_epochs=40, early_stopping_patience=999, amp=True)
        configs_and_seeds.append((c1, 42))
        
        # Config 2: Wide d256 + wd=3e-4
        c2 = make_adult_config("div_2", numeric_embedding='periodic', numeric_periodic_features=16,
                               d_model=256, n_heads=8, d_ff=512, weight_decay=3e-4,
                               batch_size=512, lr=1.5e-4, max_epochs=40, early_stopping_patience=999, amp=True)
        configs_and_seeds.append((c2, 42))
        
        # Config 3: d192 + wd=3e-4
        c3 = make_adult_config("div_3", numeric_embedding='periodic', numeric_periodic_features=16,
                               weight_decay=3e-4,
                               batch_size=512, lr=2e-4, max_epochs=40, early_stopping_patience=999, amp=True)
        configs_and_seeds.append((c3, 42))
        
        # Config 4: d192 + ft_dropout=0.1
        c4 = make_adult_config("div_4", numeric_embedding='periodic', numeric_periodic_features=16,
                               feature_token_dropout=0.1,
                               batch_size=512, lr=2e-4, max_epochs=40, early_stopping_patience=999, amp=True)
        configs_and_seeds.append((c4, 42))
        
        # Config 5: Deep 8L d256
        c5 = make_adult_config("div_5", numeric_embedding='periodic', numeric_periodic_features=16,
                               d_model=256, n_heads=8, n_layers=8, d_ff=512,
                               batch_size=512, lr=1e-4, max_epochs=40, early_stopping_patience=999, amp=True)
        configs_and_seeds.append((c5, 42))

        return "ensemble", eid, desc, configs_and_seeds
    experiments.append(exp_072)

    # --- Exp 73: 10-seed ensemble + wd=3e-4 (best regularization) ---
    def exp_073():
        eid = "exp_073_ensemble10_wd3e4"
        desc = "10-seed ensemble: batch=512 wd=3e-4 periodic k=16 lr=2e-4 SWA"
        base_config = make_adult_config(eid, numeric_embedding='periodic', numeric_periodic_features=16,
                                        batch_size=512, lr=2e-4, weight_decay=3e-4,
                                        max_epochs=40, early_stopping_patience=999, amp=True)
        seeds = [42, 123, 456, 789, 1337, 2024, 7777, 9999, 31415, 27182]
        configs_and_seeds = [(base_config, s) for s in seeds]
        return "ensemble", eid, desc, configs_and_seeds
    experiments.append(exp_073)

    # --- Exp 74: Mega diverse ensemble (10 diverse models) ---
    def exp_074():
        eid = "exp_074_mega_diverse_10"
        desc = "Mega diverse: 10 different arch+seed combos"
        configs_and_seeds = []
        
        for s in [42, 123]:
            c = make_adult_config(f"mega_{s}_base", numeric_embedding='periodic', numeric_periodic_features=16,
                                  batch_size=512, lr=2e-4, max_epochs=40, early_stopping_patience=999, amp=True)
            configs_and_seeds.append((c, s))
        
        for s in [456, 789]:
            c = make_adult_config(f"mega_{s}_wd", numeric_embedding='periodic', numeric_periodic_features=16,
                                  batch_size=512, lr=2e-4, weight_decay=3e-4,
                                  max_epochs=40, early_stopping_patience=999, amp=True)
            configs_and_seeds.append((c, s))
        
        for s in [1337, 2024]:
            c = make_adult_config(f"mega_{s}_wide", numeric_embedding='periodic', numeric_periodic_features=16,
                                  d_model=256, n_heads=8, d_ff=512,
                                  batch_size=512, lr=1.5e-4, max_epochs=40, early_stopping_patience=999, amp=True)
            configs_and_seeds.append((c, s))
        
        for s in [7777, 9999]:
            c = make_adult_config(f"mega_{s}_drop", numeric_embedding='periodic', numeric_periodic_features=16,
                                  feature_token_dropout=0.1,
                                  batch_size=512, lr=2e-4, max_epochs=40, early_stopping_patience=999, amp=True)
            configs_and_seeds.append((c, s))
        
        for s in [31415, 27182]:
            c = make_adult_config(f"mega_{s}_deep", numeric_embedding='periodic', numeric_periodic_features=16,
                                  d_model=256, n_heads=8, n_layers=8, d_ff=512,
                                  batch_size=512, lr=1e-4, max_epochs=40, early_stopping_patience=999, amp=True)
            configs_and_seeds.append((c, s))

        return "ensemble", eid, desc, configs_and_seeds
    experiments.append(exp_074)

    for i, exp_fn in enumerate(experiments):
        spec = exp_fn()
        mode = spec[0]

        print(f"\n[{i+1}/{len(experiments)}] Starting {spec[1]}...")
        try:
            if mode == "swa":
                eid, desc, config = spec[1], spec[2], spec[3]
                swa_start, total_epochs = spec[4], spec[5]
                result = run_swa_experiment(eid, desc, config, swa_start, total_epochs)
            elif mode == "ensemble":
                eid, desc, configs_and_seeds = spec[1], spec[2], spec[3]
                result = run_large_ensemble(eid, desc, configs_and_seeds)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            print(f"  -> val_loss={result.get('val_loss')}, roc_auc={result.get('roc_auc')}, duration={result.get('duration', 0):.1f}s")

        except Exception as e:
            print(f"  -> CRASH: {e}")
            traceback.print_exc()
            log_experiment(spec[1], spec[2], 'crash', None, None, None, None, 0, notes=str(e)[:200])

        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 60)
    print(f"WAVE 4 COMPLETE. Best val_loss: {get_best_val_loss():.6f}")
    print("=" * 60)
