#!/usr/bin/env python3
"""
Wave 11 — Structural experiments + seed variance + ensembles.

Wave 10 showed the config at 0.297603 is stable (near-optimal).
Now try: seed variance, periodic features, pooling, ensembles.
"""
import sys
sys.path.insert(0, '.')
import copy
import gc
import traceback
import time

import numpy as np
import torch

from run_experiment import (
    log_experiment,
    get_best_val_loss,
    update_best,
    ExperimentConfig,
    TaskConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EpisodeConfig,
)
from run_loop import run_swa_experiment, build_dataloaders, _make_criterion, _run_epoch, _move_batch, _use_amp, _build_scheduler, compute_metrics
from tabula.models.transformer import TabularTransformer
from tabula.utils import set_seed


def make_config(
    experiment_name,
    *,
    d_model=192, n_heads=6, n_layers=6, d_ff=384,
    dropout=0.1, feature_token_dropout=0.1,
    numeric_embedding='periodic', numeric_periodic_features=16,
    lr=2e-4, weight_decay=1e-4, batch_size=512,
    max_epochs=60, scheduler="cosine_warmup", warmup_epochs=3,
    label_smoothing=0.01, pooling='cls', seed=42,
):
    return ExperimentConfig(
        experiment_name=experiment_name,
        artifacts_root="artifacts",
        seed=seed,
        task=TaskConfig(mode="finetune", problem_type="binary", target_column="income"),
        data=DataConfig(
            dataset_type="prepared",
            prepared_dir="data/processed/adult_census_engineered",
            train_path="data/processed/adult_census_engineered/train.csv",
            val_path="data/processed/adult_census_engineered/val.csv",
            batch_size=batch_size,
            num_workers=0,
            standardize_numeric=True,
        ),
        model=ModelConfig(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
            dropout=dropout, feature_token_dropout=feature_token_dropout,
            norm="rmsnorm", ffn_activation="swiglu", max_categories=256,
            numeric_embedding=numeric_embedding,
            numeric_periodic_features=numeric_periodic_features,
            pooling=pooling,
        ),
        training=TrainingConfig(
            device="cuda", max_epochs=max_epochs, lr=lr,
            weight_decay=weight_decay, grad_clip_norm=1.0, log_interval=20,
            early_stopping_patience=999,
            amp=True, amp_dtype="float16",
            gradient_accumulation_steps=1,
            scheduler=scheduler, warmup_epochs=warmup_epochs, lr_min=1e-6,
            label_smoothing=label_smoothing,
        ),
        episode=EpisodeConfig(enabled=False),
    )


def run_ensemble(seeds, exp_name, desc, swa_start=40, total_epochs=60, **config_kwargs):
    """Train multiple seeds, ensemble predictions."""
    all_probs = []
    all_targets = None
    start = time.time()

    for i, seed in enumerate(seeds):
        config = make_config(exp_name, seed=seed, max_epochs=total_epochs, **config_kwargs)
        set_seed(seed)
        device = torch.device("cuda")

        train_loader, val_loader, num_numeric, num_categorical, num_text, output_dim = build_dataloaders(config)
        effective_output_dim = 1
        model = TabularTransformer(config, num_numeric, num_categorical, num_text, effective_output_dim).to(device)
        criterion = _make_criterion("binary", effective_output_dim, config.training.label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        scheduler = _build_scheduler(config, optimizer)

        state_dicts = []
        for epoch in range(1, total_epochs + 1):
            _run_epoch(model, train_loader, device, criterion, optimizer, config, effective_output_dim, scaler)
            val_result = _run_epoch(model, val_loader, device, criterion, None, config, effective_output_dim, None)
            if scheduler is not None:
                scheduler.step()
            if epoch >= swa_start:
                state_dicts.append(copy.deepcopy(model.state_dict()))
            if epoch % 10 == 0:
                print(f"  seed={seed} epoch={epoch} val_loss={val_result.loss:.4f}")

        # SWA average
        if state_dicts:
            avg_sd = {}
            for key in state_dicts[0]:
                avg_sd[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(0)
            model.load_state_dict(avg_sd)

        # Collect predictions
        model.eval()
        probs_list, targets_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                logits = model(batch)
                probs = torch.sigmoid(logits.float().squeeze(-1))
                probs_list.append(probs.cpu())
                targets_list.append(batch.y.cpu())

        all_probs.append(torch.cat(probs_list, 0))
        if all_targets is None:
            all_targets = torch.cat(targets_list, 0)

        del model, optimizer, scaler
        torch.cuda.empty_cache()
        gc.collect()

    # Ensemble average
    ensemble_probs = torch.stack(all_probs).mean(0)
    targets = all_targets.float()

    # Compute loss and metrics
    val_loss = torch.nn.functional.binary_cross_entropy(
        ensemble_probs, targets
    ).item()

    from sklearn.metrics import accuracy_score, roc_auc_score
    preds = (ensemble_probs > 0.5).long().numpy()
    acc = accuracy_score(targets.numpy(), preds)
    auc = roc_auc_score(targets.numpy(), ensemble_probs.numpy())

    duration = time.time() - start
    print(f"\nEnsemble ({len(seeds)} seeds): val_loss={val_loss:.6f} acc={acc:.4f} auc={auc:.4f} ({duration:.0f}s)")
    return val_loss, acc, auc, total_epochs, duration


def main():
    experiments = [
        # 1. Different seed — check seed variance
        ("exp_140_seed123",
         "Best config seed=123",
         lambda: (make_config("exp_140_seed123", seed=123), 40, 60)),

        # 2. Different seed — check seed variance
        ("exp_141_seed789",
         "Best config seed=789",
         lambda: (make_config("exp_141_seed789", seed=789), 40, 60)),

        # 3. Periodic k=32 (more periodic features)
        ("exp_142_periodic32",
         "Best config + periodic k=32",
         lambda: (make_config("exp_142_periodic32", numeric_periodic_features=32), 40, 60)),

        # 4. Periodic k=8 (fewer periodic features)
        ("exp_143_periodic8",
         "Best config + periodic k=8",
         lambda: (make_config("exp_143_periodic8", numeric_periodic_features=8), 40, 60)),

        # 5. Mean pooling instead of CLS
        ("exp_144_mean_pool",
         "Best config + mean pooling",
         lambda: (make_config("exp_144_mean_pool", pooling="mean"), 40, 60)),

        # 6. Best config + wd=5e-5 (tied for best in wave 10)
        ("exp_145_wd5e5",
         "Best config + wd=5e-5",
         lambda: (make_config("exp_145_wd5e5", weight_decay=5e-5), 40, 60)),

        # 7. 8 layers (between 6 and 12)
        ("exp_146_8layers",
         "Best config 8L d192",
         lambda: (make_config("exp_146_8layers", n_layers=8), 40, 60)),
    ]

    print("=" * 60)
    print("WAVE 11 — Structural Experiments + Seed Variance")
    print(f"Total experiments: {len(experiments) + 1}")
    print(f"Current best: {get_best_val_loss():.6f}")
    print("=" * 60)

    for i, (exp_id, desc, config_fn) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)+1}] Starting {exp_id}...")
        try:
            config, swa_start, total_epochs = config_fn()
            config.training.max_epochs = total_epochs
            result = run_swa_experiment(exp_id, desc, config, swa_start, total_epochs)
            print(f"  -> val_loss={result['val_loss']:.6f}, roc_auc={result['roc_auc']:.4f}, "
                  f"duration={result['duration']:.1f}s")
        except Exception as e:
            print(f"\n  CRASHED: {e}")
            traceback.print_exc()
            log_experiment(exp_id, desc, 'crash', 999, 0, 0, 0, 0)

        torch.cuda.empty_cache()
        gc.collect()

    # Final: Multi-seed ensemble (most expensive, run last)
    print(f"\n[{len(experiments)+1}/{len(experiments)+1}] Starting exp_147 ensemble...")
    try:
        seeds = [42, 123, 456, 789, 1337]
        val_loss, acc, auc, epochs, duration = run_ensemble(
            seeds, "exp_147_5seed_ensemble", "5-seed ensemble best config",
            swa_start=40, total_epochs=60,
        )
        log_experiment("exp_147_5seed_ensemble", "5-seed ensemble ls=0.01 cosine 60ep SWA40",
                       'success', val_loss, acc, auc, epochs, duration)
        prev_best = get_best_val_loss()
        if val_loss < prev_best:
            update_best("exp_147_5seed_ensemble", val_loss, auc)
            print(f"\n*** NEW BEST: {val_loss:.6f} (prev={prev_best:.6f}) ***")
        else:
            print(f"\nNo improvement: {val_loss:.6f} vs best={prev_best:.6f}")
    except Exception as e:
        print(f"\n  CRASHED: {e}")
        traceback.print_exc()
        log_experiment("exp_147_5seed_ensemble", "5-seed ensemble", 'crash', 999, 0, 0, 0, 0)

    print("\n" + "=" * 60)
    print("WAVE 11 COMPLETE")
    print(f"Best: {get_best_val_loss():.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
