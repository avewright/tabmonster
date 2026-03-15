#!/usr/bin/env python3
"""
Wave 9 — Combine winning strategies from wave 8.

Key findings from wave 8:
  - label_smoothing=0.02 → 0.299319 (NEW BEST)
  - cosine_warmup + 60ep SWA from 40 → 0.299142 (NEW BEST)
  - Architecture changes (deep/wide) don't help

Wave 9 plan: combine + refine these winning strategies.
"""
import sys
sys.path.insert(0, '.')
import gc
import traceback

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
from run_loop import run_swa_experiment
from tabula.utils import set_seed


def make_engineered_config(
    experiment_name,
    *,
    d_model=192, n_heads=6, n_layers=6, d_ff=384,
    dropout=0.1, feature_token_dropout=0.1,
    numeric_embedding='periodic', numeric_periodic_features=16,
    lr=2e-4, weight_decay=1e-4, batch_size=512,
    max_epochs=40, early_stopping_patience=999,
    scheduler="none", warmup_epochs=2,
    label_smoothing=0.0, gradient_accumulation_steps=1,
    pooling='cls',
):
    return ExperimentConfig(
        experiment_name=experiment_name,
        artifacts_root="artifacts",
        seed=42,
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
            early_stopping_patience=early_stopping_patience,
            amp=True, amp_dtype="float16",
            gradient_accumulation_steps=gradient_accumulation_steps,
            scheduler=scheduler, warmup_epochs=warmup_epochs, lr_min=1e-6,
            label_smoothing=label_smoothing,
        ),
        episode=EpisodeConfig(enabled=False),
    )


def main():
    experiments = [
        # 1. COMBINE: label_smooth=0.02 + cosine_warmup + 60ep (the two winners)
        ("exp_120_ls02_cosine60",
         "Eng + ls=0.02 + cosine_warmup 60ep SWA from 40",
         lambda: make_engineered_config("exp_120_ls02_cosine60",
                                        label_smoothing=0.02, scheduler="cosine_warmup",
                                        warmup_epochs=3, max_epochs=60),
         40, 60),

        # 2. Same combo but SWA from 35 (wider averaging window)
        ("exp_121_ls02_cosine60_swa35",
         "Eng + ls=0.02 + cosine_warmup 60ep SWA from 35",
         lambda: make_engineered_config("exp_121_ls02_cosine60_swa35",
                                        label_smoothing=0.02, scheduler="cosine_warmup",
                                        warmup_epochs=3, max_epochs=60),
         35, 60),

        # 3. ls=0.02 + cosine_warmup + 80ep + SWA from 55
        ("exp_122_ls02_cosine80",
         "Eng + ls=0.02 + cosine_warmup 80ep SWA from 55",
         lambda: make_engineered_config("exp_122_ls02_cosine80",
                                        label_smoothing=0.02, scheduler="cosine_warmup",
                                        warmup_epochs=3, lr=1.5e-4, max_epochs=80),
         55, 80),

        # 4. ls=0.01 (smaller smoothing) + cosine_warmup 60ep
        ("exp_123_ls01_cosine60",
         "Eng + ls=0.01 + cosine_warmup 60ep SWA from 40",
         lambda: make_engineered_config("exp_123_ls01_cosine60",
                                        label_smoothing=0.01, scheduler="cosine_warmup",
                                        warmup_epochs=3, max_epochs=60),
         40, 60),

        # 5. ls=0.02 + cosine_warmup + lower lr=1.5e-4 60ep
        ("exp_124_ls02_cosine_lowlr",
         "Eng + ls=0.02 + cosine_warmup lr=1.5e-4 60ep SWA from 40",
         lambda: make_engineered_config("exp_124_ls02_cosine_lowlr",
                                        label_smoothing=0.02, scheduler="cosine_warmup",
                                        warmup_epochs=3, lr=1.5e-4, max_epochs=60),
         40, 60),

        # 6. ls=0.02 + cosine_warmup + wd=3e-4
        ("exp_125_ls02_cosine_wd3e4",
         "Eng + ls=0.02 + cosine_warmup wd=3e-4 60ep SWA from 40",
         lambda: make_engineered_config("exp_125_ls02_cosine_wd3e4",
                                        label_smoothing=0.02, scheduler="cosine_warmup",
                                        warmup_epochs=3, weight_decay=3e-4, max_epochs=60),
         40, 60),

        # 7. ls=0.03 (slightly more smoothing) + cosine_warmup 60ep
        ("exp_126_ls03_cosine60",
         "Eng + ls=0.03 + cosine_warmup 60ep SWA from 40",
         lambda: make_engineered_config("exp_126_ls03_cosine60",
                                        label_smoothing=0.03, scheduler="cosine_warmup",
                                        warmup_epochs=3, max_epochs=60),
         40, 60),

        # 8. Combo + ft_dropout=0.05 (less dropout)
        ("exp_127_ls02_cosine_ftd05",
         "Eng + ls=0.02 + cosine_warmup ft_dropout=0.05 60ep SWA from 40",
         lambda: make_engineered_config("exp_127_ls02_cosine_ftd05",
                                        label_smoothing=0.02, scheduler="cosine_warmup",
                                        warmup_epochs=3, feature_token_dropout=0.05,
                                        max_epochs=60),
         40, 60),
    ]

    print("=" * 60)
    print("WAVE 9 — Combine Label Smoothing + Cosine Warmup")
    print(f"Total experiments: {len(experiments)}")
    print(f"Current best: {get_best_val_loss():.6f}")
    print("=" * 60)

    for i, (exp_id, desc, config_fn, swa_start, total_epochs) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Starting {exp_id}...")
        try:
            config = config_fn()
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

    print("\n" + "=" * 60)
    print("WAVE 9 COMPLETE")
    print(f"Best: {get_best_val_loss():.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
