#!/usr/bin/env python3
"""
Wave 8 — New strategies on engineered features:
  1. Longer training (80 epochs, SWA 50-80)
  2. Gradient accumulation (effective batch 2048)
  3. Small label smoothing (0.02)
  4. Cosine warmup + longer training (60ep)
  5. Deeper narrow model (12L, d128)
  6. Very wide model (d384, 4L)
  7. Lower lr + longer (lr=1e-4, 60ep)
  8. ft_dropout=0.15 + cosine warmup

Best so far: exp_084 = 0.300232 (d192/6L/6H, ft_dropout=0.1, engineered, SWA 25-40)
"""
import sys
sys.path.insert(0, '.')
import gc
import time
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
        # 1. Longer training: 80 epochs, SWA from 50
        ("exp_110_eng_80ep_swa50",
         "Eng + 80ep lr=1.5e-4 SWA from 50",
         lambda: make_engineered_config("exp_110_eng_80ep_swa50", max_epochs=80, lr=1.5e-4),
         50, 80),

        # 2. Gradient accumulation: effective batch 2048
        ("exp_111_eng_gradaccum4",
         "Eng + grad_accum=4 eff_batch=2048 lr=4e-4 SWA",
         lambda: make_engineered_config("exp_111_eng_gradaccum4",
                                        gradient_accumulation_steps=4, lr=4e-4),
         25, 40),

        # 3. Small label smoothing (0.02)
        ("exp_112_eng_ls002",
         "Eng + label_smoothing=0.02 SWA",
         lambda: make_engineered_config("exp_112_eng_ls002", label_smoothing=0.02),
         25, 40),

        # 4. Cosine warmup + 60 epochs
        ("exp_114_eng_cosine60",
         "Eng + cosine_warmup 60ep warmup=3 SWA from 40",
         lambda: make_engineered_config("exp_114_eng_cosine60",
                                        max_epochs=60, scheduler="cosine_warmup",
                                        warmup_epochs=3),
         40, 60),

        # 5. Deeper narrow (12L d128)
        ("exp_115_eng_deep12_d128",
         "Eng + 12L d128 4H d_ff=256 SWA",
         lambda: make_engineered_config("exp_115_eng_deep12_d128",
                                        d_model=128, n_heads=4, n_layers=12, d_ff=256),
         25, 40),

        # 6. Very wide (d384, 4L)
        ("exp_116_eng_wide384_4L",
         "Eng + d384 4L 12H d_ff=768 lr=1e-4 SWA",
         lambda: make_engineered_config("exp_116_eng_wide384_4L",
                                        d_model=384, n_heads=12, n_layers=4, d_ff=768,
                                        lr=1e-4),
         25, 40),

        # 7. Lower lr + longer
        ("exp_118_eng_lowlr60",
         "Eng + lr=1e-4 60ep SWA from 40",
         lambda: make_engineered_config("exp_118_eng_lowlr60",
                                        lr=1e-4, max_epochs=60),
         40, 60),

        # 8. ft_dropout=0.15 + cosine warmup
        ("exp_119_eng_ftd15_cosine",
         "Eng + ft_dropout=0.15 cosine_warmup 50ep SWA from 30",
         lambda: make_engineered_config("exp_119_eng_ftd15_cosine",
                                        feature_token_dropout=0.15,
                                        max_epochs=50, scheduler="cosine_warmup",
                                        warmup_epochs=3),
         30, 50),
    ]

    print("=" * 60)
    print("WAVE 8 — New Strategies on Engineered Features")
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
    print("WAVE 8 COMPLETE")
    print(f"Best: {get_best_val_loss():.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
