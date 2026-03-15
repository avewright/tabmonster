#!/usr/bin/env python3
"""Wave 5: Feature-engineered data experiments.

Uses log-transformed capital features, interaction terms, and binary indicators.
14 numeric + 8 categorical features (vs original 6 numeric + 8 categorical).
"""
import sys
sys.path.insert(0, '.')
import copy
import gc
import time
import traceback

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
from run_loop import run_swa_experiment
from tabula.utils import set_seed


def make_engineered_config(
    experiment_name,
    *,
    d_model=192, n_heads=6, n_layers=6, d_ff=384,
    dropout=0.1, feature_token_dropout=0.05,
    numeric_embedding='periodic', numeric_periodic_features=16,
    lr=2e-4, weight_decay=1e-4, batch_size=512,
    max_epochs=40, early_stopping_patience=999,
    amp=True, label_smoothing=0.0,
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
            amp=amp, amp_dtype="float16",
            gradient_accumulation_steps=1,
            scheduler="none", warmup_epochs=2, lr_min=1e-6,
            label_smoothing=label_smoothing,
        ),
        episode=EpisodeConfig(enabled=False),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("WAVE 5: FEATURE-ENGINEERED DATA EXPERIMENTS")
    print(f"Current best val_loss: {get_best_val_loss():.6f}")
    print("=" * 60)

    experiments = [
        # Exp 80: Best config with engineered features
        ("swa", "exp_080_eng_base_swa",
         "Engineered features + batch=512 periodic k=16 lr=2e-4 SWA from 25, 40ep",
         lambda: make_engineered_config("exp_080_eng_base_swa"),
         25, 40),
        
        # Exp 81: Wider d256 with engineered features
        ("swa", "exp_081_eng_wide256_swa",
         "Engineered + d256 8H batch=512 periodic k=16 lr=1.5e-4 SWA from 25, 40ep",
         lambda: make_engineered_config("exp_081_eng_wide256_swa",
                                        d_model=256, n_heads=8, d_ff=512, lr=1.5e-4),
         25, 40),

        # Exp 82: Engineered + wd=3e-4
        ("swa", "exp_082_eng_wd3e4_swa",
         "Engineered + wd=3e-4 batch=512 periodic k=16 lr=2e-4 SWA from 25, 40ep",
         lambda: make_engineered_config("exp_082_eng_wd3e4_swa", weight_decay=3e-4),
         25, 40),

        # Exp 83: Engineered + deeper 8L
        ("swa", "exp_083_eng_deep8_swa",
         "Engineered + 8L batch=512 periodic k=16 lr=2e-4 SWA from 25, 40ep",
         lambda: make_engineered_config("exp_083_eng_deep8_swa", n_layers=8),
         25, 40),

        # Exp 84: Engineered + ft_dropout=0.1
        ("swa", "exp_084_eng_ftdrop_swa",
         "Engineered + ft_dropout=0.1 batch=512 periodic k=16 lr=2e-4 SWA from 25, 40ep",
         lambda: make_engineered_config("exp_084_eng_ftdrop_swa", feature_token_dropout=0.1),
         25, 40),

        # Exp 85: Engineered + wider SWA from 15
        ("swa", "exp_085_eng_swa15_swa",
         "Engineered + SWA from 15 batch=512 periodic k=16 lr=2e-4, 40ep",
         lambda: make_engineered_config("exp_085_eng_swa15_swa"),
         15, 40),

        # Exp 86: Engineered + periodic k=32
        ("swa", "exp_086_eng_k32_swa",
         "Engineered + periodic k=32 batch=512 lr=2e-4 SWA from 25, 40ep",
         lambda: make_engineered_config("exp_086_eng_k32_swa", numeric_periodic_features=32),
         25, 40),

        # Exp 87: Engineered + d256 8L 8H (big model for more features)  
        ("swa", "exp_087_eng_d256_8L_swa",
         "Engineered + d256 8L 8H batch=512 periodic k=16 lr=1e-4 SWA from 25, 40ep",
         lambda: make_engineered_config("exp_087_eng_d256_8L_swa",
                                        d_model=256, n_heads=8, n_layers=8, d_ff=512, lr=1e-4),
         25, 40),
    ]

    for i, (mode, eid, desc, config_fn, swa_start, total_epochs) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Starting {eid}...")
        try:
            config = config_fn()
            result = run_swa_experiment(eid, desc, config, swa_start, total_epochs)
            print(f"  -> val_loss={result.get('val_loss')}, roc_auc={result.get('roc_auc')}, duration={result.get('duration', 0):.1f}s")
        except Exception as e:
            print(f"  -> CRASH: {e}")
            traceback.print_exc()
            log_experiment(eid, desc, 'crash', None, None, None, None, 0, notes=str(e)[:200])
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 60)
    print(f"WAVE 5 COMPLETE. Best val_loss: {get_best_val_loss():.6f}")
    print("=" * 60)
