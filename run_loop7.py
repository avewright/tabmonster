#!/usr/bin/env python3
"""Wave 7: v2 data (target encoding + frequency encoding + more interactions).

32 numeric + 8 categorical = 40 features. The target/freq encodings provide
pre-computed category-target statistics as numeric features.
"""
import sys
sys.path.insert(0, '.')
import gc
import time
import traceback

import numpy as np
import torch

from run_experiment import (
    log_experiment, get_best_val_loss, update_best,
    ExperimentConfig, TaskConfig, DataConfig, ModelConfig, TrainingConfig, EpisodeConfig,
)
from run_loop import run_swa_experiment
from tabula.utils import set_seed


def make_v2_config(name, **kw):
    return ExperimentConfig(
        experiment_name=name,
        artifacts_root="artifacts",
        seed=kw.pop('seed', 42),
        task=TaskConfig(mode="finetune", problem_type="binary", target_column="income"),
        data=DataConfig(
            dataset_type="prepared",
            prepared_dir="data/processed/adult_census_v2",
            train_path="data/processed/adult_census_v2/train.csv",
            val_path="data/processed/adult_census_v2/val.csv",
            batch_size=kw.pop('batch_size', 512),
            num_workers=0,
            standardize_numeric=True,
        ),
        model=ModelConfig(
            d_model=kw.pop('d_model', 192),
            n_heads=kw.pop('n_heads', 6),
            n_layers=kw.pop('n_layers', 6),
            d_ff=kw.pop('d_ff', 384),
            dropout=kw.pop('dropout', 0.1),
            feature_token_dropout=kw.pop('feature_token_dropout', 0.1),
            norm="rmsnorm", ffn_activation="swiglu", max_categories=256,
            numeric_embedding=kw.pop('numeric_embedding', 'periodic'),
            numeric_periodic_features=kw.pop('numeric_periodic_features', 16),
            pooling=kw.pop('pooling', 'cls'),
        ),
        training=TrainingConfig(
            device="cuda",
            max_epochs=kw.pop('max_epochs', 40),
            lr=kw.pop('lr', 2e-4),
            weight_decay=kw.pop('weight_decay', 1e-4),
            grad_clip_norm=1.0, log_interval=20,
            early_stopping_patience=999,
            amp=True, amp_dtype="float16",
            gradient_accumulation_steps=1,
            scheduler="none", warmup_epochs=2, lr_min=1e-6,
            label_smoothing=kw.pop('label_smoothing', 0.0),
        ),
        episode=EpisodeConfig(enabled=False),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("WAVE 7: V2 DATA (TARGET + FREQ ENCODING)")
    print(f"Current best val_loss: {get_best_val_loss():.6f}")
    print("=" * 60)

    experiments = [
        # Exp 100: v2 base (ft_dropout=0.1, same as best from wave 5)
        ("swa", "exp_100_v2_base_swa",
         "v2 data + ft_dropout=0.1 batch=512 periodic k=16 lr=2e-4 SWA from 25, 40ep",
         lambda: make_v2_config("exp_100"),
         25, 40),

        # Exp 101: v2 + wd=3e-4
        ("swa", "exp_101_v2_wd3e4_swa",
         "v2 + wd=3e-4 batch=512 periodic k=16 lr=2e-4 SWA from 25, 40ep",
         lambda: make_v2_config("exp_101", weight_decay=3e-4),
         25, 40),

        # Exp 102: v2 + wider d256
        ("swa", "exp_102_v2_d256_swa",
         "v2 + d256 8H batch=512 periodic k=16 lr=1.5e-4 SWA from 25, 40ep",
         lambda: make_v2_config("exp_102", d_model=256, n_heads=8, d_ff=512, lr=1.5e-4),
         25, 40),

        # Exp 103: v2 + deeper 8L
        ("swa", "exp_103_v2_deep8_swa",
         "v2 + 8L batch=512 periodic k=16 lr=2e-4 SWA from 25, 40ep",
         lambda: make_v2_config("exp_103", n_layers=8),
         25, 40),

        # Exp 104: v2 + d256 8L (bigger model for 40 features)
        ("swa", "exp_104_v2_d256_8L_swa",
         "v2 + d256 8L 8H batch=512 periodic k=16 lr=1e-4 SWA from 25, 40ep",
         lambda: make_v2_config("exp_104", d_model=256, n_heads=8, n_layers=8, d_ff=512, lr=1e-4),
         25, 40),

        # Exp 105: v2 + higher ft_dropout=0.15 (more features = need more regularization)
        ("swa", "exp_105_v2_ftdrop015_swa",
         "v2 + ft_dropout=0.15 batch=512 periodic k=16 lr=2e-4 SWA from 25, 40ep",
         lambda: make_v2_config("exp_105", feature_token_dropout=0.15),
         25, 40),

        # Exp 106: v2 + ft_dropout=0.05 (less dropout)
        ("swa", "exp_106_v2_ftdrop005_swa",
         "v2 + ft_dropout=0.05 batch=512 periodic k=16 lr=2e-4 SWA from 25, 40ep",
         lambda: make_v2_config("exp_106", feature_token_dropout=0.05),
         25, 40),

        # Exp 107: v2 + d256 + wd=3e-4
        ("swa", "exp_107_v2_d256_wd3e4_swa",
         "v2 + d256 8H wd=3e-4 batch=512 periodic k=16 lr=1.5e-4 SWA from 25, 40ep",
         lambda: make_v2_config("exp_107", d_model=256, n_heads=8, d_ff=512, lr=1.5e-4, weight_decay=3e-4),
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
    print(f"WAVE 7 COMPLETE. Best val_loss: {get_best_val_loss():.6f}")
    print("=" * 60)
