#!/usr/bin/env python3
"""Wave 6: Optimizing on engineered features.

Best so far: exp_084 at val_loss=0.300232 (eng + ft_dropout=0.1 + SWA)

Strategy:
- Multi-seed ensemble on engineered data 
- Combined best regularizations on engineered features
- Even more feature engineering (log fnlwgt, education buckets)
- Architecture search on engineered features
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
from run_loop4 import train_one_model_return_logits, run_large_ensemble
from tabula.data.datasets import build_dataloaders
from tabula.evaluation.metrics import compute_metrics
from tabula.utils import set_seed


def make_eng_config(name, **kw):
    d_model = kw.pop('d_model', 192)
    n_heads = kw.pop('n_heads', 6)
    n_layers = kw.pop('n_layers', 6)
    d_ff = kw.pop('d_ff', 384)
    dropout = kw.pop('dropout', 0.1)
    ft_dropout = kw.pop('feature_token_dropout', 0.05)
    numeric_embedding = kw.pop('numeric_embedding', 'periodic')
    numeric_periodic_features = kw.pop('numeric_periodic_features', 16)
    lr = kw.pop('lr', 2e-4)
    wd = kw.pop('weight_decay', 1e-4)
    batch_size = kw.pop('batch_size', 512)
    max_epochs = kw.pop('max_epochs', 40)
    label_smoothing = kw.pop('label_smoothing', 0.0)
    pooling = kw.pop('pooling', 'cls')
    scheduler = kw.pop('scheduler', 'none')
    warmup_epochs = kw.pop('warmup_epochs', 2)
    lr_min = kw.pop('lr_min', 1e-6)
    
    return ExperimentConfig(
        experiment_name=name,
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
            dropout=dropout, feature_token_dropout=ft_dropout,
            norm="rmsnorm", ffn_activation="swiglu", max_categories=256,
            numeric_embedding=numeric_embedding,
            numeric_periodic_features=numeric_periodic_features,
            pooling=pooling,
        ),
        training=TrainingConfig(
            device="cuda", max_epochs=max_epochs, lr=lr,
            weight_decay=wd, grad_clip_norm=1.0, log_interval=20,
            early_stopping_patience=999,
            amp=True, amp_dtype="float16",
            gradient_accumulation_steps=1,
            scheduler=scheduler, warmup_epochs=warmup_epochs, lr_min=lr_min,
            label_smoothing=label_smoothing,
        ),
        episode=EpisodeConfig(enabled=False),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("WAVE 6: ENGINEERED FEATURES OPTIMIZATION")
    print(f"Current best val_loss: {get_best_val_loss():.6f}")
    print("=" * 60)

    experiments = []

    # --- Exp 90: Combined ft_dropout=0.1 + wd=3e-4 ---
    experiments.append(("swa", "exp_090_eng_ftdrop_wd3e4_swa",
        "Eng + ft_dropout=0.1 wd=3e-4 batch=512 SWA from 25, 40ep",
        lambda: make_eng_config("exp_090", feature_token_dropout=0.1, weight_decay=3e-4),
        25, 40))

    # --- Exp 91: ft_dropout=0.15 ---
    experiments.append(("swa", "exp_091_eng_ftdrop015_swa",
        "Eng + ft_dropout=0.15 batch=512 SWA from 25, 40ep",
        lambda: make_eng_config("exp_091", feature_token_dropout=0.15),
        25, 40))

    # --- Exp 92: ft_dropout=0.1 + dropout=0.15 ---
    experiments.append(("swa", "exp_092_eng_ftdrop01_drop015_swa",
        "Eng + ft_dropout=0.1 dropout=0.15 batch=512 SWA from 25, 40ep",
        lambda: make_eng_config("exp_092", feature_token_dropout=0.1, dropout=0.15),
        25, 40))

    # --- Exp 93: ft_dropout=0.1 + d256 8H ---
    experiments.append(("swa", "exp_093_eng_ftdrop01_d256_swa",
        "Eng + ft_dropout=0.1 d256 8H batch=512 lr=1.5e-4 SWA from 25, 40ep",
        lambda: make_eng_config("exp_093", feature_token_dropout=0.1, d_model=256, n_heads=8, d_ff=512, lr=1.5e-4),
        25, 40))

    # --- Exp 94: 10-seed ensemble of eng + ft_dropout=0.1 ---
    def exp_094():
        eid = "exp_094_eng_ensemble10_ftdrop"
        desc = "10-seed ensemble: eng + ft_dropout=0.1 batch=512 SWA"
        seeds = [42, 123, 456, 789, 1337, 2024, 7777, 9999, 31415, 27182]
        base_config = make_eng_config(eid, feature_token_dropout=0.1)
        configs_and_seeds = [(base_config, s) for s in seeds]
        return "ensemble", eid, desc, configs_and_seeds
    experiments.append(exp_094)

    # --- Exp 95: ft_dropout=0.1 + cosine warmup ---
    experiments.append(("swa", "exp_095_eng_ftdrop01_cosine_swa",
        "Eng + ft_dropout=0.1 cosine_warmup lr=2e-4 batch=512 SWA from 25, 40ep",
        lambda: make_eng_config("exp_095", feature_token_dropout=0.1, scheduler='cosine_warmup', warmup_epochs=3, lr_min=1e-6),
        25, 40))

    # --- Exp 96: ft_dropout=0.1 + wider SWA from 20 ---
    experiments.append(("swa", "exp_096_eng_ftdrop01_swa20",
        "Eng + ft_dropout=0.1 batch=512 SWA from 20, 40ep",
        lambda: make_eng_config("exp_096", feature_token_dropout=0.1),
        20, 40))

    # --- Exp 97: SWA from epoch 10 (very wide SWA window, 30 checkpoints) ---
    experiments.append(("swa", "exp_097_eng_ftdrop01_swa10",
        "Eng + ft_dropout=0.1 batch=512 SWA from 10, 40ep (30 checkpoints)",
        lambda: make_eng_config("exp_097", feature_token_dropout=0.1),
        10, 40))

    for i, spec in enumerate(experiments):
        if callable(spec):
            spec = spec()

        mode = spec[0]
        eid = spec[1]
        desc = spec[2]

        print(f"\n[{i+1}/{len(experiments)}] Starting {eid}...")
        try:
            if mode == "swa":
                config = spec[3]() if callable(spec[3]) else spec[3]
                swa_start, total_epochs = spec[4], spec[5]
                result = run_swa_experiment(eid, desc, config, swa_start, total_epochs)
            elif mode == "ensemble":
                configs_and_seeds = spec[3]
                result = run_large_ensemble(eid, desc, configs_and_seeds)

            print(f"  -> val_loss={result.get('val_loss')}, roc_auc={result.get('roc_auc')}, duration={result.get('duration', 0):.1f}s")

        except Exception as e:
            print(f"  -> CRASH: {e}")
            traceback.print_exc()
            log_experiment(eid, desc, 'crash', None, None, None, None, 0, notes=str(e)[:200])

        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 60)
    print(f"WAVE 6 COMPLETE. Best val_loss: {get_best_val_loss():.6f}")
    print("=" * 60)
