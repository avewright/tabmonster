#!/usr/bin/env python3
"""
Wave 12 — Creative approaches + multi-seed refinement.

Plateau at 0.297603 confirmed. Try creative angles:
  1. Mean pooling + ls=0.01 + cosine (close in wave 11)
  2. batch=1024 (larger batch)  
  3. batch=256 (smaller batch)
  4. n_heads=3 (fewer heads)
  5. n_heads=12 (more heads)
  6. d_ff=768 (wider FFN, same model size)
  7. d_ff=192 (narrower FFN)
  8. 5-seed mean-pooling ensemble
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
from run_loop import run_swa_experiment, build_dataloaders, _make_criterion, _run_epoch, _move_batch, _build_scheduler, compute_metrics
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


def main():
    experiments = [
        # 1. Mean pooling + best config (0.297888 in wave 11 — close!)
        ("exp_150_mean_ls01_cosine",
         "Eng + mean pooling + ls=0.01 + cosine 60ep SWA40",
         lambda: (make_config("exp_150_mean_ls01_cosine", pooling="mean"), 40, 60)),

        # 2. Larger batch = 1024
        ("exp_151_batch1024",
         "Eng + batch=1024 lr=3e-4 ls=0.01 cosine 60ep SWA40",
         lambda: (make_config("exp_151_batch1024", batch_size=1024, lr=3e-4), 40, 60)),

        # 3. Smaller batch = 256
        ("exp_152_batch256",
         "Eng + batch=256 lr=1.5e-4 ls=0.01 cosine 60ep SWA40",
         lambda: (make_config("exp_152_batch256", batch_size=256, lr=1.5e-4), 40, 60)),

        # 4. Fewer heads (3 heads, d_model=192 → 64 per head)
        ("exp_153_3heads",
         "Eng + 3 heads ls=0.01 cosine 60ep SWA40",
         lambda: (make_config("exp_153_3heads", n_heads=3), 40, 60)),

        # 5. More heads (12 heads, 16 per head)
        ("exp_154_12heads",
         "Eng + 12 heads ls=0.01 cosine 60ep SWA40",
         lambda: (make_config("exp_154_12heads", n_heads=12), 40, 60)),

        # 6. Wider FFN (d_ff=768 vs 384)
        ("exp_155_ff768",
         "Eng + d_ff=768 ls=0.01 cosine 60ep SWA40",
         lambda: (make_config("exp_155_ff768", d_ff=768), 40, 60)),

        # 7. Narrower FFN (d_ff=192 = same as d_model)
        ("exp_156_ff192",
         "Eng + d_ff=192 ls=0.01 cosine 60ep SWA40",
         lambda: (make_config("exp_156_ff192", d_ff=192), 40, 60)),

        # 8. Mean pool + batch=256 + more epochs
        ("exp_157_mean_b256_80ep",
         "Eng + mean pool + batch=256 + ls=0.01 cosine 80ep SWA55",
         lambda: (make_config("exp_157_mean_b256_80ep", pooling="mean",
                              batch_size=256, lr=1.5e-4, max_epochs=80), 55, 80)),
    ]

    print("=" * 60)
    print("WAVE 12 — Creative Approaches")
    print(f"Total experiments: {len(experiments)}")
    print(f"Current best: {get_best_val_loss():.6f}")
    print("=" * 60)

    for i, (exp_id, desc, config_fn) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Starting {exp_id}...")
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

    print("\n" + "=" * 60)
    print("WAVE 12 COMPLETE")
    print(f"Best: {get_best_val_loss():.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
