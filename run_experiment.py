"""Autonomous experiment runner for Tabula.

Usage:
    python run_experiment.py
    
This script runs a single experiment, logs results to experiments_log.tsv,
and saves/discards based on whether it improves over previous best.
"""
from __future__ import annotations

import csv
import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch

from tabula.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TaskConfig,
    TrainingConfig,
    EpisodeConfig,
)
from tabula.training.engine import train


def _evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint_path: Path,
) -> tuple[float | None, float | None]:
    """Load a checkpoint and evaluate it on the val set. Returns (accuracy, roc_auc)."""
    device = torch.device(config.training.device)
    _, val_loader, num_numeric, num_categorical, num_text, output_dim = build_dataloaders(config)
    effective_output_dim = 1 if config.task.problem_type in {"binary", "regression"} else output_dim
    
    model = TabularTransformer(
        config, num_numeric, num_categorical, num_text, effective_output_dim
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    all_y = []
    all_logits = []
    with torch.no_grad():
        for batch in val_loader:
            from tabula.training.engine import _move_batch
            batch = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config.training.amp and device.type == "cuda"):
                logits = model(batch)
            all_y.append(batch.y.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    
    y_true = np.concatenate(all_y, axis=0)
    logits_arr = np.concatenate(all_logits, axis=0)
    metrics = compute_metrics(config.task.problem_type, y_true, logits_arr)
    return metrics.get("accuracy"), metrics.get("roc_auc")


import numpy as np

from tabula.data.datasets import build_dataloaders
from tabula.evaluation.metrics import compute_metrics
from tabula.models.transformer import TabularTransformer, EpisodicTabularTransformer


EXPERIMENTS_LOG = Path("experiments_log.tsv")
BEST_TRACKER = Path("best_experiment.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_experiment(
    experiment_id: str,
    description: str,
    status: str,
    val_loss: float | None,
    accuracy: float | None,
    roc_auc: float | None,
    steps: int | None,
    duration_s: float | None,
    notes: str = "",
) -> None:
    """Append one row to the experiments TSV log."""
    with EXPERIMENTS_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            experiment_id,
            _utc_now(),
            description,
            status,
            f"{val_loss:.6f}" if val_loss is not None else "",
            f"{accuracy:.4f}" if accuracy is not None else "",
            f"{roc_auc:.4f}" if roc_auc is not None else "",
            steps or "",
            f"{duration_s:.1f}" if duration_s is not None else "",
            notes,
        ])


def get_best_val_loss() -> float:
    """Read the current best val loss from the tracker, or inf."""
    if BEST_TRACKER.exists():
        data = json.loads(BEST_TRACKER.read_text(encoding="utf-8"))
        return float(data.get("best_val_loss", float("inf")))
    return float("inf")


def update_best(experiment_id: str, val_loss: float, roc_auc: float | None) -> None:
    """Update the best experiment tracker."""
    BEST_TRACKER.write_text(
        json.dumps({
            "experiment_id": experiment_id,
            "best_val_loss": val_loss,
            "best_roc_auc": roc_auc,
            "updated_at": _utc_now(),
        }, indent=2),
        encoding="utf-8",
    )


def make_adult_config(
    experiment_name: str,
    *,
    d_model: int = 192,
    n_heads: int = 6,
    n_layers: int = 6,
    d_ff: int = 384,
    dropout: float = 0.1,
    feature_token_dropout: float = 0.05,
    norm: str = "rmsnorm",
    ffn_activation: str = "swiglu",
    numeric_embedding: str = "linear",
    numeric_periodic_features: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 20,
    early_stopping_patience: int = 5,
    amp: bool = True,
    amp_dtype: str = "float16",
    gradient_accumulation_steps: int = 1,
    max_categories: int = 256,
    seed: int = 42,
    scheduler: str = "none",
    warmup_epochs: int = 2,
    lr_min: float = 1e-6,
    pooling: str = "cls",
    label_smoothing: float = 0.0,
) -> ExperimentConfig:
    """Build an ExperimentConfig for the adult census dataset."""
    return ExperimentConfig(
        experiment_name=experiment_name,
        artifacts_root="artifacts",
        seed=seed,
        task=TaskConfig(
            mode="finetune",
            problem_type="binary",
            target_column="income",
        ),
        data=DataConfig(
            dataset_type="prepared",
            prepared_dir="data/processed/adult_census_income",
            train_path="data/processed/adult_census_income/train.csv",
            val_path="data/processed/adult_census_income/val.csv",
            batch_size=batch_size,
            num_workers=0,
            standardize_numeric=True,
        ),
        model=ModelConfig(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            feature_token_dropout=feature_token_dropout,
            norm=norm,
            ffn_activation=ffn_activation,
            max_categories=max_categories,
            numeric_embedding=numeric_embedding,
            numeric_periodic_features=numeric_periodic_features,
            pooling=pooling,
        ),
        training=TrainingConfig(
            device="cuda",
            max_epochs=max_epochs,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip_norm=1.0,
            log_interval=20,
            early_stopping_patience=early_stopping_patience,
            amp=amp,
            amp_dtype=amp_dtype,
            gradient_accumulation_steps=gradient_accumulation_steps,
            scheduler=scheduler,
            warmup_epochs=warmup_epochs,
            lr_min=lr_min,
            label_smoothing=label_smoothing,
        ),
        episode=EpisodeConfig(enabled=False),
    )


def run_single_experiment(
    experiment_id: str,
    description: str,
    config: ExperimentConfig,
    timeout_s: float = 600,
) -> dict:
    """Run one experiment, log results, return result dict."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_id}")
    print(f"DESC: {description}")
    print(f"{'='*60}\n")
    
    torch.cuda.empty_cache()
    start = time.time()
    
    try:
        result = train(config)
        duration = time.time() - start
        
        val_loss = result.get("best_val_loss", float("inf"))
        
        # Try to get accuracy/roc_auc from progress log (streaming mode)
        progress_path = Path(config.artifacts_root) / config.experiment_name / "progress.jsonl"
        accuracy = None
        roc_auc = None
        if progress_path.exists():
            lines = progress_path.read_text(encoding="utf-8").strip().split("\n")
            for line in reversed(lines):
                entry = json.loads(line)
                if entry.get("event") == "validation":
                    metrics = entry.get("val_metrics", {})
                    accuracy = metrics.get("accuracy")
                    roc_auc = metrics.get("roc_auc")
                    break
        
        # For epoch-based training, evaluate the best checkpoint to get metrics
        if accuracy is None:
            checkpoint_path = Path(config.artifacts_root) / config.experiment_name / "best.pt"
            if checkpoint_path.exists():
                accuracy, roc_auc = _evaluate_checkpoint(config, checkpoint_path)
        
        log_experiment(
            experiment_id, description, "success",
            val_loss, accuracy, roc_auc,
            config.training.max_epochs, duration,
        )
        
        prev_best = get_best_val_loss()
        if val_loss < prev_best:
            update_best(experiment_id, val_loss, roc_auc)
            print(f"\n*** NEW BEST: val_loss={val_loss:.6f} (prev={prev_best:.6f}) ***\n")
        else:
            print(f"\nNo improvement: val_loss={val_loss:.6f} vs best={prev_best:.6f}\n")
        
        return {
            "status": "success",
            "val_loss": val_loss,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "duration": duration,
        }
        
    except Exception as e:
        duration = time.time() - start
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\nEXPERIMENT FAILED: {error_msg}\n")
        traceback.print_exc()
        
        log_experiment(
            experiment_id, description, "crash",
            None, None, None, None, duration,
            notes=error_msg[:200],
        )
        
        return {"status": "crash", "error": error_msg, "duration": duration}


if __name__ == "__main__":
    # Quick test: baseline adult config
    config = make_adult_config("exp_baseline_001")
    result = run_single_experiment("exp_baseline_001", "Baseline adult d192 6L 6H", config)
    print(json.dumps(result, indent=2, default=str))
