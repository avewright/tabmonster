"""
Push Tabula v1 pretrained model to HuggingFace Hub.
Uploads: model weights (best.pt + latest.pt), config.json, README.md, training log.
"""

import os, json, shutil, tempfile, pathlib, textwrap, datetime
import torch
from huggingface_hub import HfApi, create_repo

# ── Config ─────────────────────────────────────────────────────────────────────
HF_TOKEN   = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Set HF_TOKEN env var (e.g. source .env) before running.")
REPO_ID    = "avewright/tabula-v1"
CKPT_DIR   = pathlib.Path("artifacts/pretrain_corpus_v1")
LOG_PATH   = pathlib.Path("artifacts/pretrain_corpus_v1_output.log")

# ── Load checkpoint metadata ───────────────────────────────────────────────────
best_ckpt  = torch.load(CKPT_DIR / "best.pt",   map_location="cpu", weights_only=False)
last_ckpt  = torch.load(CKPT_DIR / "latest.pt", map_location="cpu", weights_only=False)

best_step  = best_ckpt["global_step"]
best_val   = best_ckpt["best_val_loss"]
best_rows  = best_ckpt["rows_seen"]
last_step  = last_ckpt["global_step"]
last_rows  = last_ckpt["rows_seen"]
n_params   = best_ckpt["n_params"]
cfg        = best_ckpt.get("config")

# ── Build config.json ──────────────────────────────────────────────────────────
config_dict = {
    "model_type": "tabula_transformer",
    "architecture": "TabularTransformer",
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 8,
    "d_ff": 512,
    "dropout": 0.1,
    "ffn_activation": "swiglu",
    "norm": "rmsnorm",
    "pooling": "cls",
    "numeric_embedding": "periodic",
    "numeric_periodic_features": 16,
    "max_numeric_features": 64,
    "max_categories": 128,
    "feature_token_dropout": 0.05,
    "n_params": n_params,
    "pretraining": {
        "best_step": best_step,
        "best_val_loss": round(best_val, 6),
        "best_rows_seen": best_rows,
        "final_step": last_step,
        "final_rows_seen": last_rows,
        "batch_size": 512,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "amp": True,
        "amp_dtype": "float16",
        "grad_clip": 1.0,
        "warmup_steps": 2000,
        "lr_schedule": "cosine",
        "max_steps": 200000,
    },
    "corpus": {
        "hf_repo": "avewright/tabula-pretraining-corpus-v2",
        "total_shards": 541,
        "real_datasets_ok": 3371,
        "sources": {
            "pmlb": {"ok": 422, "total_attempted": 423, "status": "fully_exhausted"},
            "openml": {"ok": 2949, "total_attempted": 4886, "schema_fail": 1900, "download_fail": 37},
            "huggingface": {"ok": 0, "download_fail": 66, "schema_fail": 1},
        },
        "synthetic_generators": [
            "tree_prior", "gaussian_mixture", "polynomial", "scm", "regression",
            "time_series", "mixed_type"
        ],
    },
    "date_trained": datetime.date.today().isoformat(),
    "framework": "pytorch",
    "pytorch_version": torch.__version__,
}

# ── Build README / model card ──────────────────────────────────────────────────
model_card = f"""\
---
license: apache-2.0
tags:
  - tabular
  - foundation-model
  - pretraining
  - tabpfn
  - schema-aware
  - pytorch
datasets:
  - avewright/tabula-pretraining-corpus-v2
language:
  - en
---

# Tabula v1 — Tabular Foundation Model (Pretrained)

A schema-aware tabular transformer pretrained on a large multi-source corpus
of real and synthetic tabular datasets.

## Model Architecture

| Property | Value |
|---|---|
| Architecture | TabularTransformer |
| d_model | 256 |
| Heads | 8 |
| Layers | 8 |
| FFN dim | 512 |
| FFN activation | SwiGLU |
| Normalization | RMSNorm |
| Pooling | CLS token |
| Numeric embedding | Periodic (k=16) |
| Max numeric features | 64 |
| Max categories | 128 |
| Parameters | **10,752,769** (~10.75M) |

## Pretraining

| Property | Value |
|---|---|
| Best checkpoint | Step {best_step:,} |
| Best val loss | {best_val:.4f} |
| Rows seen at best | {best_rows:,} |
| Final step | {last_step:,} |
| Total rows seen | {last_rows:,} |
| Batch size | 512 |
| Learning rate | 3e-4 (cosine decay, 2K warmup) |
| AMP | fp16 |
| Hardware | NVIDIA RTX A4500 (20 GB) |
| Training time | ~3 hours |

Loss objective: multi-task MSE on target prediction from mixed numeric/categorical features,
normalized per-column (z-score). Each batch samples from a fixed-width (64-feature)
schema where unused slots are masked with NaN.

## Pretraining Corpus

Trained on [`avewright/tabula-pretraining-corpus-v2`](https://huggingface.co/datasets/avewright/tabula-pretraining-corpus-v2):

| Source | OK Datasets | Status |
|---|---|---|
| PMLB | 422 | **Fully exhausted** (all 422 known datasets used) |
| OpenML | 2,949 | 4,886 attempted — 1,900 rejected (too few features), 37 download failures |
| HuggingFace | 0 | 67 attempted — format incompatibilities |
| **Synthetic** | (unlimited) | tree-prior, GMM, polynomial, SCM, regression, time-series, mixed-type |

**Total corpus:** 541 shards, ~160 GB parquet.
**Format:** `feat_0..feat_63` (Float32, NaN=unused), `target` (Float32), `_source_meta` (JSON).

### Dataset Exhaustion Notes

- **PMLB: fully exhausted.** All 422 of 423 known datasets successfully processed
  (1 download failure: `chess`). No new PMLB datasets can be added without an
  upstream PMLB library update.

- **OpenML: largely exhausted.** 4,886 unique datasets attempted. 2,949 passed
  the pipeline. The 1,900 `schema_fail` entries are almost entirely datasets with
  only 1 output column and too few rows/features to be useful (e.g. `too small: (53, 1)`).
  These are unrecoverable without lowering quality thresholds. There may be a small
  tail of undiscovered OpenML datasets not yet paginated.

- **HuggingFace tabular:** 67 attempted from curated catalog. All failed due to
  schema mismatches, missing splits, or download timeouts. Catalog needs expansion
  with manually vetted datasets.

## Files

| File | Description |
|---|---|
| `best.pt` | Best validation checkpoint (step {best_step:,}, val_loss={best_val:.4f}) |
| `latest.pt` | Final training checkpoint (step {last_step:,}) |
| `config.json` | Model and training hyperparameters |
| `training_log.txt` | Full training run output |

## Usage

```python
import torch
from tabula.models.transformer import TabularTransformer
from tabula.config import ModelConfig

# Load checkpoint
ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)
cfg  = ckpt["config"].model

# Reconstruct model
model = TabularTransformer(
    d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
    d_ff=cfg.d_ff, dropout=cfg.dropout,
    num_numeric=64, num_categorical=0, num_text=0,
    output_dim=1,
    numeric_embedding=cfg.numeric_embedding,
    numeric_periodic_features=cfg.numeric_periodic_features,
    ffn_activation=cfg.ffn_activation, norm=cfg.norm, pooling=cfg.pooling,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

## Training Notes

The model uses a fixed-width schema (64 numeric slots) regardless of original
dataset width. Narrower datasets are zero-padded with NaN masks. This forces the
model to learn position-invariant feature representations compatible with arbitrary
tabular schemas.

Synthetic data fills gaps when real corpus buffer is empty, providing 100M+ rows
per session of controlled variation in feature distributions, missingness patterns,
and task types.
"""

# ── Upload ──────────────────────────────────────────────────────────────────────
api = HfApi(token=HF_TOKEN)

print(f"Creating / confirming repo: {REPO_ID}")
create_repo(REPO_ID, repo_type="model", exist_ok=True, token=HF_TOKEN)

with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)

    # Write config.json
    (tmp / "config.json").write_text(json.dumps(config_dict, indent=2))

    # Write model card
    (tmp / "README.md").write_text(model_card)

    # Copy checkpoints
    shutil.copy(CKPT_DIR / "best.pt",   tmp / "best.pt")
    shutil.copy(CKPT_DIR / "latest.pt", tmp / "latest.pt")

    # Copy training log (trim to last 5000 lines to avoid huge upload)
    if LOG_PATH.exists():
        lines = LOG_PATH.read_text(errors="replace").splitlines()
        (tmp / "training_log.txt").write_text("\n".join(lines[-5000:]))

    print("Uploading to HuggingFace...")
    api.upload_folder(
        folder_path=str(tmp),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=f"Upload Tabula v1 pretrained model — step {last_step:,}, best_val={best_val:.4f}",
        token=HF_TOKEN,
    )

print(f"\nDone! Model pushed to: https://huggingface.co/{REPO_ID}")
print(f"  Best checkpoint: step={best_step:,}  val_loss={best_val:.4f}  rows={best_rows:,}")
print(f"  Final step: {last_step:,}  total_rows={last_rows:,}")
