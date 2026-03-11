# Live Data Sources

The repo supports two high-yield public discovery surfaces for tabular data:

- Hugging Face dataset search filtered to tabular task categories
- Kaggle dataset search filtered by tags and usability rating

## Hugging Face

Search live:

```bash
tabula data hf-auth-check
tabula data search-hf --task-category tabular-classification --limit 20
tabula data search-hf --task-category tabular-regression --query house --limit 20
```

Fetch a dataset split into `data/raw/<dataset_id>/train.csv` and record a local manifest:

```bash
tabula data fetch-hf \
  --repo-id scikit-learn/adult-census-income \
  --dataset-id hf_adult_census_income \
  --task-type binary \
  --target-column income
```

Then prepare it:

```bash
tabula data prepare --dataset hf_adult_census_income
```

The Hugging Face commands resolve `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` from `.env` first, then fall back to the process environment. This matches the documented token arguments on `huggingface_hub.HfApi(...)` and `datasets.load_dataset(...)`.

## Kaggle

Search live:

```bash
tabula data search-kaggle --tag 14101 --min-usability-rating 0.9
tabula data search-kaggle --query fraud --tag 14101 --sort-by votes
```

Fetch a live Kaggle dataset slug into `data/raw/<dataset_id>` and record a local manifest:

```bash
tabula data fetch-kaggle \
  --slug janiobachmann/bank-marketing-dataset \
  --dataset-id kaggle_live_bank_marketing \
  --task-type binary \
  --target-column deposit
```

Then prepare it:

```bash
tabula data prepare --dataset kaggle_live_bank_marketing
```

Or run the one-step path that downloads through KaggleHub and prepares immediately:

```bash
tabula data ingest-kaggle \
  --slug janiobachmann/bank-marketing-dataset \
  --dataset-id kaggle_live_bank_marketing \
  --task-type binary \
  --target-column deposit
```

## Why manifests matter

Fetched live datasets are no longer tied to the curated catalog. Each fetch writes `dataset_manifest.json` beside the raw files so the normal `prepare` pipeline can still emit:

- normalized `train.csv`, `val.csv`, `test.csv`
- a dataset card
- a generated training config

---

## OpenML (20,000+ datasets, no auth)

OpenML provides a massive free benchmark library via REST API. No API key required.

```bash
tabula data search-openml --query diabetes --limit 20
tabula data fetch-openml --dataset-id 61 --output-root data/raw
tabula data list-openml-cc18
```

The CC-18 benchmark suite (72 curated classification tasks) and CTR-23 regression suite are directly accessible:

```python
from tabula.data.openml import fetch_cc18_task_list, fetch_openml_dataset
for task in fetch_cc18_task_list()[:5]:
    fetch_openml_dataset(task.dataset_id, output_root="data/raw")
```

---

## PMLB (~400 curated benchmarks, no auth)

Penn Machine Learning Benchmarks: ≈400 clean tabular datasets served from GitHub, no API key needed.

```bash
tabula data search-pmlb --task classification --max-instances 5000
tabula data fetch-pmlb --name iris --output-root data/raw
```

Bulk download:

```python
from tabula.data.pmlb import fetch_pmlb_benchmark_suite
fetch_pmlb_benchmark_suite(task="classification", max_datasets=50, output_root="data/raw")
```

---

## Synthetic Data Generators

Seven generators produce endless training diversity without storing data:

| Generator | Description |
|---|---|
| `gaussian_mixture` | GMM features, linear/tree decision boundary |
| `tree_prior` | tabPFN-style random tree prior (correlated features) |
| `polynomial` | Polynomial decision boundaries |
| `scm` | Structural Causal Model (DAG) |
| `regression_synthetic` | Linear/additive/interaction regression targets |
| `mixed_type` | Wraps any generator, adds categorical/ordinal columns |
| `timeseries` | ARMA series with statistical feature extraction |

```bash
tabula data generate-synthetic --n-datasets 10 --seed 42
tabula data generate-synthetic --generator tree_prior --n-datasets 5 --n-samples 2000
```

---

## Time-Series Feature Extraction

Enrich time-indexed datasets with calendar, lag, rolling-window, panel, and FFT features:

```bash
tabula data extract-ts-features \
    --input data/raw/bike_demand/train.csv \
    --output data/raw/bike_demand_ts/train.csv \
    --datetime-col datetime --target-col count \
    --lags 1 24 168 --rolling-windows 24 168
```

```python
from tabula.data.timeseries import auto_extract_timeseries_features
df_rich = auto_extract_timeseries_features(df, target_col="sales")
```

---

## Auto-Discovery Pipeline

Scan all sources for new tabular datasets, skipping already-processed entries:

```bash
tabula data autodiscover \
    --sources hf openml pmlb \
    --output-root data/raw \
    --registry-file artifacts/discovery_registry.json \
    --max-new 50
```

---

## RAM-Budgeted Stream Queue Builder

Build a training queue from all available prepared datasets subject to a RAM budget:

```bash
tabula data build-queue --ram-budget-gb 8 --output queues/auto_8gb.json
```

The builder round-robins task types and applies inverse-size weighting so small datasets cycle more frequently.

```python
from tabula.data.stream_builder import StreamQueueBuilder
builder = (
    StreamQueueBuilder(ram_budget_gb=8)
    .add_from_prepared_dir("data/processed")
    .add_from_catalog("catalogs/kaggle_tabular.json")
    .add_synthetic(n_datasets=20)
)
builder.save("queues/auto_8gb.json")
```
