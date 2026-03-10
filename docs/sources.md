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
