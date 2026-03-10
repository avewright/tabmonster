# Kaggle Data Plan

The goal is not to scrape random Kaggle CSVs. The goal is to build a small, defensible corpus of strong public tabular benchmarks with enough diversity to matter for representation learning.

## Selection criteria

Datasets in `catalogs/kaggle_tabular.json` are chosen for one or more of these reasons:

- strong benchmark reputation in tabular ML
- enough rows to matter for pretraining or robust finetuning
- mixed numerical and categorical structure
- realistic missingness, sparsity, leakage risk, or temporal structure
- public accessibility through Kaggle's official CLI

## Initial priority order

1. `amex_default_prediction`
2. `home_credit_default_risk`
3. `ieee_fraud_detection`
4. `porto_seguro_safe_driver`
5. `rossmann_store_sales`
6. `santander_customer_transaction`

This mix gives binary classification, regression, dense features, sparse features, categorical-heavy features, and some temporal structure.

For immediate pipeline validation, the catalog also includes public dataset entries that do not require competition-rule acceptance, including `adult_census_income`, `bank_marketing_public`, and `credit_card_fraud_public`.

## Setup

1. Install the project dependencies, which now include `kagglehub`, and install the official Kaggle CLI if you want CLI search or zipped downloads.
2. Create a Kaggle API token from your Kaggle account.
3. Provide credentials in one of these supported forms:
   - `.env` with `KAGGLE_USERNAME=...` and `KAGGLE_KEY=...`
   - `.env` with `KAGGLE_API_TOKEN={"username":"...","key":"..."}`
   - `.env` with `KAGGLE_API_TOKEN=username:key`
   - `.env` with `KAGGLE_API_TOKEN=path/to/kaggle.json`
   - the standard Kaggle credential location for your OS
4. Accept any competition rules on the Kaggle website before downloading competition data.

The implementation follows the current Kaggle docs:

- `kagglehub.dataset_download("<owner>/<dataset>")` downloads an extracted dataset into Kaggle's local cache and returns the materialized path.
- the Kaggle CLI continues to handle live search (`kaggle datasets list`) and optional zip-style downloads (`kaggle datasets download`).

`tabula` uses `kagglehub` by default for dataset ingestion because it produces a ready-to-read directory, then mirrors those files into `data/raw/<dataset_id>/` so the normal preparation path can build train/validation/test splits and `schema.json`.

## Commands

List recommended gold-tier datasets:

```bash
tabula data auth-check
tabula data list --quality-tier gold --recommended-only
```

Show metadata for a curated dataset:

```bash
tabula data show --dataset home_credit_default_risk
```

Download and unzip a dataset:

```bash
tabula data fetch --dataset home_credit_default_risk --output-root data/raw
```

Download a live Kaggle dataset slug and immediately prepare it into the repo's stable artifact contract:

```bash
tabula data ingest-kaggle \
  --slug janiobachmann/bank-marketing-dataset \
  --dataset-id kaggle_live_bank_marketing \
  --task-type binary \
  --target-column deposit
```

If a competition download returns `401`, accept the competition rules on Kaggle first and rerun the command. Public dataset entries do not have that restriction.

Prepare the main training table into normalized train/val/test CSVs and a generated config:

```bash
tabula data prepare --dataset home_credit_default_risk --processed-root data/processed
```

Build a starter corpus from accessible public Kaggle datasets:

```bash
tabula data materialize --source-type dataset --recommended-only --processed-root data/processed
```

Inspect downloaded CSVs:

```bash
tabula data inspect --path data/raw/home_credit_default_risk
```

If you need the old CLI-only behavior, `tabula data fetch-kaggle --backend cli ...` still works, and `--no-unzip` requires that backend because KaggleHub only returns extracted files.
