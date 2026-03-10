# Data Pipeline

Prepared datasets are now a first-class training input format.

## Prepared artifact contract

Each prepared dataset directory contains:

- `train.csv`
- `val.csv`
- `test.csv`
- `schema.json`
- `dataset_card.json`
- `train_config.json`

## Why `schema.json` exists

Training should not refit encoders or normalization statistics every run. `schema.json` stores the train-split preprocessing contract:

- numeric feature names
- numeric fill values
- numeric mean and standard deviation
- categorical vocabularies
- text feature names
- text token count
- column profile vectors and heuristic type probabilities
- target classes for classification tasks

That makes the input pipeline deterministic across runs and across train/validation/test.

## Loader behavior

When `data.dataset_type` is `prepared`, the loader:

1. Reads `train.csv` and `val.csv`
2. Applies numeric imputation and optional standardization from `schema.json`
3. Applies categorical vocabularies with `0` reserved for unknown values
4. Tokenizes text-like columns into fixed-length hashed token ids for the model text encoder
5. Applies target encoding from `schema.json`

## Text handling

Text-like columns are no longer folded into the categorical path.

- Low-cardinality enum-like strings remain categorical.
- Text-like columns are detected during schema building and written under `text_features`.
- The loader tokenizes each text cell into a fixed number of hashed tokens.
- The batch also carries the raw text strings for optional pretrained encoding.
- The model embeds text cells with a dedicated text encoder before mixing them with numeric and categorical feature tokens.

Available text backends:

- `model.text_encoder = "custom"`: lightweight in-repo transformer over hashed per-cell tokens.
- `model.text_encoder = "pretrained"`: Hugging Face encoder over raw text cell strings. This requires `transformers`.

The default remains the custom encoder because it is self-contained and cheap to train. The pretrained path is there when stronger text priors matter enough to justify the extra runtime and dependency weight.

## Commands

Prepare a raw dataset into the training-ready format:

```bash
tabula data prepare --dataset adult_census_income
```

If the raw manifest does not already declare `target_column` or `task_type`, the prepare step now:

- filters out non-scalar columns that cannot be represented as flat tabular features
- resolves the target from source metadata first, then from column-name and value heuristics
- infers `binary`, `multiclass`, or `regression`
- writes the resolved target and task back into `data/raw/<dataset_id>/dataset_manifest.json`

Train directly from the generated prepared config:

```bash
tabula train --config data/processed/adult_census_income/train_config.json
```
