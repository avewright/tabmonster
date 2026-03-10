# End-to-End Pipeline Walkthrough

This document explains how `tabula` moves from an external tabular dataset to tensors consumed by the model, then to a trained checkpoint.

It is intentionally concrete. The example below uses the small Hugging Face dataset `scikit-learn/iris` so the whole flow is easy to inspect.

## The Short Version

The pipeline is:

1. Fetch a dataset into `data/raw/<dataset_id>/`
2. Inspect it and resolve the target column and task type
3. Prepare it into `data/processed/<dataset_id>/`
4. Build a schema that freezes preprocessing decisions
5. Load prepared files into tensor batches
6. Train the model and save the best checkpoint

## A Simple Example

Fetch a small dataset:

```bash
tabula data fetch-hf --repo-id scikit-learn/iris --dataset-id iris_demo
```

Prepare it:

```bash
tabula data prepare --dataset iris_demo
```

Train from the generated config:

```bash
tabula train --config data/processed/iris_demo/train_config.json
```

At that point the main directories look like this:

```text
data/
  raw/
    iris_demo/
      train.csv
      dataset_manifest.json
  processed/
    iris_demo/
      train.csv
      val.csv
      test.csv
      schema.json
      dataset_card.json
      feature_transforms.json
      train_config.json
artifacts/
  finetune_iris_demo/
    best.pt
```

## Stage 1: Fetch Raw Data

For Hugging Face datasets, the entry point is [`fetch_huggingface_dataset()`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/tabula/src/tabula/data/huggingface.py).

What it does:

- calls `datasets.load_dataset(...)`
- converts the selected split to a pandas frame
- drops nested or unsupported non-scalar columns through the inspection path
- resolves the target column and task type
- writes the cleaned frame to `data/raw/<dataset_id>/train.csv`
- writes `dataset_manifest.json`

For `scikit-learn/iris`, the raw frame looks roughly like:

```text
Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
1, 5.1, 3.5, 1.4, 0.2, Iris-setosa
...
```

The raw manifest stores source metadata and the resolved supervision contract:

- dataset id
- provider
- source URL
- task type
- target column
- selected train file

## Stage 2: Inspect The Table

The inspection logic lives in [`inspect_supervised_frame()`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/tabula/src/tabula/data/inspection.py).

Its job is to answer:

- Which columns are valid flat tabular columns?
- Which column is the label?
- Is the task `binary`, `multiclass`, or `regression`?

The inspection step:

- keeps scalar columns like numbers, booleans, datetimes, and strings
- drops nested lists, dicts, arrays, and other unsupported objects
- prefers an explicit target from the manifest or dataset metadata
- otherwise uses name and value heuristics to infer the target

For the iris example, it resolves:

- `target_column = Species`
- `task_type = multiclass`

## Stage 3: Prepare The Dataset

The main preparation entry point is [`prepare_dataset()`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/tabula/src/tabula/data/prep.py).

This is the most important data step. It takes one raw training table and turns it into a stable training contract.

### 3.1 Load and normalize

The prepare step:

- reads the source file from `data/raw/<dataset_id>/`
- re-runs inspection if needed
- drops rows with missing targets
- normalizes column names to lowercase snake-like names

For iris, columns become roughly:

```text
id
sepal_length_cm
sepal_width_cm
petal_length_cm
petal_width_cm
species
```

### 3.2 Split train, validation, and test

The frame is split into:

- `train.csv`
- `val.csv`
- `test.csv`

Classification tasks use stratified splits when possible.

### 3.3 Drop obvious identifier columns

Columns like `id` or `customer_id` can be removed before training because they often leak row identity without helping generalization.

This behavior is controlled by `drop_identifier_columns`.

### 3.4 Apply train-only feature engineering

The prep path now fits feature transforms on the training split only, then applies those fitted transforms to validation and test.

Current transform types:

- missingness indicators such as `amount_is_missing`
- `log1p` transforms for skewed non-negative numeric columns
- frequency encoding for high-cardinality categorical columns
- rare-category collapsing

Those fitted transforms are written to:

```text
data/processed/<dataset_id>/feature_transforms.json
```

This matters because it prevents validation and test leakage.

### 3.5 Write the prepared artifacts

The prepare step writes:

- `train.csv`
- `val.csv`
- `test.csv`
- `schema.json`
- `dataset_card.json`
- `feature_transforms.json`
- `train_config.json`

## Stage 4: Freeze Preprocessing In `schema.json`

The schema builder lives in [`build_schema()`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/tabula/src/tabula/data/schema.py).

`schema.json` is the contract between preparation and training. Training should not rediscover encoders every run.

It stores:

- target metadata
- numeric feature names
- numeric fill values
- numeric mean and standard deviation
- categorical vocabularies
- text feature declarations
- hashed column-name token inputs
- profile vectors and soft type probabilities

In plain terms:

- numeric columns are standardized using train-split statistics
- categorical columns are mapped to integer ids
- text-like columns are kept on a separate path, not forced into categorical ids
- the target label encoding is frozen

## Stage 5: Build Dataloaders

The tensorization path lives in [`build_dataloaders()`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/tabula/src/tabula/data/datasets.py).

When `data.dataset_type == "prepared"`, the loader:

1. reads prepared `train.csv` and `val.csv`
2. loads `schema.json`
3. encodes numeric features
4. encodes categorical features
5. tokenizes text features
6. encodes the target
7. builds PyTorch `DataLoader` objects

### What a batch contains

The loader returns a `TabularBatch` with both row values and schema metadata.

Important fields:

- `x_num`: numeric values
- `x_cat`: categorical ids
- `x_text_token_ids`: tokenized text cells
- `x_num_mask`, `x_cat_mask`, `x_text_mask`: present or missing masks
- `num_name_token_ids`, `cat_name_token_ids`, `text_name_token_ids`: hashed column-name tokens
- `num_profile_vectors`, `cat_profile_vectors`, `text_profile_vectors`: per-column metadata
- `y`: target tensor

This is how the model sees both:

- the row values
- metadata about what each column means

## Stage 6: Train

The training loop is [`train()`](/c:/Users/AWright/OneDrive%20-%20Kahua,%20Inc/Projects/tabula/src/tabula/training/engine.py).

It does the following:

1. sets the random seed
2. builds train and validation dataloaders
3. constructs the `TabularTransformer`
4. chooses the loss function based on task type
5. runs training and validation epochs
6. tracks the best validation loss
7. writes the best checkpoint to `artifacts/<experiment_name>/best.pt`

The generated `train_config.json` from preparation is designed to be directly consumable by this step.

## What Happens For The Iris Example

For `scikit-learn/iris`, the end-to-end path is:

1. `tabula data fetch-hf --repo-id scikit-learn/iris --dataset-id iris_demo`
2. raw `train.csv` and `dataset_manifest.json` are written under `data/raw/iris_demo/`
3. `tabula data prepare --dataset iris_demo`
4. `species` is treated as the label and the task resolves to multiclass classification
5. prepared files and schema are written under `data/processed/iris_demo/`
6. `tabula train --config data/processed/iris_demo/train_config.json`
7. the loader encodes the prepared CSVs into tensors
8. the model trains and saves `artifacts/finetune_iris_demo/best.pt`

## How The Files Relate To Each Other

- `data/raw/<dataset_id>/dataset_manifest.json`
  Explains where the dataset came from and what the target should be.

- `data/processed/<dataset_id>/train.csv`, `val.csv`, `test.csv`
  The actual model-ready row splits.

- `data/processed/<dataset_id>/schema.json`
  Freezes preprocessing and encoding decisions.

- `data/processed/<dataset_id>/feature_transforms.json`
  Freezes fitted feature engineering transforms.

- `data/processed/<dataset_id>/dataset_card.json`
  Human-readable summary of what preparation decided.

- `data/processed/<dataset_id>/train_config.json`
  A runnable experiment config for `tabula train`.

- `artifacts/<experiment_name>/best.pt`
  The best model checkpoint from training.

## Why This Design Exists

This design is trying to separate concerns cleanly:

- fetchers deal with external providers
- preparation resolves ambiguous raw tables into a stable contract
- schema building freezes preprocessing
- dataloaders convert prepared files into tensors
- training only consumes the prepared contract

That separation is important because it keeps training deterministic and makes debugging easier. If model behavior changes, you can inspect the exact prepared files and schema that produced the tensors.

## What This Pipeline Does Not Do Yet

The current path is intentionally simple. It does not yet:

- join relational side tables
- build multi-table aggregates automatically
- train with `test.csv` in the loop
- use `EpisodeBatch` as the default training mode
- do large-scale cross-dataset pretraining

Those are future extensions, not part of the baseline path described here.
