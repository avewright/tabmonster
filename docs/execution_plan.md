# Execution Plan

This is the concrete plan from the current metadata-aware flat transformer to a TabPFN-class training pipeline.

## Current state

The repo already has:

- prepared datasets with stable `schema.json`
- numeric, categorical, and text feature handling
- column-name and column-profile metadata
- custom and pretrained text cell encoders
- a working supervised trainer

The main bottleneck is not ingestion anymore. It is the mismatch between the model objective and the target behavior:

- current training is row-supervised
- target training should be context/query episodic
- current model is flat over feature tokens
- target model should be hierarchical over cells and rows

## Iteration order

### Iteration 1: Episodic scaffolding

Goal:

- define a stable context/query batch contract without replacing the current trainer

Deliverables:

- `EpisodeConfig` in experiment configs
- `EpisodeBatch` type
- a sampler that converts a `TabularBatch` into support/query subsets

Success criteria:

- we can build and inspect context/query batches from prepared datasets
- the next model can target this batch interface directly

### Iteration 2: Hierarchical model skeleton

Goal:

- add a cell/row model path without deleting the current baseline

Deliverables:

- `CellEncoder`
- `RowEncoder`
- `EpisodeEncoder`
- a second model entrypoint alongside `TabularTransformer`

Success criteria:

- forward pass works on an `EpisodeBatch`
- query predictions are produced from support plus query rows

### Iteration 3: Episodic trainer

Goal:

- train with support/query objectives over one prepared dataset at a time

Deliverables:

- episode-aware dataloader loop
- target masking for query rows
- classification and regression loss paths

Success criteria:

- one-dataset episodic smoke training completes

### Iteration 4: Multi-dataset episodic mixing

Goal:

- sample episodes across many prepared datasets

Deliverables:

- corpus manifest of prepared datasets
- dataset-balanced or temperature-weighted sampling
- per-dataset metric logging

Success criteria:

- one training run consumes multiple prepared datasets cleanly

### Iteration 5: Benchmarking and scaling

Goal:

- make the loop comparable to serious tabular baselines

Deliverables:

- benchmark harness
- ranking and regret metrics
- larger corpus materialization and caching

Success criteria:

- stable benchmark reports over a fixed dataset suite

## Data and batch contract

### Current supervised batch

```text
TabularBatch
  x_num: [rows, num_numeric]
  x_cat: [rows, num_categorical]
  x_text_token_ids: [rows, num_text, text_token_count]
  x_text_values: list[list[str]]
  x_num_mask: [rows, num_numeric]
  x_cat_mask: [rows, num_categorical]
  x_text_mask: [rows, num_text]
  num_name_token_ids: [num_numeric, name_token_count]
  cat_name_token_ids: [num_categorical, name_token_count]
  text_name_token_ids: [num_text, name_token_count]
  num_profile_vectors: [num_numeric, profile_dim]
  cat_profile_vectors: [num_categorical, profile_dim]
  text_profile_vectors: [num_text, profile_dim]
  y: [rows]
```

### Iteration-1 episodic batch

```text
EpisodeBatch
  support: TabularBatch
  query: TabularBatch
```

This is intentionally simple. The next model can add role embeddings and target masking internally without forcing another loader rewrite.

## Recommended immediate focus

The next engineering move should be:

1. keep the current supervised path intact
2. build the hierarchical episode model in parallel
3. switch training only after the new path has smoke tests

That is the lowest-risk route. The repo now has enough data and metadata plumbing to support it.
