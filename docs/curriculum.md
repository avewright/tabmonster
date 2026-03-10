# Curriculum Training

Curriculum training lets Tabula train on many HuggingFace datasets **sequentially** without holding any data in memory beyond a single streaming batch.  A persistent **queue** file tracks which datasets still need work; an append-only **ledger** file records every training session so the worker can crash-recover and resume exactly where it left off.

---

## Core concepts

| Concept | What it is |
|---|---|
| **Queue** (`artifacts/curriculum_queue.json`) | Ordered JSON list of dataset entries.  Each entry carries its own step budgets, HF repo reference, and prepared-data location.  The worker pops the highest-priority `pending` entry on every cycle. |
| **Ledger** (`artifacts/curriculum_ledger.jsonl`) | Append-only JSONL log.  Every completed or interrupted session writes one line: dataset id, steps trained, best val loss, exit reason, checkpoint path, trunk source. |
| **Trunk transfer** | When moving to a new dataset, all transformer backbone weights whose *shape matches* the new model are copied from the previous best checkpoint.  Dataset-specific heads (numeric projections, categorical embeddings, output layer) are re-initialised from scratch. |
| **VRAM safety** | Each dataset is streamed row-by-row from HuggingFace; no full snapshot is ever materialised on GPU.  The `hf_cache_dir` option lets you pin the HF dataset cache to a local SSD to avoid repeated re-downloads. |

---

## Quick start

### 1 — Prepare datasets

Each dataset in the queue needs a **prepared directory** containing:
- `schema.json` — column types and target info
- `val.csv` — fixed validation split
- `train_config.json` — base model/training hyperparameters

Use the normal `tabula data prepare` or `tabula data ingest-kaggle` pipeline to produce these.

### 2 — Build the queue

```bash
# Add the first dataset
tabula curriculum queue add \
  --dataset-id hf_adult \
  --prepared-dir data/processed/hf_adult_census_income \
  --repo-id scikit-learn/adult-census-income \
  --steps-per-cycle 2000 \
  --max-total-steps 20000 \
  --priority 10

# Add more datasets (lower priority number = trained first)
tabula curriculum queue add \
  --dataset-id hf_otto \
  --prepared-dir data/processed/otto_group_product_classification \
  --repo-id inversion/otto-group-product-classification \
  --steps-per-cycle 2000 \
  --max-total-steps 30000 \
  --priority 20
```

### 3 — Inspect the queue

```bash
tabula curriculum queue list
tabula curriculum queue status
```

### 4 — Start the worker

```bash
# GPU run, cycles indefinitely until all datasets are done
tabula curriculum-worker \
  --device cuda \
  --shuffle-buffer-size 10000 \
  --cache-dir /tmp/hf_cache \
  --val-interval-steps 500 \
  --checkpoint-interval-steps 500 \
  --sleep-seconds 5

# CPU smoke test: 2 cycles max
tabula curriculum-worker \
  --device cpu \
  --max-cycles 2 \
  --sleep-seconds 0
```

The worker runs forever (or until `--max-cycles`) in a loop:
1. Load the queue, pick the highest-priority `pending` entry.
2. Warm-start the model from the best checkpoint in the ledger (trunk transfer).
3. Run `steps_per_cycle` optimizer steps against the streaming HF dataset, saving checkpoints periodically.
4. Update the queue (steps/rows/best loss) and append a session record to the ledger.
5. If `total_steps >= max_total_steps`, mark the entry `done`.
6. Sleep `sleep_seconds`, then repeat.

### 5 — Monitor progress

```bash
# Live queue status
tabula curriculum queue status

# Last 20 sessions from the ledger
tabula curriculum ledger

# Sessions for one dataset only
tabula curriculum ledger --dataset-id hf_adult

# Artifacts sit in the normal per-experiment directories
ls artifacts/curriculum_hf_adult/
```

---

## Crash recovery

If the worker is killed mid-session:

- The entry is left in `in_progress` status in the queue.  
- The latest checkpoint is always written every `checkpoint_interval_steps` steps, so at most that many steps are lost.
- On the next launch the worker sees `in_progress` entries as stuck; manually reset them:

```bash
tabula curriculum queue reset --dataset-id hf_adult
```

Then restart the worker normally — `training.resume = True` is set automatically, so the streaming state is restored.

---

## Re-queueing a finished dataset

Mark a `done` entry back to `pending` to train it for more steps (e.g. after increasing `max_total_steps`):

```bash
tabula curriculum queue reset --dataset-id hf_adult
```

---

## Trunk weight transfer

By default the worker copies all matching-shape parameters from the previous best checkpoint into the freshly-initialised model before training each new dataset.  This is done by `tabula.training.trunk.load_trunk_weights` with strict shape-matching; mismatches are silently skipped.

Typically **60–80 % of parameters transfer** (transformer attention/FFN layers), while the dataset-specific components (per-feature numeric projections, categorical embeddings, output head) are re-initialised.  This gives the backbone a warm start and meaningfully reduces the steps needed to converge on each new task.

Disable trunk transfer with `--no-trunk-transfer` if you want clean-slate training on every dataset.

---

## Configuration reference

### `CurriculumEntry` fields (in `curriculum_queue.json`)

| Field | Default | Description |
|---|---|---|
| `dataset_id` | — | Unique local id |
| `prepared_dir` | — | Path to prepared dataset directory |
| `hf_repo_id` | — | HuggingFace repo to stream from |
| `hf_config_name` | `null` | Optional HF config name |
| `hf_split` | `"train"` | HF split name |
| `status` | `"pending"` | `pending` / `in_progress` / `done` / `failed` |
| `priority` | `100` | Lower = scheduled sooner |
| `steps_per_cycle` | `2000` | Steps per worker cycle |
| `max_total_steps` | `20000` | Lifetime step cap |
| `total_steps` | `0` | Accumulated steps (managed by worker) |
| `total_rows_seen` | `0` | Accumulated rows (managed by worker) |
| `best_val_loss` | `null` | Best validation loss across all sessions |
| `experiment_name` | `null` | Defaults to `curriculum_{dataset_id}` |

### `curriculum-worker` CLI flags

| Flag | Default | Description |
|---|---|---|
| `--artifacts-root` | `artifacts` | Queue, ledger, and run artifact root |
| `--device` | `cpu` | Training device (`cpu` or `cuda`) |
| `--batch-size` | dataset default | Override batch size |
| `--shuffle-buffer-size` | `10000` | HF streaming shuffle buffer |
| `--cache-dir` | — | HF dataset cache directory |
| `--val-interval-steps` | `500` | Validation cadence |
| `--checkpoint-interval-steps` | `500` | Checkpoint write cadence |
| `--sleep-seconds` | `30` | Idle sleep between cycles |
| `--max-cycles` | — | Hard cap on total cycles |
| `--no-trunk-transfer` | off | Disable backbone warm-start |
