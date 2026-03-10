from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pprint import pprint
from pathlib import Path
import random
import time

from tabula.config import StreamJobConfig, load_config, load_stream_queue_config, stream_queue_config_to_dict


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tabula", description="Train tabular foundation model experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run a training job from a YAML config.")
    train_parser.add_argument("--config", required=True, help="Path to experiment config.")

    train_stream_parser = subparsers.add_parser("train-hf-stream", help="Run resumable training against a streaming Hugging Face dataset using an existing prepared schema.")
    train_stream_parser.add_argument("--prepared-dir", required=True, help="Prepared dataset directory containing schema.json, val.csv, and train_config.json.")
    train_stream_parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id to stream from.")
    train_stream_parser.add_argument("--config-name", help="Optional Hugging Face config name.")
    train_stream_parser.add_argument("--split", default="train", help="Streaming split name.")
    train_stream_parser.add_argument("--experiment-name", help="Override experiment name.")
    train_stream_parser.add_argument("--device", help="Override training device, e.g. cuda or cpu.")
    train_stream_parser.add_argument("--max-steps", type=int, default=2000, help="Maximum streaming optimization steps.")
    train_stream_parser.add_argument("--val-interval-steps", type=int, default=500, help="Validation interval in optimizer steps.")
    train_stream_parser.add_argument("--checkpoint-interval-steps", type=int, default=500, help="Latest-checkpoint write interval in optimizer steps.")
    train_stream_parser.add_argument("--batch-size", type=int, help="Override batch size.")
    train_stream_parser.add_argument("--shuffle-buffer-size", type=int, default=10000, help="Streaming shuffle buffer size.")
    train_stream_parser.add_argument("--cache-dir", help="Optional Hugging Face cache directory.")
    train_stream_parser.add_argument("--max-stream-rows", type=int, help="Optional hard cap on streamed rows for smoke tests or bounded runs.")

    train_stream_worker_parser = subparsers.add_parser("train-hf-stream-worker", help="Run a resumable background-friendly worker loop for streamed Hugging Face training.")
    train_stream_worker_parser.add_argument("--prepared-dir", required=True, help="Prepared dataset directory containing schema.json, val.csv, and train_config.json.")
    train_stream_worker_parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id to stream from.")
    train_stream_worker_parser.add_argument("--config-name", help="Optional Hugging Face config name.")
    train_stream_worker_parser.add_argument("--split", default="train", help="Streaming split name.")
    train_stream_worker_parser.add_argument("--experiment-name", help="Override experiment name.")
    train_stream_worker_parser.add_argument("--device", help="Override training device, e.g. cuda or cpu.")
    train_stream_worker_parser.add_argument("--steps-per-cycle", type=int, default=1000, help="Additional steps to train each worker cycle.")
    train_stream_worker_parser.add_argument("--max-total-steps", type=int, default=10000, help="Stop after this total step count.")
    train_stream_worker_parser.add_argument("--val-interval-steps", type=int, default=500, help="Validation interval in optimizer steps.")
    train_stream_worker_parser.add_argument("--checkpoint-interval-steps", type=int, default=500, help="Latest-checkpoint write interval in optimizer steps.")
    train_stream_worker_parser.add_argument("--batch-size", type=int, help="Override batch size.")
    train_stream_worker_parser.add_argument("--shuffle-buffer-size", type=int, default=10000, help="Streaming shuffle buffer size.")
    train_stream_worker_parser.add_argument("--cache-dir", help="Optional Hugging Face cache directory.")
    train_stream_worker_parser.add_argument("--max-stream-rows", type=int, help="Optional hard cap on streamed rows for smoke tests or bounded runs.")
    train_stream_worker_parser.add_argument("--sleep-seconds", type=int, default=15, help="Idle sleep between worker cycles.")
    train_stream_worker_parser.add_argument("--max-cycles", type=int, help="Optional hard cap on worker cycles.")
    train_stream_queue_parser = subparsers.add_parser("train-hf-stream-queue", help="Run a queue of streamed Hugging Face jobs with persistent registry updates.")
    train_stream_queue_parser.add_argument("--queue-config", required=True, help="Path to a JSON stream queue config.")

    

    data_parser = subparsers.add_parser("data", help="Curate and fetch external datasets.")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)

    list_parser = data_subparsers.add_parser("list", help="List curated Kaggle tabular datasets.")
    list_parser.add_argument("--quality-tier", choices=["gold", "silver"], help="Filter by catalog tier.")
    list_parser.add_argument("--task-type", choices=["binary", "multiclass", "regression"], help="Filter by task type.")
    list_parser.add_argument("--recommended-only", action="store_true", help="Show only recommended datasets.")

    show_parser = data_subparsers.add_parser("show", help="Show metadata for one curated dataset.")
    show_parser.add_argument("--dataset", required=True, help="Dataset id from the catalog.")

    fetch_parser = data_subparsers.add_parser("fetch", help="Download a curated dataset via the Kaggle CLI.")
    fetch_parser.add_argument("--dataset", required=True, help="Dataset id from the catalog.")
    fetch_parser.add_argument("--output-root", default="data/raw", help="Root directory for downloaded files.")
    fetch_parser.add_argument("--no-unzip", action="store_true", help="Keep the downloaded zip archive compressed.")
    fetch_parser.add_argument("--force", action="store_true", help="Pass --force to the Kaggle CLI.")

    search_kaggle_parser = data_subparsers.add_parser("search-kaggle", help="Search Kaggle datasets live.")
    search_kaggle_parser.add_argument("--query", help="Optional search string.")
    search_kaggle_parser.add_argument("--tag", action="append", help="Kaggle tag id or slug. Repeatable.")
    search_kaggle_parser.add_argument("--sort-by", choices=["hottest", "votes", "updated", "active"], default="votes")
    search_kaggle_parser.add_argument("--page", type=int, default=1, help="Kaggle results page.")
    search_kaggle_parser.add_argument("--min-usability-rating", type=float, default=0.9, help="Minimum Kaggle usability rating.")

    fetch_kaggle_parser = data_subparsers.add_parser("fetch-kaggle", help="Download a Kaggle dataset by slug.")
    fetch_kaggle_parser.add_argument("--slug", required=True, help="Kaggle dataset slug, e.g. uciml/adult-census-income.")
    fetch_kaggle_parser.add_argument("--dataset-id", help="Local dataset id under data/raw.")
    fetch_kaggle_parser.add_argument("--output-root", default="data/raw", help="Root directory for downloaded files.")
    fetch_kaggle_parser.add_argument("--no-unzip", action="store_true", help="Keep the downloaded zip archive compressed.")
    fetch_kaggle_parser.add_argument("--force", action="store_true", help="Pass --force to the Kaggle CLI.")
    fetch_kaggle_parser.add_argument("--title", help="Optional local title override.")
    fetch_kaggle_parser.add_argument("--task-type", choices=["binary", "multiclass", "regression"], help="Optional task type to store in the manifest.")
    fetch_kaggle_parser.add_argument("--target-column", help="Optional target column to store in the manifest.")
    fetch_kaggle_parser.add_argument("--notes", default="", help="Optional notes to store in the manifest.")
    fetch_kaggle_parser.add_argument("--backend", choices=["auto", "hub", "cli"], default="auto", help="Download backend. `auto` prefers KaggleHub for extracted dataset downloads.")

    ingest_kaggle_parser = data_subparsers.add_parser("ingest-kaggle", help="Download a Kaggle dataset and prepare it into train/val/test artifacts.")
    ingest_kaggle_parser.add_argument("--slug", required=True, help="Kaggle dataset slug, e.g. uciml/adult-census-income.")
    ingest_kaggle_parser.add_argument("--dataset-id", help="Local dataset id under data/raw and data/processed.")
    ingest_kaggle_parser.add_argument("--output-root", default="data/raw", help="Root directory for downloaded files.")
    ingest_kaggle_parser.add_argument("--processed-root", default="data/processed", help="Output directory for prepared artifacts.")
    ingest_kaggle_parser.add_argument("--backend", choices=["auto", "hub", "cli"], default="auto", help="Download backend. `auto` prefers KaggleHub for extracted dataset downloads.")
    ingest_kaggle_parser.add_argument("--no-unzip", action="store_true", help="Keep the downloaded zip archive compressed. This requires `--backend cli`.")
    ingest_kaggle_parser.add_argument("--force", action="store_true", help="Force a redownload from Kaggle.")
    ingest_kaggle_parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    ingest_kaggle_parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction.")
    ingest_kaggle_parser.add_argument("--test-fraction", type=float, default=0.1, help="Test fraction.")
    ingest_kaggle_parser.add_argument("--max-rows", type=int, help="Optional row cap for large datasets.")
    ingest_kaggle_parser.add_argument("--keep-identifiers", action="store_true", help="Do not drop obvious identifier columns.")
    ingest_kaggle_parser.add_argument("--title", help="Optional local title override.")
    ingest_kaggle_parser.add_argument("--task-type", choices=["binary", "multiclass", "regression"], help="Task type for preparation.")
    ingest_kaggle_parser.add_argument("--target-column", help="Target column for preparation.")
    ingest_kaggle_parser.add_argument("--train-file", help="Override the source file to prepare when multiple are present.")
    ingest_kaggle_parser.add_argument("--notes", default="", help="Optional notes to store in the manifest and dataset card.")
    ingest_kaggle_parser.add_argument("--no-feature-engineering", action="store_true", help="Disable train-only feature engineering during preparation.")

    data_subparsers.add_parser("auth-check", help="Validate local Kaggle CLI and credential discovery.")
    data_subparsers.add_parser("hf-auth-check", help="Validate Hugging Face token discovery from .env or environment.")

    search_hf_parser = data_subparsers.add_parser("search-hf", help="Search Hugging Face datasets live.")
    search_hf_parser.add_argument("--query", help="Optional search string.")
    search_hf_parser.add_argument(
        "--task-category",
        default="tabular-classification",
        help="Hugging Face task category, e.g. tabular-classification or tabular-regression.",
    )
    search_hf_parser.add_argument("--limit", type=int, default=20, help="Maximum number of results.")
    search_hf_parser.add_argument("--sort", default="downloads", help="Sort key, default downloads.")

    fetch_hf_parser = data_subparsers.add_parser("fetch-hf", help="Download a Hugging Face dataset split into the raw data directory.")
    fetch_hf_parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id.")
    fetch_hf_parser.add_argument("--dataset-id", help="Local dataset id under data/raw.")
    fetch_hf_parser.add_argument("--output-root", default="data/raw", help="Root directory for downloaded files.")
    fetch_hf_parser.add_argument("--config-name", help="Optional dataset config name.")
    fetch_hf_parser.add_argument("--split", default="train", help="Dataset split to materialize.")
    fetch_hf_parser.add_argument("--max-rows", type=int, help="Optional row cap applied during fetch.")
    fetch_hf_parser.add_argument("--title", help="Optional local title override.")
    fetch_hf_parser.add_argument("--task-type", choices=["binary", "multiclass", "regression"], help="Optional task type to store in the manifest.")
    fetch_hf_parser.add_argument("--target-column", help="Optional target column to store in the manifest.")
    fetch_hf_parser.add_argument("--notes", default="", help="Optional notes to store in the manifest.")

    prepare_parser = data_subparsers.add_parser("prepare", help="Prepare a downloaded dataset into train/val/test CSVs.")
    prepare_parser.add_argument("--dataset", required=True, help="Local dataset id or curated catalog dataset id.")
    prepare_parser.add_argument("--raw-root", default="data/raw", help="Root directory containing downloaded raw data.")
    prepare_parser.add_argument("--processed-root", default="data/processed", help="Output directory for prepared artifacts.")
    prepare_parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    prepare_parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction.")
    prepare_parser.add_argument("--test-fraction", type=float, default=0.1, help="Test fraction.")
    prepare_parser.add_argument("--max-rows", type=int, help="Optional row cap for large datasets.")
    prepare_parser.add_argument("--keep-identifiers", action="store_true", help="Do not drop obvious identifier columns.")
    prepare_parser.add_argument("--task-type", choices=["binary", "multiclass", "regression"], help="Override task type for uncatalogued datasets.")
    prepare_parser.add_argument("--target-column", help="Override target column for uncatalogued datasets.")
    prepare_parser.add_argument("--train-file", help="Override the source file to prepare when multiple are present.")
    prepare_parser.add_argument("--title", help="Override the dataset title in the generated dataset card.")
    prepare_parser.add_argument("--notes", help="Override notes in the generated dataset card.")
    prepare_parser.add_argument("--no-feature-engineering", action="store_true", help="Disable train-only feature engineering during preparation.")

    materialize_parser = data_subparsers.add_parser(
        "materialize",
        help="Fetch and prepare multiple curated datasets in one pass.",
    )
    materialize_parser.add_argument("--dataset", action="append", help="Dataset id to include. Repeatable.")
    materialize_parser.add_argument("--quality-tier", choices=["gold", "silver"], help="Filter by catalog tier.")
    materialize_parser.add_argument("--task-type", choices=["binary", "multiclass", "regression"], help="Filter by task type.")
    materialize_parser.add_argument("--source-type", choices=["dataset", "competition"], default="dataset", help="Filter by Kaggle source type.")
    materialize_parser.add_argument("--recommended-only", action="store_true", help="Only include recommended datasets.")
    materialize_parser.add_argument("--raw-root", default="data/raw", help="Root directory for downloaded raw data.")
    materialize_parser.add_argument("--processed-root", default="data/processed", help="Root directory for prepared artifacts.")
    materialize_parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    materialize_parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction.")
    materialize_parser.add_argument("--test-fraction", type=float, default=0.1, help="Test fraction.")
    materialize_parser.add_argument("--max-rows", type=int, help="Optional row cap for large datasets.")
    materialize_parser.add_argument("--keep-identifiers", action="store_true", help="Do not drop obvious identifier columns.")
    materialize_parser.add_argument("--force", action="store_true", help="Pass --force to Kaggle download.")
    materialize_parser.add_argument("--skip-fetch", action="store_true", help="Assume raw files already exist.")

    inspect_parser = data_subparsers.add_parser("inspect", help="Inspect CSV files under a local dataset directory.")
    inspect_parser.add_argument("--path", required=True, help="Directory containing downloaded dataset files.")

    # ------------------------------------------------------------------
    # curriculum-worker  (top-level background loop)
    # ------------------------------------------------------------------

    cw_parser = subparsers.add_parser(
        "curriculum-worker",
        help=(
            "Background worker: iterate the curriculum queue, training each dataset "
            "one cycle at a time with VRAM-safe streaming.  Logs every session to the "
            "curriculum ledger and resumes automatically after crashes."
        ),
    )
    cw_parser.add_argument("--artifacts-root", default="artifacts", help="Root directory for queue, ledger, and run artifacts.")
    cw_parser.add_argument("--device", default="cpu", help="Training device (cpu or cuda).")
    cw_parser.add_argument("--batch-size", type=int, help="Override batch size for every dataset.")
    cw_parser.add_argument("--shuffle-buffer-size", type=int, default=10000, help="HuggingFace streaming shuffle buffer.")
    cw_parser.add_argument("--cache-dir", help="Optional HuggingFace dataset cache directory.")
    cw_parser.add_argument("--val-interval-steps", type=int, default=500, help="Validation interval in optimizer steps.")
    cw_parser.add_argument("--checkpoint-interval-steps", type=int, default=500, help="Checkpoint write interval in steps.")
    cw_parser.add_argument("--sleep-seconds", type=int, default=30, help="Idle sleep between cycles when no pending work.")
    cw_parser.add_argument("--max-cycles", type=int, help="Hard cap on total worker cycles.")
    cw_parser.add_argument("--no-trunk-transfer", action="store_true", help="Disable trunk weight transfer between datasets.")

    # ------------------------------------------------------------------
    # curriculum  (queue/ledger management sub-commands)
    # ------------------------------------------------------------------

    curriculum_parser = subparsers.add_parser("curriculum", help="Manage the curriculum training queue and ledger.")
    curriculum_subparsers = curriculum_parser.add_subparsers(dest="curriculum_command", required=True)

    # --- queue ---
    queue_parser = curriculum_subparsers.add_parser("queue", help="Manage the dataset queue.")
    queue_subparsers = queue_parser.add_subparsers(dest="queue_command", required=True)

    queue_add = queue_subparsers.add_parser("add", help="Add a dataset entry to the curriculum queue.")
    queue_add.add_argument("--dataset-id", required=True, help="Unique local id for this dataset, e.g. hf_adult.")
    queue_add.add_argument("--prepared-dir", required=True, help="Prepared dataset directory (schema.json + val.csv + train_config.json).")
    queue_add.add_argument("--repo-id", required=True, help="Hugging Face repo id to stream from.")
    queue_add.add_argument("--config-name", help="Optional HuggingFace dataset config name.")
    queue_add.add_argument("--split", default="train", help="HuggingFace split to stream.")
    queue_add.add_argument("--steps-per-cycle", type=int, default=2000, help="Optimizer steps per worker cycle.")
    queue_add.add_argument("--max-total-steps", type=int, default=20000, help="Lifetime step cap for this dataset.")
    queue_add.add_argument("--priority", type=int, default=100, help="Scheduling priority (lower = earlier).")
    queue_add.add_argument("--experiment-name", help="Override artifact directory name.")
    queue_add.add_argument("--notes", default="", help="Free-text notes stored in the queue.")
    queue_add.add_argument("--tags", nargs="*", default=[], help="Optional tags.")
    queue_add.add_argument("--artifacts-root", default="artifacts", help="Queue file location root.")

    queue_list = queue_subparsers.add_parser("list", help="List all entries in the curriculum queue.")
    queue_list.add_argument("--artifacts-root", default="artifacts", help="Queue file location root.")
    queue_list.add_argument("--status", help="Filter by status (pending/in_progress/done/failed).")

    queue_status = queue_subparsers.add_parser("status", help="Print a summary of queue progress.")
    queue_status.add_argument("--artifacts-root", default="artifacts", help="Queue file location root.")

    queue_reset = queue_subparsers.add_parser("reset", help="Reset a failed or done entry back to pending.")
    queue_reset.add_argument("--dataset-id", required=True, help="Dataset id to reset.")
    queue_reset.add_argument("--artifacts-root", default="artifacts", help="Queue file location root.")

    # --- ledger ---
    ledger_parser = curriculum_subparsers.add_parser("ledger", help="Inspect the training session ledger.")
    ledger_parser.add_argument("--artifacts-root", default="artifacts", help="Ledger file location root.")
    ledger_parser.add_argument("--last", type=int, default=20, help="Show the N most recent sessions.")
    ledger_parser.add_argument("--dataset-id", help="Filter sessions to a specific dataset id.")

    auto_hf_parser = subparsers.add_parser(
        "autocurriculum-hf",
        help="Discover Hugging Face tabular datasets, bootstrap local prepared dirs, enqueue them, and start the curriculum worker.",
    )
    auto_hf_parser.add_argument("--artifacts-root", default="artifacts", help="Root directory for queue, ledger, and run artifacts.")
    auto_hf_parser.add_argument("--raw-root", default="data/raw", help="Root directory for bootstrapped raw dataset samples.")
    auto_hf_parser.add_argument("--processed-root", default="data/processed", help="Root directory for prepared dataset artifacts.")
    auto_hf_parser.add_argument("--query", help="Optional Hugging Face dataset search query.")
    auto_hf_parser.add_argument(
        "--task-category",
        action="append",
        dest="task_categories",
        help="Hugging Face task category to search. Repeatable. Defaults to tabular-classification and tabular-regression.",
    )
    auto_hf_parser.add_argument("--limit", type=int, default=100, help="Maximum results to pull per task category.")
    auto_hf_parser.add_argument("--sort", default="downloads", help="Hugging Face search sort key.")
    auto_hf_parser.add_argument("--bootstrap-rows", type=int, default=2048, help="Number of streamed rows to sample locally for schema and validation bootstrap.")
    auto_hf_parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset sampling and splitting.")
    auto_hf_parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction for the local bootstrap sample.")
    auto_hf_parser.add_argument("--test-fraction", type=float, default=0.1, help="Test fraction for the local bootstrap sample.")
    auto_hf_parser.add_argument("--keep-identifiers", action="store_true", help="Do not drop identifier-like columns during bootstrap preparation.")
    auto_hf_parser.add_argument("--no-feature-engineering", action="store_true", help="Disable train-only feature engineering during bootstrap preparation.")
    auto_hf_parser.add_argument("--steps-per-cycle", type=int, default=1000, help="Optimizer steps per dataset per curriculum cycle.")
    auto_hf_parser.add_argument("--max-total-steps", type=int, default=10000, help="Lifetime optimizer step budget per dataset.")
    auto_hf_parser.add_argument("--priority-base", type=int, default=100, help="Base queue priority for the first discovered dataset.")
    auto_hf_parser.add_argument("--device", default="cpu", help="Training device for the worker, e.g. cuda.")
    auto_hf_parser.add_argument("--batch-size", type=int, help="Override batch size for all queued datasets.")
    auto_hf_parser.add_argument("--shuffle-buffer-size", type=int, default=10000, help="Hugging Face streaming shuffle buffer.")
    auto_hf_parser.add_argument("--cache-dir", help="Optional Hugging Face cache directory.")
    auto_hf_parser.add_argument("--val-interval-steps", type=int, default=500, help="Validation interval in optimizer steps.")
    auto_hf_parser.add_argument("--checkpoint-interval-steps", type=int, default=500, help="Checkpoint write interval in steps.")
    auto_hf_parser.add_argument("--sleep-seconds", type=int, default=30, help="Idle sleep between worker cycles.")
    auto_hf_parser.add_argument("--max-cycles", type=int, help="Optional hard cap on worker cycles.")
    auto_hf_parser.add_argument("--no-trunk-transfer", action="store_true", help="Disable trunk warm-start transfer between datasets.")
    auto_hf_parser.add_argument("--max-new-datasets", type=int, help="Optional cap on how many newly discovered datasets to enqueue during this bootstrap pass.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    def build_stream_config_from_job(job: StreamJobConfig) -> tuple[object, Path]:
        prepared_dir = Path(job.prepared_dir)
        base_config_path = prepared_dir / "train_config.json"
        if not base_config_path.exists():
            raise FileNotFoundError(f"Could not find base train config at `{base_config_path}`.")
        config = load_config(base_config_path)
        config.data.dataset_type = "hf_stream"
        config.data.prepared_dir = str(prepared_dir)
        config.data.train_path = None
        config.data.val_path = str(prepared_dir / "val.csv")
        config.data.hf_repo_id = job.repo_id
        config.data.hf_config_name = job.config_name
        config.data.hf_split = job.split
        config.data.hf_streaming = True
        config.data.hf_shuffle_buffer_size = job.shuffle_buffer_size
        config.data.hf_cache_dir = job.cache_dir
        config.data.hf_max_stream_rows = job.max_stream_rows
        if job.batch_size:
            config.data.batch_size = job.batch_size
        config.training.val_interval_steps = job.val_interval_steps
        config.training.checkpoint_interval_steps = job.checkpoint_interval_steps
        if job.device:
            config.training.device = job.device
        if job.experiment_name:
            config.experiment_name = job.experiment_name
        else:
            config.experiment_name = f"{config.experiment_name}_stream"
        generated_config_path = prepared_dir / "train_config.hf_stream.json"
        generated_config_path.write_text(json.dumps(config, default=lambda obj: obj.__dict__, indent=2), encoding="utf-8")
        return config, generated_config_path

    if args.command == "train":
        from tabula.training.engine import train

        config = load_config(args.config)
        pprint(train(config))
    elif args.command == "train-hf-stream":
        from tabula.training.engine import train

        config, generated_config_path = build_stream_config_from_job(
            StreamJobConfig(
                prepared_dir=args.prepared_dir,
                repo_id=args.repo_id,
                config_name=args.config_name,
                split=args.split,
                experiment_name=args.experiment_name,
                device=args.device,
                steps_per_cycle=args.max_steps,
                max_total_steps=args.max_steps,
                val_interval_steps=args.val_interval_steps,
                checkpoint_interval_steps=args.checkpoint_interval_steps,
                batch_size=args.batch_size,
                shuffle_buffer_size=args.shuffle_buffer_size,
                cache_dir=args.cache_dir,
                max_stream_rows=args.max_stream_rows,
            )
        )
        config.training.max_steps = args.max_steps
        pprint({"generated_config": str(generated_config_path)})
        pprint(train(config))
    elif args.command == "train-hf-stream-worker":
        from tabula.training.engine import train
        from tabula.training.registry import update_registry_job

        job = StreamJobConfig(
            prepared_dir=args.prepared_dir,
            repo_id=args.repo_id,
            config_name=args.config_name,
            split=args.split,
            experiment_name=args.experiment_name,
            device=args.device,
            steps_per_cycle=args.steps_per_cycle,
            max_total_steps=args.max_total_steps,
            val_interval_steps=args.val_interval_steps,
            checkpoint_interval_steps=args.checkpoint_interval_steps,
            batch_size=args.batch_size,
            shuffle_buffer_size=args.shuffle_buffer_size,
            cache_dir=args.cache_dir,
            max_stream_rows=args.max_stream_rows,
        )
        config, generated_config_path = build_stream_config_from_job(job)
        pprint({"generated_config": str(generated_config_path)})
        state_path = Path("artifacts") / config.experiment_name / "train_state.json"
        cycle = 0
        while True:
            cycle += 1
            current_step = 0
            if state_path.exists():
                current_step = int(json.loads(state_path.read_text(encoding="utf-8")).get("global_step", 0))
            if current_step >= args.max_total_steps:
                update_registry_job(
                    config.experiment_name,
                    status="complete",
                    prepared_dir=job.prepared_dir,
                    repo_id=job.repo_id,
                    config_name=job.config_name,
                    split=job.split,
                    target_max_steps=args.max_total_steps,
                    current_step=current_step,
                )
                pprint({"worker_status": "complete", "global_step": current_step})
                break
            config.training.max_steps = min(current_step + args.steps_per_cycle, args.max_total_steps)
            update_registry_job(
                config.experiment_name,
                status="running",
                prepared_dir=job.prepared_dir,
                repo_id=job.repo_id,
                config_name=job.config_name,
                split=job.split,
                target_max_steps=config.training.max_steps,
                current_step=current_step,
            )
            pprint({"worker_cycle": cycle, "target_max_steps": config.training.max_steps, "current_step": current_step})
            pprint(train(config))
            if args.max_cycles is not None and cycle >= args.max_cycles:
                update_registry_job(
                    config.experiment_name,
                    status="paused",
                    prepared_dir=job.prepared_dir,
                    repo_id=job.repo_id,
                    config_name=job.config_name,
                    split=job.split,
                    target_max_steps=config.training.max_steps,
                    current_step=int(json.loads(state_path.read_text(encoding="utf-8")).get("global_step", current_step)) if state_path.exists() else current_step,
                )
                pprint({"worker_status": "max_cycles_reached", "cycles": cycle})
                break
            time.sleep(max(args.sleep_seconds, 0))
    elif args.command == "train-hf-stream-queue":
        from tabula.training.engine import train
        from tabula.training.registry import load_registry, reconcile_registry, refresh_dashboard, update_registry_job
        from tabula.data.cache import trim_cache_to_budget

        queue = load_stream_queue_config(args.queue_config)
        pprint(stream_queue_config_to_dict(queue))
        rng = random.Random(42)
        cycle = 0
        while True:
            cycle += 1
            reconcile_registry(stale_after_seconds=queue.stale_after_seconds)
            registry = load_registry()
            refresh_dashboard()
            runnable_jobs: list[StreamJobConfig] = []
            weights: list[float] = []
            for job in queue.jobs:
                config, generated_config_path = build_stream_config_from_job(job)
                state_path = Path("artifacts") / config.experiment_name / "train_state.json"
                current_step = 0
                if state_path.exists():
                    current_step = int(json.loads(state_path.read_text(encoding="utf-8")).get("global_step", 0))
                if current_step >= job.max_total_steps:
                    update_registry_job(
                        config.experiment_name,
                        status="complete",
                        prepared_dir=job.prepared_dir,
                        repo_id=job.repo_id,
                        config_name=job.config_name,
                        split=job.split,
                        target_max_steps=job.max_total_steps,
                        current_step=current_step,
                    )
                    continue
                registry_job = dict(registry.get("jobs", {}).get(config.experiment_name, {}))
                if registry_job.get("status") == "failed":
                    update_registry_job(
                        config.experiment_name,
                        status="failed",
                        prepared_dir=job.prepared_dir,
                        repo_id=job.repo_id,
                        config_name=job.config_name,
                        split=job.split,
                        target_max_steps=job.max_total_steps,
                        current_step=current_step,
                        heartbeat_at_utc=_utc_now().isoformat(),
                        failure_count=int(registry_job.get("failure_count", 0)),
                        last_error=registry_job.get("last_error"),
                        cooldown_until_utc=registry_job.get("cooldown_until_utc"),
                    )
                    continue
                heartbeat = _parse_utc(registry_job.get("heartbeat_at_utc"))
                if registry_job.get("status") == "running" and heartbeat is not None:
                    if (_utc_now() - heartbeat).total_seconds() > queue.stale_after_seconds:
                        update_registry_job(
                            config.experiment_name,
                            status="stale",
                            prepared_dir=job.prepared_dir,
                            repo_id=job.repo_id,
                            config_name=job.config_name,
                            split=job.split,
                            target_max_steps=job.max_total_steps,
                            current_step=current_step,
                            heartbeat_at_utc=_utc_now().isoformat(),
                            failure_count=int(registry_job.get("failure_count", 0)),
                            last_error=registry_job.get("last_error"),
                            cooldown_until_utc=registry_job.get("cooldown_until_utc"),
                        )
                        registry_job["status"] = "stale"
                if registry_job.get("status") == "stale":
                    update_registry_job(
                        config.experiment_name,
                        status="recovered",
                        prepared_dir=job.prepared_dir,
                        repo_id=job.repo_id,
                        config_name=job.config_name,
                        split=job.split,
                        target_max_steps=job.max_total_steps,
                        current_step=current_step,
                        heartbeat_at_utc=_utc_now().isoformat(),
                        failure_count=int(registry_job.get("failure_count", 0)),
                        last_error=registry_job.get("last_error"),
                        cooldown_until_utc=registry_job.get("cooldown_until_utc"),
                    )
                    registry_job["status"] = "recovered"
                cooldown_until = _parse_utc(registry_job.get("cooldown_until_utc"))
                if cooldown_until is not None and cooldown_until > _utc_now():
                    update_registry_job(
                        config.experiment_name,
                        status="cooldown",
                        prepared_dir=job.prepared_dir,
                        repo_id=job.repo_id,
                        config_name=job.config_name,
                        split=job.split,
                        target_max_steps=job.max_total_steps,
                        current_step=current_step,
                        heartbeat_at_utc=_utc_now().isoformat(),
                        failure_count=int(registry_job.get("failure_count", 0)),
                        last_error=registry_job.get("last_error"),
                        cooldown_until_utc=registry_job.get("cooldown_until_utc"),
                    )
                    continue
                runnable_jobs.append(job)
                weights.append(max(float(job.weight), 1e-6))

            if runnable_jobs:
                remaining_jobs = list(runnable_jobs)
                remaining_weights = list(weights)
                for _ in range(len(runnable_jobs)):
                    if not remaining_jobs:
                        break
                    selected_index = rng.choices(range(len(remaining_jobs)), weights=remaining_weights, k=1)[0]
                    job = remaining_jobs.pop(selected_index)
                    remaining_weights.pop(selected_index)
                    config, generated_config_path = build_stream_config_from_job(job)
                    state_path = Path("artifacts") / config.experiment_name / "train_state.json"
                    current_step = 0
                    if state_path.exists():
                        current_step = int(json.loads(state_path.read_text(encoding="utf-8")).get("global_step", 0))
                    config.training.max_steps = min(current_step + job.steps_per_cycle, job.max_total_steps)
                    update_registry_job(
                        config.experiment_name,
                        status="running",
                        prepared_dir=job.prepared_dir,
                        repo_id=job.repo_id,
                        config_name=job.config_name,
                        split=job.split,
                        target_max_steps=config.training.max_steps,
                        current_step=current_step,
                        heartbeat_at_utc=_utc_now().isoformat(),
                    )
                    pprint({"generated_config": str(generated_config_path), "worker_cycle": cycle, "experiment_name": config.experiment_name, "current_step": current_step, "target_max_steps": config.training.max_steps, "weight": job.weight})
                    try:
                        pprint(train(config))
                        latest_step = current_step
                        if state_path.exists():
                            latest_step = int(json.loads(state_path.read_text(encoding="utf-8")).get("global_step", current_step))
                        update_registry_job(
                            config.experiment_name,
                            status="complete" if latest_step >= job.max_total_steps else "idle",
                            prepared_dir=job.prepared_dir,
                            repo_id=job.repo_id,
                            config_name=job.config_name,
                            split=job.split,
                            target_max_steps=job.max_total_steps,
                            current_step=latest_step,
                            heartbeat_at_utc=_utc_now().isoformat(),
                            failure_count=0,
                            last_error=None,
                            cooldown_until_utc=None,
                        )
                    except Exception as exc:
                        registry_after = load_registry()
                        registry_job = dict(registry_after.get("jobs", {}).get(config.experiment_name, {}))
                        failure_count = int(registry_job.get("failure_count", 0)) + 1
                        cooldown_until = (_utc_now() + timedelta(seconds=max(job.retry_backoff_seconds, 0))).isoformat()
                        status = "failed" if failure_count >= job.max_retries else "retry_wait"
                        update_registry_job(
                            config.experiment_name,
                            status=status,
                            prepared_dir=job.prepared_dir,
                            repo_id=job.repo_id,
                            config_name=job.config_name,
                            split=job.split,
                            target_max_steps=config.training.max_steps,
                            current_step=current_step,
                            heartbeat_at_utc=_utc_now().isoformat(),
                            failure_count=failure_count,
                            last_error=str(exc),
                            cooldown_until_utc=cooldown_until,
                        )
                        pprint({"experiment_name": config.experiment_name, "status": status, "failure_count": failure_count, "error": str(exc)})
            if queue.max_cycles is not None and cycle >= queue.max_cycles:
                refresh_dashboard()
                break
            if queue.hf_cache_max_gb is not None:
                cache_dirs = sorted({job.cache_dir for job in queue.jobs if job.cache_dir})
                for cache_dir in cache_dirs:
                    pprint(trim_cache_to_budget(cache_dir, queue.hf_cache_max_gb))
            refresh_dashboard()
            time.sleep(max(queue.sleep_seconds, 0))
    elif args.command == "data":
        from tabula.data import (
            discover_csvs,
            download_dataset,
            download_kaggle_slug,
            filter_catalog,
            fetch_huggingface_dataset,
            get_dataset_entry,
            huggingface_auth_status,
            kaggle_auth_status,
            load_kaggle_catalog,
            ingest_kaggle_dataset,
            prepare_dataset,
            prepared_dataset_to_dict,
            search_huggingface_datasets,
            search_kaggle_datasets,
        )

        if args.data_command == "list":
            entries = filter_catalog(
                load_kaggle_catalog(),
                quality_tier=args.quality_tier,
                task_type=args.task_type,
                recommended_only=args.recommended_only,
            )
            for entry in entries:
                print(
                    f"{entry.id}: tier={entry.quality_tier} task={entry.task_type} "
                    f"rows~={entry.est_rows} slug={entry.kaggle_slug}"
                )
        elif args.data_command == "show":
            pprint(get_dataset_entry(args.dataset))
        elif args.data_command == "fetch":
            output_dir = download_dataset(
                args.dataset,
                output_root=args.output_root,
                unzip=not args.no_unzip,
                force=args.force,
            )
            print(output_dir)
        elif args.data_command == "search-kaggle":
            results = search_kaggle_datasets(
                search=args.query,
                tags=args.tag,
                sort_by=args.sort_by,
                page=args.page,
                min_usability_rating=args.min_usability_rating,
            )
            for item in results:
                print(
                    f"{item.slug}: usability={item.usability_rating:.2f} votes={item.vote_count} "
                    f"downloads={item.download_count} size={item.size_bytes} updated={item.last_updated}"
                )
        elif args.data_command == "fetch-kaggle":
            output_dir = download_kaggle_slug(
                args.slug,
                output_root=args.output_root,
                dataset_id=args.dataset_id,
                unzip=not args.no_unzip,
                force=args.force,
                backend=args.backend,
                title=args.title,
                task_type=args.task_type,
                target_column=args.target_column,
                notes=args.notes,
            )
            print(output_dir)
        elif args.data_command == "ingest-kaggle":
            prepared = ingest_kaggle_dataset(
                args.slug,
                output_root=args.output_root,
                processed_root=args.processed_root,
                dataset_id=args.dataset_id,
                unzip=not args.no_unzip,
                force=args.force,
                backend=args.backend,
                seed=args.seed,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                max_rows=args.max_rows,
                drop_identifier_columns=not args.keep_identifiers,
                title=args.title,
                task_type=args.task_type,
                target_column=args.target_column,
                train_file=args.train_file,
                notes=args.notes,
                feature_engineering=not args.no_feature_engineering,
            )
            pprint(prepared)
        elif args.data_command == "auth-check":
            pprint(kaggle_auth_status())
        elif args.data_command == "hf-auth-check":
            pprint(huggingface_auth_status())
        elif args.data_command == "search-hf":
            results = search_huggingface_datasets(
                query=args.query,
                task_category=args.task_category,
                limit=args.limit,
                sort=args.sort,
            )
            for item in results:
                print(
                    f"{item.repo_id}: downloads={item.downloads} likes={item.likes} "
                    f"updated={item.last_modified}"
                )
        elif args.data_command == "fetch-hf":
            output_dir = fetch_huggingface_dataset(
                args.repo_id,
                output_root=args.output_root,
                dataset_id=args.dataset_id,
                config_name=args.config_name,
                split=args.split,
                max_rows=args.max_rows,
                title=args.title,
                task_type=args.task_type,
                target_column=args.target_column,
                notes=args.notes,
            )
            print(output_dir)
        elif args.data_command == "prepare":
            prepared = prepare_dataset(
                args.dataset,
                raw_root=args.raw_root,
                processed_root=args.processed_root,
                seed=args.seed,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                max_rows=args.max_rows,
                drop_identifier_columns=not args.keep_identifiers,
                task_type=args.task_type,
                target_column=args.target_column,
                train_file=args.train_file,
                title=args.title,
                notes=args.notes,
                feature_engineering=not args.no_feature_engineering,
            )
            pprint(prepared_dataset_to_dict(prepared))
        elif args.data_command == "materialize":
            entries = load_kaggle_catalog()
            if args.dataset:
                requested_ids = set(args.dataset)
                entries = [entry for entry in entries if entry.id in requested_ids]
            else:
                entries = filter_catalog(
                    entries,
                    quality_tier=args.quality_tier,
                    task_type=args.task_type,
                    recommended_only=args.recommended_only,
                )
                entries = [entry for entry in entries if entry.source_type == args.source_type]

            summary: list[dict[str, object]] = []
            for entry in entries:
                item: dict[str, object] = {"dataset_id": entry.id, "source_type": entry.source_type}
                try:
                    if not args.skip_fetch:
                        download_dataset(
                            entry.id,
                            output_root=args.raw_root,
                            unzip=True,
                            force=args.force,
                        )
                    prepared = prepare_dataset(
                        entry.id,
                        raw_root=args.raw_root,
                        processed_root=args.processed_root,
                        seed=args.seed,
                        val_fraction=args.val_fraction,
                        test_fraction=args.test_fraction,
                        max_rows=args.max_rows,
                        drop_identifier_columns=not args.keep_identifiers,
                        feature_engineering=True,
                    )
                    item["status"] = "ok"
                    item["config_path"] = prepared.config_path
                    item["rows"] = prepared.train_rows + prepared.val_rows + prepared.test_rows
                except Exception as exc:
                    item["status"] = "error"
                    item["error"] = str(exc)
                summary.append(item)
            pprint(summary)
        elif args.data_command == "inspect":
            for summary in discover_csvs(args.path):
                print(f"{summary.path}: rows={summary.row_count} columns={len(summary.columns)}")
                print(", ".join(summary.columns))
    elif args.command == "curriculum-worker":
        _run_curriculum_worker(args)
    elif args.command == "curriculum":
        _run_curriculum_management(args)
    elif args.command == "autocurriculum-hf":
        _run_autocurriculum_hf(args)


def _seed_curriculum_queue_from_hf_search(args: object) -> dict[str, object]:
    from tabula.data import bootstrap_huggingface_stream_sample, prepare_dataset, search_huggingface_datasets, sanitize_dataset_id
    from tabula.training.curriculum import CurriculumEntry, CurriculumQueue, queue_path

    task_categories = list(getattr(args, "task_categories", []) or ["tabular-classification", "tabular-regression"])
    limit = int(getattr(args, "limit", 100))
    max_new_datasets = getattr(args, "max_new_datasets", None)
    artifacts_root = getattr(args, "artifacts_root", "artifacts")
    q_path = queue_path(artifacts_root)
    queue = CurriculumQueue.load(q_path)
    existing_ids = {entry.dataset_id for entry in queue.entries}
    existing_repo_ids = {entry.hf_repo_id for entry in queue.entries}

    discovered: list[dict[str, object]] = []
    seen_repo_ids: set[str] = set()
    for task_category in task_categories:
        results = search_huggingface_datasets(
            query=getattr(args, "query", None),
            task_category=task_category,
            limit=limit,
            sort=getattr(args, "sort", "downloads"),
        )
        for item in results:
            if item.repo_id in seen_repo_ids:
                continue
            seen_repo_ids.add(item.repo_id)
            discovered.append(
                {
                    "repo_id": item.repo_id,
                    "task_category": task_category,
                    "downloads": item.downloads,
                    "likes": item.likes,
                }
            )

    added: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []

    for item in discovered:
        if max_new_datasets is not None and len(added) >= int(max_new_datasets):
            break
        repo_id = str(item["repo_id"])
        dataset_id = sanitize_dataset_id(repo_id)
        if dataset_id in existing_ids or repo_id in existing_repo_ids:
            skipped.append({"repo_id": repo_id, "reason": "already_enqueued"})
            continue

        prepared_dir = Path(getattr(args, "processed_root", "data/processed")) / dataset_id
        try:
            if not (prepared_dir / "train_config.json").exists():
                bootstrap_huggingface_stream_sample(
                    repo_id,
                    output_root=getattr(args, "raw_root", "data/raw"),
                    dataset_id=dataset_id,
                    sample_rows=int(getattr(args, "bootstrap_rows", 2048)),
                    shuffle_buffer_size=int(getattr(args, "shuffle_buffer_size", 10000)),
                    seed=int(getattr(args, "seed", 42)),
                )
                prepare_dataset(
                    dataset_id,
                    raw_root=getattr(args, "raw_root", "data/raw"),
                    processed_root=getattr(args, "processed_root", "data/processed"),
                    seed=int(getattr(args, "seed", 42)),
                    val_fraction=float(getattr(args, "val_fraction", 0.1)),
                    test_fraction=float(getattr(args, "test_fraction", 0.1)),
                    max_rows=int(getattr(args, "bootstrap_rows", 2048)),
                    drop_identifier_columns=not bool(getattr(args, "keep_identifiers", False)),
                    feature_engineering=not bool(getattr(args, "no_feature_engineering", False)),
                )

            entry = CurriculumEntry(
                dataset_id=dataset_id,
                prepared_dir=str(prepared_dir),
                hf_repo_id=repo_id,
                steps_per_cycle=int(getattr(args, "steps_per_cycle", 1000)),
                max_total_steps=int(getattr(args, "max_total_steps", 10000)),
                priority=int(getattr(args, "priority_base", 100)) + len(added),
            )
            queue.add(entry)
            existing_ids.add(dataset_id)
            existing_repo_ids.add(repo_id)
            added.append(
                {
                    "dataset_id": dataset_id,
                    "repo_id": repo_id,
                    "prepared_dir": str(prepared_dir),
                    "task_category": str(item["task_category"]),
                }
            )
        except Exception as exc:
            failed.append({"repo_id": repo_id, "error": str(exc)})

    queue.save(q_path)
    summary = {
        "queue_path": str(q_path),
        "discovered_count": len(discovered),
        "added_count": len(added),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "added": added[:10],
        "failed": failed[:10],
    }
    return summary


def _run_autocurriculum_hf(args: object) -> None:
    summary = _seed_curriculum_queue_from_hf_search(args)
    pprint(summary)
    _run_curriculum_worker(args)


def _run_curriculum_worker(args: object) -> None:
    """Background loop: pop pending entries, train one cycle, sleep, repeat."""
    import time
    from tabula.training.curriculum import (
        CurriculumLedger,
        CurriculumQueue,
        LedgerSession,
        ledger_path,
        queue_path,
    )
    from tabula.training.engine import train

    artifacts_root = getattr(args, "artifacts_root", "artifacts")
    q_path = queue_path(artifacts_root)
    l_path = ledger_path(artifacts_root)
    transfer_trunk: bool = not getattr(args, "no_trunk_transfer", False)

    cycle = 0
    while True:
        queue = CurriculumQueue.load(q_path)
        entry = queue.next_pending()

        if entry is None:
            pending_states = {e.status for e in queue.entries}
            if not pending_states or pending_states <= {"done", "failed"}:
                pprint({"worker_status": "all_done", "cycle": cycle})
                break
            sleep_secs = getattr(args, "sleep_seconds", 30)
            print(f"[curriculum-worker] No pending entries - sleeping {sleep_secs}s ...")
            time.sleep(sleep_secs)
            continue

        cycle += 1
        if args.max_cycles is not None and cycle > args.max_cycles:  # type: ignore[attr-defined]
            pprint({"worker_status": "max_cycles_reached", "cycles": cycle - 1})
            break

        expname = entry.effective_experiment_name()
        target_steps = entry.next_cycle_target()

        print(
            f"\n[curriculum-worker] cycle={cycle} dataset={entry.dataset_id} "
            f"experiment={expname} steps={entry.total_steps}->{target_steps}"
        )

        # Mark in-progress before training (so a crash is detectable)
        queue.mark_in_progress(entry.dataset_id)
        queue.save(q_path)

        # Resolve trunk source (best checkpoint from ledger so far)
        trunk_source: str | None = None
        if transfer_trunk:
            trunk_source = CurriculumLedger.latest_best_checkpoint(l_path)

        # Build config
        prepared_dir = Path(entry.prepared_dir)
        base_config_path = prepared_dir / "train_config.json"
        if not base_config_path.exists():
            print(f"[curriculum-worker] ERROR: {base_config_path} not found - marking failed.")
            queue.mark_failed(entry.dataset_id)
            queue.save(q_path)
            continue

        config = load_config(base_config_path)
        config.data.dataset_type = "hf_stream"
        config.data.prepared_dir = str(prepared_dir)
        config.data.train_path = None
        config.data.val_path = str(prepared_dir / "val.csv")
        config.data.hf_repo_id = entry.hf_repo_id
        config.data.hf_config_name = entry.hf_config_name
        config.data.hf_split = entry.hf_split
        config.data.hf_streaming = True
        config.data.hf_shuffle_buffer_size = getattr(args, "shuffle_buffer_size", 10000)
        config.data.hf_cache_dir = getattr(args, "cache_dir", None)
        if getattr(args, "batch_size", None):
            config.data.batch_size = args.batch_size  # type: ignore[attr-defined]
        config.training.device = getattr(args, "device", "cpu")
        config.training.max_steps = target_steps
        config.training.val_interval_steps = getattr(args, "val_interval_steps", 500)
        config.training.checkpoint_interval_steps = getattr(args, "checkpoint_interval_steps", 500)
        config.training.resume = True
        config.experiment_name = expname
        config.artifacts_root = artifacts_root

        session_id = LedgerSession.new_session_id()
        started_at = LedgerSession.utc_now()
        exit_reason = "interrupted"
        result: dict = {}
        error_msg: str | None = None

        try:
            result = train(config, pretrained_trunk_path=trunk_source)
            steps_before = entry.total_steps
            new_total = int(
                json.loads((Path(artifacts_root) / expname / "train_state.json").read_text(encoding="utf-8"))
                .get("global_step", entry.total_steps)
                if (Path(artifacts_root) / expname / "train_state.json").exists()
                else entry.total_steps
            )
            steps_trained = max(0, new_total - steps_before)
            rows_delta = int(
                json.loads((Path(artifacts_root) / expname / "train_state.json").read_text(encoding="utf-8"))
                .get("rows_seen", 0)
                if (Path(artifacts_root) / expname / "train_state.json").exists()
                else 0
            ) - entry.total_rows_seen
            rows_delta = max(0, rows_delta)
            exit_reason = "max_total_steps" if new_total >= entry.max_total_steps else "max_steps"
        except Exception as exc:
            error_msg = str(exc)
            exit_reason = "error"
            steps_trained = 0
            rows_delta = 0
            new_total = entry.total_steps
            print(f"[curriculum-worker] Training error for {entry.dataset_id}: {exc}")

        ended_at = LedgerSession.utc_now()
        checkpoint_path = result.get("checkpoint") or str(Path(artifacts_root) / expname / "best.pt")

        # Reload queue to get latest state (may have changed in train_state.json)
        queue = CurriculumQueue.load(q_path)
        queue.update_progress(
            entry.dataset_id,
            steps_delta=steps_trained,
            rows_delta=rows_delta,
            best_val_loss=result.get("best_val_loss"),
            session_timestamp=ended_at,
        )
        if error_msg:
            queue.mark_failed(entry.dataset_id)
        queue.save(q_path)

        # Re-read updated entry for cumulative totals
        updated = queue.get(entry.dataset_id)
        sess = LedgerSession(
            session_id=session_id,
            dataset_id=entry.dataset_id,
            experiment_name=expname,
            started_at=started_at,
            ended_at=ended_at,
            steps_trained=steps_trained,
            cumulative_steps=updated.total_steps if updated else new_total,
            rows_seen=rows_delta,
            cumulative_rows=updated.total_rows_seen if updated else 0,
            exit_reason=exit_reason,
            best_val_loss=result.get("best_val_loss"),
            checkpoint_path=checkpoint_path,
            trunk_source=trunk_source,
            error=error_msg,
        )
        CurriculumLedger.append(sess, l_path)
        pprint({"session": session_id, "dataset": entry.dataset_id, "steps_trained": steps_trained,
                "exit_reason": exit_reason, "best_val_loss": result.get("best_val_loss")})

        sleep_secs = getattr(args, "sleep_seconds", 30)
        if sleep_secs > 0:
            time.sleep(sleep_secs)


def _run_curriculum_management(args: object) -> None:
    """Handle ``tabula curriculum queue/ledger`` sub-commands."""
    from tabula.training.curriculum import (
        CurriculumEntry,
        CurriculumLedger,
        CurriculumQueue,
        ledger_path,
        queue_path,
    )

    curriculum_command = getattr(args, "curriculum_command", None)

    if curriculum_command == "queue":
        queue_command = getattr(args, "queue_command", None)
        artifacts_root = getattr(args, "artifacts_root", "artifacts")
        q_path = queue_path(artifacts_root)

        if queue_command == "add":
            queue = CurriculumQueue.load(q_path)
            entry = CurriculumEntry(
                dataset_id=args.dataset_id,  # type: ignore[attr-defined]
                prepared_dir=args.prepared_dir,  # type: ignore[attr-defined]
                hf_repo_id=args.repo_id,  # type: ignore[attr-defined]
                hf_config_name=getattr(args, "config_name", None),
                hf_split=getattr(args, "split", "train"),
                steps_per_cycle=getattr(args, "steps_per_cycle", 2000),
                max_total_steps=getattr(args, "max_total_steps", 20000),
                priority=getattr(args, "priority", 100),
                experiment_name=getattr(args, "experiment_name", None),
                notes=getattr(args, "notes", ""),
                tags=list(getattr(args, "tags", []) or []),
            )
            queue.add(entry)
            queue.save(q_path)
            pprint({"added": entry.dataset_id, "queue_size": len(queue.entries), "queue_path": str(q_path)})

        elif queue_command == "list":
            queue = CurriculumQueue.load(q_path)
            filter_status = getattr(args, "status", None)
            entries = queue.entries if not filter_status else [e for e in queue.entries if e.status == filter_status]
            for e in entries:
                remaining = e.remaining_steps()
                print(
                    f"{e.dataset_id}: status={e.status} priority={e.priority} "
                    f"steps={e.total_steps}/{e.max_total_steps} remaining={remaining} "
                    f"best_loss={e.best_val_loss} last={e.last_session_at or 'never'}"
                )

        elif queue_command == "status":
            queue = CurriculumQueue.load(q_path)
            from collections import Counter
            counts = Counter(e.status for e in queue.entries)
            total_steps = sum(e.total_steps for e in queue.entries)
            total_budget = sum(e.max_total_steps for e in queue.entries)
            pprint({
                "queue_path": str(q_path),
                "total_entries": len(queue.entries),
                "by_status": dict(counts),
                "total_steps_trained": total_steps,
                "total_steps_budget": total_budget,
                "pct_complete": round(100 * total_steps / total_budget, 1) if total_budget else 0.0,
            })

        elif queue_command == "reset":
            queue = CurriculumQueue.load(q_path)
            queue.reset_to_pending(args.dataset_id)  # type: ignore[attr-defined]
            queue.save(q_path)
            pprint({"reset": args.dataset_id, "status": "pending"})  # type: ignore[attr-defined]

    elif curriculum_command == "ledger":
        artifacts_root = getattr(args, "artifacts_root", "artifacts")
        l_path = ledger_path(artifacts_root)
        sessions = CurriculumLedger.load(l_path)
        filter_ds = getattr(args, "dataset_id", None)
        if filter_ds:
            sessions = [s for s in sessions if s.dataset_id == filter_ds]
        last_n = getattr(args, "last", 20)
        for sess in sessions[-last_n:]:
            print(
                f"{sess.started_at[:19]} [{sess.session_id}] "
                f"{sess.dataset_id} steps={sess.steps_trained} cuml={sess.cumulative_steps} "
                f"exit={sess.exit_reason} best_loss={sess.best_val_loss} "
                f"trunk={'yes' if sess.trunk_source else 'no'}"
            )


if __name__ == "__main__":
    main()
