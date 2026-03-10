from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

ProblemType = Literal["binary", "multiclass", "regression"]
TaskMode = Literal["pretrain", "finetune"]
DatasetType = Literal["synthetic", "csv", "prepared", "hf_stream"]


@dataclass
class TaskConfig:
    mode: TaskMode = "pretrain"
    problem_type: ProblemType = "multiclass"
    target_column: str = "target"


@dataclass
class DataConfig:
    dataset_type: DatasetType = "synthetic"
    prepared_dir: str | None = None
    train_path: str | None = None
    val_path: str | None = None
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    num_numeric_features: int = 8
    num_categorical_features: int = 0
    categorical_cardinality: int = 8
    num_classes: int = 2
    train_size: int = 4096
    val_size: int = 1024
    batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = False
    standardize_numeric: bool = True
    hf_repo_id: str | None = None
    hf_config_name: str | None = None
    hf_split: str = "train"
    hf_streaming: bool = False
    hf_shuffle_buffer_size: int = 10000
    hf_cache_dir: str | None = None
    hf_max_stream_rows: int | None = None
    hf_skip_rows: int = 0


@dataclass
class ModelConfig:
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 384
    dropout: float = 0.1
    feature_token_dropout: float = 0.05
    norm: Literal["layernorm", "rmsnorm"] = "rmsnorm"
    ffn_activation: Literal["gelu", "swiglu"] = "swiglu"
    max_categories: int = 128
    numeric_embedding: Literal["linear", "periodic"] = "linear"
    numeric_periodic_features: int = 8
    text_encoder: Literal["custom", "pretrained"] = "custom"
    text_encoder_layers: int = 1
    text_encoder_heads: int = 4
    text_max_tokens: int = 16
    text_pretrained_model_name: str = "distilbert-base-uncased"
    text_pretrained_max_length: int = 32
    text_pretrained_trainable: bool = False
    schema_encoder: Literal["hash", "pretrained"] = "pretrained"
    schema_pretrained_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    schema_pretrained_max_length: int = 32
    schema_pretrained_trainable: bool = False


@dataclass
class TrainingConfig:
    device: str = "cpu"
    max_epochs: int = 20
    max_steps: int | None = None
    val_interval_steps: int = 500
    checkpoint_interval_steps: int = 500
    amp: bool = False
    amp_dtype: Literal["float16", "bfloat16"] = "float16"
    gradient_accumulation_steps: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    log_interval: int = 20
    early_stopping_patience: int = 5
    resume: bool = True


@dataclass
class EpisodeConfig:
    enabled: bool = False
    support_size: int = 64
    query_size: int = 64
    sample_with_replacement: bool = False


@dataclass
class CurriculumConfig:
    """Global settings for the curriculum background worker.

    Per-dataset budgets (``steps_per_cycle``, ``max_total_steps``) live on each
    :class:`~tabula.training.curriculum.CurriculumEntry` in the queue file.
    These fields control the *worker process* itself.
    """

    artifacts_root: str = "artifacts"
    """Root directory for the queue file, ledger file, and per-dataset artifact dirs."""

    device: str = "cpu"
    """Training device override applied to every dataset in the curriculum.
    Use ``cuda`` for GPU or ``cpu`` as a fallback."""

    batch_size: int | None = None
    """If set, overrides the batch size from each dataset's ``train_config.json``."""

    shuffle_buffer_size: int = 10000
    """Streaming shuffle buffer passed to every HuggingFace dataset loader."""

    hf_cache_dir: str | None = None
    """Optional local directory for the HuggingFace dataset cache.  Useful to avoid
    exhausting VRAM/disk on repeated re-downloads."""

    val_interval_steps: int = 500
    val_checkpoint_interval_steps: int = 500

    sleep_seconds: int = 30
    """Seconds the worker sleeps between cycles when the queue has no pending entries."""

    max_cycles: int | None = None
    """Hard cap on the total number of worker cycles.  ``None`` means run forever."""

    transfer_trunk: bool = True
    """If ``True`` (default), warm-start each new dataset from the best checkpoint of
    the *previous* dataset via shape-matched weight transfer.  Disable with
    ``transfer_trunk: false`` to always train from scratch."""


@dataclass
class ExperimentConfig:
    experiment_name: str = "tabula_run"
    artifacts_root: str = "artifacts"
    seed: int = 42
    task: TaskConfig = field(default_factory=TaskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)


@dataclass
class StreamJobConfig:
    prepared_dir: str
    repo_id: str
    config_name: str | None = None
    split: str = "train"
    experiment_name: str | None = None
    device: str | None = None
    steps_per_cycle: int = 1000
    max_total_steps: int = 10000
    val_interval_steps: int = 500
    checkpoint_interval_steps: int = 500
    batch_size: int | None = None
    shuffle_buffer_size: int = 10000
    cache_dir: str | None = None
    max_stream_rows: int | None = None
    weight: float = 1.0
    max_retries: int = 3
    retry_backoff_seconds: int = 30


@dataclass
class StreamQueueConfig:
    jobs: list[StreamJobConfig] = field(default_factory=list)
    sleep_seconds: int = 15
    max_cycles: int | None = None
    stale_after_seconds: int = 300
    hf_cache_max_gb: float | None = None


def load_stream_queue_config(path: str | Path) -> StreamQueueConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    jobs = [StreamJobConfig(**item) for item in raw.get("jobs", [])]
    return StreamQueueConfig(
        jobs=jobs,
        sleep_seconds=int(raw.get("sleep_seconds", 15)),
        max_cycles=raw.get("max_cycles"),
        stale_after_seconds=int(raw.get("stale_after_seconds", 300)),
        hf_cache_max_gb=raw.get("hf_cache_max_gb"),
    )


def stream_queue_config_to_dict(config: StreamQueueConfig) -> dict[str, Any]:
    return asdict(config)


def _merge_dataclass(cls: type[Any], values: dict[str, Any] | None) -> Any:
    values = values or {}
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        raw = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "PyYAML is not installed. Install project dependencies with `pip install -e .` "
                "or provide the config in JSON format."
            ) from exc
    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", "tabula_run"),
        artifacts_root=raw.get("artifacts_root", "artifacts"),
        seed=raw.get("seed", 42),
        task=_merge_dataclass(TaskConfig, raw.get("task")),
        data=_merge_dataclass(DataConfig, raw.get("data")),
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        training=_merge_dataclass(TrainingConfig, raw.get("training")),
        episode=_merge_dataclass(EpisodeConfig, raw.get("episode")),
    )
