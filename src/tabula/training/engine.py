from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import hashlib

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from tabula.config import ExperimentConfig
from tabula.data.datasets import TabularBatch, build_dataloaders
from tabula.data.episodes import EpisodeBatch, sample_episode_batch
from tabula.evaluation.metrics import compute_metrics
from tabula.models.transformer import EpisodicTabularTransformer, TabularTransformer
from tabula.training.trunk import load_trunk_weights
from tabula.utils import set_seed


@dataclass
class EpochResult:
    loss: float
    metrics: dict[str, float]


@dataclass
class StreamTrainState:
    global_step: int = 0
    rows_seen: int = 0
    source_rows_seen: int = 0
    best_val_loss: float = float("inf")
    patience: int = 0


def _make_criterion(problem_type: str, output_dim: int, label_smoothing: float = 0.0) -> nn.Module:
    if problem_type == "regression":
        return nn.MSELoss()
    if output_dim == 1:
        return nn.BCEWithLogitsLoss(reduction="mean")
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def _move_batch(batch: TabularBatch, device: torch.device) -> TabularBatch:
    return TabularBatch(
        x_num=batch.x_num.to(device),
        x_cat=batch.x_cat.to(device),
        x_text_token_ids=batch.x_text_token_ids.to(device),
        x_text_values=batch.x_text_values,
        x_num_mask=batch.x_num_mask.to(device),
        x_cat_mask=batch.x_cat_mask.to(device),
        x_text_mask=batch.x_text_mask.to(device),
        num_schema_texts=batch.num_schema_texts,
        cat_schema_texts=batch.cat_schema_texts,
        text_schema_texts=batch.text_schema_texts,
        num_name_token_ids=batch.num_name_token_ids.to(device),
        cat_name_token_ids=batch.cat_name_token_ids.to(device),
        text_name_token_ids=batch.text_name_token_ids.to(device),
        num_profile_vectors=batch.num_profile_vectors.to(device),
        cat_profile_vectors=batch.cat_profile_vectors.to(device),
        text_profile_vectors=batch.text_profile_vectors.to(device),
        y=batch.y.to(device),
    )


def _move_episode_batch(episode: EpisodeBatch, device: torch.device) -> EpisodeBatch:
    """Move both halves of an :class:`~tabula.data.episodes.EpisodeBatch` to *device*."""
    return EpisodeBatch(
        support=_move_batch(episode.support, device),
        query=_move_batch(episode.query, device),
    )


def _compute_loss(problem_type: str, output_dim: int, criterion: nn.Module, logits: torch.Tensor, y: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    if problem_type == "regression":
        return criterion(logits.reshape(-1), y.float())
    if output_dim == 1:
        targets = y.float()
        if label_smoothing > 0:
            targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
        return criterion(logits.reshape(-1), targets)
    return criterion(logits, y.long())


def _autocast_dtype(config: ExperimentConfig) -> torch.dtype:
    return torch.bfloat16 if config.training.amp_dtype == "bfloat16" else torch.float16


def _use_amp(config: ExperimentConfig, device: torch.device) -> bool:
    return bool(config.training.amp and device.type == "cuda")


def _run_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: ExperimentConfig,
    output_dim: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    all_y: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []

    iterator = tqdm(loader, disable=False, leave=False)
    accumulation_steps = max(int(config.training.gradient_accumulation_steps), 1)
    if is_train:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(iterator, start=1):
        batch = _move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, dtype=_autocast_dtype(config), enabled=_use_amp(config, device)):
                logits = model(batch)
                _ls = getattr(config.training, 'label_smoothing', 0.0) if is_train else 0.0
                raw_loss = _compute_loss(config.task.problem_type, output_dim, criterion, logits, batch.y, _ls)
                loss = raw_loss / accumulation_steps if is_train else raw_loss
            if is_train:
                if scaler is not None and _use_amp(config, device):
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if step % accumulation_steps == 0 or step == len(loader):
                    if scaler is not None and _use_amp(config, device):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        losses.append(float(raw_loss.item() if is_train else loss.item()))
        all_y.append(batch.y.detach().cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())
        if is_train and step % config.training.log_interval == 0:
            iterator.set_description(f"train loss={np.mean(losses):.4f}")

    y_true = np.concatenate(all_y, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    metrics = compute_metrics(config.task.problem_type, y_true, logits)
    return EpochResult(loss=float(np.mean(losses)), metrics=metrics)


def _run_episode_epoch(
    model: EpisodicTabularTransformer,
    loader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: ExperimentConfig,
    output_dim: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> EpochResult:
    """Run one epoch of episode-mode training or validation.

    Each flat batch from *loader* is re-sampled into a support/query split via
    :func:`~tabula.data.episodes.sample_episode_batch`.  The episodic model
    sees the full episode and is supervised only on query-row labels, matching
    the few-shot evaluation protocol.
    """
    ep_cfg = config.episode
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    all_y: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []

    iterator = tqdm(loader, disable=False, leave=False)
    accumulation_steps = max(int(config.training.gradient_accumulation_steps), 1)
    if is_train:
        optimizer.zero_grad(set_to_none=True)
    for step, flat_batch in enumerate(iterator, start=1):
        flat_batch = _move_batch(flat_batch, device)
        episode = sample_episode_batch(
            flat_batch,
            support_size=ep_cfg.support_size,
            query_size=ep_cfg.query_size,
            sample_with_replacement=ep_cfg.sample_with_replacement,
        )
        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, dtype=_autocast_dtype(config), enabled=_use_amp(config, device)):
                logits = model(episode)
                _ls = getattr(config.training, 'label_smoothing', 0.0) if is_train else 0.0
                raw_loss = _compute_loss(
                    config.task.problem_type, output_dim, criterion, logits, episode.query.y, _ls
                )
                loss = raw_loss / accumulation_steps if is_train else raw_loss
            if is_train:
                if scaler is not None and _use_amp(config, device):
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if step % accumulation_steps == 0 or step == len(loader):
                    if scaler is not None and _use_amp(config, device):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        losses.append(float(raw_loss.item() if is_train else loss.item()))
        all_y.append(episode.query.y.detach().cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())
        if is_train and step % config.training.log_interval == 0:
            iterator.set_description(f"episode train loss={np.mean(losses):.4f}")

    y_true = np.concatenate(all_y, axis=0)
    logits_arr = np.concatenate(all_logits, axis=0)
    metrics = compute_metrics(config.task.problem_type, y_true, logits_arr)
    return EpochResult(loss=float(np.mean(losses)), metrics=metrics)


def _state_dict_path(output_dir: Path) -> Path:
    return output_dir / "train_state.json"


def _progress_log_path(output_dir: Path) -> Path:
    return output_dir / "progress.jsonl"


def _latest_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "latest.pt"


def _load_stream_state_from_artifacts(experiment_name: str, artifacts_root: str | Path = "artifacts") -> StreamTrainState | None:
    path = Path(artifacts_root) / experiment_name / "train_state.json"
    if not path.exists():
        return None
    return StreamTrainState(**json.loads(path.read_text(encoding="utf-8")))


def _save_stream_state(output_dir: Path, state: StreamTrainState) -> None:
    _state_dict_path(output_dir).write_text(json.dumps(state.__dict__, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_manifest_path(output_dir: Path) -> Path:
    return output_dir / "run_manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_run_manifest(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    state: StreamTrainState | None,
    best_checkpoint: Path,
    latest_checkpoint: Path | None = None,
) -> None:
    prepared_dir = Path(config.data.prepared_dir) if config.data.prepared_dir else None
    schema_path = prepared_dir / "schema.json" if prepared_dir else None
    transforms_path = prepared_dir / "feature_transforms.json" if prepared_dir else None
    payload = {
        "experiment_name": config.experiment_name,
        "dataset_type": config.data.dataset_type,
        "prepared_dir": str(prepared_dir) if prepared_dir else None,
        "hf_repo_id": config.data.hf_repo_id,
        "hf_config_name": config.data.hf_config_name,
        "hf_split": config.data.hf_split,
        "hf_streaming": config.data.hf_streaming,
        "schema_path": str(schema_path) if schema_path and schema_path.exists() else None,
        "schema_sha256": _sha256_file(schema_path) if schema_path and schema_path.exists() else None,
        "feature_transforms_path": str(transforms_path) if transforms_path and transforms_path.exists() else None,
        "feature_transforms_sha256": _sha256_file(transforms_path) if transforms_path and transforms_path.exists() else None,
        "best_checkpoint": str(best_checkpoint),
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else None,
        "state": state.__dict__ if state is not None else None,
        "updated_at_utc": _utc_now(),
    }
    manifest_path = _run_manifest_path(output_dir)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["created_at_utc"] = existing.get("created_at_utc", payload["updated_at_utc"])
    else:
        payload["created_at_utc"] = payload["updated_at_utc"]
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_progress(output_dir: Path, payload: dict[str, object]) -> None:
    path = _progress_log_path(output_dir)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    *,
    num_numeric: int,
    num_categorical: int,
    num_text: int,
    output_dim: int,
    episode_mode: bool,
    state: StreamTrainState | None = None,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "num_numeric": num_numeric,
            "num_categorical": num_categorical,
            "num_text": num_text,
            "output_dim": output_dim,
            "episode_mode": episode_mode,
            "train_state": state.__dict__ if state is not None else None,
        },
        path,
    )


def _load_stream_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> StreamTrainState | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    state_payload = payload.get("train_state")
    return StreamTrainState(**state_payload) if state_payload else None


def _evaluate_model(
    model: nn.Module,
    loader,
    device: torch.device,
    criterion: nn.Module,
    config: ExperimentConfig,
    output_dim: int,
    *,
    use_episodes: bool,
) -> EpochResult:
    run_epoch = _run_episode_epoch if use_episodes else _run_epoch  # type: ignore[assignment]
    return run_epoch(model, loader, device, criterion, None, config, output_dim)


def _train_streaming(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    output_dir: Path,
    checkpoint_path: Path,
    *,
    num_numeric: int,
    num_categorical: int,
    num_text: int,
    output_dim: int,
    use_episodes: bool,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    max_steps = config.training.max_steps
    if max_steps is None or max_steps < 1:
        raise ValueError("Streaming training requires training.max_steps >= 1.")

    state = StreamTrainState()
    latest_checkpoint = _latest_checkpoint_path(output_dir)
    if config.training.resume:
        restored = _load_stream_checkpoint(latest_checkpoint, model, optimizer)
        if restored is not None:
            state = restored
    _update_run_manifest(output_dir, config, state=state, best_checkpoint=checkpoint_path, latest_checkpoint=latest_checkpoint)

    model.train(True)
    train_iter = iter(train_loader)
    recent_losses: list[float] = []
    accumulation_steps = max(int(config.training.gradient_accumulation_steps), 1)
    optimizer.zero_grad(set_to_none=True)

    while state.global_step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = _move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=_autocast_dtype(config), enabled=_use_amp(config, device)):
            logits = model(batch)
            _ls = getattr(config.training, 'label_smoothing', 0.0)
            raw_loss = _compute_loss(config.task.problem_type, output_dim, criterion, logits, batch.y, _ls)
            loss = raw_loss / accumulation_steps
        if scaler is not None and _use_amp(config, device):
            scaler.scale(loss).backward()
        else:
            loss.backward()

        state.global_step += 1
        state.rows_seen += int(batch.y.shape[0])
        state.source_rows_seen += int(batch.y.shape[0])
        if state.global_step % accumulation_steps == 0 or state.global_step == max_steps:
            if scaler is not None and _use_amp(config, device):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        recent_losses.append(float(raw_loss.item()))
        if len(recent_losses) > max(config.training.log_interval, 100):
            recent_losses.pop(0)

        if state.global_step % config.training.log_interval == 0:
            avg_loss = float(np.mean(recent_losses))
            print(f"step={state.global_step} rows={state.rows_seen} train_loss={avg_loss:.4f}")
            _append_progress(
                output_dir,
                {
                    "event": "train_step",
                    "step": state.global_step,
                    "rows_seen": state.rows_seen,
                    "train_loss": avg_loss,
                },
            )

        if state.global_step % config.training.val_interval_steps == 0 or state.global_step == max_steps:
            val_result = _evaluate_model(
                model,
                val_loader,
                device,
                criterion,
                config,
                output_dim,
                use_episodes=use_episodes,
            )
            print(
                f"step={state.global_step} rows={state.rows_seen} "
                f"val_loss={val_result.loss:.4f} val_metrics={val_result.metrics}"
            )
            _append_progress(
                output_dir,
                {
                    "event": "validation",
                    "step": state.global_step,
                    "rows_seen": state.rows_seen,
                    "val_loss": val_result.loss,
                    "val_metrics": val_result.metrics,
                },
            )
            if val_result.loss < state.best_val_loss:
                state.best_val_loss = val_result.loss
                state.patience = 0
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    config,
                    num_numeric=num_numeric,
                    num_categorical=num_categorical,
                    num_text=num_text,
                    output_dim=output_dim,
                    episode_mode=use_episodes,
                    state=state,
                )
            else:
                state.patience += 1
                if state.patience >= config.training.early_stopping_patience:
                    break

        if state.global_step % config.training.checkpoint_interval_steps == 0 or state.global_step == max_steps:
            _save_checkpoint(
                latest_checkpoint,
                model,
                optimizer,
                config,
                num_numeric=num_numeric,
                num_categorical=num_categorical,
                num_text=num_text,
                output_dim=output_dim,
                episode_mode=use_episodes,
                state=state,
            )
            _save_stream_state(output_dir, state)
            _update_run_manifest(output_dir, config, state=state, best_checkpoint=checkpoint_path, latest_checkpoint=latest_checkpoint)

    _save_checkpoint(
        latest_checkpoint,
        model,
        optimizer,
        config,
        num_numeric=num_numeric,
        num_categorical=num_categorical,
        num_text=num_text,
        output_dim=output_dim,
        episode_mode=use_episodes,
        state=state,
    )
    _save_stream_state(output_dir, state)
    _update_run_manifest(output_dir, config, state=state, best_checkpoint=checkpoint_path, latest_checkpoint=latest_checkpoint)
    return {"best_val_loss": state.best_val_loss, "checkpoint": str(checkpoint_path), "latest_checkpoint": str(latest_checkpoint)}


def _build_scheduler(
    config: ExperimentConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build an optional learning rate scheduler based on config."""
    sched_kind = getattr(config.training, "scheduler", "none")
    if sched_kind == "none":
        return None
    if sched_kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.max_epochs,
            eta_min=getattr(config.training, "lr_min", 1e-6),
        )
    if sched_kind == "cosine_warmup":
        warmup_epochs = getattr(config.training, "warmup_epochs", 2)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(config.training.max_epochs - warmup_epochs, 1),
            eta_min=getattr(config.training, "lr_min", 1e-6),
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
    raise ValueError(f"Unsupported scheduler kind '{sched_kind}'.")


def train(
    config: ExperimentConfig,
    *,
    pretrained_trunk_path: str | Path | None = None,
) -> dict[str, float]:
    """Run a training experiment.

    Parameters
    ----------
    config:
        Full experiment configuration.
    pretrained_trunk_path:
        Optional path to a previous checkpoint whose *matching* parameter tensors
        will be copied into the freshly-initialised model before training begins.
        Shape mismatches (e.g. different number of features) are silently skipped so
        the shared transformer backbone transfers while dataset-specific heads are
        re-initialised.  Useful in curriculum training to warm-start from the best
        checkpoint of the previous dataset.
    """
    set_seed(config.seed)
    if config.data.dataset_type == "hf_stream" and config.training.resume:
        resumed_state = _load_stream_state_from_artifacts(config.experiment_name, config.artifacts_root)
        if resumed_state is not None:
            config.data.hf_skip_rows = int(resumed_state.source_rows_seen)
    device = torch.device(config.training.device)
    train_loader, val_loader, num_numeric, num_categorical, num_text, output_dim = build_dataloaders(config)
    effective_output_dim = 1 if config.task.problem_type in {"binary", "regression"} else output_dim

    use_episodes = config.episode.enabled

    if use_episodes:
        model: nn.Module = EpisodicTabularTransformer(
            config, num_numeric, num_categorical, num_text, effective_output_dim
        ).to(device)
    else:
        model = TabularTransformer(
            config, num_numeric, num_categorical, num_text, effective_output_dim
        ).to(device)

    trunk_summary: dict[str, object] | None = None
    if pretrained_trunk_path is not None:
        trunk_summary = load_trunk_weights(model, pretrained_trunk_path, device=device)

    criterion = _make_criterion(config.task.problem_type, effective_output_dim, getattr(config.training, 'label_smoothing', 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=_use_amp(config, device))

    # Build learning rate scheduler
    scheduler = _build_scheduler(config, optimizer)

    best_val = float("inf")
    patience = 0
    output_dir = Path(config.artifacts_root) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    _update_run_manifest(output_dir, config, state=None, best_checkpoint=checkpoint_path)

    run_epoch = _run_episode_epoch if use_episodes else _run_epoch  # type: ignore[assignment]

    if config.data.dataset_type == "hf_stream":
        result = _train_streaming(
            model,
            train_loader,
            val_loader,
            device,
            criterion,
            optimizer,
            config,
            output_dir,
            checkpoint_path,
            num_numeric=num_numeric,
            num_categorical=num_categorical,
            num_text=num_text,
            output_dim=effective_output_dim,
            use_episodes=use_episodes,
            scaler=scaler,
        )
        if trunk_summary is not None:
            result["trunk_transferred"] = trunk_summary["transferred_count"]
            result["trunk_source"] = trunk_summary["checkpoint"]
        return result

    for epoch in range(1, config.training.max_epochs + 1):
        train_result = run_epoch(model, train_loader, device, criterion, optimizer, config, effective_output_dim, scaler)
        val_result = run_epoch(model, val_loader, device, criterion, None, config, effective_output_dim, None)
        if scheduler is not None:
            scheduler.step()
        print(
            f"epoch={epoch} "
            f"train_loss={train_result.loss:.4f} "
            f"val_loss={val_result.loss:.4f} "
            f"val_metrics={val_result.metrics}"
        )

        if val_result.loss < best_val:
            best_val = val_result.loss
            patience = 0
            _save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                config,
                num_numeric=num_numeric,
                num_categorical=num_categorical,
                num_text=num_text,
                output_dim=effective_output_dim,
                episode_mode=use_episodes,
            )
        else:
            patience += 1
            if patience >= config.training.early_stopping_patience:
                break

    result = {"best_val_loss": best_val, "checkpoint": str(checkpoint_path)}
    if trunk_summary is not None:
        result["trunk_transferred"] = trunk_summary["transferred_count"]  # type: ignore[assignment]
        result["trunk_source"] = trunk_summary["checkpoint"]  # type: ignore[assignment]
    return result
