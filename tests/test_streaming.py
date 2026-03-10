from __future__ import annotations

import json
import sys
import types

import pandas as pd
from tabula.config import load_stream_queue_config
from tabula.training.registry import alerts_path, dashboard_csv_path, dashboard_path, load_registry, reconcile_registry, update_registry_job

from tabula.config import DataConfig, ExperimentConfig, ModelConfig, TaskConfig, TrainingConfig
from tabula.data.cache import trim_cache_to_budget
from tabula.data.datasets import build_dataloaders
from tabula.training.engine import train


def _write_prepared_dir(base, dataset_id: str = "stream_demo") -> str:
    processed_dir = base / dataset_id
    processed_dir.mkdir(parents=True)
    train_frame = pd.DataFrame(
        [
            {"num": 1.0, "cat": "a", "target": "no"},
            {"num": 2.0, "cat": "b", "target": "yes"},
            {"num": 3.0, "cat": "a", "target": "no"},
            {"num": 4.0, "cat": "b", "target": "yes"},
        ]
    )
    val_frame = pd.DataFrame(
        [
            {"num": 1.5, "cat": "a", "target": "no"},
            {"num": 3.5, "cat": "b", "target": "yes"},
        ]
    )
    schema = {
        "target": {"column": "target", "problem_type": "binary", "classes": ["no", "yes"]},
        "numeric_features": [
            {
                "name": "num",
                "fill_value": 2.5,
                "mean": 2.5,
                "std": 1.118,
                "profile": {
                    "name": "num",
                    "name_tokens": ["num", "", "", ""],
                    "missing_ratio": 0.0,
                    "unique_ratio": 1.0,
                    "top_frequency_ratio": 0.25,
                    "avg_string_length": 3.0,
                    "parse_numeric_ratio": 1.0,
                    "parse_datetime_ratio": 0.0,
                    "type_probabilities": {"numeric": 1.0, "categorical": 0.0, "text": 0.0, "identifier": 0.0, "datetime": 0.0},
                },
            }
        ],
        "categorical_features": [
            {
                "name": "cat",
                "categories": ["a", "b"],
                "unknown_index": 0,
                "profile": {
                    "name": "cat",
                    "name_tokens": ["cat", "", "", ""],
                    "missing_ratio": 0.0,
                    "unique_ratio": 0.5,
                    "top_frequency_ratio": 0.5,
                    "avg_string_length": 1.0,
                    "parse_numeric_ratio": 0.0,
                    "parse_datetime_ratio": 0.0,
                    "type_probabilities": {"numeric": 0.0, "categorical": 1.0, "text": 0.0, "identifier": 0.0, "datetime": 0.0},
                },
            }
        ],
        "text_features": [],
        "metadata": {
            "profile_vector_fields": [],
            "type_names": ["numeric", "categorical", "text", "identifier", "datetime"],
            "name_token_count": 4,
            "text_token_count": 16,
        },
    }
    (processed_dir / "train.csv").write_text(train_frame.to_csv(index=False), encoding="utf-8")
    (processed_dir / "val.csv").write_text(val_frame.to_csv(index=False), encoding="utf-8")
    (processed_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (processed_dir / "train_config.json").write_text(json.dumps({"experiment_name": "stream_demo"}, indent=2), encoding="utf-8")
    return str(processed_dir)


def test_build_dataloaders_hf_stream(monkeypatch, tmp_path):
    processed_dir = _write_prepared_dir(tmp_path)

    class FakeStream:
        def shuffle(self, seed, buffer_size):
            return self

        def __iter__(self):
            yield {"num": 10.0, "cat": "a", "target": "no"}
            yield {"num": 11.0, "cat": "b", "target": "yes"}

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=lambda *args, **kwargs: FakeStream()))

    config = ExperimentConfig(
        experiment_name="stream_loader",
        task=TaskConfig(mode="finetune", problem_type="binary", target_column="target"),
        data=DataConfig(
            dataset_type="hf_stream",
            prepared_dir=processed_dir,
            val_path=str((tmp_path / "stream_demo" / "val.csv")),
            batch_size=2,
            hf_repo_id="acme/demo",
            hf_streaming=True,
        ),
        model=ModelConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0, feature_token_dropout=0.0, norm="layernorm", ffn_activation="gelu", max_categories=8, schema_encoder="hash"),
        training=TrainingConfig(device="cpu", max_steps=1, val_interval_steps=1, checkpoint_interval_steps=1),
    )
    train_loader, val_loader, num_numeric, num_categorical, num_text, output_dim = build_dataloaders(config)
    batch = next(iter(train_loader))
    assert batch.x_num.shape == (2, 1)
    assert batch.x_cat.shape == (2, 1)
    assert num_numeric == 1
    assert num_categorical == 1
    assert num_text == 0
    assert output_dim == 2
    assert len(next(iter(val_loader)).y) == 2


def test_stream_train_resume(monkeypatch, tmp_path):
    processed_dir = _write_prepared_dir(tmp_path)

    class FakeStream:
        def shuffle(self, seed, buffer_size):
            return self

        def __iter__(self):
            while True:
                yield {"num": 10.0, "cat": "a", "target": "no"}
                yield {"num": 11.0, "cat": "b", "target": "yes"}

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=lambda *args, **kwargs: FakeStream()))
    monkeypatch.chdir(tmp_path)

    config = ExperimentConfig(
        experiment_name="stream_resume",
        task=TaskConfig(mode="finetune", problem_type="binary", target_column="target"),
        data=DataConfig(
            dataset_type="hf_stream",
            prepared_dir=processed_dir,
            val_path=str((tmp_path / "stream_demo" / "val.csv")),
            batch_size=2,
            hf_repo_id="acme/demo",
            hf_streaming=True,
        ),
        model=ModelConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32, dropout=0.0, feature_token_dropout=0.0, norm="layernorm", ffn_activation="gelu", max_categories=8, schema_encoder="hash"),
        training=TrainingConfig(device="cpu", max_steps=2, val_interval_steps=1, checkpoint_interval_steps=1, log_interval=1, early_stopping_patience=3, resume=True),
    )
    first = train(config)
    assert "latest_checkpoint" in first
    manifest_path = tmp_path / "artifacts" / "stream_resume" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["hf_repo_id"] == "acme/demo"
    assert manifest["dataset_type"] == "hf_stream"
    config.training.max_steps = 4
    second = train(config)
    state_path = tmp_path / "artifacts" / "stream_resume" / "train_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["global_step"] == 4
    assert state["source_rows_seen"] == 8
    assert second["latest_checkpoint"].endswith("latest.pt")


def test_stream_queue_config_loader(tmp_path):
    path = tmp_path / "queue.json"
    path.write_text(
        json.dumps(
            {
                "sleep_seconds": 0,
                "max_cycles": 2,
                "stale_after_seconds": 45,
                "jobs": [
                    {
                        "prepared_dir": "data/processed/foo",
                        "repo_id": "acme/foo",
                        "steps_per_cycle": 10,
                        "max_total_steps": 100,
                        "weight": 2.5,
                        "max_retries": 4,
                        "retry_backoff_seconds": 9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    queue = load_stream_queue_config(path)
    assert queue.sleep_seconds == 0
    assert queue.max_cycles == 2
    assert queue.stale_after_seconds == 45
    assert queue.jobs[0].repo_id == "acme/foo"
    assert queue.jobs[0].weight == 2.5
    assert queue.jobs[0].max_retries == 4
    assert queue.jobs[0].retry_backoff_seconds == 9


def test_registry_updates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = update_registry_job(
        "demo_exp",
        status="running",
        prepared_dir="data/processed/foo",
        repo_id="acme/foo",
        config_name=None,
        split="train",
        target_max_steps=100,
        current_step=12,
        heartbeat_at_utc="2026-03-09T00:00:00+00:00",
        failure_count=2,
        last_error="boom",
        cooldown_until_utc="2026-03-09T00:01:00+00:00",
    )
    registry = load_registry()
    dashboard = json.loads(dashboard_path().read_text(encoding="utf-8"))
    dashboard_csv = dashboard_csv_path().read_text(encoding="utf-8")
    assert path.exists()
    assert registry["jobs"]["demo_exp"]["current_step"] == 12
    assert registry["jobs"]["demo_exp"]["status"] == "running"
    assert registry["jobs"]["demo_exp"]["failure_count"] == 2
    assert registry["jobs"]["demo_exp"]["last_error"] == "boom"
    assert dashboard["job_count"] == 1
    assert dashboard["by_status"]["running"] == 1
    assert "demo_exp" in dashboard_csv


def test_registry_reconcile_marks_orphaned(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = update_registry_job(
        "orphan_exp",
        status="running",
        prepared_dir="data/processed/foo",
        repo_id="acme/foo",
        config_name=None,
        split="train",
        target_max_steps=10,
        current_step=3,
        heartbeat_at_utc=None,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["jobs"]["orphan_exp"]["heartbeat_at_utc"] = None
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    reconcile_registry(stale_after_seconds=1)
    registry = load_registry()
    alerts = alerts_path().read_text(encoding="utf-8")
    assert registry["jobs"]["orphan_exp"]["status"] == "orphaned"
    assert "orphan_exp" in alerts


def test_trim_cache_to_budget(tmp_path):
    cache_dir = tmp_path / "hf_cache"
    old_dir = cache_dir / "old"
    new_dir = cache_dir / "new"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old_dir / "a.bin").write_bytes(b"x" * 1024)
    (new_dir / "b.bin").write_bytes(b"x" * 1024)
    result = trim_cache_to_budget(cache_dir, max_gb=0.000001)
    assert result["exists"] is True
    assert isinstance(result["removed"], list)
