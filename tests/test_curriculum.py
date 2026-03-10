"""Tests for curriculum training: queue, ledger, and trunk weight transfer."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tabula.cli import _seed_curriculum_queue_from_hf_search
from tabula.training.curriculum import (
    CurriculumEntry,
    CurriculumLedger,
    CurriculumQueue,
    LedgerSession,
    ledger_path,
    queue_path,
)
from tabula.training.trunk import load_trunk_weights


def _make_entry(dataset_id: str = "ds_a", priority: int = 100, status: str = "pending") -> CurriculumEntry:
    entry = CurriculumEntry(
        dataset_id=dataset_id,
        prepared_dir=f"data/processed/{dataset_id}",
        hf_repo_id=f"org/{dataset_id}",
        steps_per_cycle=500,
        max_total_steps=2000,
        priority=priority,
    )
    entry.status = status
    return entry


class TestCurriculumEntry:
    def test_effective_experiment_name_default(self) -> None:
        assert _make_entry("my_ds").effective_experiment_name() == "curriculum_my_ds"

    def test_effective_experiment_name_override(self) -> None:
        entry = _make_entry("my_ds")
        entry.experiment_name = "custom_exp"
        assert entry.effective_experiment_name() == "custom_exp"

    def test_remaining_steps(self) -> None:
        entry = _make_entry()
        entry.total_steps = 800
        assert entry.remaining_steps() == 1200

    def test_remaining_steps_exhausted(self) -> None:
        entry = _make_entry()
        entry.total_steps = 2500
        assert entry.remaining_steps() == 0

    def test_next_cycle_target_normal(self) -> None:
        entry = _make_entry()
        entry.total_steps = 600
        assert entry.next_cycle_target() == 1100

    def test_next_cycle_target_clamp(self) -> None:
        entry = _make_entry()
        entry.total_steps = 1800
        assert entry.next_cycle_target() == 2000


class TestCurriculumQueue:
    def test_roundtrip(self, tmp_path: Path) -> None:
        queue_file = tmp_path / "queue.json"
        queue = CurriculumQueue([_make_entry("a"), _make_entry("b")])
        queue.save(queue_file)
        loaded = CurriculumQueue.load(queue_file)
        assert len(loaded.entries) == 2
        assert loaded.entries[0].dataset_id == "a"

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert CurriculumQueue.load(tmp_path / "nonexistent.json").entries == []

    def test_next_pending_priority_order(self) -> None:
        queue = CurriculumQueue([_make_entry("a", priority=50), _make_entry("b", priority=10), _make_entry("c", priority=200)])
        assert queue.next_pending().dataset_id == "b"

    def test_next_pending_skips_done(self) -> None:
        queue = CurriculumQueue([_make_entry("done_ds", status="done"), _make_entry("pending_ds", priority=5)])
        assert queue.next_pending().dataset_id == "pending_ds"

    def test_next_pending_empty_when_all_done(self) -> None:
        assert CurriculumQueue([_make_entry("x", status="done")]).next_pending() is None

    def test_add_duplicate_raises(self) -> None:
        queue = CurriculumQueue([_make_entry("x")])
        with pytest.raises(ValueError, match="already exists"):
            queue.add(_make_entry("x"))

    def test_mark_in_progress(self) -> None:
        queue = CurriculumQueue([_make_entry("ds")])
        queue.mark_in_progress("ds")
        assert queue.get("ds").status == "in_progress"

    def test_mark_done(self) -> None:
        queue = CurriculumQueue([_make_entry("ds")])
        queue.mark_done("ds")
        assert queue.get("ds").status == "done"

    def test_reset_to_pending(self) -> None:
        queue = CurriculumQueue([_make_entry("ds", status="failed")])
        queue.reset_to_pending("ds")
        assert queue.get("ds").status == "pending"

    def test_update_progress_marks_done_when_exhausted(self) -> None:
        queue = CurriculumQueue([_make_entry("ds")])
        queue.get("ds").status = "in_progress"
        queue.update_progress("ds", steps_delta=2000, rows_delta=1000, best_val_loss=0.5, session_timestamp="now")
        assert queue.get("ds").status == "done"
        assert queue.get("ds").total_steps == 2000

    def test_update_progress_keeps_pending_when_not_exhausted(self) -> None:
        queue = CurriculumQueue([_make_entry("ds")])
        queue.get("ds").status = "in_progress"
        queue.update_progress("ds", steps_delta=500, rows_delta=200, best_val_loss=0.8, session_timestamp="now")
        assert queue.get("ds").status == "pending"
        assert queue.get("ds").total_steps == 500

    def test_update_progress_tracks_best_val_loss(self) -> None:
        queue = CurriculumQueue([_make_entry("ds")])
        queue.update_progress("ds", steps_delta=0, rows_delta=0, best_val_loss=0.9, session_timestamp="t1")
        queue.update_progress("ds", steps_delta=0, rows_delta=0, best_val_loss=0.4, session_timestamp="t2")
        queue.update_progress("ds", steps_delta=0, rows_delta=0, best_val_loss=0.7, session_timestamp="t3")
        assert queue.get("ds").best_val_loss == pytest.approx(0.4)

    def test_summary_structure(self) -> None:
        summary = CurriculumQueue([_make_entry("ds")]).summary()
        assert isinstance(summary, list)
        assert summary[0]["dataset_id"] == "ds"
        assert "status" in summary[0]


class TestCurriculumLedger:
    def _make_session(self, dataset_id: str = "ds", ckpt: str | None = None) -> LedgerSession:
        return LedgerSession(
            session_id=LedgerSession.new_session_id(),
            dataset_id=dataset_id,
            experiment_name=f"curriculum_{dataset_id}",
            started_at=LedgerSession.utc_now(),
            ended_at=LedgerSession.utc_now(),
            steps_trained=100,
            cumulative_steps=100,
            rows_seen=2000,
            cumulative_rows=2000,
            exit_reason="max_steps",
            best_val_loss=0.5,
            checkpoint_path=ckpt,
        )

    def test_append_and_load(self, tmp_path: Path) -> None:
        ledger_file = tmp_path / "ledger.jsonl"
        session = self._make_session()
        CurriculumLedger.append(session, ledger_file)
        CurriculumLedger.append(session, ledger_file)
        assert len(CurriculumLedger.load(ledger_file)) == 2

    def test_load_empty_returns_empty(self, tmp_path: Path) -> None:
        assert CurriculumLedger.load(tmp_path / "ledger.jsonl") == []

    def test_latest_best_checkpoint_returns_existing(self, tmp_path: Path) -> None:
        ledger_file = tmp_path / "ledger.jsonl"
        checkpoint = tmp_path / "best.pt"
        checkpoint.touch()
        CurriculumLedger.append(self._make_session(ckpt=str(checkpoint)), ledger_file)
        assert CurriculumLedger.latest_best_checkpoint(ledger_file) == str(checkpoint)

    def test_latest_best_checkpoint_skips_missing_files(self, tmp_path: Path) -> None:
        ledger_file = tmp_path / "ledger.jsonl"
        CurriculumLedger.append(self._make_session(ckpt="/nonexistent/best.pt"), ledger_file)
        assert CurriculumLedger.latest_best_checkpoint(ledger_file) is None

    def test_helper_paths(self, tmp_path: Path) -> None:
        assert queue_path(tmp_path).name == "curriculum_queue.json"
        assert ledger_path(tmp_path).name == "curriculum_ledger.jsonl"


def _save_fake_checkpoint(path: Path, model: nn.Module) -> None:
    torch.save({"model_state_dict": model.state_dict()}, path)


class TestLoadTrunkWeights:
    def test_identical_model_full_transfer(self, tmp_path: Path) -> None:
        model_a = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        model_b = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        checkpoint = tmp_path / "best.pt"
        _save_fake_checkpoint(checkpoint, model_a)
        summary = load_trunk_weights(model_b, checkpoint, verbose=False)
        assert summary["transferred_count"] == len(list(model_a.state_dict()))
        assert summary["skipped_shape_count"] == 0
        for key in model_a.state_dict():
            assert torch.allclose(model_a.state_dict()[key], model_b.state_dict()[key])

    def test_shape_mismatch_skipped(self, tmp_path: Path) -> None:
        model_a = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        model_b = nn.Sequential(nn.Linear(4, 16), nn.Linear(16, 2))
        checkpoint = tmp_path / "best.pt"
        _save_fake_checkpoint(checkpoint, model_a)
        summary = load_trunk_weights(model_b, checkpoint, verbose=False)
        assert summary["skipped_shape_count"] > 0
        assert summary["transferred_count"] >= 0

    def test_missing_key_skipped(self, tmp_path: Path) -> None:
        model_a = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        model_b = nn.Linear(4, 8)
        checkpoint = tmp_path / "best.pt"
        _save_fake_checkpoint(checkpoint, model_a)
        summary = load_trunk_weights(model_b, checkpoint, verbose=False)
        assert summary["skipped_missing_count"] > 0

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_trunk_weights(nn.Linear(4, 2), tmp_path / "nonexistent.pt", verbose=False)

    def test_model_unchanged_when_all_shapes_mismatch(self, tmp_path: Path) -> None:
        model_a = nn.Linear(4, 8)
        model_b = nn.Linear(100, 200)
        original_weight = model_b.weight.detach().clone()
        checkpoint = tmp_path / "best.pt"
        _save_fake_checkpoint(checkpoint, model_a)
        load_trunk_weights(model_b, checkpoint, verbose=False)
        assert torch.allclose(model_b.weight, original_weight)


def test_seed_curriculum_queue_from_hf_search(monkeypatch, tmp_path: Path) -> None:
    import tabula.data as data_mod

    def fake_search_huggingface_datasets(query=None, task_category=None, limit=20, sort="downloads"):
        assert task_category in {"tabular-classification", "tabular-regression"}
        return [SimpleNamespace(repo_id=f"acme/{task_category.replace('-', '_')}_demo", downloads=10, likes=1)]

    def fake_bootstrap_huggingface_stream_sample(repo_id, output_root="data/raw", dataset_id=None, **kwargs):
        raw_dir = Path(output_root) / (dataset_id or "dataset")
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "train.csv").write_text("x,target\n1,yes\n2,no\n", encoding="utf-8")
        (raw_dir / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "id": dataset_id,
                    "title": repo_id,
                    "provider": "huggingface",
                    "source_type": "dataset",
                    "external_ref": repo_id,
                    "source_url": f"https://huggingface.co/datasets/{repo_id}",
                    "task_type": "binary",
                    "target_column": "target",
                    "train_file": "train.csv",
                    "notes": "",
                }
            ),
            encoding="utf-8",
        )
        return raw_dir

    def fake_prepare_dataset(dataset_id, raw_root="data/raw", processed_root="data/processed", **kwargs):
        processed_dir = Path(processed_root) / dataset_id
        processed_dir.mkdir(parents=True, exist_ok=True)
        (processed_dir / "train_config.json").write_text(json.dumps({"experiment_name": dataset_id}), encoding="utf-8")
        return SimpleNamespace(processed_dir=str(processed_dir))

    monkeypatch.setattr(data_mod, "search_huggingface_datasets", fake_search_huggingface_datasets)
    monkeypatch.setattr(data_mod, "bootstrap_huggingface_stream_sample", fake_bootstrap_huggingface_stream_sample)
    monkeypatch.setattr(data_mod, "prepare_dataset", fake_prepare_dataset)

    args = SimpleNamespace(
        artifacts_root=str(tmp_path / "artifacts"),
        raw_root=str(tmp_path / "data" / "raw"),
        processed_root=str(tmp_path / "data" / "processed"),
        query=None,
        task_categories=["tabular-classification", "tabular-regression"],
        limit=5,
        sort="downloads",
        bootstrap_rows=32,
        shuffle_buffer_size=128,
        seed=42,
        val_fraction=0.1,
        test_fraction=0.1,
        keep_identifiers=False,
        no_feature_engineering=False,
        steps_per_cycle=25,
        max_total_steps=250,
        priority_base=100,
        max_new_datasets=None,
    )

    summary = _seed_curriculum_queue_from_hf_search(args)
    queue = CurriculumQueue.load(queue_path(args.artifacts_root))

    assert summary["added_count"] == 2
    assert summary["failed_count"] == 0
    assert len(queue.entries) == 2
    assert queue.entries[0].steps_per_cycle == 25
    assert queue.entries[0].max_total_steps == 250
