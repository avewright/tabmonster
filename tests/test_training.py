"""Training-side tests: episode sampling, episodic forward pass, and smoke runs.

These complement the existing data-pipeline tests and address the previous gap
around training coverage.  All tests use fully synthetic, tiny configurations
so they finish quickly on CPU with no external data.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tabula.config import (
    DataConfig,
    EpisodeConfig,
    ExperimentConfig,
    ModelConfig,
    TaskConfig,
    TrainingConfig,
)
from tabula.data.datasets import TabularBatch, TabularDataset, collate_tabular
from tabula.data.episodes import EpisodeBatch, sample_episode_batch
from tabula.models.transformer import EpisodicTabularTransformer, TabularTransformer
from tabula.training.engine import train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_config(
    *,
    episode_enabled: bool = False,
    support_size: int = 32,
    query_size: int = 32,
    replace: bool = False,
    num_numeric: int = 4,
    num_categorical: int = 2,
    train_size: int = 128,
    val_size: int = 64,
    batch_size: int = 64,
    max_epochs: int = 1,
    num_classes: int = 2,
) -> ExperimentConfig:
    """Return a fully in-memory synthetic ExperimentConfig suitable for fast tests."""
    return ExperimentConfig(
        experiment_name="test_smoke",
        seed=0,
        task=TaskConfig(mode="finetune", problem_type="binary", target_column="target"),
        data=DataConfig(
            dataset_type="synthetic",
            num_numeric_features=num_numeric,
            num_categorical_features=num_categorical,
            categorical_cardinality=8,
            num_classes=num_classes,
            train_size=train_size,
            val_size=val_size,
            batch_size=batch_size,
            num_workers=0,
            standardize_numeric=True,
        ),
        model=ModelConfig(
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            feature_token_dropout=0.0,
            norm="layernorm",
            ffn_activation="gelu",
            max_categories=16,
        ),
        training=TrainingConfig(
            device="cpu",
            max_epochs=max_epochs,
            lr=1e-3,
            weight_decay=0.0,
            grad_clip_norm=1.0,
            log_interval=9999,
            early_stopping_patience=5,
        ),
        episode=EpisodeConfig(
            enabled=episode_enabled,
            support_size=support_size,
            query_size=query_size,
            sample_with_replacement=replace,
        ),
    )


def _make_flat_batch(
    n_rows: int = 100,
    num_numeric: int = 4,
    num_categorical: int = 2,
) -> TabularBatch:
    """Build a minimal TabularBatch with random data, on CPU."""
    n_num = max(num_numeric, 0)
    n_cat = max(num_categorical, 0)
    ds = TabularDataset(
        x_num=np.random.randn(n_rows, n_num).astype(np.float32),
        x_cat=np.random.randint(0, 8, size=(n_rows, n_cat)).astype(np.int64),
        y=np.random.randint(0, 2, size=n_rows).astype(np.int64),
    )
    return collate_tabular([ds[i] for i in range(n_rows)])


# ---------------------------------------------------------------------------
# Episode-sampling tests
# ---------------------------------------------------------------------------


class TestSampleEpisodeBatch:
    def test_output_shapes_without_replacement(self):
        batch = _make_flat_batch(n_rows=120)
        episode = sample_episode_batch(batch, support_size=40, query_size=30)

        assert isinstance(episode, EpisodeBatch)
        assert episode.support.x_num.shape[0] == 40
        assert episode.query.x_num.shape[0] == 30

    def test_output_shapes_with_replacement(self):
        batch = _make_flat_batch(n_rows=20)
        episode = sample_episode_batch(
            batch, support_size=50, query_size=50, sample_with_replacement=True
        )
        assert episode.support.x_num.shape[0] == 50
        assert episode.query.x_num.shape[0] == 50

    def test_no_overlap_without_replacement(self):
        """Support and query should draw from disjoint row indices."""
        batch = _make_flat_batch(n_rows=200)
        # Use y values as a proxy for row identity (unique per original index after
        # assigning incrementing targets).
        y_arr = torch.arange(200, dtype=torch.long)
        batch = TabularBatch(
            x_num=batch.x_num,
            x_cat=batch.x_cat,
            x_text_token_ids=batch.x_text_token_ids,
            x_text_values=batch.x_text_values,
            x_num_mask=batch.x_num_mask,
            x_cat_mask=batch.x_cat_mask,
            x_text_mask=batch.x_text_mask,
            num_schema_texts=batch.num_schema_texts,
            cat_schema_texts=batch.cat_schema_texts,
            text_schema_texts=batch.text_schema_texts,
            num_name_token_ids=batch.num_name_token_ids,
            cat_name_token_ids=batch.cat_name_token_ids,
            text_name_token_ids=batch.text_name_token_ids,
            num_profile_vectors=batch.num_profile_vectors,
            cat_profile_vectors=batch.cat_profile_vectors,
            text_profile_vectors=batch.text_profile_vectors,
            y=y_arr,
        )
        episode = sample_episode_batch(batch, support_size=60, query_size=60)
        support_ids = set(episode.support.y.tolist())
        query_ids = set(episode.query.y.tolist())
        assert len(support_ids & query_ids) == 0, "Support and query should be disjoint."

    def test_error_on_empty_batch(self):
        batch = _make_flat_batch(n_rows=10)
        with pytest.raises(ValueError, match="empty batch"):
            # Create truly empty batch
            empty = TabularBatch(
                x_num=batch.x_num[:0],
                x_cat=batch.x_cat[:0],
                x_text_token_ids=batch.x_text_token_ids[:0],
                x_text_values=[],
                x_num_mask=batch.x_num_mask[:0],
                x_cat_mask=batch.x_cat_mask[:0],
                x_text_mask=batch.x_text_mask[:0],
                num_schema_texts=batch.num_schema_texts,
                cat_schema_texts=batch.cat_schema_texts,
                text_schema_texts=batch.text_schema_texts,
                num_name_token_ids=batch.num_name_token_ids,
                cat_name_token_ids=batch.cat_name_token_ids,
                text_name_token_ids=batch.text_name_token_ids,
                num_profile_vectors=batch.num_profile_vectors,
                cat_profile_vectors=batch.cat_profile_vectors,
                text_profile_vectors=batch.text_profile_vectors,
                y=batch.y[:0],
            )
            sample_episode_batch(empty, support_size=4, query_size=4)

    def test_error_without_replacement_when_too_few_rows(self):
        batch = _make_flat_batch(n_rows=10)
        with pytest.raises(ValueError, match="replacement"):
            sample_episode_batch(batch, support_size=8, query_size=8, sample_with_replacement=False)

    def test_labels_are_subset_of_original(self):
        """Each episode row y value should appear in the original batch y values."""
        batch = _make_flat_batch(n_rows=100)
        episode = sample_episode_batch(batch, support_size=30, query_size=20)
        all_original = set(batch.y.tolist())
        assert set(episode.support.y.tolist()).issubset(all_original)
        assert set(episode.query.y.tolist()).issubset(all_original)

    def test_feature_column_count_preserved(self):
        batch = _make_flat_batch(n_rows=80, num_numeric=6, num_categorical=3)
        episode = sample_episode_batch(batch, support_size=20, query_size=20)
        assert episode.support.x_num.shape[1] == 6
        assert episode.support.x_cat.shape[1] == 3
        assert episode.query.x_num.shape[1] == 6
        assert episode.query.x_cat.shape[1] == 3

    def test_deterministic_with_generator(self):
        batch = _make_flat_batch(n_rows=100)
        g1 = torch.Generator()
        g1.manual_seed(7)
        ep1 = sample_episode_batch(batch, support_size=20, query_size=20, generator=g1)

        g2 = torch.Generator()
        g2.manual_seed(7)
        ep2 = sample_episode_batch(batch, support_size=20, query_size=20, generator=g2)

        assert torch.equal(ep1.support.y, ep2.support.y)
        assert torch.equal(ep1.query.y, ep2.query.y)


# ---------------------------------------------------------------------------
# EpisodicTabularTransformer forward-pass tests
# ---------------------------------------------------------------------------


class TestEpisodicTransformerForward:
    def _build_model_and_episode(
        self,
        support: int = 16,
        query: int = 8,
        num_numeric: int = 4,
        num_categorical: int = 2,
    ):
        cfg = _make_tiny_config(num_numeric=num_numeric, num_categorical=num_categorical)
        model = EpisodicTabularTransformer(cfg, num_numeric, num_categorical, 0, 1)
        model.eval()
        flat_batch = _make_flat_batch(
            n_rows=support + query + 10, num_numeric=num_numeric, num_categorical=num_categorical
        )
        episode = sample_episode_batch(flat_batch, support_size=support, query_size=query)
        return model, episode

    def test_output_shape(self):
        model, episode = self._build_model_and_episode(support=16, query=8)
        with torch.no_grad():
            logits = model(episode)
        # binary → output_dim=1 → (query, 1)
        assert logits.shape == (8, 1)

    def test_output_is_finite(self):
        model, episode = self._build_model_and_episode()
        with torch.no_grad():
            logits = model(episode)
        assert torch.isfinite(logits).all()

    def test_type_error_on_plain_batch(self):
        cfg = _make_tiny_config()
        model = EpisodicTabularTransformer(cfg, 4, 2, 0, 1)
        plain = _make_flat_batch()
        with pytest.raises(TypeError, match="EpisodeBatch"):
            model(plain)  # type: ignore[arg-type]

    def test_multiclass_output_shape(self):
        cfg = _make_tiny_config(num_classes=4)
        # For multiclass problem_type would be multiclass; here we just check
        # the model's output_dim parameter wiring.
        model = EpisodicTabularTransformer(cfg, 4, 2, 0, 4)
        flat_batch = _make_flat_batch(n_rows=60)
        episode = sample_episode_batch(flat_batch, support_size=20, query_size=10)
        with torch.no_grad():
            logits = model(episode)
        assert logits.shape == (10, 4)

    def test_support_context_changes_logits(self):
        """Using a different support set should produce different query logits."""
        cfg = _make_tiny_config()
        model = EpisodicTabularTransformer(cfg, 4, 2, 0, 1)
        model.eval()

        flat_batch = _make_flat_batch(n_rows=200)
        # Seed two separate episodes with the SAME query rows but different support.
        g = torch.Generator().manual_seed(0)
        ep_a = sample_episode_batch(flat_batch, 30, 20, generator=g)

        g2 = torch.Generator().manual_seed(99)
        # Replace query with ep_a's query so we isolate support difference.
        ep_b = EpisodeBatch(
            support=sample_episode_batch(flat_batch, 30, 1, generator=g2).support,
            query=ep_a.query,
        )

        with torch.no_grad():
            logits_a = model(ep_a)
            logits_b = model(ep_b)

        # The context vector differs → logits should differ (with overwhelming probability).
        assert not torch.allclose(logits_a, logits_b, atol=1e-6), (
            "Different support sets should yield different query logits."
        )


# ---------------------------------------------------------------------------
# Baseline (flat) training smoke test
# ---------------------------------------------------------------------------


def test_baseline_trainer_one_epoch(tmp_path, monkeypatch):
    """Train the standard TabularTransformer for one epoch on synthetic data."""
    monkeypatch.chdir(tmp_path)
    cfg = _make_tiny_config(episode_enabled=False, train_size=64, val_size=32, batch_size=32)
    result = train(cfg)
    assert "best_val_loss" in result
    assert result["best_val_loss"] < float("inf")
    checkpoint = tmp_path / "artifacts" / "test_smoke" / "best.pt"
    assert checkpoint.exists()
    ckpt = torch.load(str(checkpoint), weights_only=False)
    assert ckpt["episode_mode"] is False


def test_baseline_trainer_amp_flag_on_cpu(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _make_tiny_config(episode_enabled=False, train_size=64, val_size=32, batch_size=32)
    cfg.training.amp = True
    result = train(cfg)
    assert result["best_val_loss"] < float("inf")


# ---------------------------------------------------------------------------
# Episodic training smoke test
# ---------------------------------------------------------------------------


def test_episode_trainer_one_epoch(tmp_path, monkeypatch):
    """Train EpisodicTabularTransformer for one epoch with episode sampling."""
    monkeypatch.chdir(tmp_path)
    # batch_size must be >= support_size + query_size
    cfg = _make_tiny_config(
        episode_enabled=True,
        support_size=16,
        query_size=16,
        replace=False,
        train_size=128,
        val_size=64,
        batch_size=64,
        max_epochs=1,
    )
    result = train(cfg)
    assert "best_val_loss" in result
    assert result["best_val_loss"] < float("inf")
    checkpoint = tmp_path / "artifacts" / "test_smoke" / "best.pt"
    assert checkpoint.exists()
    ckpt = torch.load(str(checkpoint), weights_only=False)
    assert ckpt["episode_mode"] is True


def test_episode_trainer_with_replacement(tmp_path, monkeypatch):
    """Episode training with replacement handles batches smaller than support+query."""
    monkeypatch.chdir(tmp_path)
    # batch_size (32) < support_size + query_size (40+40=80) → needs replacement
    cfg = _make_tiny_config(
        episode_enabled=True,
        support_size=20,
        query_size=20,
        replace=True,
        train_size=64,
        val_size=32,
        batch_size=32,
        max_epochs=1,
    )
    result = train(cfg)
    assert result["best_val_loss"] < float("inf")
