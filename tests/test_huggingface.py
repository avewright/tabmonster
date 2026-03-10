from __future__ import annotations

import json
import sys
import types

from tabula.data import huggingface as hf


def test_huggingface_auth_status_prefers_env_file(monkeypatch):
    monkeypatch.setattr(
        hf,
        "load_repo_env_file",
        lambda: {"HF_TOKEN": "hf_12345678"},
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    status = hf.huggingface_auth_status()

    assert status["token_resolved"] is True
    assert status["token_source"] == ".env"
    assert status["token_hint"] == "hf_1...5678"


def test_search_huggingface_datasets_passes_token_to_hf_api(monkeypatch):
    calls: dict[str, object] = {}

    class FakeDatasetInfo:
        id = "acme/demo"
        downloads = 12
        likes = 3
        lastModified = "2026-03-09T00:00:00.000Z"
        tags = ["tabular"]

    class FakeHfApi:
        def __init__(self, token=None):
            calls["token"] = token

        def list_datasets(self, **kwargs):
            calls["kwargs"] = kwargs
            return [FakeDatasetInfo()]

    monkeypatch.setattr(hf, "_load_huggingface_token", lambda: "hf_test_token")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeHfApi))

    results = hf.search_huggingface_datasets(query="demo", limit=5)

    assert calls["token"] == "hf_test_token"
    assert calls["kwargs"] == {
        "filter": "task_categories:tabular-classification",
        "search": "demo",
        "sort": "downloads",
        "direction": -1,
        "limit": 5,
        "expand": ["downloads", "likes", "lastModified", "tags"],
    }
    assert results[0].repo_id == "acme/demo"


def test_fetch_huggingface_dataset_passes_token_to_load_dataset(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    class FakeDataset:
        info = types.SimpleNamespace(supervised_keys=None)
        features = {}

        def to_pandas(self):
            import pandas as pd

            return pd.DataFrame([{"x": 1, "target": 0}])

    def fake_load_dataset(repo_id, name=None, split=None, token=None):
        calls["repo_id"] = repo_id
        calls["name"] = name
        calls["split"] = split
        calls["token"] = token
        return FakeDataset()

    monkeypatch.setattr(hf, "_load_huggingface_token", lambda: "hf_test_token")
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    output_dir = hf.fetch_huggingface_dataset(
        "acme/demo",
        output_root=tmp_path,
        dataset_id="demo_ds",
        config_name="default",
        split="train",
        max_rows=10,
    )

    assert output_dir == tmp_path / "demo_ds"
    assert calls == {
        "repo_id": "acme/demo",
        "name": "default",
        "split": "train[:10]",
        "token": "hf_test_token",
    }
    assert (output_dir / "train.csv").exists()
    assert (output_dir / "dataset_manifest.json").exists()


def test_fetch_huggingface_dataset_infers_target_and_drops_nested_columns(monkeypatch, tmp_path):
    class FakeClassLabel:
        def __init__(self, names):
            self.names = names

    class FakeDataset:
        info = types.SimpleNamespace(supervised_keys=("feature_a", "label"))
        features = {
            "feature_a": object(),
            "label": FakeClassLabel(["no", "yes"]),
            "metadata": object(),
        }

        def to_pandas(self):
            import pandas as pd

            return pd.DataFrame(
                [
                    {"feature_a": 1.0, "label": "yes", "metadata": [1, 2]},
                    {"feature_a": 2.5, "label": "no", "metadata": [3, 4]},
                ]
            )

    def fake_load_dataset(repo_id, name=None, split=None, token=None):
        return FakeDataset()

    monkeypatch.setattr(hf, "_load_huggingface_token", lambda: "hf_test_token")
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    output_dir = hf.fetch_huggingface_dataset("acme/demo", output_root=tmp_path, dataset_id="demo_ds")

    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_column"] == "label"
    assert manifest["task_type"] == "binary"

    import pandas as pd

    saved_frame = pd.read_csv(output_dir / "train.csv")
    assert list(saved_frame.columns) == ["feature_a", "label"]


def test_bootstrap_huggingface_stream_sample_uses_streaming(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    class FakeClassLabel:
        def __init__(self, names):
            self.names = names

    class FakeStream:
        info = types.SimpleNamespace(supervised_keys=("feature_a", "label"))
        features = {
            "feature_a": object(),
            "label": FakeClassLabel(["no", "yes"]),
            "metadata": object(),
        }

        def shuffle(self, seed=None, buffer_size=None):
            calls["shuffle"] = {"seed": seed, "buffer_size": buffer_size}
            return self

        def __iter__(self):
            yield {"feature_a": 1.0, "label": "yes", "metadata": [1, 2]}
            yield {"feature_a": 2.0, "label": "no", "metadata": [3, 4]}
            yield {"feature_a": 3.0, "label": "yes", "metadata": [5, 6]}

    def fake_load_dataset(repo_id, name=None, split=None, streaming=None, token=None):
        calls["repo_id"] = repo_id
        calls["name"] = name
        calls["split"] = split
        calls["streaming"] = streaming
        calls["token"] = token
        return FakeStream()

    monkeypatch.setattr(hf, "_load_huggingface_token", lambda: "hf_test_token")
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))

    output_dir = hf.bootstrap_huggingface_stream_sample(
        "acme/demo",
        output_root=tmp_path,
        dataset_id="demo_stream",
        sample_rows=2,
        shuffle_buffer_size=50,
        seed=7,
    )

    assert output_dir == tmp_path / "demo_stream"
    assert calls["streaming"] is True
    assert calls["split"] == "train"
    assert calls["token"] == "hf_test_token"
    assert calls["shuffle"] == {"seed": 7, "buffer_size": 50}

    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_column"] == "label"
    assert manifest["task_type"] == "binary"

    import pandas as pd

    saved_frame = pd.read_csv(output_dir / "train.csv")
    assert len(saved_frame) == 2
    assert list(saved_frame.columns) == ["feature_a", "label"]
