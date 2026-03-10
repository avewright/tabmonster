from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tabula.data.prep import prepare_dataset


def _write_manifest(raw_dir, dataset_id: str, train_file: str) -> None:
    payload = {
        "id": dataset_id,
        "title": "Demo dataset",
        "provider": "local",
        "source_type": "dataset",
        "external_ref": dataset_id,
        "source_url": "https://example.com/demo",
        "task_type": None,
        "target_column": None,
        "train_file": train_file,
        "notes": "",
    }
    (raw_dir / "dataset_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_prepare_dataset_infers_target_and_persists_manifest(tmp_path):
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    dataset_id = "survival_demo"
    raw_dir = raw_root / dataset_id
    raw_dir.mkdir(parents=True)

    frame = pd.DataFrame(
        [
            {"passenger_id": 1, "fare": 7.25, "age": 22, "survived": 0},
            {"passenger_id": 2, "fare": 71.28, "age": 38, "survived": 1},
            {"passenger_id": 3, "fare": 7.92, "age": 26, "survived": 1},
            {"passenger_id": 4, "fare": 53.10, "age": 35, "survived": 1},
            {"passenger_id": 5, "fare": 8.05, "age": 35, "survived": 0},
            {"passenger_id": 6, "fare": 8.46, "age": 27, "survived": 0},
            {"passenger_id": 7, "fare": 51.86, "age": 54, "survived": 1},
            {"passenger_id": 8, "fare": 21.07, "age": 2, "survived": 1},
            {"passenger_id": 9, "fare": 11.13, "age": 27, "survived": 0},
            {"passenger_id": 10, "fare": 30.07, "age": 14, "survived": 1},
        ]
    )
    frame.to_csv(raw_dir / "train.csv", index=False)
    _write_manifest(raw_dir, dataset_id=dataset_id, train_file="train.csv")

    prepared = prepare_dataset(dataset_id, raw_root=raw_root, processed_root=processed_root, test_fraction=0.2)

    assert prepared.target_column == "survived"

    manifest = json.loads((raw_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_column"] == "survived"
    assert manifest["task_type"] == "binary"

    dataset_card = json.loads((processed_root / dataset_id / "dataset_card.json").read_text(encoding="utf-8"))
    assert dataset_card["original_target_column"] == "survived"
    assert dataset_card["tabular_inspection"]["target_source"] == "heuristic"


def test_prepare_dataset_raises_on_ambiguous_target_inference(tmp_path):
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    dataset_id = "ambiguous_demo"
    raw_dir = raw_root / dataset_id
    raw_dir.mkdir(parents=True)

    frame = pd.DataFrame(
        [
            {"feature_a": 0, "feature_b": 1, "flag_one": 0, "flag_two": 1},
            {"feature_a": 1, "feature_b": 0, "flag_one": 1, "flag_two": 0},
            {"feature_a": 0, "feature_b": 1, "flag_one": 0, "flag_two": 1},
            {"feature_a": 1, "feature_b": 0, "flag_one": 1, "flag_two": 0},
        ]
    )
    frame.to_csv(raw_dir / "train.csv", index=False)
    _write_manifest(raw_dir, dataset_id=dataset_id, train_file="train.csv")

    with pytest.raises(ValueError, match="Could not infer the target column confidently"):
        prepare_dataset(dataset_id, raw_root=raw_root, processed_root=processed_root)


def test_prepare_dataset_applies_train_only_feature_engineering(tmp_path):
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    dataset_id = "feature_demo"
    raw_dir = raw_root / dataset_id
    raw_dir.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    for i in range(48):
        if i < 24:
            segment = "core_a"
        elif i < 38:
            segment = "core_b"
        else:
            segment = f"rare_{i}"
        rows.append(
            {
                "customer_id": i + 1,
                "amount": float(np.exp((i % 8) + 1)),
                "segment": None if i % 11 == 0 else segment,
                "approved": int(i % 3 == 0 or i % 5 == 0),
            }
        )
    frame = pd.DataFrame(rows)
    frame.loc[[3, 17, 33], "amount"] = None
    frame.to_csv(raw_dir / "train.csv", index=False)
    _write_manifest(raw_dir, dataset_id=dataset_id, train_file="train.csv")

    prepared = prepare_dataset(
        dataset_id,
        raw_root=raw_root,
        processed_root=processed_root,
        seed=7,
        test_fraction=0.2,
    )

    processed_dir = processed_root / dataset_id
    train_frame = pd.read_csv(processed_dir / "train.csv")
    transform_spec = json.loads((processed_dir / "feature_transforms.json").read_text(encoding="utf-8"))
    dataset_card = json.loads((processed_dir / "dataset_card.json").read_text(encoding="utf-8"))

    assert prepared.transform_artifact_path.endswith("feature_transforms.json")
    assert "amount_is_missing" in train_frame.columns
    assert "amount_log1p" in train_frame.columns
    assert "segment_frequency" in train_frame.columns
    assert any(item["source"] == "segment" for item in transform_spec["rare_category_collapses"])
    assert any(item["source"] == "amount" for item in transform_spec["log_transforms"])
    assert dataset_card["feature_engineering_enabled"] is True
    assert dataset_card["feature_transform_artifact_path"].endswith("feature_transforms.json")
