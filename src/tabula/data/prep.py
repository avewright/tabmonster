from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.model_selection import train_test_split

from tabula.data.inspection import inspect_supervised_frame
from tabula.data.manifest import DatasetManifest, resolve_manifest, write_manifest
from tabula.data.schema import build_schema, write_schema


@dataclass(frozen=True)
class PreparedDataset:
    dataset_id: str
    raw_source_path: str
    processed_dir: str
    train_rows: int
    val_rows: int
    test_rows: int
    target_column: str
    numeric_columns: list[str]
    categorical_columns: list[str]
    config_path: str
    dataset_card_path: str
    transform_artifact_path: str


FEATURE_TRANSFORMS_FILENAME = "feature_transforms.json"
RARE_CATEGORY_TOKEN = "__RARE__"


def _normalize_name(name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return normalized or "column"


def _make_unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    output: list[str] = []
    for name in names:
        base = name
        suffix = seen.get(base, 0)
        while name in seen:
            suffix += 1
            name = f"{base}_{suffix}"
        seen[base] = suffix
        seen[name] = 0
        output.append(name)
    return output


def _normalize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    original_columns = list(frame.columns)
    normalized_columns = _make_unique([_normalize_name(column) for column in original_columns])
    mapping = dict(zip(original_columns, normalized_columns, strict=True))
    renamed = frame.rename(columns=mapping).copy()
    return renamed, mapping


def _resolve_source_file(raw_dir: Path, train_file: str | None) -> Path:
    if train_file:
        direct = raw_dir / train_file
        if direct.exists():
            return direct

        matches = sorted(raw_dir.rglob(train_file))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Found multiple candidates for `{train_file}` under `{raw_dir}`. "
                "Narrow the dataset layout manually before preparing it."
            )
        raise FileNotFoundError(
            f"Could not find `{train_file}` under `{raw_dir}`. "
            "Run `tabula data inspect` to see available files."
        )

    candidates: list[Path] = []
    for pattern in ("*.csv", "*.parquet", "*.jsonl", "*.tsv"):
        candidates.extend(sorted(raw_dir.rglob(pattern)))
    if len(candidates) == 1:
        return candidates[0]
    for preferred_name in ("train.csv", "training.csv", "train.parquet"):
        for candidate in candidates:
            if candidate.name.lower() == preferred_name:
                return candidate
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a supported tabular file under `{raw_dir}`. "
            "Run `tabula data inspect` to see available files."
        )
    raise RuntimeError(
        f"Found multiple candidate tabular files under `{raw_dir}`. "
        "Pass `--train-file` to select one."
    )


def _read_table(path: str | Path) -> pd.DataFrame:
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source_path)
    if suffix == ".tsv":
        return pd.read_csv(source_path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(source_path)
    if suffix == ".jsonl":
        return pd.read_json(source_path, lines=True)
    raise ValueError(f"Unsupported tabular file type: {source_path.suffix}")


def _resolve_spec(
    dataset_id: str,
    raw_root: str | Path,
    task_type: str | None,
    target_column: str | None,
    train_file: str | None,
    title: str | None,
    notes: str | None,
) -> DatasetManifest:
    spec = resolve_manifest(dataset_id, raw_root=raw_root)
    return DatasetManifest(
        id=spec.id,
        title=title or spec.title,
        provider=spec.provider,
        source_type=spec.source_type,
        external_ref=spec.external_ref,
        source_url=spec.source_url,
        task_type=task_type or spec.task_type,
        target_column=target_column or spec.target_column,
        train_file=train_file or spec.train_file,
        notes=notes if notes is not None else spec.notes,
    )


def _sample_frame(
    frame: pd.DataFrame,
    target_column: str,
    task_type: str,
    max_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame
    stratify = frame[target_column] if task_type != "regression" else None
    sampled, _ = _train_test_split(frame, train_size=max_rows, random_state=seed, stratify=stratify)
    return sampled.reset_index(drop=True)


def _split_frame(
    frame: pd.DataFrame,
    target_column: str,
    task_type: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be >= 0 and sum to less than 1.")

    stratify = frame[target_column] if task_type != "regression" else None
    train_frame, temp_frame = _train_test_split(
        frame,
        test_size=val_fraction + test_fraction,
        random_state=seed,
        stratify=stratify,
    )

    if len(temp_frame) == 0:
        return train_frame.reset_index(drop=True), temp_frame.copy(), temp_frame.copy()

    if test_fraction == 0:
        return train_frame.reset_index(drop=True), temp_frame.reset_index(drop=True), temp_frame.iloc[0:0].copy()

    test_share_of_temp = test_fraction / (val_fraction + test_fraction)
    temp_stratify = temp_frame[target_column] if task_type != "regression" else None
    val_frame, test_frame = _train_test_split(
        temp_frame,
        test_size=test_share_of_temp,
        random_state=seed,
        stratify=temp_stratify,
    )
    return train_frame.reset_index(drop=True), val_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def _train_test_split(*args, **kwargs):
    try:
        return train_test_split(*args, **kwargs)
    except ValueError as exc:
        if kwargs.get("stratify") is None:
            raise
        message = str(exc)
        if "least populated class" not in message and "should be greater or equal to the number of classes" not in message:
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["stratify"] = None
        return train_test_split(*args, **fallback_kwargs)


def _infer_feature_types(frame: pd.DataFrame, target_column: str) -> tuple[list[str], list[str], list[str]]:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    maybe_identifier_columns: list[str] = []

    for column in frame.columns:
        if column == target_column:
            continue
        lower = column.lower()
        if lower in {"id", "row_id", "customer_id"} or lower.endswith("_id"):
            maybe_identifier_columns.append(column)
        if is_numeric_dtype(frame[column]) and not is_bool_dtype(frame[column]):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return numeric_columns, categorical_columns, maybe_identifier_columns


def _safe_float_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _fit_feature_transforms(
    train_frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Any]:
    missing_indicators: list[dict[str, str]] = []
    log_transforms: list[dict[str, str]] = []
    frequency_encodings: list[dict[str, Any]] = []
    rare_category_collapses: list[dict[str, Any]] = []

    for column in numeric_columns:
        series = _safe_float_series(train_frame, column)
        if series.isna().any():
            missing_indicators.append({"source": column, "name": f"{column}_is_missing"})
        non_null = series.dropna()
        if len(non_null) < 16 or non_null.nunique() < 8:
            continue
        if (non_null < 0).any():
            continue
        skew = float(non_null.skew())
        if pd.notna(skew) and skew >= 1.0 and float(non_null.max()) > 0.0:
            log_transforms.append({"source": column, "name": f"{column}_log1p"})

    row_count = max(len(train_frame), 1)
    for column in categorical_columns:
        series = train_frame[column]
        if series.isna().any():
            missing_indicators.append({"source": column, "name": f"{column}_is_missing"})
        values = series.fillna("__MISSING__").astype(str)
        unique_count = int(values.nunique())
        unique_ratio = float(unique_count / row_count)
        counts = values.value_counts(dropna=False)
        min_count = max(2, int(len(values) * 0.01))
        frequent_values = sorted(counts[counts >= min_count].index.astype(str).tolist())
        if 1 < len(frequent_values) < unique_count:
            rare_category_collapses.append(
                {
                    "source": column,
                    "name": column,
                    "keep_values": frequent_values,
                    "replacement": RARE_CATEGORY_TOKEN,
                    "min_count": min_count,
                }
            )
        if unique_count >= 32 or unique_ratio >= 0.2:
            frequencies = (counts / len(values)).to_dict()
            frequency_encodings.append(
                {
                    "source": column,
                    "name": f"{column}_frequency",
                    "mapping": {str(key): float(value) for key, value in frequencies.items()},
                    "default": 0.0,
                }
            )

    return {
        "version": 1,
        "missing_indicators": missing_indicators,
        "log_transforms": log_transforms,
        "frequency_encodings": frequency_encodings,
        "rare_category_collapses": rare_category_collapses,
    }


def _apply_feature_transforms(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    transformed = frame.copy()

    for item in spec.get("rare_category_collapses", []):
        source = str(item["source"])
        if source not in transformed.columns:
            continue
        keep_values = set(str(value) for value in item.get("keep_values", []))
        replacement = str(item.get("replacement", RARE_CATEGORY_TOKEN))
        values = transformed[source].fillna("__MISSING__").astype(str)
        transformed[source] = values.where(values.isin(keep_values), replacement)

    for item in spec.get("missing_indicators", []):
        source = str(item["source"])
        name = str(item["name"])
        transformed[name] = transformed[source].isna().astype("int8") if source in transformed.columns else 1

    for item in spec.get("log_transforms", []):
        source = str(item["source"])
        name = str(item["name"])
        series = _safe_float_series(transformed, source) if source in transformed.columns else pd.Series([0.0] * len(transformed))
        transformed[name] = series.clip(lower=0).fillna(0.0).map(lambda value: float(np.log1p(value)))

    for item in spec.get("frequency_encodings", []):
        source = str(item["source"])
        name = str(item["name"])
        mapping = {str(key): float(value) for key, value in dict(item.get("mapping", {})).items()}
        default = float(item.get("default", 0.0))
        if source in transformed.columns:
            values = transformed[source].fillna("__MISSING__").astype(str)
            transformed[name] = values.map(lambda value: mapping.get(value, default)).astype(float)
        else:
            transformed[name] = default

    return transformed


def _transformed_feature_types(
    numeric_columns: list[str],
    categorical_columns: list[str],
    spec: dict[str, Any],
) -> tuple[list[str], list[str]]:
    transformed_numeric = list(numeric_columns)
    transformed_categorical = list(categorical_columns)
    for item in spec.get("missing_indicators", []):
        name = str(item["name"])
        if name not in transformed_numeric:
            transformed_numeric.append(name)
    for item in spec.get("log_transforms", []):
        name = str(item["name"])
        if name not in transformed_numeric:
            transformed_numeric.append(name)
    for item in spec.get("frequency_encodings", []):
        name = str(item["name"])
        if name not in transformed_numeric:
            transformed_numeric.append(name)
    return transformed_numeric, transformed_categorical


def _write_feature_transforms(processed_dir: Path, spec: dict[str, Any]) -> Path:
    path = processed_dir / FEATURE_TRANSFORMS_FILENAME
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return path


def _build_training_config(
    spec: DatasetManifest,
    processed_dir: Path,
    target_column: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    text_columns: list[str],
    seed: int,
) -> dict[str, object]:
    return {
        "experiment_name": f"finetune_{spec.id}",
        "seed": seed,
        "task": {
            "mode": "finetune",
            "problem_type": spec.task_type,
            "target_column": target_column,
        },
        "data": {
            "dataset_type": "prepared",
            "prepared_dir": str(processed_dir),
            "train_path": str(processed_dir / "train.csv"),
            "val_path": str(processed_dir / "val.csv"),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "text_columns": text_columns,
            "batch_size": 256,
            "num_workers": 0,
            "standardize_numeric": True,
        },
        "model": {
            "d_model": 192,
            "n_heads": 6,
            "n_layers": 6,
            "d_ff": 384,
            "dropout": 0.1,
            "max_categories": 256,
        },
        "training": {
            "device": "cpu",
            "max_epochs": 20,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "grad_clip_norm": 1.0,
            "log_interval": 20,
            "early_stopping_patience": 5,
        },
    }


def prepare_dataset(
    dataset_id: str,
    raw_root: str | Path = "data/raw",
    processed_root: str | Path = "data/processed",
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    max_rows: int | None = None,
    drop_identifier_columns: bool = True,
    task_type: str | None = None,
    target_column: str | None = None,
    train_file: str | None = None,
    title: str | None = None,
    notes: str | None = None,
    feature_engineering: bool = True,
) -> PreparedDataset:
    spec = _resolve_spec(
        dataset_id,
        raw_root=raw_root,
        task_type=task_type,
        target_column=target_column,
        train_file=train_file,
        title=title,
        notes=notes,
    )
    raw_dir = Path(raw_root) / spec.id
    source_path = _resolve_source_file(raw_dir, spec.train_file)
    frame = _read_table(source_path)
    inspection = inspect_supervised_frame(
        frame,
        preferred_target_column=spec.target_column,
        preferred_task_type=spec.task_type,
    )
    frame = inspection.frame
    resolved_target_column = inspection.target_column
    resolved_task_type = inspection.task_type
    relative_source_path = source_path.relative_to(raw_dir) if source_path.is_relative_to(raw_dir) else Path(source_path.name)
    resolved_spec = DatasetManifest(
        id=spec.id,
        title=spec.title,
        provider=spec.provider,
        source_type=spec.source_type,
        external_ref=spec.external_ref,
        source_url=spec.source_url,
        task_type=resolved_task_type,
        target_column=resolved_target_column,
        train_file=str(relative_source_path).replace("\\", "/"),
        notes=spec.notes,
    )
    write_manifest(raw_dir, resolved_spec)

    frame = frame.dropna(subset=[resolved_target_column]).reset_index(drop=True)
    frame = _sample_frame(frame, resolved_target_column, resolved_task_type, max_rows=max_rows, seed=seed)
    frame, rename_map = _normalize_columns(frame)
    normalized_target = rename_map[resolved_target_column]

    train_frame, val_frame, test_frame = _split_frame(
        frame,
        target_column=normalized_target,
        task_type=resolved_task_type,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    numeric_columns, categorical_columns, maybe_identifier_columns = _infer_feature_types(
        train_frame,
        target_column=normalized_target,
    )
    dropped_identifier_columns: list[str] = []
    if drop_identifier_columns and maybe_identifier_columns:
        dropped_identifier_columns = list(maybe_identifier_columns)
        train_frame = train_frame.drop(columns=dropped_identifier_columns, errors="ignore")
        val_frame = val_frame.drop(columns=dropped_identifier_columns, errors="ignore")
        test_frame = test_frame.drop(columns=dropped_identifier_columns, errors="ignore")
        numeric_columns = [column for column in numeric_columns if column not in dropped_identifier_columns]
        categorical_columns = [column for column in categorical_columns if column not in dropped_identifier_columns]

    feature_transform_spec = {
        "version": 1,
        "missing_indicators": [],
        "log_transforms": [],
        "frequency_encodings": [],
        "rare_category_collapses": [],
    }
    if feature_engineering:
        feature_transform_spec = _fit_feature_transforms(
            train_frame,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )
        train_frame = _apply_feature_transforms(train_frame, feature_transform_spec)
        val_frame = _apply_feature_transforms(val_frame, feature_transform_spec)
        test_frame = _apply_feature_transforms(test_frame, feature_transform_spec)
        numeric_columns, categorical_columns = _transformed_feature_types(
            numeric_columns,
            categorical_columns,
            feature_transform_spec,
        )

    processed_dir = Path(processed_root) / spec.id
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"
    train_frame.to_csv(train_path, index=False)
    val_frame.to_csv(val_path, index=False)
    test_frame.to_csv(test_path, index=False)
    schema = build_schema(
        train_frame,
        target_column=normalized_target,
        problem_type=resolved_task_type,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    schema_output_path = write_schema(processed_dir, schema)
    transform_artifact_path = _write_feature_transforms(processed_dir, feature_transform_spec)
    text_columns = [str(spec["name"]) for spec in schema.get("text_features", [])]
    categorical_columns = [str(spec["name"]) for spec in schema.get("categorical_features", [])]

    config = _build_training_config(
        resolved_spec,
        processed_dir=processed_dir,
        target_column=normalized_target,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        text_columns=text_columns,
        seed=seed,
    )
    config_path = processed_dir / "train_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    dataset_card = {
        "dataset_id": resolved_spec.id,
        "title": resolved_spec.title,
        "provider": resolved_spec.provider,
        "source_type": resolved_spec.source_type,
        "external_ref": resolved_spec.external_ref,
        "source_url": resolved_spec.source_url,
        "task_type": resolved_task_type,
        "raw_source_path": str(source_path),
        "processed_dir": str(processed_dir),
        "original_target_column": resolved_target_column,
        "target_column": normalized_target,
        "train_rows": len(train_frame),
        "val_rows": len(val_frame),
        "test_rows": len(test_frame),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "text_columns": text_columns,
        "tabular_inspection": inspection.to_metadata(),
        "feature_engineering_enabled": feature_engineering,
        "feature_transforms": feature_transform_spec,
        "maybe_identifier_columns": maybe_identifier_columns,
        "dropped_identifier_columns": dropped_identifier_columns,
        "schema_path": str(schema_output_path),
        "feature_transform_artifact_path": str(transform_artifact_path),
        "original_to_normalized_columns": rename_map,
        "notes": resolved_spec.notes,
        "limitations": [
            "This first-pass preparation only uses the main training table.",
            "Relational side tables are not joined yet.",
            "Identifier detection uses simple name-based heuristics.",
            "Feature engineering uses train-only heuristics and may need per-dataset tuning.",
        ],
    }
    dataset_card_path = processed_dir / "dataset_card.json"
    dataset_card_path.write_text(json.dumps(dataset_card, indent=2), encoding="utf-8")

    return PreparedDataset(
        dataset_id=resolved_spec.id,
        raw_source_path=str(source_path),
        processed_dir=str(processed_dir),
        train_rows=len(train_frame),
        val_rows=len(val_frame),
        test_rows=len(test_frame),
        target_column=normalized_target,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        config_path=str(config_path),
        dataset_card_path=str(dataset_card_path),
        transform_artifact_path=str(transform_artifact_path),
    )


def prepared_dataset_to_dict(prepared: PreparedDataset) -> dict[str, object]:
    return asdict(prepared)
