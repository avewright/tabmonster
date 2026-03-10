from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, IterableDataset

from tabula.config import DataConfig, ExperimentConfig
from tabula.data.schema import encode_frame, encode_target, load_schema, schema_feature_metadata


NAME_HASH_VOCAB_SIZE = 8192
FEATURE_TRANSFORMS_FILENAME = "feature_transforms.json"


@dataclass
class TabularBatch:
    x_num: torch.Tensor
    x_cat: torch.Tensor
    x_text_token_ids: torch.Tensor
    x_text_values: list[list[str]]
    x_num_mask: torch.Tensor
    x_cat_mask: torch.Tensor
    x_text_mask: torch.Tensor
    num_schema_texts: list[str]
    cat_schema_texts: list[str]
    text_schema_texts: list[str]
    num_name_token_ids: torch.Tensor
    cat_name_token_ids: torch.Tensor
    text_name_token_ids: torch.Tensor
    num_profile_vectors: torch.Tensor
    cat_profile_vectors: torch.Tensor
    text_profile_vectors: torch.Tensor
    y: torch.Tensor


class TabularDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        y: np.ndarray,
        x_num_mask: np.ndarray | None = None,
        x_cat_mask: np.ndarray | None = None,
        x_text_token_ids: np.ndarray | None = None,
        x_text_values: list[list[str]] | None = None,
        x_text_mask: np.ndarray | None = None,
        num_schema_texts: list[str] | None = None,
        cat_schema_texts: list[str] | None = None,
        text_schema_texts: list[str] | None = None,
        num_name_token_ids: np.ndarray | None = None,
        cat_name_token_ids: np.ndarray | None = None,
        text_name_token_ids: np.ndarray | None = None,
        num_profile_vectors: np.ndarray | None = None,
        cat_profile_vectors: np.ndarray | None = None,
        text_profile_vectors: np.ndarray | None = None,
    ) -> None:
        self.x_num = torch.as_tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.y = torch.as_tensor(y)
        self.x_num_mask = torch.as_tensor(
            x_num_mask if x_num_mask is not None else np.ones_like(x_num, dtype=np.bool_),
            dtype=torch.bool,
        )
        self.x_cat_mask = torch.as_tensor(
            x_cat_mask if x_cat_mask is not None else np.ones_like(x_cat, dtype=np.bool_),
            dtype=torch.bool,
        )
        self.x_text_token_ids = torch.as_tensor(
            x_text_token_ids if x_text_token_ids is not None else np.zeros((self.x_num.shape[0], 0, 0), dtype=np.int64),
            dtype=torch.long,
        )
        self.x_text_values = (
            x_text_values
            if x_text_values is not None
            else [["" for _ in range(self.x_text_token_ids.shape[1])] for _ in range(self.x_num.shape[0])]
        )
        self.x_text_mask = torch.as_tensor(
            x_text_mask if x_text_mask is not None else np.zeros((self.x_num.shape[0], 0), dtype=np.bool_),
            dtype=torch.bool,
        )
        self.num_schema_texts = list(num_schema_texts or [])
        self.cat_schema_texts = list(cat_schema_texts or [])
        self.text_schema_texts = list(text_schema_texts or [])
        self.num_name_token_ids = torch.as_tensor(
            num_name_token_ids if num_name_token_ids is not None else np.zeros((self.x_num.shape[1], 0), dtype=np.int64),
            dtype=torch.long,
        )
        self.cat_name_token_ids = torch.as_tensor(
            cat_name_token_ids if cat_name_token_ids is not None else np.zeros((self.x_cat.shape[1], 0), dtype=np.int64),
            dtype=torch.long,
        )
        self.text_name_token_ids = torch.as_tensor(
            text_name_token_ids if text_name_token_ids is not None else np.zeros((self.x_text_token_ids.shape[1], 0), dtype=np.int64),
            dtype=torch.long,
        )
        self.num_profile_vectors = torch.as_tensor(
            num_profile_vectors if num_profile_vectors is not None else np.zeros((self.x_num.shape[1], 0), dtype=np.float32),
            dtype=torch.float32,
        )
        self.cat_profile_vectors = torch.as_tensor(
            cat_profile_vectors if cat_profile_vectors is not None else np.zeros((self.x_cat.shape[1], 0), dtype=np.float32),
            dtype=torch.float32,
        )
        self.text_profile_vectors = torch.as_tensor(
            text_profile_vectors if text_profile_vectors is not None else np.zeros((self.x_text_token_ids.shape[1], 0), dtype=np.float32),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.x_num[idx],
            self.x_cat[idx],
            self.x_text_token_ids[idx],
            self.x_text_values[idx],
            self.x_num_mask[idx],
            self.x_cat_mask[idx],
            self.x_text_mask[idx],
            self.num_schema_texts,
            self.cat_schema_texts,
            self.text_schema_texts,
            self.num_name_token_ids,
            self.cat_name_token_ids,
            self.text_name_token_ids,
            self.num_profile_vectors,
            self.cat_profile_vectors,
            self.text_profile_vectors,
            self.y[idx],
        )


def collate_tabular(batch: list[tuple[torch.Tensor, ...]]) -> TabularBatch:
    x_num, x_cat, x_text_token_ids, x_text_values, x_num_mask, x_cat_mask, x_text_mask, num_schema_texts, cat_schema_texts, text_schema_texts, num_name_token_ids, cat_name_token_ids, text_name_token_ids, num_profile_vectors, cat_profile_vectors, text_profile_vectors, y = zip(*batch)
    return TabularBatch(
        x_num=torch.stack(list(x_num)),
        x_cat=torch.stack(list(x_cat)),
        x_text_token_ids=torch.stack(list(x_text_token_ids)),
        x_text_values=[list(row) for row in x_text_values],
        x_num_mask=torch.stack(list(x_num_mask)),
        x_cat_mask=torch.stack(list(x_cat_mask)),
        x_text_mask=torch.stack(list(x_text_mask)),
        num_schema_texts=list(num_schema_texts[0]),
        cat_schema_texts=list(cat_schema_texts[0]),
        text_schema_texts=list(text_schema_texts[0]),
        num_name_token_ids=num_name_token_ids[0],
        cat_name_token_ids=cat_name_token_ids[0],
        text_name_token_ids=text_name_token_ids[0],
        num_profile_vectors=num_profile_vectors[0],
        cat_profile_vectors=cat_profile_vectors[0],
        text_profile_vectors=text_profile_vectors[0],
        y=torch.stack(list(y)),
    )


def _hash_name_token(token: str) -> int:
    if not token:
        return 0
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % (NAME_HASH_VOCAB_SIZE - 1)) + 1


def _feature_metadata_arrays(schema: dict[str, object]) -> dict[str, object]:
    metadata = schema_feature_metadata(schema)
    num_name_ids = np.asarray(
        [[_hash_name_token(token) for token in tokens] for tokens in metadata["numeric_name_tokens"]],
        dtype=np.int64,
    ) if metadata["numeric_name_tokens"] else np.zeros((0, 0), dtype=np.int64)
    cat_name_ids = np.asarray(
        [[_hash_name_token(token) for token in tokens] for tokens in metadata["categorical_name_tokens"]],
        dtype=np.int64,
    ) if metadata["categorical_name_tokens"] else np.zeros((0, 0), dtype=np.int64)
    text_name_ids = np.asarray(
        [[_hash_name_token(token) for token in tokens] for tokens in metadata["text_name_tokens"]],
        dtype=np.int64,
    ) if metadata["text_name_tokens"] else np.zeros((0, 0), dtype=np.int64)
    num_profiles = np.asarray(metadata["numeric_profile_vectors"], dtype=np.float32) if metadata["numeric_profile_vectors"] else np.zeros((0, 0), dtype=np.float32)
    cat_profiles = np.asarray(metadata["categorical_profile_vectors"], dtype=np.float32) if metadata["categorical_profile_vectors"] else np.zeros((0, 0), dtype=np.float32)
    text_profiles = np.asarray(metadata["text_profile_vectors"], dtype=np.float32) if metadata["text_profile_vectors"] else np.zeros((0, 0), dtype=np.float32)
    return {
        "num_name_ids": num_name_ids,
        "cat_name_ids": cat_name_ids,
        "text_name_ids": text_name_ids,
        "num_schema_texts": [str(text) for text in metadata["numeric_schema_texts"]],
        "cat_schema_texts": [str(text) for text in metadata["categorical_schema_texts"]],
        "text_schema_texts": [str(text) for text in metadata["text_schema_texts"]],
        "num_profiles": num_profiles,
        "cat_profiles": cat_profiles,
        "text_profiles": text_profiles,
    }


def _tokenize_text_value(value: object, token_count: int) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return [0] * token_count
    text = str(value).strip().lower()
    parts = [part for part in text.replace("_", " ").split() if part]
    if not parts:
        parts = [text] if text else []
    hashed = [_hash_name_token(part) for part in parts[:token_count]]
    if len(hashed) < token_count:
        hashed.extend([0] * (token_count - len(hashed)))
    return hashed


def _encode_text_frame(frame: pd.DataFrame, schema: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    text_specs = schema.get("text_features", [])
    if not text_specs:
        return np.zeros((len(frame), 0, 0), dtype=np.int64), np.zeros((len(frame), 0), dtype=np.bool_)
    token_count = int(schema.get("metadata", {}).get("text_token_count", 16))
    encoded_columns: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for spec in text_specs:
        column = str(spec["name"])
        values = frame[column] if column in frame.columns else pd.Series([None] * len(frame))
        masks.append(values.notna().to_numpy(dtype=np.bool_))
        encoded_columns.append(
            np.asarray([_tokenize_text_value(value, token_count) for value in values.tolist()], dtype=np.int64)
        )
    return np.stack(encoded_columns, axis=1), np.stack(masks, axis=1)


def _raw_text_values(frame: pd.DataFrame, schema: dict[str, object]) -> list[list[str]]:
    text_specs = schema.get("text_features", [])
    if not text_specs:
        return [[""] * 0 for _ in range(len(frame))]
    values_by_column: list[list[str]] = []
    for spec in text_specs:
        column = str(spec["name"])
        series = frame[column] if column in frame.columns else pd.Series([None] * len(frame))
        values_by_column.append(["" if pd.isna(value) else str(value) for value in series.tolist()])
    return [list(row) for row in zip(*values_by_column, strict=False)]


def _make_synthetic_frame(config: DataConfig, n_rows: int) -> pd.DataFrame:
    rng = np.random.default_rng()
    frame = pd.DataFrame()
    linear_terms: list[np.ndarray] = []

    for i in range(config.num_numeric_features):
        col = rng.normal(size=n_rows)
        frame[f"num_{i}"] = col
        linear_terms.append(col * rng.uniform(0.3, 1.2))

    for i in range(config.num_categorical_features):
        col = rng.integers(0, config.categorical_cardinality, size=n_rows)
        frame[f"cat_{i}"] = col.astype(str)
        linear_terms.append((col % max(config.num_classes, 2)) * rng.uniform(0.1, 0.5))

    score = np.sum(np.vstack(linear_terms), axis=0) + rng.normal(scale=0.5, size=n_rows)
    if config.num_classes <= 2:
        target = (score > np.median(score)).astype(int)
    else:
        bins = np.quantile(score, np.linspace(0, 1, config.num_classes + 1)[1:-1])
        target = np.digitize(score, bins)

    frame["target"] = target
    return frame


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_prepared_frame(path: str | Path) -> pd.DataFrame:
    source_path = Path(path)
    if source_path.suffix.lower() == ".parquet":
        return pd.read_parquet(source_path)
    return pd.read_csv(source_path)


def _load_feature_transform_spec(prepared_dir: str | Path | None) -> dict[str, object]:
    if not prepared_dir:
        return {}
    path = Path(prepared_dir) / FEATURE_TRANSFORMS_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_transforms_to_row(row: dict[str, object], spec: dict[str, object]) -> dict[str, object]:
    output = dict(row)
    for item in spec.get("rare_category_collapses", []):
        source = str(item["source"])
        keep_values = {str(value) for value in item.get("keep_values", [])}
        replacement = str(item.get("replacement", "__RARE__"))
        value = "__MISSING__" if output.get(source) is None or pd.isna(output.get(source)) else str(output.get(source))
        output[source] = value if value in keep_values else replacement
    for item in spec.get("missing_indicators", []):
        source = str(item["source"])
        name = str(item["name"])
        value = output.get(source)
        output[name] = int(value is None or pd.isna(value))
    for item in spec.get("log_transforms", []):
        source = str(item["source"])
        name = str(item["name"])
        value = output.get(source)
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        numeric = 0.0 if pd.isna(numeric) else max(float(numeric), 0.0)
        output[name] = float(np.log1p(numeric))
    for item in spec.get("frequency_encodings", []):
        source = str(item["source"])
        name = str(item["name"])
        mapping = {str(key): float(value) for key, value in dict(item.get("mapping", {})).items()}
        raw = output.get(source)
        key = "__MISSING__" if raw is None or pd.isna(raw) else str(raw)
        output[name] = float(mapping.get(key, float(item.get("default", 0.0))))
    return output


def _encode_stream_row(
    row: dict[str, object],
    schema: dict[str, object],
    feature_metadata: dict[str, object],
    standardize_numeric: bool,
    transform_spec: dict[str, object],
) -> tuple[torch.Tensor, ...]:
    transformed = _apply_transforms_to_row(row, transform_spec) if transform_spec else dict(row)
    numeric_specs = schema.get("numeric_features", [])
    categorical_specs = schema.get("categorical_features", [])
    text_specs = schema.get("text_features", [])
    target_spec = schema["target"]

    x_num_values: list[float] = []
    x_num_mask_values: list[bool] = []
    for spec in numeric_specs:
        column = str(spec["name"])
        raw = transformed.get(column)
        numeric = pd.to_numeric(pd.Series([raw]), errors="coerce").iloc[0]
        present = not pd.isna(numeric)
        if not present:
            numeric = float(spec["fill_value"])
        value = float(numeric)
        if standardize_numeric:
            value = (value - float(spec["mean"])) / float(spec["std"])
        x_num_values.append(value)
        x_num_mask_values.append(present)

    x_cat_values: list[int] = []
    x_cat_mask_values: list[bool] = []
    for spec in categorical_specs:
        column = str(spec["name"])
        raw = transformed.get(column)
        present = raw is not None and not pd.isna(raw)
        value = "__MISSING__" if not present else str(raw)
        mapping = {category: idx + 1 for idx, category in enumerate(spec["categories"])}
        x_cat_values.append(int(mapping.get(value, 0)))
        x_cat_mask_values.append(present)

    token_count = int(schema.get("metadata", {}).get("text_token_count", 16))
    x_text_columns: list[list[int]] = []
    x_text_mask_values: list[bool] = []
    raw_text_values: list[str] = []
    for spec in text_specs:
        column = str(spec["name"])
        raw = transformed.get(column)
        present = raw is not None and not pd.isna(raw)
        x_text_columns.append(_tokenize_text_value(raw, token_count))
        x_text_mask_values.append(present)
        raw_text_values.append("" if not present else str(raw))

    problem_type = str(target_spec["problem_type"])
    target_column = str(target_spec["column"])
    target_raw = transformed.get(target_column)
    if problem_type == "regression":
        y = torch.tensor(float(target_raw), dtype=torch.float32)
    else:
        classes = [str(item) for item in target_spec["classes"]]
        mapping = {label: idx for idx, label in enumerate(classes)}
        encoded = mapping.get("__MISSING__" if target_raw is None or pd.isna(target_raw) else str(target_raw))
        if encoded is None:
            raise ValueError(f"Found unseen target label during streaming encode: {target_raw}")
        y = torch.tensor(encoded, dtype=torch.long)

    return (
        torch.tensor(x_num_values, dtype=torch.float32),
        torch.tensor(x_cat_values, dtype=torch.long),
        torch.tensor(x_text_columns, dtype=torch.long) if x_text_columns else torch.zeros((0, token_count), dtype=torch.long),
        raw_text_values,
        torch.tensor(x_num_mask_values, dtype=torch.bool),
        torch.tensor(x_cat_mask_values, dtype=torch.bool),
        torch.tensor(x_text_mask_values, dtype=torch.bool),
        list(feature_metadata["num_schema_texts"]),
        list(feature_metadata["cat_schema_texts"]),
        list(feature_metadata["text_schema_texts"]),
        torch.as_tensor(feature_metadata["num_name_ids"], dtype=torch.long),
        torch.as_tensor(feature_metadata["cat_name_ids"], dtype=torch.long),
        torch.as_tensor(feature_metadata["text_name_ids"], dtype=torch.long),
        torch.as_tensor(feature_metadata["num_profiles"], dtype=torch.float32),
        torch.as_tensor(feature_metadata["cat_profiles"], dtype=torch.float32),
        torch.as_tensor(feature_metadata["text_profiles"], dtype=torch.float32),
        y,
    )


class HuggingFaceStreamingDataset(IterableDataset[tuple[torch.Tensor, ...]]):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        if not config.data.prepared_dir:
            raise ValueError("hf_stream mode requires data.prepared_dir so an existing schema can be reused.")
        if not config.data.hf_repo_id:
            raise ValueError("hf_stream mode requires data.hf_repo_id.")
        self.config = config
        self.schema = load_schema(config.data.prepared_dir)
        self.feature_metadata = _feature_metadata_arrays(self.schema)
        self.transform_spec = _load_feature_transform_spec(config.data.prepared_dir)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        from datasets import load_dataset

        data_cfg = self.config.data
        dataset = load_dataset(
            data_cfg.hf_repo_id,
            name=data_cfg.hf_config_name,
            split=data_cfg.hf_split,
            streaming=data_cfg.hf_streaming,
            cache_dir=data_cfg.hf_cache_dir,
        )
        if data_cfg.hf_streaming:
            dataset = dataset.shuffle(seed=self.config.seed, buffer_size=data_cfg.hf_shuffle_buffer_size)
        row_limit = data_cfg.hf_max_stream_rows
        skip_rows = max(int(data_cfg.hf_skip_rows), 0)
        for row_index, row in enumerate(dataset):
            if row_index < skip_rows:
                continue
            if row_limit is not None and row_index >= row_limit:
                break
            yield _encode_stream_row(
                dict(row),
                self.schema,
                self.feature_metadata,
                standardize_numeric=data_cfg.standardize_numeric,
                transform_spec=self.transform_spec,
            )


def _encode_frame(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[TabularDataset, TabularDataset, int]:
    data_cfg = config.data
    target = config.task.target_column
    numeric_columns = data_cfg.numeric_columns or [c for c in train_df.columns if c.startswith("num_")]
    categorical_columns = data_cfg.categorical_columns or [c for c in train_df.columns if c.startswith("cat_")]

    train_num = train_df[numeric_columns].to_numpy(dtype=np.float32) if numeric_columns else np.zeros((len(train_df), 0), dtype=np.float32)
    val_num = val_df[numeric_columns].to_numpy(dtype=np.float32) if numeric_columns else np.zeros((len(val_df), 0), dtype=np.float32)

    if data_cfg.standardize_numeric and numeric_columns:
        scaler = StandardScaler()
        train_num = scaler.fit_transform(train_num).astype(np.float32)
        val_num = scaler.transform(val_num).astype(np.float32)

    train_cat_parts: list[np.ndarray] = []
    val_cat_parts: list[np.ndarray] = []
    for column in categorical_columns:
        encoder = LabelEncoder()
        encoder.fit(pd.concat([train_df[column], val_df[column]], axis=0).astype(str))
        train_cat_parts.append(encoder.transform(train_df[column].astype(str)))
        val_cat_parts.append(encoder.transform(val_df[column].astype(str)))

    if train_cat_parts:
        train_cat = np.stack(train_cat_parts, axis=1).astype(np.int64)
        val_cat = np.stack(val_cat_parts, axis=1).astype(np.int64)
    else:
        train_cat = np.zeros((len(train_df), 0), dtype=np.int64)
        val_cat = np.zeros((len(val_df), 0), dtype=np.int64)

    if config.task.problem_type == "regression":
        y_train = train_df[target].to_numpy(dtype=np.float32)
        y_val = val_df[target].to_numpy(dtype=np.float32)
        output_dim = 1
    else:
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train_df[target])
        y_val = label_encoder.transform(val_df[target])
        output_dim = len(label_encoder.classes_)

    train_num_mask = ~np.isnan(train_df[numeric_columns].to_numpy(dtype=np.float32)) if numeric_columns else np.zeros((len(train_df), 0), dtype=np.bool_)
    val_num_mask = ~np.isnan(val_df[numeric_columns].to_numpy(dtype=np.float32)) if numeric_columns else np.zeros((len(val_df), 0), dtype=np.bool_)
    train_cat_mask = np.ones_like(train_cat, dtype=np.bool_)
    val_cat_mask = np.ones_like(val_cat, dtype=np.bool_)
    train_ds = TabularDataset(train_num, train_cat, y_train, train_num_mask, train_cat_mask)
    val_ds = TabularDataset(val_num, val_cat, y_val, val_num_mask, val_cat_mask)
    return train_ds, val_ds, output_dim


def _encode_prepared_frame(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[TabularDataset, TabularDataset, int]:
    if not config.data.prepared_dir:
        raise ValueError("Prepared dataset mode requires data.prepared_dir.")
    schema = load_schema(config.data.prepared_dir)
    train_num, train_cat = encode_frame(train_df, schema, standardize_numeric=config.data.standardize_numeric)
    val_num, val_cat = encode_frame(val_df, schema, standardize_numeric=config.data.standardize_numeric)
    train_text_token_ids, train_text_mask = _encode_text_frame(train_df, schema)
    val_text_token_ids, val_text_mask = _encode_text_frame(val_df, schema)
    train_text_values = _raw_text_values(train_df, schema)
    val_text_values = _raw_text_values(val_df, schema)
    y_train, output_dim = encode_target(train_df, schema)
    y_val, _ = encode_target(val_df, schema)
    feature_metadata = _feature_metadata_arrays(schema)
    numeric_names = feature_metadata["num_profiles"].shape[0]
    categorical_names = feature_metadata["cat_profiles"].shape[0]
    train_num_mask = (
        ~pd.concat([pd.to_numeric(train_df[column], errors="coerce") for column in train_df.columns if column in [spec["name"] for spec in schema.get("numeric_features", [])]], axis=1).isna().to_numpy(dtype=np.bool_)
        if numeric_names > 0
        else np.zeros((len(train_df), 0), dtype=np.bool_)
    )
    val_num_mask = (
        ~pd.concat([pd.to_numeric(val_df[column], errors="coerce") for column in val_df.columns if column in [spec["name"] for spec in schema.get("numeric_features", [])]], axis=1).isna().to_numpy(dtype=np.bool_)
        if numeric_names > 0
        else np.zeros((len(val_df), 0), dtype=np.bool_)
    )
    train_cat_mask = (
        train_df[[spec["name"] for spec in schema.get("categorical_features", [])]].notna().to_numpy(dtype=np.bool_)
        if categorical_names > 0
        else np.zeros((len(train_df), 0), dtype=np.bool_)
    )
    val_cat_mask = (
        val_df[[spec["name"] for spec in schema.get("categorical_features", [])]].notna().to_numpy(dtype=np.bool_)
        if categorical_names > 0
        else np.zeros((len(val_df), 0), dtype=np.bool_)
    )
    return (
        TabularDataset(
            train_num,
            train_cat,
            y_train,
            train_num_mask,
            train_cat_mask,
            train_text_token_ids,
            train_text_values,
            train_text_mask,
            feature_metadata["num_schema_texts"],
            feature_metadata["cat_schema_texts"],
            feature_metadata["text_schema_texts"],
            feature_metadata["num_name_ids"],
            feature_metadata["cat_name_ids"],
            feature_metadata["text_name_ids"],
            feature_metadata["num_profiles"],
            feature_metadata["cat_profiles"],
            feature_metadata["text_profiles"],
        ),
        TabularDataset(
            val_num,
            val_cat,
            y_val,
            val_num_mask,
            val_cat_mask,
            val_text_token_ids,
            val_text_values,
            val_text_mask,
            feature_metadata["num_schema_texts"],
            feature_metadata["cat_schema_texts"],
            feature_metadata["text_schema_texts"],
            feature_metadata["num_name_ids"],
            feature_metadata["cat_name_ids"],
            feature_metadata["text_name_ids"],
            feature_metadata["num_profiles"],
            feature_metadata["cat_profiles"],
            feature_metadata["text_profiles"],
        ),
        output_dim,
    )


def build_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader, int, int, int, int]:
    data_cfg = config.data
    if data_cfg.dataset_type == "synthetic":
        train_df = _make_synthetic_frame(data_cfg, data_cfg.train_size)
        val_df = _make_synthetic_frame(data_cfg, data_cfg.val_size)
        train_ds, val_ds, output_dim = _encode_frame(train_df, val_df, config)
    else:
        if not data_cfg.train_path or not data_cfg.val_path:
            if data_cfg.dataset_type != "hf_stream":
                raise ValueError("CSV and prepared modes require train_path and val_path.")
        if data_cfg.dataset_type == "prepared":
            train_df = _load_prepared_frame(data_cfg.train_path)
            val_df = _load_prepared_frame(data_cfg.val_path)
            train_ds, val_ds, output_dim = _encode_prepared_frame(train_df, val_df, config)
        elif data_cfg.dataset_type == "hf_stream":
            if not data_cfg.val_path or not data_cfg.prepared_dir:
                raise ValueError("hf_stream mode requires data.val_path and data.prepared_dir.")
            val_df = _load_prepared_frame(data_cfg.val_path)
            schema = load_schema(data_cfg.prepared_dir)
            y_val, output_dim = encode_target(val_df, schema)
            del y_val
            train_ds = HuggingFaceStreamingDataset(config)
            _, val_ds, _ = _encode_prepared_frame(val_df, val_df, config)
        else:
            train_df = _read_csv(data_cfg.train_path)
            val_df = _read_csv(data_cfg.val_path)
            train_ds, val_ds, output_dim = _encode_frame(train_df, val_df, config)
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        shuffle=data_cfg.dataset_type != "hf_stream",
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_tabular,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_tabular,
    )
    if data_cfg.dataset_type == "hf_stream":
        schema = load_schema(data_cfg.prepared_dir or "")
        num_numeric = len(schema.get("numeric_features", []))
        num_categorical = len(schema.get("categorical_features", []))
        num_text = len(schema.get("text_features", []))
    else:
        num_numeric = train_ds.x_num.shape[1]
        num_categorical = train_ds.x_cat.shape[1]
        num_text = train_ds.x_text_token_ids.shape[1]
    return train_loader, val_loader, num_numeric, num_categorical, num_text, output_dim
