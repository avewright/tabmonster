from __future__ import annotations

import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd


SCHEMA_FILENAME = "schema.json"
MISSING_CATEGORY_TOKEN = "__MISSING__"
UNKNOWN_CATEGORY_INDEX = 0
NAME_TOKEN_COUNT = 4
TYPE_NAMES = ["numeric", "categorical", "text", "identifier", "datetime"]
TEXT_TOKEN_COUNT = 16


def _column_name_tokens(name: str, max_tokens: int = NAME_TOKEN_COUNT) -> list[str]:
    parts = [part for part in re.split(r"[^a-zA-Z0-9]+", name.lower()) if part]
    if not parts:
        parts = [name.lower() or "column"]
    tokens = parts[:max_tokens]
    if len(tokens) < max_tokens:
        tokens.extend([""] * (max_tokens - len(tokens)))
    return tokens


def _safe_float(value: float | int) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def _heuristic_type_probabilities(
    series: pd.Series,
    normalized_name: str,
    prefer_numeric: bool,
) -> dict[str, float]:
    values = series.dropna()
    values_as_str = values.astype(str)
    row_count = max(len(series), 1)
    unique_ratio = float(values_as_str.nunique(dropna=True) / max(len(values_as_str), 1)) if len(values_as_str) else 0.0
    numeric_ratio = float(pd.to_numeric(values_as_str, errors="coerce").notna().mean()) if len(values_as_str) else 0.0
    datetime_ratio = _datetime_parse_ratio(values_as_str)
    avg_length = float(values_as_str.str.len().mean()) if len(values_as_str) else 0.0
    lower_name = normalized_name.lower()
    identifier_name = lower_name in {"id", "row_id", "customer_id"} or lower_name.endswith("_id")

    scores = {
        "numeric": 0.15,
        "categorical": 0.15,
        "text": 0.10,
        "identifier": 0.10,
        "datetime": 0.10,
    }
    scores["numeric"] += 1.2 * numeric_ratio + (0.3 if prefer_numeric else 0.0)
    scores["categorical"] += 0.9 * max(0.0, 0.2 - unique_ratio) + (0.25 if not prefer_numeric else 0.0)
    scores["text"] += 0.8 * max(0.0, avg_length / 32.0) + 0.5 * unique_ratio
    scores["identifier"] += (1.2 if identifier_name else 0.0) + 0.9 * max(0.0, unique_ratio - 0.95)
    scores["datetime"] += 1.3 * datetime_ratio

    total = sum(max(score, 1e-6) for score in scores.values())
    return {name: float(max(score, 1e-6) / total) for name, score in scores.items()}


def _build_column_profile(
    train_df: pd.DataFrame,
    column: str,
    prefer_numeric: bool,
) -> dict[str, object]:
    series = train_df[column]
    values = series.dropna()
    values_as_str = values.astype(str)
    missing_ratio = float(series.isna().mean())
    unique_ratio = float(values_as_str.nunique(dropna=True) / max(len(values_as_str), 1)) if len(values_as_str) else 0.0
    top_frequency_ratio = (
        float(values_as_str.value_counts(normalize=True, dropna=True).iloc[0])
        if len(values_as_str)
        else 0.0
    )
    avg_string_length = float(values_as_str.str.len().mean()) if len(values_as_str) else 0.0
    parse_numeric_ratio = float(pd.to_numeric(values_as_str, errors="coerce").notna().mean()) if len(values_as_str) else 0.0
    parse_datetime_ratio = _datetime_parse_ratio(values_as_str)

    return {
        "name": column,
        "name_tokens": _column_name_tokens(column),
        "missing_ratio": missing_ratio,
        "unique_ratio": unique_ratio,
        "top_frequency_ratio": top_frequency_ratio,
        "avg_string_length": avg_string_length,
        "parse_numeric_ratio": parse_numeric_ratio,
        "parse_datetime_ratio": parse_datetime_ratio,
        "type_probabilities": _heuristic_type_probabilities(series, column, prefer_numeric=prefer_numeric),
    }


def _profile_vector(profile: dict[str, object]) -> list[float]:
    type_probabilities = profile["type_probabilities"]
    return [
        _safe_float(profile["missing_ratio"]),
        _safe_float(profile["unique_ratio"]),
        _safe_float(profile["top_frequency_ratio"]),
        _safe_float(profile["avg_string_length"]),
        _safe_float(profile["parse_numeric_ratio"]),
        _safe_float(profile["parse_datetime_ratio"]),
        *[_safe_float(type_probabilities[name]) for name in TYPE_NAMES],
    ]


def _schema_text(profile: dict[str, object], feature_kind: str) -> str:
    type_probabilities = profile["type_probabilities"]
    dominant_type = max(TYPE_NAMES, key=lambda name: float(type_probabilities.get(name, 0.0)))
    return " ".join(
        [
            f"name {profile['name']}",
            f"feature_kind {feature_kind}",
            f"dominant_type {dominant_type}",
            f"missing_ratio {_safe_float(profile['missing_ratio']):.3f}",
            f"unique_ratio {_safe_float(profile['unique_ratio']):.3f}",
            f"top_frequency_ratio {_safe_float(profile['top_frequency_ratio']):.3f}",
            f"avg_length {_safe_float(profile['avg_string_length']):.3f}",
            f"numeric_parse_ratio {_safe_float(profile['parse_numeric_ratio']):.3f}",
            f"datetime_parse_ratio {_safe_float(profile['parse_datetime_ratio']):.3f}",
        ]
    )


def _datetime_parse_ratio(values_as_str: pd.Series) -> float:
    if len(values_as_str) == 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(values_as_str, errors="coerce")
    return float(parsed.notna().mean())


def _is_text_feature(profile: dict[str, object]) -> bool:
    probs = profile["type_probabilities"]
    return bool(
        probs["text"] >= max(probs["categorical"], probs["identifier"])
        and (
            _safe_float(profile["avg_string_length"]) >= 12.0
            or _safe_float(profile["unique_ratio"]) >= 0.25
        )
    )


def build_schema(
    train_df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, object]:
    numeric_features: list[dict[str, float | str]] = []
    for column in numeric_columns:
        series = pd.to_numeric(train_df[column], errors="coerce")
        fill_value = float(series.median()) if not series.dropna().empty else 0.0
        filled = series.fillna(fill_value)
        mean = float(filled.mean())
        std = float(filled.std(ddof=0))
        profile = _build_column_profile(train_df, column, prefer_numeric=True)
        numeric_features.append(
            {
                "name": column,
                "fill_value": fill_value,
                "mean": mean,
                "std": std if std > 0 else 1.0,
                "profile": profile,
            }
        )

    categorical_features: list[dict[str, object]] = []
    text_features: list[dict[str, object]] = []
    for column in categorical_columns:
        values = train_df[column].fillna(MISSING_CATEGORY_TOKEN).astype(str)
        profile = _build_column_profile(train_df, column, prefer_numeric=False)
        if _is_text_feature(profile):
            text_features.append(
                {
                    "name": column,
                    "token_count": TEXT_TOKEN_COUNT,
                    "profile": profile,
                }
            )
        else:
            categories = sorted(values.unique().tolist())
            categorical_features.append(
                {
                    "name": column,
                    "categories": categories,
                    "unknown_index": UNKNOWN_CATEGORY_INDEX,
                    "profile": profile,
                }
            )

    if problem_type == "regression":
        target = {"column": target_column, "problem_type": problem_type, "classes": None}
    else:
        classes = sorted(train_df[target_column].fillna(MISSING_CATEGORY_TOKEN).astype(str).unique().tolist())
        target = {"column": target_column, "problem_type": problem_type, "classes": classes}

    return {
        "target": target,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "text_features": text_features,
        "metadata": {
            "profile_vector_fields": [
                "missing_ratio",
                "unique_ratio",
                "top_frequency_ratio",
                "avg_string_length",
                "parse_numeric_ratio",
                "parse_datetime_ratio",
                *[f"type_probability_{name}" for name in TYPE_NAMES],
            ],
            "type_names": TYPE_NAMES,
            "name_token_count": NAME_TOKEN_COUNT,
            "text_token_count": TEXT_TOKEN_COUNT,
        },
    }


def schema_path(processed_dir: str | Path) -> Path:
    return Path(processed_dir) / SCHEMA_FILENAME


def write_schema(processed_dir: str | Path, schema: dict[str, object]) -> Path:
    path = schema_path(processed_dir)
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return path


def load_schema(processed_dir: str | Path) -> dict[str, object]:
    return json.loads(schema_path(processed_dir).read_text(encoding="utf-8"))


def encode_frame(
    frame: pd.DataFrame,
    schema: dict[str, object],
    standardize_numeric: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    numeric_specs = schema.get("numeric_features", [])
    categorical_specs = schema.get("categorical_features", [])

    numeric_parts: list[np.ndarray] = []
    for spec in numeric_specs:
        column = str(spec["name"])
        series = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series([np.nan] * len(frame))
        filled = series.fillna(float(spec["fill_value"])).to_numpy(dtype=np.float32)
        if standardize_numeric:
            filled = ((filled - float(spec["mean"])) / float(spec["std"])).astype(np.float32)
        numeric_parts.append(filled)
    x_num = np.stack(numeric_parts, axis=1).astype(np.float32) if numeric_parts else np.zeros((len(frame), 0), dtype=np.float32)

    categorical_parts: list[np.ndarray] = []
    for spec in categorical_specs:
        column = str(spec["name"])
        values = frame[column].fillna(MISSING_CATEGORY_TOKEN).astype(str) if column in frame.columns else pd.Series([MISSING_CATEGORY_TOKEN] * len(frame))
        mapping = {value: idx + 1 for idx, value in enumerate(spec["categories"])}
        encoded = values.map(lambda value: mapping.get(value, UNKNOWN_CATEGORY_INDEX)).to_numpy(dtype=np.int64)
        categorical_parts.append(encoded)
    x_cat = np.stack(categorical_parts, axis=1).astype(np.int64) if categorical_parts else np.zeros((len(frame), 0), dtype=np.int64)

    return x_num, x_cat


def encode_target(frame: pd.DataFrame, schema: dict[str, object]) -> tuple[np.ndarray, int]:
    target_spec = schema["target"]
    target_column = str(target_spec["column"])
    problem_type = str(target_spec["problem_type"])
    if problem_type == "regression":
        return frame[target_column].to_numpy(dtype=np.float32), 1

    classes = [str(item) for item in target_spec["classes"]]
    mapping = {label: idx for idx, label in enumerate(classes)}
    encoded = frame[target_column].fillna(MISSING_CATEGORY_TOKEN).astype(str).map(mapping)
    if encoded.isna().any():
        unknown = frame.loc[encoded.isna(), target_column].astype(str).unique().tolist()
        raise ValueError(f"Found unseen target labels during encoding: {unknown}")
    return encoded.to_numpy(dtype=np.int64), len(classes)


def schema_feature_metadata(schema: dict[str, object]) -> dict[str, list[object]]:
    numeric_specs = schema.get("numeric_features", [])
    categorical_specs = schema.get("categorical_features", [])
    text_specs = schema.get("text_features", [])
    return {
        "numeric_names": [str(spec["name"]) for spec in numeric_specs],
        "categorical_names": [str(spec["name"]) for spec in categorical_specs],
        "text_names": [str(spec["name"]) for spec in text_specs],
        "numeric_name_tokens": [list(spec["profile"]["name_tokens"]) for spec in numeric_specs],
        "categorical_name_tokens": [list(spec["profile"]["name_tokens"]) for spec in categorical_specs],
        "text_name_tokens": [list(spec["profile"]["name_tokens"]) for spec in text_specs],
        "numeric_schema_texts": [_schema_text(spec["profile"], "numeric") for spec in numeric_specs],
        "categorical_schema_texts": [_schema_text(spec["profile"], "categorical") for spec in categorical_specs],
        "text_schema_texts": [_schema_text(spec["profile"], "text") for spec in text_specs],
        "numeric_profile_vectors": [_profile_vector(spec["profile"]) for spec in numeric_specs],
        "categorical_profile_vectors": [_profile_vector(spec["profile"]) for spec in categorical_specs],
        "text_profile_vectors": [_profile_vector(spec["profile"]) for spec in text_specs],
    }
