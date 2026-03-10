from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time
import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_object_dtype, is_scalar


_EXACT_TARGET_NAME_WEIGHTS = {
    "target": 8.0,
    "label": 8.0,
    "labels": 7.5,
    "class": 7.0,
    "classes": 6.5,
    "outcome": 6.0,
    "response": 6.0,
    "output": 5.5,
    "y": 5.0,
}

_TARGET_KEYWORD_WEIGHTS = {
    "target": 4.5,
    "label": 4.5,
    "class": 4.0,
    "outcome": 3.5,
    "response": 3.5,
    "status": 2.5,
    "result": 2.5,
    "price": 2.5,
    "score": 2.0,
    "rating": 2.0,
    "risk": 2.0,
    "income": 2.0,
    "quality": 2.0,
    "churn": 2.0,
    "default": 2.0,
    "fraud": 2.0,
    "approved": 2.0,
    "approval": 2.0,
    "survived": 2.0,
    "diagnosis": 2.0,
    "species": 1.5,
    "deposit": 1.5,
    "sale": 1.0,
    "sales": 1.0,
}


@dataclass(frozen=True)
class TargetCandidate:
    column: str
    score: float
    task_type: str
    name_score: float
    unique_count: int
    unique_ratio: float
    reason: str


@dataclass
class TabularInspection:
    frame: pd.DataFrame
    scalar_columns: list[str]
    dropped_columns: dict[str, str]
    target_column: str
    task_type: str
    target_source: str
    target_candidates: list[TargetCandidate]

    def to_metadata(self) -> dict[str, object]:
        return {
            "scalar_columns": list(self.scalar_columns),
            "dropped_columns": dict(self.dropped_columns),
            "target_source": self.target_source,
            "target_candidates": [asdict(candidate) for candidate in self.target_candidates],
        }


def inspect_supervised_frame(
    frame: pd.DataFrame,
    preferred_target_column: str | None = None,
    preferred_task_type: str | None = None,
    metadata_target_column: str | None = None,
    metadata_task_type: str | None = None,
) -> TabularInspection:
    normalized_frame = frame.rename(columns={column: str(column) for column in frame.columns}).copy()
    scalar_columns, dropped_columns = _partition_tabular_columns(normalized_frame)
    if not scalar_columns:
        raise ValueError("The dataset has no flat scalar columns that can be used for tabular training.")

    filtered_frame = normalized_frame.loc[:, scalar_columns].copy()
    target_column, task_type, target_source, target_candidates = _resolve_target_column(
        filtered_frame,
        preferred_target_column=preferred_target_column,
        preferred_task_type=preferred_task_type,
        metadata_target_column=metadata_target_column,
        metadata_task_type=metadata_task_type,
        dropped_columns=dropped_columns,
    )
    feature_columns = [column for column in scalar_columns if column != target_column]
    if not feature_columns:
        raise ValueError(
            f"The dataset only contains the resolved target column `{target_column}` after filtering unsupported columns."
        )

    return TabularInspection(
        frame=filtered_frame,
        scalar_columns=scalar_columns,
        dropped_columns=dropped_columns,
        target_column=target_column,
        task_type=task_type,
        target_source=target_source,
        target_candidates=target_candidates,
    )


def infer_huggingface_target_metadata(dataset: Any) -> tuple[str | None, str | None]:
    info = getattr(dataset, "info", None)
    features = getattr(dataset, "features", None) or getattr(info, "features", None)

    metadata_target: str | None = None
    metadata_task: str | None = None

    supervised_keys = getattr(info, "supervised_keys", None)
    if isinstance(supervised_keys, (tuple, list)) and len(supervised_keys) >= 2 and supervised_keys[1]:
        metadata_target = str(supervised_keys[1])

    class_label_candidates: list[tuple[str, str]] = []
    if hasattr(features, "items"):
        for column, feature in features.items():
            if type(feature).__name__ != "ClassLabel":
                continue
            names = list(getattr(feature, "names", []) or [])
            task_type = "binary" if len(names) == 2 else "multiclass"
            class_label_candidates.append((str(column), task_type))

    if metadata_target:
        for column, task_type in class_label_candidates:
            if column == metadata_target:
                metadata_task = task_type
                break
    elif len(class_label_candidates) == 1:
        metadata_target, metadata_task = class_label_candidates[0]

    return metadata_target, metadata_task


def _partition_tabular_columns(frame: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    scalar_columns: list[str] = []
    dropped_columns: dict[str, str] = {}
    for column in frame.columns:
        reason = _unsupported_column_reason(frame[column])
        if reason is None:
            scalar_columns.append(str(column))
        else:
            dropped_columns[str(column)] = reason
    return scalar_columns, dropped_columns


def _unsupported_column_reason(series: pd.Series) -> str | None:
    if is_numeric_dtype(series) or is_bool_dtype(series) or is_datetime64_any_dtype(series):
        return None
    if not is_object_dtype(series):
        return None

    sample = series.dropna().head(128).tolist()
    for value in sample:
        reason = _unsupported_value_reason(value)
        if reason is not None:
            return reason
    return None


def _unsupported_value_reason(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return "contains binary bytes values"
    if isinstance(value, np.ndarray):
        return "contains ndarray values"
    if isinstance(value, (list, tuple, set, frozenset)):
        return "contains sequence values"
    if isinstance(value, dict):
        return "contains mapping values"
    if not is_scalar(value):
        return f"contains unsupported {type(value).__name__} values"
    if isinstance(value, (str, int, float, bool, np.generic, datetime, date, time, pd.Timestamp, pd.Timedelta)):
        return None
    return None


def _resolve_target_column(
    frame: pd.DataFrame,
    preferred_target_column: str | None,
    preferred_task_type: str | None,
    metadata_target_column: str | None,
    metadata_task_type: str | None,
    dropped_columns: dict[str, str],
) -> tuple[str, str, str, list[TargetCandidate]]:
    if preferred_target_column:
        if preferred_target_column in dropped_columns:
            raise ValueError(
                f"Resolved target column `{preferred_target_column}` is not tabular: {dropped_columns[preferred_target_column]}."
            )
        if preferred_target_column not in frame.columns:
            raise KeyError(f"Target column `{preferred_target_column}` is missing from the dataset.")
        task_type = preferred_task_type or metadata_task_type or _infer_task_type(frame[preferred_target_column])
        return preferred_target_column, task_type, "manifest_or_cli", []

    if metadata_target_column:
        if metadata_target_column in dropped_columns:
            raise ValueError(
                f"Hugging Face metadata points at non-tabular target column `{metadata_target_column}`: "
                f"{dropped_columns[metadata_target_column]}."
            )
        if metadata_target_column in frame.columns:
            task_type = preferred_task_type or metadata_task_type or _infer_task_type(frame[metadata_target_column])
            return metadata_target_column, task_type, "source_metadata", []

    candidates = sorted(
        (_score_target_candidate(frame[column], column) for column in frame.columns),
        key=lambda candidate: candidate.score,
        reverse=True,
    )
    if not candidates:
        raise ValueError("Could not infer a target column because the dataset has no usable columns.")

    top_candidate = candidates[0]
    second_candidate = candidates[1] if len(candidates) > 1 else None
    score_gap = top_candidate.score - second_candidate.score if second_candidate is not None else top_candidate.score
    if top_candidate.name_score >= 6.0 and (second_candidate is None or top_candidate.name_score > second_candidate.name_score):
        task_type = preferred_task_type or metadata_task_type or top_candidate.task_type
        return top_candidate.column, task_type, "heuristic", candidates[:5]
    if top_candidate.score >= 3.0 and (top_candidate.name_score > 0 or score_gap >= 1.5):
        task_type = preferred_task_type or metadata_task_type or top_candidate.task_type
        return top_candidate.column, task_type, "heuristic", candidates[:5]

    formatted_candidates = ", ".join(
        f"{candidate.column}<{candidate.task_type}> score={candidate.score:.2f}"
        for candidate in candidates[:5]
    )
    raise ValueError(
        "Could not infer the target column confidently. "
        "Provide `--target-column` or store it in the dataset manifest. "
        f"Top candidates: {formatted_candidates}"
    )


def _score_target_candidate(series: pd.Series, column: str) -> TargetCandidate:
    values = series.dropna()
    string_values = values.astype(str)
    unique_count = int(string_values.nunique(dropna=True))
    unique_ratio = float(unique_count / max(len(string_values), 1)) if len(string_values) else 0.0
    avg_length = float(string_values.str.len().mean()) if len(string_values) else 0.0
    task_type = _infer_task_type(series)
    normalized_name = _normalize_name(column)
    name_score, keyword_hits = _target_name_score(normalized_name)

    score = name_score
    reasons: list[str] = []
    if name_score > 0:
        if keyword_hits:
            reasons.append(f"name includes target-like token(s): {', '.join(keyword_hits[:3])}")
        else:
            reasons.append("name matches a canonical target label")

    if _looks_like_identifier(column):
        score -= 5.0
        reasons.append("identifier-like column name")
    elif unique_ratio >= 0.98 and len(values) >= 32 and name_score == 0:
        score -= 2.5
        reasons.append("almost every row is unique")

    if unique_count <= 1:
        score -= 10.0
        reasons.append("constant or empty column")
    elif task_type == "binary":
        score += 3.0
        reasons.append("binary label cardinality")
    elif task_type == "multiclass":
        if unique_count <= 16:
            score += 2.5
            reasons.append("classification-like cardinality")
        else:
            score += 1.0
            reasons.append("discrete values")
    else:
        score += 1.0
        reasons.append("continuous numeric values")

    if task_type != "regression" and unique_count > max(64, int(len(values) * 0.2)) and name_score == 0:
        score -= 2.0
        reasons.append("too many classes for a likely label")
    if task_type != "regression" and avg_length > 48 and name_score == 0:
        score -= 1.5
        reasons.append("long free-form text values")

    return TargetCandidate(
        column=column,
        score=score,
        task_type=task_type,
        name_score=name_score,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        reason="; ".join(reasons) if reasons else "fallback heuristic",
    )


def _infer_task_type(series: pd.Series) -> str:
    values = series.dropna()
    if len(values) == 0:
        return "multiclass"
    if is_bool_dtype(series):
        return "binary"

    string_values = values.astype(str)
    unique_count = int(string_values.nunique(dropna=True))
    if unique_count <= 2:
        return "binary"

    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_ratio = float(numeric_values.notna().mean()) if len(values) else 0.0
    if numeric_ratio < 0.95:
        return "multiclass"

    dense_numeric = numeric_values.dropna().astype(float)
    if len(dense_numeric) == 0:
        return "multiclass"
    integer_like = bool(np.isclose(np.mod(dense_numeric, 1.0), 0.0).all())
    unique_ratio = float(unique_count / max(len(values), 1))
    if integer_like and unique_count <= min(32, max(3, len(values) // 20)):
        return "multiclass"
    if unique_ratio <= 0.1 and unique_count <= 64:
        return "multiclass"
    return "regression"


def _target_name_score(normalized_name: str) -> tuple[float, list[str]]:
    if normalized_name in _EXACT_TARGET_NAME_WEIGHTS:
        return _EXACT_TARGET_NAME_WEIGHTS[normalized_name], []

    keyword_hits = [
        keyword
        for keyword in _TARGET_KEYWORD_WEIGHTS
        if keyword in normalized_name
    ]
    if not keyword_hits:
        return 0.0, []
    score = min(sum(_TARGET_KEYWORD_WEIGHTS[keyword] for keyword in keyword_hits), 5.0)
    return score, keyword_hits


def _looks_like_identifier(column: str) -> bool:
    normalized_name = _normalize_name(column)
    return (
        normalized_name in {"id", "index", "row_id", "row_number", "record_id", "customer_id"}
        or normalized_name.endswith("_id")
        or normalized_name.endswith("_key")
        or normalized_name.endswith("_uuid")
    )


def _normalize_name(name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")
    return normalized or "column"
