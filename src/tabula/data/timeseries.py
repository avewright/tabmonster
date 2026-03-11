"""Real-data time-series-to-tabular feature extractor.

This module transforms datasets that contain temporal columns (datetime,
timestamps, integer time indices, etc.) into rich tabular feature matrices
suitable for training.

Unlike ``synthetic.py`` which *generates* ARMA time series, this module works
on *real* dataframes that happen to contain time-indexed observations.

Key transformations
-------------------
1. **Temporal calendar features** – year, month, week-of-year, day-of-week,
   hour, minute, is_weekend, quarter, is_month_start/end flags.
2. **Lag features** – autoregressive lags of a target column.
3. **Rolling-window statistics** – mean, std, min, max, median over configurable
   windows in a sorted time axis.
4. **Panel-level aggregations** – per-entity (group) mean, std, count, rank.
5. **Frequency-domain features** – FFT magnitudes for dominant frequencies
   (requires series of numeric target).

Usage
-----
    from tabula.data.timeseries import TimeSeriesToTabularTransformer

    t = TimeSeriesToTabularTransformer(
        datetime_col="date",
        target_col="sales",
        entity_col="store_id",
        lags=[1, 7, 14, 28],
        rolling_windows=[7, 28],
    )
    df_features = t.fit_transform(raw_df)

CLI helper (see cli.py)::

    tabula data extract-ts-features --input data/raw/sales/train.csv \
        --datetime-col date --target-col sales --output data/raw/sales_ts/train.csv
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utility: detect temporal columns
# ---------------------------------------------------------------------------


def detect_temporal_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that look like datetime or numeric time indices.

    Detection heuristics:
      - dtype is already datetime64
      - dtype is object and >= 80% of non-null values parse as dates
      - column name contains common temporal keywords and values are
        monotonically increasing integers
    """
    temporal: list[str] = []
    datetime_keywords = {
        "date", "time", "timestamp", "datetime", "year", "month", "week",
        "day", "hour", "period", "quarter", "dt", "ts",
    }
    for col in df.columns:
        s = df[col]
        # already datetime
        if pd.api.types.is_datetime64_any_dtype(s):
            temporal.append(col)
            continue
        # try to parse as date string
        if s.dtype == object or s.dtype.name == "string":
            sample = s.dropna().head(200)
            if len(sample) == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
                frac_valid = parsed.notna().mean()
                if frac_valid >= 0.8:
                    temporal.append(col)
                    continue
            except Exception:
                pass
        # numeric column with temporal-sounding name
        col_lower = col.lower().replace(" ", "_")
        if pd.api.types.is_numeric_dtype(s):
            for kw in datetime_keywords:
                if kw in col_lower:
                    temporal.append(col)
                    break
    return temporal


# ---------------------------------------------------------------------------
# Calendar feature extraction
# ---------------------------------------------------------------------------


def extract_calendar_features(
    df: pd.DataFrame,
    datetime_col: str,
    prefix: str | None = None,
    drop_original: bool = False,
) -> pd.DataFrame:
    """Add calendar features derived from a datetime column.

    New columns added (all prefixed with ``<prefix>_`` if prefix given):
      - year, month, quarter, week_of_year, day_of_month, day_of_week, hour,
        minute, is_weekend, is_month_start, is_month_end

    The original column is converted to datetime if not already;
    ``NaT`` values get 0-filled calendar features.
    """
    df = df.copy()
    pfx = (prefix or datetime_col) + "_"

    # coerce to datetime
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

    dt = df[datetime_col]
    df[pfx + "year"] = dt.dt.year.fillna(0).astype(int)
    df[pfx + "month"] = dt.dt.month.fillna(0).astype(int)
    df[pfx + "quarter"] = dt.dt.quarter.fillna(0).astype(int)
    df[pfx + "week_of_year"] = dt.dt.isocalendar().week.fillna(0).astype(int)
    df[pfx + "day_of_month"] = dt.dt.day.fillna(0).astype(int)
    df[pfx + "day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)
    df[pfx + "hour"] = dt.dt.hour.fillna(0).astype(int)
    df[pfx + "minute"] = dt.dt.minute.fillna(0).astype(int)
    df[pfx + "is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df[pfx + "is_month_start"] = dt.dt.is_month_start.astype(int)
    df[pfx + "is_month_end"] = dt.dt.is_month_end.astype(int)

    # Cyclical encoding for month and day-of-week
    df[pfx + "month_sin"] = np.sin(2 * math.pi * dt.dt.month.fillna(0) / 12)
    df[pfx + "month_cos"] = np.cos(2 * math.pi * dt.dt.month.fillna(0) / 12)
    df[pfx + "dow_sin"] = np.sin(2 * math.pi * dt.dt.dayofweek.fillna(0) / 7)
    df[pfx + "dow_cos"] = np.cos(2 * math.pi * dt.dt.dayofweek.fillna(0) / 7)

    if drop_original:
        df = df.drop(columns=[datetime_col])
    return df


# ---------------------------------------------------------------------------
# Lag / rolling features
# ---------------------------------------------------------------------------


def extract_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int],
    group_col: str | None = None,
    sort_col: str | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add autoregressive lag features for ``target_col``.

    Parameters
    ----------
    lags : list[int]
        Lag steps, e.g. ``[1, 7, 14]``.
    group_col : str, optional
        If provided, lags are computed within each group (panel data).
    sort_col : str, optional
        Sort by this column before computing lags.  If None, uses current
        row order.
    """
    df = df.copy()
    pfx = (prefix or target_col) + "_lag"
    if sort_col and sort_col in df.columns:
        df = df.sort_values(sort_col)
    series = df.groupby(group_col)[target_col] if group_col else df[target_col]
    for lag in lags:
        col_name = f"{pfx}{lag}"
        if group_col:
            df[col_name] = series.shift(lag).reset_index(drop=True)
        else:
            df[col_name] = df[target_col].shift(lag)
    return df


def extract_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: list[int],
    group_col: str | None = None,
    sort_col: str | None = None,
    stats: list[str] | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add rolling-window statistics for ``target_col``.

    Parameters
    ----------
    windows : list[int]
        Rolling window sizes.
    stats : list[str], optional
        Stats to compute.  Defaults to ``["mean", "std", "min", "max"]``.
    """
    df = df.copy()
    pfx = prefix or target_col
    if stats is None:
        stats = ["mean", "std", "min", "max"]
    if sort_col and sort_col in df.columns:
        df = df.sort_values(sort_col)

    for window in windows:
        series = (
            df.groupby(group_col)[target_col]
            if group_col
            else df[target_col]
        )
        for stat in stats:
            col_name = f"{pfx}_roll{window}_{stat}"
            rolled = series.rolling(window, min_periods=1)
            if group_col:
                # groupby rolling returns multi-index; reset
                vals = getattr(rolled, stat)().reset_index(level=0, drop=True)
            else:
                vals = getattr(rolled, stat)()
            df[col_name] = vals.astype(float)
    return df


# ---------------------------------------------------------------------------
# Panel / entity level features
# ---------------------------------------------------------------------------


def extract_panel_features(
    df: pd.DataFrame,
    target_col: str,
    entity_col: str,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add within-group aggregation features.

    Adds per-entity (group) mean, std, count, and rank (within group) of
    ``target_col``.  Useful for panel / longitudinal data.
    """
    df = df.copy()
    pfx = prefix or target_col

    grouped = df.groupby(entity_col)[target_col]
    df[f"{pfx}_entity_mean"] = grouped.transform("mean")
    df[f"{pfx}_entity_std"] = grouped.transform("std").fillna(0)
    df[f"{pfx}_entity_count"] = grouped.transform("count")
    df[f"{pfx}_entity_rank"] = grouped.rank(pct=True)
    return df


# ---------------------------------------------------------------------------
# Frequency-domain features
# ---------------------------------------------------------------------------


def extract_fft_features(
    series: pd.Series | np.ndarray,
    n_components: int = 10,
    prefix: str = "fft",
) -> dict[str, float]:
    """Return the magnitudes of dominant FFT frequency bins.

    Parameters
    ----------
    series : array-like
        1D numeric time series.
    n_components : int
        Number of top frequency magnitudes to return.

    Returns
    -------
    dict mapping ``fft_0``, ``fft_1``, … to magnitudes.
    """
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 4:
        return {f"{prefix}_{i}": 0.0 for i in range(n_components)}
    spectrum = np.abs(np.fft.rfft(arr - arr.mean()))
    # Normalize
    if spectrum.max() > 0:
        spectrum = spectrum / spectrum.max()
    # Return top n_components bins (skip DC component at index 0)
    top_n = spectrum[1 : n_components + 1]
    result = {}
    for i, mag in enumerate(top_n):
        result[f"{prefix}_{i}"] = float(mag)
    # Pad if series shorter than n_components
    for i in range(len(top_n), n_components):
        result[f"{prefix}_{i}"] = 0.0
    return result


# ---------------------------------------------------------------------------
# Main transformer class
# ---------------------------------------------------------------------------


@dataclass
class TimeSeriesToTabularTransformer:
    """Transform a time-indexed DataFrame into a rich tabular feature matrix.

    This class is a stateless transformer (no learned parameters).  It wraps
    all the individual feature extraction functions in a single
    ``fit_transform`` / ``transform`` interface compatible with sklearn-style
    pipelines.

    Parameters
    ----------
    datetime_col : str, optional
        Column containing datetime values.  If None, calendar features are
        skipped.
    target_col : str, optional
        Numeric target column for lag/rolling/FFT features.  If None, those
        feature groups are skipped.
    entity_col : str, optional
        Grouping column for panel data.
    sort_col : str, optional
        Column to sort by before lag/rolling computations.  Defaults to
        ``datetime_col`` if set.
    lags : list[int]
        Lag offsets.
    rolling_windows : list[int]
        Window sizes for rolling statistics.
    rolling_stats : list[str]
        Which statistics to compute in rolling windows.
    add_calendar : bool
        Whether to add calendar features.
    add_panel : bool
        Whether to add panel (entity-level) aggregation features.
    fft_components : int
        Number of FFT magnitude features to add (0 = skip).
    drop_datetime_col : bool
        If True, remove the datetime column from the output.
    drop_na_rows : bool
        If True, drop rows with NaN in any new feature.  Useful when large
        lags produce leading NaN rows.
    """

    datetime_col: str | None = None
    target_col: str | None = None
    entity_col: str | None = None
    sort_col: str | None = None
    lags: list[int] = field(default_factory=lambda: [1, 7, 14])
    rolling_windows: list[int] = field(default_factory=lambda: [7, 28])
    rolling_stats: list[str] = field(
        default_factory=lambda: ["mean", "std", "min", "max"]
    )
    add_calendar: bool = True
    add_panel: bool = True
    fft_components: int = 10
    drop_datetime_col: bool = True
    drop_na_rows: bool = False

    def _effective_sort_col(self) -> str | None:
        return self.sort_col or self.datetime_col

    def fit(self, df: pd.DataFrame) -> "TimeSeriesToTabularTransformer":
        """No-op; transformer is stateless."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all enabled feature extraction steps."""
        df = df.copy()

        # Sort by time
        sort_col = self._effective_sort_col()
        if sort_col and sort_col in df.columns:
            df = df.sort_values(sort_col).reset_index(drop=True)

        # Calendar features
        if self.add_calendar and self.datetime_col and self.datetime_col in df.columns:
            df = extract_calendar_features(
                df,
                datetime_col=self.datetime_col,
                drop_original=False,
            )

        # Lag features
        if self.target_col and self.target_col in df.columns and self.lags:
            df = extract_lag_features(
                df,
                target_col=self.target_col,
                lags=self.lags,
                group_col=self.entity_col,
                sort_col=sort_col,
            )

        # Rolling features
        if self.target_col and self.target_col in df.columns and self.rolling_windows:
            df = extract_rolling_features(
                df,
                target_col=self.target_col,
                windows=self.rolling_windows,
                group_col=self.entity_col,
                sort_col=sort_col,
                stats=self.rolling_stats,
            )

        # Panel features
        if (
            self.add_panel
            and self.entity_col
            and self.entity_col in df.columns
            and self.target_col
            and self.target_col in df.columns
        ):
            df = extract_panel_features(
                df,
                target_col=self.target_col,
                entity_col=self.entity_col,
            )

        # FFT features — computed once per entity (or globally) and joined
        if self.fft_components > 0 and self.target_col and self.target_col in df.columns:
            df = self._add_fft_features(df)

        # Drop the datetime column if requested
        if self.drop_datetime_col and self.datetime_col and self.datetime_col in df.columns:
            df = df.drop(columns=[self.datetime_col])

        if self.drop_na_rows:
            df = df.dropna().reset_index(drop=True)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _add_fft_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute FFT features and broadcast back to row level."""
        if self.entity_col and self.entity_col in df.columns:
            # Compute per entity
            fft_rows: dict[Any, dict[str, float]] = {}
            for entity_val, grp in df.groupby(self.entity_col):
                fft_rows[entity_val] = extract_fft_features(
                    grp[self.target_col], n_components=self.fft_components
                )
            fft_col_names = list(next(iter(fft_rows.values())).keys())
            for col in fft_col_names:
                df[col] = df[self.entity_col].map(
                    {k: v[col] for k, v in fft_rows.items()}
                )
        else:
            fft_vals = extract_fft_features(
                df[self.target_col], n_components=self.fft_components
            )
            for col, val in fft_vals.items():
                df[col] = val
        return df


# ---------------------------------------------------------------------------
# Auto-transform: detect temporal columns and apply transformer
# ---------------------------------------------------------------------------


def auto_extract_timeseries_features(
    df: pd.DataFrame,
    target_col: str | None = None,
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    drop_datetime: bool = True,
) -> pd.DataFrame:
    """High-level convenience: detect temporal columns and enrich a DataFrame.

    Useful for augmenting a raw dataset before passing to ``prepare_dataset``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with at least one temporal column.
    target_col : str, optional
        Override target column for lag/rolling/FFT features.  If None, the
        function picks the first numeric non-temporal column.
    lags / rolling_windows : list[int], optional
        Override defaults.

    Returns
    -------
    Enriched DataFrame.  If no temporal columns detected, returns df unchanged.
    """
    temporal_cols = detect_temporal_columns(df)
    if not temporal_cols:
        return df

    dt_col = temporal_cols[0]

    # Pick target as first numeric column that isn't the datetime col
    if target_col is None:
        for col in df.columns:
            if col != dt_col and pd.api.types.is_numeric_dtype(df[col]):
                target_col = col
                break

    transformer = TimeSeriesToTabularTransformer(
        datetime_col=dt_col,
        target_col=target_col,
        lags=lags or [1, 7, 14],
        rolling_windows=rolling_windows or [7, 28],
        drop_datetime_col=drop_datetime,
    )
    return transformer.fit_transform(df)
