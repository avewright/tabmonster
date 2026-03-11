"""Synthetic tabular data generators.

Implements several generative strategies that produce diverse synthetic tabular
datasets for pretraining.  Every generator follows the same interface and emits
a ``pandas.DataFrame`` + a ``dict`` metadata payload so the rest of the pipeline
can apply the standard ``build_schema`` → ``encode_frame`` path.

Generators
----------
GaussianMixtureGenerator
    Features drawn from a Gaussian mixture model, labels from a learned linear
    or tree decision boundary.
TreePriorGenerator
    tabPFN-style prior: features from a multivariate Gaussian, labels from a
    random decision tree applied to those features.
PolynomialGenerator
    Polynomial decision boundaries over Gaussian or uniform features.
SCMGenerator
    Structural Causal Model: each feature is a noisy linear combination of a
    random subset of prior features, label is a threshold over a leaf variable.
TimeSeriesSyntheticGenerator
    Generates AR/ARMA time series and extracts statistical tabular features from
    each series.  Useful as a source of numeric-only panels that differ from
    standard cross-sectional tabular data.
MixedTypeGenerator
    Wraps any numeric generator and adds synthetic categorical and ordinal
    columns to exercise the mixed-type embedding path.

Usage
-----
    df, meta = TreePriorGenerator(n_samples=1024, n_features=20).generate(seed=0)
    df, meta = MixedTypeGenerator(TreePriorGenerator()).generate(seed=42)
    df, meta = TimeSeriesSyntheticGenerator(n_series=500, series_length=50).generate(seed=1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _clip_feature_count(n: int, lo: int = 2, hi: int = 128) -> int:
    return max(lo, min(hi, n))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


@dataclass
class SyntheticDatasetMeta:
    generator: str
    n_samples: int
    n_features: int
    task_type: str          # "binary" | "multiclass" | "regression"
    n_classes: int | None   # None for regression
    feature_names: list[str]
    target_name: str
    extra: dict[str, Any] = field(default_factory=dict)


class BaseSyntheticGenerator:
    """Interface contract – subclasses implement ``generate``."""

    def generate(self, seed: int | None = None) -> tuple[pd.DataFrame, SyntheticDatasetMeta]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Gaussian Mixture Generator
# ---------------------------------------------------------------------------


class GaussianMixtureGenerator(BaseSyntheticGenerator):
    """Draw features from a Gaussian mixture; assign labels from a random
    linear threshold or a shallow decision tree.

    Parameters
    ----------
    n_samples : int
        Number of rows.
    n_features : int
        Number of numeric features.
    n_classes : int
        Number of classes (2 → binary, >2 → multiclass).
    n_components : int
        Number of Gaussian mixture components.
    label_strategy : "linear" | "quadratic" | "tree"
        How labels are derived from features.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_classes: int = 2,
        n_components: int = 4,
        label_strategy: str = "linear",
    ) -> None:
        self.n_samples = n_samples
        self.n_features = _clip_feature_count(n_features)
        self.n_classes = n_classes
        self.n_components = n_components
        self.label_strategy = label_strategy

    def generate(self, seed: int | None = None) -> tuple[pd.DataFrame, SyntheticDatasetMeta]:
        rng = _rng(seed)
        n, d = self.n_samples, self.n_features

        # --- build mixture components ---
        comp_means = rng.standard_normal((self.n_components, d)) * 2.0
        comp_scales = np.abs(rng.standard_normal((self.n_components, d))) + 0.3
        comp_weights = rng.dirichlet(np.ones(self.n_components))
        comp_ids = rng.choice(self.n_components, size=n, p=comp_weights)

        X = np.zeros((n, d), dtype=np.float32)
        for k in range(self.n_components):
            mask = comp_ids == k
            count = mask.sum()
            if count:
                X[mask] = rng.normal(comp_means[k], comp_scales[k], size=(count, d)).astype(np.float32)

        # --- derive labels ---
        y = self._labels(X, rng)

        feature_names = [f"feat_{i}" for i in range(d)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y

        task_type = "binary" if self.n_classes == 2 else "multiclass"
        meta = SyntheticDatasetMeta(
            generator="GaussianMixture",
            n_samples=n,
            n_features=d,
            task_type=task_type,
            n_classes=self.n_classes,
            feature_names=feature_names,
            target_name="target",
            extra={"n_components": self.n_components, "label_strategy": self.label_strategy},
        )
        return df, meta

    def _labels(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n, d = X.shape
        if self.label_strategy == "linear":
            w = rng.standard_normal(d).astype(np.float32)
            scores = X @ w
            if self.n_classes == 2:
                return (scores > np.median(scores)).astype(np.int64)
            thresholds = np.quantile(scores, np.linspace(0, 1, self.n_classes + 1)[1:-1])
            return np.digitize(scores, thresholds).astype(np.int64)
        elif self.label_strategy == "quadratic":
            W = rng.standard_normal((d, d)).astype(np.float32) * 0.3
            scores = np.einsum("ni,ij,nj->n", X, W, X) + X @ rng.standard_normal(d).astype(np.float32)
            if self.n_classes == 2:
                return (scores > np.median(scores)).astype(np.int64)
            thresholds = np.quantile(scores, np.linspace(0, 1, self.n_classes + 1)[1:-1])
            return np.digitize(scores, thresholds).astype(np.int64)
        elif self.label_strategy == "tree":
            return _random_tree_labels(X, rng, n_classes=self.n_classes, max_depth=4)
        else:
            raise ValueError(f"Unknown label_strategy: {self.label_strategy!r}")


# ---------------------------------------------------------------------------
# Tree Prior Generator  (tabPFN-style)
# ---------------------------------------------------------------------------


def _random_tree_labels(
    X: np.ndarray,
    rng: np.random.Generator,
    n_classes: int = 2,
    max_depth: int = 5,
) -> np.ndarray:
    """Assign labels using a randomly-grown decision tree applied to X."""
    n, d = X.shape
    labels = np.zeros(n, dtype=np.int64)
    _grow_tree(X, labels, rng, node_indices=np.arange(n), depth=0, max_depth=max_depth, n_classes=n_classes)
    return labels


def _grow_tree(
    X: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    node_indices: np.ndarray,
    depth: int,
    max_depth: int,
    n_classes: int,
) -> None:
    if len(node_indices) == 0:
        return
    if depth >= max_depth or len(node_indices) < 2:
        labels[node_indices] = rng.integers(0, n_classes)
        return
    d = X.shape[1]
    feat = rng.integers(0, d)
    col = X[node_indices, feat]
    lo, hi = col.min(), col.max()
    if lo >= hi:
        labels[node_indices] = rng.integers(0, n_classes)
        return
    threshold = float(rng.uniform(lo, hi))
    left = node_indices[col <= threshold]
    right = node_indices[col > threshold]
    if rng.random() < 0.1:
        labels[node_indices] = rng.integers(0, n_classes)
        return
    _grow_tree(X, labels, rng, left, depth + 1, max_depth, n_classes)
    _grow_tree(X, labels, rng, right, depth + 1, max_depth, n_classes)


class TreePriorGenerator(BaseSyntheticGenerator):
    """tabPFN-style synthetic prior.

    Features from a multivariate normal with a random covariance; labels from
    a randomly grown decision tree.  The depth and number of features are
    sampled at generation time to produce diverse datasets from a single
    generator instance.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int | tuple[int, int] = (5, 30),
        n_classes: int | tuple[int, int] = (2, 5),
        max_depth: int | tuple[int, int] = (2, 7),
        correlated_features: bool = True,
    ) -> None:
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.correlated_features = correlated_features

    def _sample_int(self, spec: int | tuple[int, int], rng: np.random.Generator) -> int:
        if isinstance(spec, int):
            return spec
        lo, hi = spec
        return int(rng.integers(lo, hi + 1))

    def generate(self, seed: int | None = None) -> tuple[pd.DataFrame, SyntheticDatasetMeta]:
        rng = _rng(seed)
        n = self.n_samples
        d = _clip_feature_count(self._sample_int(self.n_features, rng))
        n_classes = max(2, self._sample_int(self.n_classes, rng))
        max_depth = max(1, self._sample_int(self.max_depth, rng))

        if self.correlated_features:
            # Sample a random correlation matrix via random Cholesky
            A = rng.standard_normal((d, d)).astype(np.float32)
            cov = (A @ A.T) / d + np.eye(d, dtype=np.float32) * 0.1
            mean = rng.standard_normal(d).astype(np.float32)
            X = rng.multivariate_normal(mean.astype(np.float64), cov.astype(np.float64), size=n).astype(np.float32)
        else:
            X = rng.standard_normal((n, d)).astype(np.float32)

        y = _random_tree_labels(X, rng, n_classes=n_classes, max_depth=max_depth)

        feature_names = [f"feat_{i}" for i in range(d)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y

        task_type = "binary" if n_classes == 2 else "multiclass"
        meta = SyntheticDatasetMeta(
            generator="TreePrior",
            n_samples=n,
            n_features=d,
            task_type=task_type,
            n_classes=n_classes,
            feature_names=feature_names,
            target_name="target",
            extra={"max_depth": max_depth},
        )
        return df, meta


# ---------------------------------------------------------------------------
# Polynomial Generator
# ---------------------------------------------------------------------------


class PolynomialGenerator(BaseSyntheticGenerator):
    """Polynomial decision boundaries over Gaussian / uniform features.

    Useful for stress-testing numeric tokenizers because the decision boundary
    is smooth and cannot be approximated by axis-aligned splits.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        degree: int = 2,
        n_classes: int = 2,
        noise: float = 0.1,
        feature_distribution: str = "normal",
    ) -> None:
        self.n_samples = n_samples
        self.n_features = _clip_feature_count(n_features)
        self.degree = degree
        self.n_classes = n_classes
        self.noise = noise
        self.feature_distribution = feature_distribution

    def generate(self, seed: int | None = None) -> tuple[pd.DataFrame, SyntheticDatasetMeta]:
        rng = _rng(seed)
        n, d = self.n_samples, self.n_features

        if self.feature_distribution == "uniform":
            X = rng.uniform(-2, 2, size=(n, d)).astype(np.float32)
        else:
            X = rng.standard_normal((n, d)).astype(np.float32)

        # build polynomial features up to requested degree
        powers = []
        for deg in range(1, self.degree + 1):
            # random pairs for interaction terms
            cols = rng.integers(0, d, size=(d,))
            powers.append(X[:, cols] ** deg)
        poly_X = np.concatenate(powers, axis=1)
        w = rng.standard_normal(poly_X.shape[1]).astype(np.float32)
        scores = poly_X @ w + rng.normal(0, self.noise, size=n).astype(np.float32)

        if self.n_classes == 2:
            y = (scores > np.median(scores)).astype(np.int64)
        else:
            thresholds = np.quantile(scores, np.linspace(0, 1, self.n_classes + 1)[1:-1])
            y = np.digitize(scores, thresholds).astype(np.int64)

        feature_names = [f"x_{i}" for i in range(d)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y

        task_type = "binary" if self.n_classes == 2 else "multiclass"
        meta = SyntheticDatasetMeta(
            generator="Polynomial",
            n_samples=n,
            n_features=d,
            task_type=task_type,
            n_classes=self.n_classes,
            feature_names=feature_names,
            target_name="target",
            extra={"degree": self.degree, "noise": self.noise},
        )
        return df, meta


# ---------------------------------------------------------------------------
# Regression synthetic generator
# ---------------------------------------------------------------------------


class RegressionSyntheticGenerator(BaseSyntheticGenerator):
    """Generate a regression dataset with configurable noise and function form.

    Supports ``linear``, ``additive``, and ``interaction`` response functions.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        response_type: str = "additive",
        noise_std: float = 0.1,
        feature_distribution: str = "normal",
    ) -> None:
        self.n_samples = n_samples
        self.n_features = _clip_feature_count(n_features)
        self.response_type = response_type
        self.noise_std = noise_std
        self.feature_distribution = feature_distribution

    def generate(self, seed: int | None = None) -> tuple[pd.DataFrame, SyntheticDatasetMeta]:
        rng = _rng(seed)
        n, d = self.n_samples, self.n_features

        if self.feature_distribution == "uniform":
            X = rng.uniform(-3, 3, size=(n, d)).astype(np.float32)
        elif self.feature_distribution == "lognormal":
            X = rng.lognormal(0, 1, size=(n, d)).astype(np.float32)
        else:
            X = rng.standard_normal((n, d)).astype(np.float32)

        w = rng.standard_normal(d).astype(np.float32)

        if self.response_type == "linear":
            y_clean = X @ w
        elif self.response_type == "additive":
            # Random nonlinear per-feature transforms
            transforms = rng.choice(["id", "sq", "sin", "abs"], size=d)
            parts = []
            for i, t in enumerate(transforms):
                xi = X[:, i]
                parts.append(
                    xi if t == "id" else
                    xi ** 2 if t == "sq" else
                    np.sin(xi * float(w[i] * 3)) if t == "sin" else
                    np.abs(xi)
                )
            y_clean = np.stack(parts, axis=1) @ w
        elif self.response_type == "interaction":
            W = rng.standard_normal((d, d)).astype(np.float32) * 0.5
            y_clean = np.einsum("ni,ij,nj->n", X, W, X).astype(np.float32)
        else:
            raise ValueError(f"Unknown response_type: {self.response_type!r}")

        noise = rng.normal(0, self.noise_std, size=n).astype(np.float32)
        y = (y_clean + noise).astype(np.float32)

        feature_names = [f"x_{i}" for i in range(d)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y

        meta = SyntheticDatasetMeta(
            generator="RegressionSynthetic",
            n_samples=n,
            n_features=d,
            task_type="regression",
            n_classes=None,
            feature_names=feature_names,
            target_name="target",
            extra={"response_type": self.response_type, "noise_std": self.noise_std},
        )
        return df, meta


# ---------------------------------------------------------------------------
# SCM Generator
# ---------------------------------------------------------------------------


class SCMGenerator(BaseSyntheticGenerator):
    """Structural Causal Model generator.

    Each feature is a noisy linear combination of a random subset of
    previously generated features.  The label is computed from the last
    (downstream) feature via a threshold or linear rule.  This creates
    datasets with causal structure that differs qualitatively from i.i.d.
    Gaussian noise.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 15,
        n_classes: int = 2,
        noise_std: float = 0.5,
        edge_probability: float = 0.3,
    ) -> None:
        self.n_samples = n_samples
        self.n_features = _clip_feature_count(n_features)
        self.n_classes = n_classes
        self.noise_std = noise_std
        self.edge_probability = edge_probability

    def generate(self, seed: int | None = None) -> tuple[pd.DataFrame, SyntheticDatasetMeta]:
        rng = _rng(seed)
        n, d = self.n_samples, self.n_features

        X = np.zeros((n, d), dtype=np.float32)
        for i in range(d):
            noise = rng.normal(0, self.noise_std, size=n).astype(np.float32)
            if i == 0:
                X[:, i] = rng.standard_normal(n).astype(np.float32)
            else:
                parents = [j for j in range(i) if rng.random() < self.edge_probability]
                if not parents:
                    parents = [rng.integers(0, i)]
                w = rng.standard_normal(len(parents)).astype(np.float32)
                X[:, i] = X[:, parents] @ w + noise

        # label from downstream feature
        scores = X[:, -1]
        if self.n_classes == 2:
            y = (scores > np.median(scores)).astype(np.int64)
        else:
            thresholds = np.quantile(scores, np.linspace(0, 1, self.n_classes + 1)[1:-1])
            y = np.digitize(scores, thresholds).astype(np.int64)

        feature_names = [f"x_{i}" for i in range(d)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y

        task_type = "binary" if self.n_classes == 2 else "multiclass"
        meta = SyntheticDatasetMeta(
            generator="SCM",
            n_samples=n,
            n_features=d,
            task_type=task_type,
            n_classes=self.n_classes,
            feature_names=feature_names,
            target_name="target",
            extra={"edge_probability": self.edge_probability, "noise_std": self.noise_std},
        )
        return df, meta


# ---------------------------------------------------------------------------
# Mixed Type Generator (wrapper)
# ---------------------------------------------------------------------------


_ORDINAL_WORDS = [
    ["low", "medium", "high"],
    ["small", "medium", "large", "xlarge"],
    ["poor", "fair", "good", "excellent"],
    ["never", "rarely", "sometimes", "often", "always"],
    ["strongly_disagree", "disagree", "neutral", "agree", "strongly_agree"],
]


class MixedTypeGenerator(BaseSyntheticGenerator):
    """Wraps any numeric generator and adds synthetic categorical columns.

    The extra categorical columns are derived from existing numeric columns via
    binning so they are not independent noise — they carry real signal.  This
    exercises the mixed-type embedding path of the model.
    """

    def __init__(
        self,
        base_generator: BaseSyntheticGenerator,
        n_categorical: int = 4,
        n_ordinal: int = 2,
        n_binary_indicator: int = 3,
    ) -> None:
        self.base = base_generator
        self.n_categorical = n_categorical
        self.n_ordinal = n_ordinal
        self.n_binary_indicator = n_binary_indicator

    def generate(self, seed: int | None = None) -> tuple[pd.DataFrame, SyntheticDatasetMeta]:
        rng = _rng(seed)
        df, meta = self.base.generate(seed=seed)

        num_cols = meta.feature_names
        if not num_cols:
            return df, meta

        new_cols: list[str] = []

        # Nominal categorical: hash-bin a numeric feature into random category labels
        for i in range(self.n_categorical):
            src = num_cols[rng.integers(0, len(num_cols))]
            n_cats = int(rng.integers(3, 12))
            labels = [f"cat{i}_v{k}" for k in range(n_cats)]
            bins = np.quantile(df[src].dropna(), np.linspace(0, 1, n_cats + 1))
            bins = np.unique(bins)
            if len(bins) < 2:
                bins = np.array([df[src].min() - 1, df[src].max() + 1])
            indices = np.clip(np.digitize(df[src].values, bins[1:]), 0, len(labels) - 1)
            col_name = f"cat_{i}"
            df[col_name] = [labels[idx] for idx in indices]
            new_cols.append(col_name)

        # Ordinal: bin a numeric feature into ordered words
        for i in range(self.n_ordinal):
            src = num_cols[rng.integers(0, len(num_cols))]
            words = _ORDINAL_WORDS[i % len(_ORDINAL_WORDS)]
            n_cats = len(words)
            bins = np.quantile(df[src].dropna(), np.linspace(0, 1, n_cats + 1))
            bins = np.unique(bins)
            if len(bins) < 2:
                bins = np.array([df[src].min() - 1, df[src].max() + 1])
            indices = np.clip(np.digitize(df[src].values, bins[1:]), 0, n_cats - 1)
            col_name = f"ord_{i}"
            df[col_name] = [words[idx] for idx in indices]
            new_cols.append(col_name)

        # Binary indicator: threshold a numeric feature
        for i in range(self.n_binary_indicator):
            src = num_cols[rng.integers(0, len(num_cols))]
            median = float(df[src].median())
            col_name = f"flag_{i}"
            df[col_name] = (df[src] > median).astype(int)
            new_cols.append(col_name)

        all_feature_names = meta.feature_names + new_cols
        updated_meta = SyntheticDatasetMeta(
            generator=f"MixedType({meta.generator})",
            n_samples=meta.n_samples,
            n_features=len(all_feature_names),
            task_type=meta.task_type,
            n_classes=meta.n_classes,
            feature_names=all_feature_names,
            target_name=meta.target_name,
            extra={**meta.extra, "n_categorical": self.n_categorical, "n_ordinal": self.n_ordinal},
        )
        return df, updated_meta


# ---------------------------------------------------------------------------
# Time Series Synthetic Generator
# ---------------------------------------------------------------------------


class TimeSeriesSyntheticGenerator(BaseSyntheticGenerator):
    """Generate AR/ARMA synthetic time series and extract tabular features.

    Each row in the output corresponds to one time series (a "series ID").
    Features are statistical summaries extracted from the series.  The label
    is generated from the mean of the series (binary/multiclass) or the last
    value (regression).
    """

    def __init__(
        self,
        n_series: int = 500,
        series_length: int = 64,
        ar_order: int = 3,
        ma_order: int = 1,
        task_type: str = "binary",
        n_classes: int = 2,
        add_seasonal: bool = True,
        add_trend: bool = True,
    ) -> None:
        self.n_series = n_series
        self.series_length = series_length
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.task_type = task_type
        self.n_classes = n_classes
        self.add_seasonal = add_seasonal
        self.add_trend = add_trend

    def generate(self, seed: int | None = None) -> tuple[pd.DataFrame, SyntheticDatasetMeta]:
        rng = _rng(seed)
        n, T = self.n_series, self.series_length

        # generate distinct ARMA parameters per series
        ar_coeffs = rng.uniform(-0.4, 0.4, size=(n, self.ar_order)).astype(np.float32)
        ma_coeffs = rng.uniform(-0.3, 0.3, size=(n, self.ma_order)).astype(np.float32)
        noise_std = rng.uniform(0.05, 1.5, size=n).astype(np.float32)
        trends = rng.uniform(-0.02, 0.02, size=n).astype(np.float32) if self.add_trend else np.zeros(n, np.float32)
        periods = rng.integers(4, 32, size=n) if self.add_seasonal else None
        amplitudes = rng.uniform(0.0, 1.5, size=n).astype(np.float32) if self.add_seasonal else np.zeros(n, np.float32)

        rows = []
        raw_series = np.zeros((n, T), dtype=np.float32)

        for i in range(n):
            eps = rng.normal(0, float(noise_std[i]), size=T + self.ma_order).astype(np.float32)
            series = np.zeros(T + max(self.ar_order, self.ma_order), dtype=np.float32)

            for t in range(max(self.ar_order, self.ma_order), T + max(self.ar_order, self.ma_order)):
                ar_part = sum(ar_coeffs[i, k] * series[t - 1 - k] for k in range(self.ar_order))
                ma_part = sum(ma_coeffs[i, k] * eps[t - k] for k in range(self.ma_order))
                trend_part = trends[i] * t
                if self.add_seasonal and periods is not None:
                    seasonal_part = amplitudes[i] * math.sin(2 * math.pi * t / float(periods[i]))
                else:
                    seasonal_part = 0.0
                series[t] = ar_part + ma_part + trend_part + seasonal_part + eps[t]

            ts = series[max(self.ar_order, self.ma_order):]
            raw_series[i] = ts.astype(np.float32)
            rows.append(_extract_ts_features(ts))

        feature_names = list(rows[0].keys())
        df = pd.DataFrame(rows)

        # label from series aggregate
        series_mean = raw_series.mean(axis=1)
        if self.task_type == "regression":
            df["target"] = series_mean.astype(np.float32)
        else:
            if self.n_classes == 2:
                df["target"] = (series_mean > np.median(series_mean)).astype(np.int64)
            else:
                thresholds = np.quantile(series_mean, np.linspace(0, 1, self.n_classes + 1)[1:-1])
                df["target"] = np.digitize(series_mean, thresholds).astype(np.int64)

        meta = SyntheticDatasetMeta(
            generator="TimeSeriesSynthetic",
            n_samples=n,
            n_features=len(feature_names),
            task_type=self.task_type,
            n_classes=self.n_classes if self.task_type != "regression" else None,
            feature_names=feature_names,
            target_name="target",
            extra={"series_length": T, "ar_order": self.ar_order, "ma_order": self.ma_order},
        )
        return df, meta


def _extract_ts_features(ts: np.ndarray) -> dict[str, float]:
    """Compute statistical features from a 1-D time series array."""
    n = len(ts)
    mean = float(ts.mean())
    std = float(ts.std()) + 1e-8
    features: dict[str, float] = {
        "ts_mean": mean,
        "ts_std": std,
        "ts_min": float(ts.min()),
        "ts_max": float(ts.max()),
        "ts_range": float(ts.max() - ts.min()),
        "ts_median": float(np.median(ts)),
        "ts_q25": float(np.percentile(ts, 25)),
        "ts_q75": float(np.percentile(ts, 75)),
        "ts_iqr": float(np.percentile(ts, 75) - np.percentile(ts, 25)),
    }
    # skewness (manual for speed)
    skew = float(np.mean(((ts - mean) / std) ** 3))
    kurt = float(np.mean(((ts - mean) / std) ** 4)) - 3.0
    features["ts_skewness"] = skew
    features["ts_kurtosis"] = kurt
    # trend slope via simple least squares
    t_axis = np.arange(n, dtype=np.float32)
    t_mean = t_axis.mean()
    cov = float(np.mean((t_axis - t_mean) * (ts - mean)))
    t_var = float(np.var(t_axis)) + 1e-8
    features["ts_trend_slope"] = cov / t_var
    # autocorrelation at lag 1
    if n > 1:
        features["ts_acf1"] = float(np.corrcoef(ts[:-1], ts[1:])[0, 1])
    else:
        features["ts_acf1"] = 0.0
    # autocorrelation at lag 2
    if n > 2:
        features["ts_acf2"] = float(np.corrcoef(ts[:-2], ts[2:])[0, 1])
    else:
        features["ts_acf2"] = 0.0
    # energy (mean squared value)
    features["ts_energy"] = float(np.mean(ts ** 2))
    # zero crossing rate
    crossings = int(np.sum(np.diff(np.sign(ts - mean)) != 0))
    features["ts_zero_crossing_rate"] = crossings / max(n - 1, 1)
    # first and last value
    features["ts_first"] = float(ts[0])
    features["ts_last"] = float(ts[-1])
    features["ts_change"] = float(ts[-1] - ts[0])
    return features


# ---------------------------------------------------------------------------
# Convenience: sample random generator
# ---------------------------------------------------------------------------


_GENERATOR_CLASSES = [
    GaussianMixtureGenerator,
    TreePriorGenerator,
    PolynomialGenerator,
    SCMGenerator,
    RegressionSyntheticGenerator,
    TimeSeriesSyntheticGenerator,
]


def sample_random_generator(seed: int | None = None) -> BaseSyntheticGenerator:
    """Return a randomly configured generator instance for diversity."""
    rng = _rng(seed)
    cls = _GENERATOR_CLASSES[int(rng.integers(0, len(_GENERATOR_CLASSES)))]
    n_samples = int(rng.integers(256, 8192))
    n_features = int(rng.integers(4, 32))
    n_classes = int(rng.integers(2, 6))
    if cls == RegressionSyntheticGenerator:
        return cls(
            n_samples=n_samples,
            n_features=n_features,
            response_type=str(rng.choice(["linear", "additive", "interaction"])),
            noise_std=float(rng.uniform(0.01, 0.5)),
        )
    if cls == TimeSeriesSyntheticGenerator:
        task = str(rng.choice(["binary", "regression"]))
        return cls(
            n_series=n_samples,
            series_length=int(rng.integers(32, 128)),
            ar_order=int(rng.integers(1, 5)),
            ma_order=int(rng.integers(0, 3)),
            task_type=task,
            n_classes=n_classes,
        )
    if cls == GaussianMixtureGenerator:
        return cls(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_components=int(rng.integers(2, 8)),
            label_strategy=str(rng.choice(["linear", "quadratic", "tree"])),
        )
    if cls == TreePriorGenerator:
        return MixedTypeGenerator(
            cls(n_samples=n_samples),
            n_categorical=int(rng.integers(0, 6)),
            n_ordinal=int(rng.integers(0, 3)),
            n_binary_indicator=int(rng.integers(0, 4)),
        )
    if cls == PolynomialGenerator:
        return cls(
            n_samples=n_samples,
            n_features=n_features,
            degree=int(rng.integers(2, 5)),
            n_classes=n_classes,
        )
    if cls == SCMGenerator:
        return cls(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            edge_probability=float(rng.uniform(0.2, 0.6)),
        )
    return cls(n_samples=n_samples, n_features=n_features)  # type: ignore[call-arg]


def generate_synthetic_batch(
    n_datasets: int = 16,
    seed: int | None = None,
) -> list[tuple[pd.DataFrame, SyntheticDatasetMeta]]:
    """Generate ``n_datasets`` diverse synthetic datasets.

    Returns a list of ``(DataFrame, SyntheticDatasetMeta)`` tuples.
    Each dataset is generated from a randomly selected and configured generator,
    producing diverse task types, feature counts, and generative mechanisms.
    """
    rng = _rng(seed)
    results: list[tuple[pd.DataFrame, SyntheticDatasetMeta]] = []
    for i in range(n_datasets):
        child_seed = int(rng.integers(0, 2**31))
        gen = sample_random_generator(seed=child_seed)
        df, meta = gen.generate(seed=child_seed)
        results.append((df, meta))
    return results
