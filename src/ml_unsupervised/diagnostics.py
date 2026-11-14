# Path: src/ml_unsupervised/diagnostics.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-helmi

"""Interpretation and diagnostics helpers for unsupervised workflows.

These utilities add senior data-science diagnostics (profiles, comparisons,
stability checks) while keeping core estimators simple and reusable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import adjusted_rand_score

from .clustering import ClusteringEngine
from .evaluation import ClusteringEvaluator, ClusteringMetrics
from .utils import as_2d_float_array


@dataclass(frozen=True, slots=True)
class MethodComparison:
    method: str
    params: dict[str, Any]
    labels: NDArray[np.int_]
    metrics: ClusteringMetrics | None
    n_clusters: int
    noise_ratio: float


def cluster_profile(
    frame: pd.DataFrame, labels: ArrayLike, *, feature_columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Summarize clusters by size and per-feature means.

    Args:
        frame: Original dataframe (or feature dataframe).
        labels: Cluster labels aligned with rows.
        feature_columns: Numeric columns to profile. Defaults to numeric columns.
    """
    data = frame.copy()
    labels_arr = np.asarray(labels)
    if len(labels_arr) != len(data):
        raise ValueError("labels must align with frame rows.")
    data["cluster"] = labels_arr
    if feature_columns is None:
        feature_columns = (
            data.select_dtypes(include=[np.number]).columns.drop("cluster", errors="ignore").tolist()
        )
    if not feature_columns:
        raise ValueError("No numeric feature columns available for profiling.")

    grouped = data.groupby("cluster", dropna=False)
    counts = grouped.size().rename("count")
    means = grouped[list(feature_columns)].mean(numeric_only=True)
    profile = pd.concat([counts, means], axis=1).reset_index()
    profile["ratio"] = profile["count"] / max(len(data), 1)
    # Order noise (-1) last for readability when present.
    profile["_sort"] = profile["cluster"].apply(lambda v: (1, v) if v == -1 else (0, v))
    return profile.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)


def select_k(
    X: ArrayLike,
    ks: Sequence[int] = tuple(range(2, 9)),
    *,
    method: str = "kmeans",
    base_params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sweep cluster counts and compute internal metrics for quick model selection."""
    values = as_2d_float_array(X)
    rows: list[dict[str, Any]] = []
    params = dict(base_params or {})
    for k in ks:
        k_params = dict(params)
        if method in {"kmeans", "agglomerative", "spectral", "birch"}:
            k_params.setdefault("n_clusters", int(k))
        elif method in {"gmm", "gaussian_mixture"}:
            k_params.setdefault("n_components", int(k))
        else:
            raise ValueError(f"select_k supports centroid/partition methods with explicit k; got {method!r}.")
        if "random_state" not in k_params and method in {"kmeans", "gmm", "gaussian_mixture", "spectral"}:
            k_params["random_state"] = random_state
        engine = ClusteringEngine(method, k_params)
        labels = engine.fit_predict(values)
        try:
            metrics = ClusteringEvaluator.evaluate(values, labels)
            row = {"k": int(k), **metrics.to_dict()}
        except ValueError:
            row = {"k": int(k), "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
        rows.append(row)
    return pd.DataFrame(rows)


def compare_methods(
    X: ArrayLike,
    methods: Sequence[tuple[str, dict[str, Any]]],
) -> list[MethodComparison]:
    """Fit multiple clustering methods on the same data and collect metrics."""
    values = as_2d_float_array(X)
    results: list[MethodComparison] = []
    for method, params in methods:
        engine = ClusteringEngine(method, dict(params))
        labels = engine.fit_predict(values)
        try:
            metrics = ClusteringEvaluator.evaluate(values, labels)
        except ValueError:
            metrics = None
        unique = set(np.unique(labels).tolist())
        n_clusters = len([u for u in unique if u != -1])
        noise_ratio = float(np.mean(labels == -1)) if len(labels) else 0.0
        results.append(
            MethodComparison(
                method=method,
                params=dict(params),
                labels=labels,
                metrics=metrics,
                n_clusters=n_clusters,
                noise_ratio=noise_ratio,
            )
        )
    return results


def bootstrap_stability(
    X: ArrayLike,
    *,
    method: str = "kmeans",
    params: dict[str, Any] | None = None,
    n_runs: int = 10,
    sample_fraction: float = 0.8,
    random_state: int = 42,
) -> dict[str, float]:
    """Estimate clustering stability via subsample agreement (ARI).

    For each run, fit on a bootstrap/subsample and compare labels on that
    subsample against a reference fit on the full dataset (restricted to the
    same indices). Higher mean ARI indicates more stable structure.
    """
    values = as_2d_float_array(X)
    n = len(values)
    if n < 5 or n_runs < 2:
        return {"mean_ari": float("nan"), "std_ari": float("nan"), "n_runs": 0}
    rng = np.random.default_rng(random_state)
    base = ClusteringEngine(method, dict(params or {})).fit_predict(values)
    sample_size = max(2, int(n * sample_fraction))
    sample_size = min(sample_size, n)
    scores: list[float] = []
    for _ in range(n_runs):
        idx = rng.choice(n, size=sample_size, replace=False)
        labels = ClusteringEngine(method, dict(params or {})).fit_predict(values[idx])
        scores.append(float(adjusted_rand_score(base[idx], labels)))
    arr = np.asarray(scores, dtype=float)
    return {"mean_ari": float(np.mean(arr)), "std_ari": float(np.std(arr)), "n_runs": int(len(scores))}


def anomaly_summary(
    frame: pd.DataFrame,
    labels: ArrayLike,
    *,
    feature_columns: Sequence[str] | None = None,
    top_n: int = 3,
) -> dict[str, Any]:
    """Summarize anomalies and top deviating features by absolute z-score."""
    labels_arr = np.asarray(labels)
    if len(labels_arr) != len(frame):
        raise ValueError("labels must align with frame rows.")
    data = frame.copy()
    data["is_outlier"] = labels_arr.astype(int)
    if feature_columns is None:
        feature_columns = (
            data.select_dtypes(include=[np.number]).columns.drop("is_outlier", errors="ignore").tolist()
        )
    summary: dict[str, Any] = {
        "n_samples": int(len(data)),
        "n_outliers": int(np.sum(labels_arr == 1)),
        "outlier_ratio": float(np.mean(labels_arr == 1)) if len(data) else 0.0,
        "top_features": [],
    }
    if not feature_columns or summary["n_outliers"] == 0:
        return summary
    numeric = data[list(feature_columns)].astype(float)
    mu = numeric.mean()
    sigma = numeric.std(ddof=0).replace(0, np.nan)
    z = ((numeric - mu) / sigma).abs()
    outlier_z = z.loc[data["is_outlier"] == 1].mean().dropna().sort_values(ascending=False)
    summary["top_features"] = [
        {"feature": k, "mean_abs_z": float(v)} for k, v in outlier_z.head(top_n).items()
    ]
    return summary


def hopkins_statistic(X: ArrayLike, *, sample_size: int | None = None, random_state: int = 42) -> float:
    """Estimate cluster tendency with the Hopkins statistic.

    Values closer to 1 suggest clusterable structure; ~0.5 suggests uniform/random;
    closer to 0 suggests regularly spaced data. Uses a simple uniform-reference
    approximation over the feature bounding box.
    """
    values = as_2d_float_array(X)
    n, d = values.shape
    if n < 2:
        raise ValueError("Hopkins statistic requires at least two samples.")
    m = sample_size or min(max(int(0.1 * n), 10), n - 1)
    m = max(1, min(m, n - 1))
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(n, size=m, replace=False)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    span = np.where((maxs - mins) == 0, 1.0, maxs - mins)
    uniform = mins + rng.random((m, d)) * span

    # Nearest-neighbor distances (excluding identical sampled point when applicable)
    from sklearn.neighbors import NearestNeighbors

    nn_data = NearestNeighbors(n_neighbors=2).fit(values)
    w_dist, _ = nn_data.kneighbors(values[sample_idx], n_neighbors=2)
    w = w_dist[:, 1] if w_dist.shape[1] > 1 else w_dist[:, 0]
    nn_uniform = NearestNeighbors(n_neighbors=1).fit(values)
    u = nn_uniform.kneighbors(uniform, n_neighbors=1)[0][:, 0]
    denom = float(np.sum(u) + np.sum(w))
    if denom == 0:
        return 0.5
    return float(np.sum(u) / denom)


def embedding_trustworthiness(
    X: ArrayLike,
    X_embedded: ArrayLike,
    *,
    n_neighbors: int = 5,
    metric: str = "euclidean",
) -> float:
    """Compute trustworthiness of a low-dimensional embedding.

    Uses sklearn's implementation: 1 is perfect preservation of local neighborhoods.
    """
    from sklearn.manifold import trustworthiness

    X_arr = as_2d_float_array(X)
    X_emb = as_2d_float_array(X_embedded)
    if len(X_arr) != len(X_emb):
        raise ValueError("X and X_embedded must have the same number of samples.")
    k = min(max(int(n_neighbors), 1), max(len(X_arr) - 1, 1))
    return float(trustworthiness(X_arr, X_emb, n_neighbors=k, metric=metric))
