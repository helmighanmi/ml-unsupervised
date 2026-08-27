# Path: src/ml_unsupervised/clustering.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Reusable clustering estimators and backwards-compatible helper functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

from .exceptions import NotFittedError, OptionalDependencyError, UnsupportedAlgorithmError
from .utils import as_2d_float_array, normalize_name

_SUPPORTED = {
    "kmeans",
    "dbscan",
    "hdbscan",
    "agglomerative",
    "gmm",
    "gaussian_mixture",
    "spectral",
    "birch",
}


@dataclass(slots=True)
class ClusteringEngine:
    """Fit and apply a clustering algorithm behind one stable project API.

    Parameters are passed directly to the selected estimator. Expensive or optional
    dependencies such as ``hdbscan`` are imported lazily, allowing the rest of the
    package to remain usable when that optional wheel is unavailable.
    """

    method: str = "kmeans"
    params: dict[str, Any] = field(default_factory=dict)
    model_: Any | None = field(default=None, init=False, repr=False)
    labels_: NDArray[np.int_] | None = field(default=None, init=False, repr=False)

    def _build_model(self) -> Any:
        method = normalize_name(self.method)
        params = dict(self.params)

        if method == "kmeans":
            params.setdefault("n_clusters", 3)
            params.setdefault("random_state", 42)
            params.setdefault("n_init", 10)
            return KMeans(**params)
        if method == "dbscan":
            params.setdefault("eps", 0.5)
            params.setdefault("min_samples", 5)
            return DBSCAN(**params)
        if method == "hdbscan":
            params.setdefault("min_cluster_size", 10)
            try:
                import hdbscan  # type: ignore
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise OptionalDependencyError(
                    "HDBSCAN requires the 'hdbscan' package. Install the project dependencies first."
                ) from exc
            return hdbscan.HDBSCAN(**params)
        if method == "agglomerative":
            params.setdefault("n_clusters", 3)
            return AgglomerativeClustering(**params)
        if method in {"gmm", "gaussian_mixture"}:
            params.setdefault("n_components", 3)
            params.setdefault("random_state", 42)
            return GaussianMixture(**params)
        if method == "spectral":
            params.setdefault("n_clusters", 3)
            params.setdefault("affinity", "nearest_neighbors")
            params.setdefault("random_state", 42)
            return SpectralClustering(**params)
        if method == "birch":
            params.setdefault("n_clusters", 3)
            return Birch(**params)
        raise UnsupportedAlgorithmError(
            f"Unsupported clustering method {self.method!r}. Supported methods: {sorted(_SUPPORTED)}"
        )

    def fit_predict(self, X: ArrayLike) -> NDArray[np.int_]:
        """Fit the selected clustering model and return integer cluster labels."""
        values = as_2d_float_array(X)
        self.model_ = self._build_model()
        labels = self.model_.fit_predict(values)
        self.labels_ = np.asarray(labels, dtype=int)
        return self.labels_.copy()

    def fit(self, X: ArrayLike) -> "ClusteringEngine":
        """Fit the selected clustering model."""
        self.fit_predict(X)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.int_]:
        """Predict clusters for estimators that support out-of-sample prediction.

        Density and graph-based estimators such as DBSCAN, AgglomerativeClustering,
        SpectralClustering and standard HDBSCAN do not expose a standard ``predict``
        method. In those cases, callers receive an explicit error instead of a
        misleading result.
        """
        if self.model_ is None:
            raise NotFittedError("ClusteringEngine must be fitted before predict().")
        if not hasattr(self.model_, "predict"):
            raise UnsupportedAlgorithmError(
                f"{self.method!r} does not provide out-of-sample predict() in this API. "
                "Use fit_predict() for this estimator."
            )
        values = as_2d_float_array(X)
        return np.asarray(self.model_.predict(values), dtype=int)

    @property
    def model(self) -> Any:
        """Expose the fitted underlying estimator for advanced users."""
        if self.model_ is None:
            raise NotFittedError("ClusteringEngine has not been fitted yet.")
        return self.model_


# Convenience functions retained for simple scripts and notebook migration.
def run_kmeans(X: ArrayLike, n_clusters: int = 3, random_state: int = 42):
    engine = ClusteringEngine(
        "kmeans", {"n_clusters": n_clusters, "random_state": random_state, "n_init": 10}
    )
    labels = engine.fit_predict(X)
    return labels, np.asarray(engine.model.cluster_centers_)


def run_dbscan(X: ArrayLike, eps: float = 0.5, min_samples: int = 5):
    return ClusteringEngine("dbscan", {"eps": eps, "min_samples": min_samples}).fit_predict(X)


def run_hdbscan(X: ArrayLike, min_cluster_size: int = 10):
    return ClusteringEngine("hdbscan", {"min_cluster_size": min_cluster_size}).fit_predict(X)


def run_agglomerative(X: ArrayLike, n_clusters: int = 3):
    return ClusteringEngine("agglomerative", {"n_clusters": n_clusters}).fit_predict(X)


def run_gmm(X: ArrayLike, n_components: int = 3, random_state: int = 42):
    engine = ClusteringEngine("gmm", {"n_components": n_components, "random_state": random_state})
    labels = engine.fit_predict(X)
    return labels, engine.model


def run_spectral(X: ArrayLike, n_clusters: int = 3, random_state: int = 42):
    engine = ClusteringEngine(
        "spectral", {"n_clusters": n_clusters, "affinity": "nearest_neighbors", "random_state": random_state}
    )
    labels = engine.fit_predict(X)
    return labels, engine.model


def run_birch(X: ArrayLike, n_clusters: int = 3):
    engine = ClusteringEngine("birch", {"n_clusters": n_clusters})
    labels = engine.fit_predict(X)
    return labels, engine.model
