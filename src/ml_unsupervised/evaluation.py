# Path: src/ml_unsupervised/evaluation.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Clustering quality metrics with noise-aware validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from .utils import as_2d_float_array


@dataclass(frozen=True, slots=True)
class ClusteringMetrics:
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    evaluated_samples: int
    n_clusters: int
    ignored_noise_samples: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class ClusteringEvaluator:
    """Evaluate clusters while optionally excluding DBSCAN/HDBSCAN noise label -1."""

    @staticmethod
    def evaluate(X: ArrayLike, labels: ArrayLike, *, ignore_noise: bool = True) -> ClusteringMetrics:
        values = as_2d_float_array(X)
        labels_array = np.asarray(labels)
        if labels_array.ndim != 1 or len(labels_array) != len(values):
            raise ValueError("labels must be a 1-D array with one label per input sample.")

        ignored = 0
        if ignore_noise:
            mask = labels_array != -1
            ignored = int((~mask).sum())
            values = values[mask]
            labels_array = labels_array[mask]

        unique = np.unique(labels_array)
        if len(values) < 2 or len(unique) < 2:
            raise ValueError("Clustering metrics require at least two non-empty clusters.")
        if len(unique) >= len(values):
            raise ValueError("Clustering metrics require fewer clusters than samples.")

        return ClusteringMetrics(
            silhouette=float(silhouette_score(values, labels_array)),
            davies_bouldin=float(davies_bouldin_score(values, labels_array)),
            calinski_harabasz=float(calinski_harabasz_score(values, labels_array)),
            evaluated_samples=int(len(values)),
            n_clusters=int(len(unique)),
            ignored_noise_samples=ignored,
        )


def evaluate_clustering(X: ArrayLike, labels: ArrayLike) -> dict[str, float | int | str]:
    """Backwards-compatible dictionary API used by older notebooks."""
    try:
        return ClusteringEvaluator.evaluate(X, labels).to_dict()
    except ValueError as exc:
        return {"error": str(exc)}
