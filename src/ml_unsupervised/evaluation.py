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
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .utils import as_2d_float_array


MetricValue = float | int
EvaluationValue = float | int | str


@dataclass(frozen=True, slots=True)
class ClusteringMetrics:
    """Container for clustering evaluation metrics."""

    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    evaluated_samples: int
    n_clusters: int
    ignored_noise_samples: int = 0

    def to_dict(self) -> dict[str, MetricValue]:
        """Return the metrics as a serializable dictionary."""
        return asdict(self)


class ClusteringEvaluator:
    """Evaluate clustering quality with optional noise exclusion.

    DBSCAN and HDBSCAN commonly use label ``-1`` for samples classified
    as noise. When ``ignore_noise`` is enabled, those samples are excluded
    before computing clustering quality metrics.
    """

    @staticmethod
    def evaluate(
        X: ArrayLike,
        labels: ArrayLike,
        *,
        ignore_noise: bool = True,
    ) -> ClusteringMetrics:
        """Compute clustering quality metrics.

        Args:
            X:
                Two-dimensional feature matrix.
            labels:
                One-dimensional cluster label array with one label for
                every input sample.
            ignore_noise:
                Exclude samples whose cluster label is ``-1``.

        Returns:
            Calculated clustering metrics.

        Raises:
            ValueError:
                If the labels are invalid or the clustering result cannot
                be evaluated meaningfully.
        """
        values = as_2d_float_array(X)
        labels_array = np.asarray(labels)

        if labels_array.ndim != 1:
            raise ValueError("labels must be a 1-D array.")

        if len(labels_array) != len(values):
            raise ValueError(
                "labels must contain exactly one label per input sample."
            )

        ignored_noise_samples = 0

        if ignore_noise:
            non_noise_mask = labels_array != -1
            ignored_noise_samples = int((~non_noise_mask).sum())

            values = values[non_noise_mask]
            labels_array = labels_array[non_noise_mask]

        unique_labels = np.unique(labels_array)
        n_samples = len(values)
        n_clusters = len(unique_labels)

        if n_samples < 2:
            raise ValueError(
                "Clustering metrics require at least two evaluated samples."
            )

        if n_clusters < 2:
            raise ValueError(
                "Clustering metrics require at least two non-empty clusters."
            )

        if n_clusters >= n_samples:
            raise ValueError(
                "Clustering metrics require fewer clusters than samples."
            )

        return ClusteringMetrics(
            silhouette=float(
                silhouette_score(values, labels_array)
            ),
            davies_bouldin=float(
                davies_bouldin_score(values, labels_array)
            ),
            calinski_harabasz=float(
                calinski_harabasz_score(values, labels_array)
            ),
            evaluated_samples=int(n_samples),
            n_clusters=int(n_clusters),
            ignored_noise_samples=ignored_noise_samples,
        )


def evaluate_clustering(
    X: ArrayLike,
    labels: ArrayLike,
) -> dict[str, EvaluationValue]:
    """Evaluate clustering using the legacy dictionary-based API.

    This function is retained for backwards compatibility with older
    notebooks and examples.

    Successful evaluations return numeric metrics. Invalid clustering
    configurations return an ``error`` entry rather than raising
    ``ValueError``.

    Args:
        X:
            Two-dimensional feature matrix.
        labels:
            One-dimensional cluster labels.

    Returns:
        Dictionary containing clustering metrics or an error message.
    """
    try:
        metrics = ClusteringEvaluator.evaluate(X, labels)

        # Build the broader legacy return type explicitly.
        #
        # ClusteringMetrics.to_dict() correctly returns:
        #     dict[str, float | int]
        #
        # This compatibility function may additionally return:
        #     {"error": str}
        #
        # Using an explicitly typed dictionary avoids dict invariance
        # problems detected by mypy.
        result: dict[str, EvaluationValue] = {}
        result.update(metrics.to_dict())

        return result

    except ValueError as exc:
        return {"error": str(exc)}