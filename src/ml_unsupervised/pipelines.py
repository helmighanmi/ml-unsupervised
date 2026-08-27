# Path: src/ml_unsupervised/pipelines.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Composable unsupervised-learning pipelines for training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .clustering import ClusteringEngine
from .dimensionality_reduction import DimensionalityReducer
from .evaluation import ClusteringEvaluator, ClusteringMetrics
from .exceptions import NotFittedError
from .preprocessing import FeaturePreprocessor
from .utils import as_2d_float_array


@dataclass(slots=True)
class PipelineResult:
    """Artifacts produced by a clustering workflow."""

    labels: NDArray[np.int_]
    transformed: NDArray[np.float64]
    metrics: ClusteringMetrics | None = None


@dataclass(slots=True)
class ClusteringPipeline:
    """End-to-end preprocessing -> optional reduction -> clustering workflow.

    This class is the primary production API for reusable clustering workflows.
    It can be imported by applications, executed from the CLI, serialized with
    :mod:`joblib`, and demonstrated from notebooks without duplicating logic.
    """

    scaler: str = "robust"
    clustering_method: str = "kmeans"
    clustering_params: dict[str, Any] = field(default_factory=dict)
    reduction_method: str | None = None
    n_components: int = 2
    reduction_params: dict[str, Any] = field(default_factory=dict)
    preprocessor_: FeaturePreprocessor | None = field(default=None, init=False, repr=False)
    reducer_: DimensionalityReducer | None = field(default=None, init=False, repr=False)
    clusterer_: ClusteringEngine | None = field(default=None, init=False, repr=False)
    fitted_: bool = field(default=False, init=False)

    def _fit_transform_features(self, X: ArrayLike) -> NDArray[np.float64]:
        values = as_2d_float_array(X)
        self.preprocessor_ = FeaturePreprocessor(self.scaler)
        transformed = self.preprocessor_.fit_transform(values)
        if self.reduction_method:
            self.reducer_ = DimensionalityReducer(
                self.reduction_method,
                n_components=self.n_components,
                params=dict(self.reduction_params),
            )
            transformed = self.reducer_.fit_transform(transformed)
        else:
            self.reducer_ = None
        return transformed

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Apply fitted preprocessing/reduction stages to new data."""
        if not self.fitted_ or self.preprocessor_ is None:
            raise NotFittedError("ClusteringPipeline must be fitted before transform().")
        transformed = self.preprocessor_.transform(X)
        if self.reducer_ is not None:
            transformed = self.reducer_.transform(transformed)
        return transformed

    def fit_predict(self, X: ArrayLike, *, evaluate: bool = True) -> PipelineResult:
        transformed = self._fit_transform_features(X)
        self.clusterer_ = ClusteringEngine(self.clustering_method, dict(self.clustering_params))
        labels = self.clusterer_.fit_predict(transformed)
        self.fitted_ = True

        metrics: ClusteringMetrics | None = None
        if evaluate:
            try:
                metrics = ClusteringEvaluator.evaluate(transformed, labels)
            except ValueError:
                # Some valid workflows (single cluster, tiny datasets) cannot be scored.
                metrics = None
        return PipelineResult(labels=labels, transformed=transformed, metrics=metrics)

    def fit(self, X: ArrayLike) -> "ClusteringPipeline":
        self.fit_predict(X)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.int_]:
        if not self.fitted_ or self.clusterer_ is None:
            raise NotFittedError("ClusteringPipeline must be fitted before predict().")
        return self.clusterer_.predict(self.transform(X))

    def save(self, path: str | Path) -> Path:
        """Persist the fitted pipeline for later inference."""
        if not self.fitted_:
            raise NotFittedError("Only a fitted ClusteringPipeline can be saved.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ClusteringPipeline":
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"Artifact does not contain {cls.__name__}.")
        return loaded


def clustering_pipeline(n_clusters: int = 3, scaler: str = "robust") -> ClusteringPipeline:
    """Compatibility helper returning a production ``ClusteringPipeline``."""
    return ClusteringPipeline(
        scaler=scaler,
        clustering_method="kmeans",
        clustering_params={"n_clusters": n_clusters, "random_state": 42, "n_init": 10},
    )
