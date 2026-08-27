# Path: src/ml_unsupervised/anomaly_detection.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Production-oriented anomaly-detection API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from .exceptions import NotFittedError, UnsupportedAlgorithmError
from .utils import as_2d_float_array, normalize_name


@dataclass(frozen=True, slots=True)
class AnomalyResult:
    """Normalized anomaly output: 0=inlier, 1=outlier."""

    labels: NDArray[np.int_]
    scores: NDArray[np.float64] | None = None

    @property
    def outlier_ratio(self) -> float:
        return float(np.mean(self.labels == 1))


@dataclass(slots=True)
class AnomalyDetector:
    """Fit anomaly detectors through one consistent inference interface."""

    method: str = "isolation_forest"
    params: dict[str, Any] = field(default_factory=dict)
    model_: Any | None = field(default=None, init=False, repr=False)

    def _build_model(self) -> Any:
        method = normalize_name(self.method)
        params = dict(self.params)
        if method in {"isolation_forest", "iforest"}:
            params.setdefault("contamination", 0.1)
            params.setdefault("random_state", 42)
            return IsolationForest(**params)
        if method in {"one_class_svm", "oneclass_svm", "ocsvm"}:
            params.setdefault("nu", 0.1)
            params.setdefault("kernel", "rbf")
            params.setdefault("gamma", "scale")
            return OneClassSVM(**params)
        if method in {"lof", "local_outlier_factor"}:
            params.setdefault("n_neighbors", 20)
            params.setdefault("contamination", 0.1)
            # novelty=True is required for predicting on future/unseen samples.
            params.setdefault("novelty", True)
            return LocalOutlierFactor(**params)
        raise UnsupportedAlgorithmError(f"Unsupported anomaly-detection method {self.method!r}.")

    def fit(self, X: ArrayLike) -> "AnomalyDetector":
        self.model_ = self._build_model()
        self.model_.fit(as_2d_float_array(X))
        return self

    def predict(self, X: ArrayLike) -> AnomalyResult:
        if self.model_ is None:
            raise NotFittedError("AnomalyDetector must be fitted before predict().")
        values = as_2d_float_array(X)
        raw_labels = np.asarray(self.model_.predict(values), dtype=int)
        labels = np.where(raw_labels == -1, 1, 0).astype(int)
        scores: NDArray[np.float64] | None = None
        if hasattr(self.model_, "decision_function"):
            # Higher values are made more anomalous for a consistent public contract.
            scores = -np.asarray(self.model_.decision_function(values), dtype=float)
        return AnomalyResult(labels=labels, scores=scores)

    def fit_predict(self, X: ArrayLike) -> AnomalyResult:
        values = as_2d_float_array(X)
        return self.fit(values).predict(values)

    @property
    def model(self) -> Any:
        if self.model_ is None:
            raise NotFittedError("AnomalyDetector has not been fitted yet.")
        return self.model_


# Legacy-style wrappers retain the original -1/1 labels for backwards compatibility.
def _legacy_labels(result: AnomalyResult) -> NDArray[np.int_]:
    return np.where(result.labels == 1, -1, 1).astype(int)


def run_isolation_forest(X: ArrayLike, contamination: float = 0.1, random_state: int = 42):
    detector = AnomalyDetector(
        "isolation_forest", {"contamination": contamination, "random_state": random_state}
    )
    result = detector.fit_predict(X)
    return _legacy_labels(result), detector.model


def run_oneclass_svm(X: ArrayLike, nu: float = 0.1, kernel: str = "rbf", gamma: str = "scale"):
    detector = AnomalyDetector("one_class_svm", {"nu": nu, "kernel": kernel, "gamma": gamma})
    result = detector.fit_predict(X)
    return _legacy_labels(result), detector.model


def run_lof(X: ArrayLike, n_neighbors: int = 20, contamination: float = 0.1):
    detector = AnomalyDetector("lof", {"n_neighbors": n_neighbors, "contamination": contamination})
    result = detector.fit_predict(X)
    return _legacy_labels(result), detector.model
