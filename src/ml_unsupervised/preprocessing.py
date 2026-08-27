# Path: src/ml_unsupervised/preprocessing.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Reusable preprocessing components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from numpy.typing import ArrayLike, NDArray
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .exceptions import NotFittedError, UnsupportedAlgorithmError
from .utils import as_2d_float_array, normalize_name


def get_scaler(name: str = "robust"):
    """Create a supported scikit-learn scaler by name."""
    normalized = normalize_name(name).replace("_scaler", "")
    if normalized == "standard":
        return StandardScaler()
    if normalized == "minmax":
        return MinMaxScaler()
    if normalized == "robust":
        return RobustScaler()
    raise UnsupportedAlgorithmError(f"Unknown scaler {name!r}; choose standard, minmax or robust.")


@dataclass(slots=True)
class FeaturePreprocessor:
    """Small stateful wrapper used consistently by pipelines, CLI and UI."""

    scaler: str = "robust"
    scaler_params: dict[str, Any] = field(default_factory=dict)
    scaler_: Any | None = field(default=None, init=False, repr=False)

    def fit(self, X: ArrayLike) -> "FeaturePreprocessor":
        values = as_2d_float_array(X)
        self.scaler_ = get_scaler(self.scaler)
        if self.scaler_params:
            self.scaler_.set_params(**self.scaler_params)
        self.scaler_.fit(values)
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.scaler_ is None:
            raise NotFittedError("FeaturePreprocessor must be fitted before transform().")
        return np.asarray(self.scaler_.transform(as_2d_float_array(X)), dtype=float)

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        values = as_2d_float_array(X)
        return self.fit(values).transform(values)
