# Path: src/ml_unsupervised/dimensionality_reduction.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Dimensionality-reduction classes and convenience helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import FactorAnalysis, FastICA, KernelPCA, NMF, PCA
from sklearn.manifold import TSNE

from .exceptions import NotFittedError, OptionalDependencyError, UnsupportedAlgorithmError
from .utils import as_2d_float_array, normalize_name


@dataclass(slots=True)
class DimensionalityReducer:
    """Stable facade over supported dimensionality-reduction estimators."""

    method: str = "pca"
    n_components: int = 2
    params: dict[str, Any] = field(default_factory=dict)
    model_: Any | None = field(default=None, init=False, repr=False)
    embedding_: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def _build_model(self) -> Any:
        method = normalize_name(self.method)
        params = dict(self.params)
        params.setdefault("n_components", self.n_components)

        if method == "pca":
            return PCA(**params)
        if method in {"tsne", "t_sne"}:
            params.setdefault("random_state", 42)
            params.setdefault("init", "pca")
            return TSNE(**params)
        if method == "umap":
            params.setdefault("random_state", 42)
            try:
                import umap  # type: ignore
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise OptionalDependencyError(
                    "UMAP requires the 'umap-learn' package. Install the project dependencies first."
                ) from exc
            return umap.UMAP(**params)
        if method in {"kernel_pca", "kpca"}:
            params.setdefault("kernel", "rbf")
            return KernelPCA(**params)
        if method in {"ica", "fastica", "fast_ica"}:
            params.setdefault("random_state", 42)
            return FastICA(**params)
        if method == "nmf":
            params.setdefault("random_state", 42)
            params.setdefault("init", "nndsvda")
            return NMF(**params)
        if method in {"factor_analysis", "fa"}:
            params.setdefault("random_state", 42)
            return FactorAnalysis(**params)
        raise UnsupportedAlgorithmError(f"Unsupported dimensionality-reduction method {self.method!r}.")

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        values = as_2d_float_array(X)
        self.model_ = self._build_model()
        reduced = self.model_.fit_transform(values)
        self.embedding_ = np.asarray(reduced, dtype=float)
        return self.embedding_.copy()

    def fit(self, X: ArrayLike) -> "DimensionalityReducer":
        self.fit_transform(X)
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        if self.model_ is None:
            raise NotFittedError("DimensionalityReducer must be fitted before transform().")
        if not hasattr(self.model_, "transform"):
            raise UnsupportedAlgorithmError(
                f"{self.method!r} does not support out-of-sample transform(). Use fit_transform()."
            )
        return np.asarray(self.model_.transform(as_2d_float_array(X)), dtype=float)

    @property
    def model(self) -> Any:
        if self.model_ is None:
            raise NotFittedError("DimensionalityReducer has not been fitted yet.")
        return self.model_


def run_pca(X: ArrayLike, n_components: int = 2):
    reducer = DimensionalityReducer("pca", n_components)
    return reducer.fit_transform(X), reducer.model


def run_tsne(X: ArrayLike, n_components: int = 2, perplexity: float = 30, random_state: int = 42):
    reducer = DimensionalityReducer(
        "tsne", n_components, {"perplexity": perplexity, "random_state": random_state}
    )
    return reducer.fit_transform(X)


def run_umap(X: ArrayLike, n_components: int = 2, random_state: int = 42):
    reducer = DimensionalityReducer("umap", n_components, {"random_state": random_state})
    return reducer.fit_transform(X)


def run_kernel_pca(X: ArrayLike, n_components: int = 2, kernel: str = "rbf"):
    reducer = DimensionalityReducer("kernel_pca", n_components, {"kernel": kernel})
    return reducer.fit_transform(X), reducer.model


def run_ica(X: ArrayLike, n_components: int = 2, random_state: int = 42):
    reducer = DimensionalityReducer("ica", n_components, {"random_state": random_state})
    return reducer.fit_transform(X), reducer.model


def run_nmf(X: ArrayLike, n_components: int = 2, random_state: int = 42):
    reducer = DimensionalityReducer("nmf", n_components, {"random_state": random_state})
    return reducer.fit_transform(X), reducer.model


def run_factor_analysis(X: ArrayLike, n_components: int = 2, random_state: int = 42):
    reducer = DimensionalityReducer("factor_analysis", n_components, {"random_state": random_state})
    return reducer.fit_transform(X), reducer.model
