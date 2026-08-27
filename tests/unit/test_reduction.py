# Path: tests/unit/test_reduction.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

import numpy as np
import pytest

from ml_unsupervised import DimensionalityReducer
from ml_unsupervised.exceptions import NotFittedError


def test_pca_fit_transform_and_transform() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 6))
    reducer = DimensionalityReducer("pca", n_components=2)
    reduced = reducer.fit_transform(X)
    assert reduced.shape == (50, 2)
    assert reducer.transform(X[:3]).shape == (3, 2)


def test_transform_before_fit_fails() -> None:
    with pytest.raises(NotFittedError):
        DimensionalityReducer("pca").transform(np.ones((3, 2)))


def test_invalid_input_rejects_nan() -> None:
    X = np.array([[1.0, np.nan], [2.0, 3.0]])
    with pytest.raises(ValueError, match="NaN"):
        DimensionalityReducer("pca").fit_transform(X)
