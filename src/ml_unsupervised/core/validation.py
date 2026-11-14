# Path: src/ml_unsupervised/core/validation.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-helmi

"""Shared validation helpers used across package layers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_2d_float_array(data: ArrayLike, *, name: str = "X") -> NDArray[np.float64]:
    """Convert input to a finite 2-D floating-point NumPy array."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array; received shape {array.shape!r}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and one feature.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def normalize_name(value: str) -> str:
    """Normalize user-facing algorithm names for internal dispatch."""
    return value.strip().lower().replace("-", "_").replace(" ", "_")
