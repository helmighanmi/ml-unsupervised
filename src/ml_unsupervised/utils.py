# Path: src/ml_unsupervised/utils.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Shared validation and parsing helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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


def coerce_scalar(value: str) -> Any:
    """Convert a CLI key=value value to bool/int/float/None/string."""
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_key_value_params(items: list[str] | None) -> dict[str, Any]:
    """Parse repeated CLI parameters formatted as ``key=value``."""
    params: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid parameter {item!r}; expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid parameter {item!r}; key cannot be empty.")
        params[key] = coerce_scalar(value.strip())
    return params


def merge_params(base: Mapping[str, Any] | None, override: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a shallow merged parameter dictionary."""
    merged = dict(base or {})
    merged.update(override or {})
    return merged
