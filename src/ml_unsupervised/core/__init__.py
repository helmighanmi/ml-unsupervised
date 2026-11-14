"""Core contracts and validation."""

from .exceptions import (
    ConfigurationError,
    MLUnsupervisedError,
    NotFittedError,
    OptionalDependencyError,
    UnsupportedAlgorithmError,
)
from .validation import as_2d_float_array, normalize_name

__all__ = [
    "ConfigurationError",
    "MLUnsupervisedError",
    "NotFittedError",
    "OptionalDependencyError",
    "UnsupportedAlgorithmError",
    "as_2d_float_array",
    "normalize_name",
]
