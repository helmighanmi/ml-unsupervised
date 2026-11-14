"""Compatibility shim for validation and CLI parsing helpers."""

from .core.validation import as_2d_float_array, normalize_name
from .interfaces.cli import coerce_scalar, parse_key_value_params

__all__ = ["as_2d_float_array", "normalize_name", "coerce_scalar", "parse_key_value_params"]
