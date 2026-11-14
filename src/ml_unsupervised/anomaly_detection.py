"""Compatibility shim. Use ml_unsupervised.algorithms.anomaly instead."""

from .algorithms.anomaly import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
