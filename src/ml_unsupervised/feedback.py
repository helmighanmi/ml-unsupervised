"""Compatibility shim. Use ml_unsupervised.algorithms.feedback instead."""

from .algorithms.feedback import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
