"""Compatibility shim. Use ml_unsupervised.reporting.plots instead."""

from .reporting.plots import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
