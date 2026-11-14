"""Compatibility shim. Use ml_unsupervised.evaluation.diagnostics instead."""

from .evaluation.diagnostics import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
