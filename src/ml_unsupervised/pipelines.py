"""Compatibility shim. Use ml_unsupervised.workflows.pipeline instead."""

from .workflows.pipeline import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
