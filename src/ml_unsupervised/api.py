"""Compatibility shim. Use ml_unsupervised.interfaces.api instead."""

from .interfaces.api import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
