"""Compatibility shim. Use ml_unsupervised.infrastructure.config instead."""

from .infrastructure.config import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
