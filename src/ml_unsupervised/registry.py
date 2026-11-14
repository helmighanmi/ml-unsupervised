"""Compatibility shim. Use ml_unsupervised.infrastructure.registry instead."""

from .infrastructure.registry import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
