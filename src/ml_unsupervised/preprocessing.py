"""Compatibility shim. Use ml_unsupervised.data.preprocessing instead."""

from .data.preprocessing import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
