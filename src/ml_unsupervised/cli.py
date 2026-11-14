"""Compatibility shim. Use ml_unsupervised.interfaces.cli instead."""

from .interfaces.cli import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
