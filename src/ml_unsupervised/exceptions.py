# Path: src/ml_unsupervised/exceptions.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Domain-specific exceptions for the unsupervised ML toolkit."""


class MLUnsupervisedError(Exception):
    """Base exception for toolkit errors."""


class ConfigurationError(MLUnsupervisedError):
    """Raised when configuration is missing or invalid."""


class UnsupportedAlgorithmError(MLUnsupervisedError):
    """Raised when an unknown algorithm is requested."""


class NotFittedError(MLUnsupervisedError):
    """Raised when inference is requested before fitting a component."""


class OptionalDependencyError(MLUnsupervisedError):
    """Raised when a requested feature requires an optional dependency."""
