# Path: src/ml_unsupervised/__init__.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Production-ready unsupervised machine-learning toolkit."""

from .algorithms.anomaly import AnomalyDetector, AnomalyResult
from .algorithms.clustering import ClusteringEngine
from .algorithms.reduction import DimensionalityReducer
from .data.preprocessing import FeaturePreprocessor
from .evaluation.metrics import ClusteringEvaluator, ClusteringMetrics
from .workflows.pipeline import ClusteringPipeline, PipelineResult

__all__ = [
    "AnomalyDetector",
    "AnomalyResult",
    "ClusteringEngine",
    "ClusteringEvaluator",
    "ClusteringMetrics",
    "ClusteringPipeline",
    "DimensionalityReducer",
    "FeaturePreprocessor",
    "PipelineResult",
]

__version__ = "1.0.0"
