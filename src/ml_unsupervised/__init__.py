# Path: src/ml_unsupervised/__init__.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Production-ready unsupervised machine-learning toolkit."""

from .anomaly_detection import AnomalyDetector, AnomalyResult
from .clustering import ClusteringEngine
from .dimensionality_reduction import DimensionalityReducer
from .evaluation import ClusteringEvaluator, ClusteringMetrics
from .pipelines import ClusteringPipeline, PipelineResult
from .preprocessing import FeaturePreprocessor

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
