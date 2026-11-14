"""Evaluation and diagnostics."""

from .diagnostics import (
    MethodComparison,
    anomaly_summary,
    bootstrap_stability,
    cluster_profile,
    compare_methods,
    embedding_trustworthiness,
    hopkins_statistic,
    select_k,
)
from .metrics import ClusteringEvaluator, ClusteringMetrics, evaluate_clustering

__all__ = [
    "MethodComparison",
    "ClusteringEvaluator",
    "ClusteringMetrics",
    "evaluate_clustering",
    "cluster_profile",
    "select_k",
    "compare_methods",
    "bootstrap_stability",
    "anomaly_summary",
    "hopkins_statistic",
    "embedding_trustworthiness",
]
