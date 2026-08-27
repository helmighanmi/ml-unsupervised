# Path: tests/unit/test_evaluation.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

import numpy as np

from ml_unsupervised import ClusteringEvaluator
from ml_unsupervised.evaluation import evaluate_clustering


def test_evaluator_ignores_density_noise() -> None:
    X = np.array([[0, 0], [0, 1], [10, 10], [10, 11], [100, 100]], dtype=float)
    labels = np.array([0, 0, 1, 1, -1])
    metrics = ClusteringEvaluator.evaluate(X, labels)
    assert metrics.n_clusters == 2
    assert metrics.evaluated_samples == 4
    assert metrics.ignored_noise_samples == 1


def test_compatibility_api_returns_error_for_single_cluster() -> None:
    result = evaluate_clustering(np.ones((5, 2)), np.zeros(5, dtype=int))
    assert "error" in result
