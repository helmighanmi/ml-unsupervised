# Path: tests/unit/test_anomaly_detection.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

import numpy as np

from ml_unsupervised import AnomalyDetector
from ml_unsupervised.anomaly_detection import run_isolation_forest


def test_anomaly_detector_normalizes_labels_and_scores() -> None:
    rng = np.random.default_rng(42)
    normal = rng.normal(size=(100, 2))
    outliers = np.array([[9.0, 9.0], [-9.0, -9.0]])
    X = np.vstack([normal, outliers])

    result = AnomalyDetector(
        "isolation_forest", {"contamination": 0.05, "random_state": 42}
    ).fit_predict(X)

    assert set(np.unique(result.labels)).issubset({0, 1})
    assert result.scores is not None
    assert 0.0 <= result.outlier_ratio <= 1.0


def test_legacy_anomaly_helper_keeps_minus_one_contract() -> None:
    X = np.arange(40, dtype=float).reshape(20, 2)
    labels, _ = run_isolation_forest(X, contamination=0.1)
    assert set(np.unique(labels)).issubset({-1, 1})
