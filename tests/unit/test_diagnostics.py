# Path: tests/unit/test_diagnostics.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-helmi

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from ml_unsupervised.diagnostics import (
    anomaly_summary,
    bootstrap_stability,
    cluster_profile,
    compare_methods,
    select_k,
)


def test_cluster_profile_and_select_k() -> None:
    X, y = make_blobs(n_samples=60, centers=3, n_features=3, random_state=42)
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    prof = cluster_profile(df, y)
    assert "cluster" in prof.columns and "count" in prof.columns
    assert prof["count"].sum() == 60

    sweep = select_k(X, ks=[2, 3, 4], method="kmeans", random_state=42)
    assert set(["k", "silhouette"]).issubset(sweep.columns)
    assert len(sweep) == 3


def test_compare_methods_and_stability() -> None:
    X, _ = make_blobs(n_samples=80, centers=3, n_features=4, random_state=0)
    results = compare_methods(
        X,
        [
            ("kmeans", {"n_clusters": 3, "random_state": 0, "n_init": 10}),
            ("agglomerative", {"n_clusters": 3}),
        ],
    )
    assert len(results) == 2
    assert results[0].metrics is not None

    stab = bootstrap_stability(
        X, method="kmeans", params={"n_clusters": 3, "random_state": 0, "n_init": 10}, n_runs=3
    )
    assert "mean_ari" in stab


def test_anomaly_summary() -> None:
    df = pd.DataFrame({"x": [0, 0.1, 0.2, 10.0], "y": [0, 0.1, 0.2, 10.0]})
    labels = np.array([0, 0, 0, 1])
    summary = anomaly_summary(df, labels)
    assert summary["n_outliers"] == 1
    assert summary["outlier_ratio"] == 0.25
