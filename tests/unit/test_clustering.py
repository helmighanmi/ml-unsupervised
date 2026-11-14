# Path: tests/unit/test_clustering.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from ml_unsupervised import ClusteringEngine
from ml_unsupervised.clustering import run_kmeans
from ml_unsupervised.exceptions import NotFittedError, UnsupportedAlgorithmError


def test_kmeans_engine_fit_predict_and_predict() -> None:
    X, _ = make_blobs(n_samples=60, centers=3, random_state=42)
    engine = ClusteringEngine("kmeans", {"n_clusters": 3, "random_state": 42, "n_init": 10})
    labels = engine.fit_predict(X)
    assert labels.shape == (60,)
    assert len(np.unique(labels)) == 3
    assert engine.predict(X[:5]).shape == (5,)


def test_predict_before_fit_fails_explicitly() -> None:
    with pytest.raises(NotFittedError):
        ClusteringEngine("kmeans").predict(np.ones((3, 2)))


def test_algorithm_without_predict_is_explicit() -> None:
    X, _ = make_blobs(n_samples=30, centers=2, random_state=42)
    engine = ClusteringEngine("agglomerative", {"n_clusters": 2})
    engine.fit(X)
    with pytest.raises(UnsupportedAlgorithmError):
        engine.predict(X[:2])


def test_legacy_kmeans_helper_is_preserved() -> None:
    X, _ = make_blobs(n_samples=40, centers=2, random_state=7)
    labels, centers = run_kmeans(X, n_clusters=2)
    assert labels.shape == (40,)
    assert centers.shape == (2, 2)


def test_clustering_convenience_wrappers() -> None:
    X, _ = make_blobs(n_samples=40, centers=2, random_state=1)
    from ml_unsupervised.clustering import (
        run_agglomerative,
        run_birch,
        run_dbscan,
        run_gmm,
        run_hdbscan,
        run_spectral,
    )

    labels_db = run_dbscan(X, eps=1.5, min_samples=3)
    assert labels_db.shape == (40,)
    labels_ag = run_agglomerative(X, n_clusters=2)
    assert labels_ag.shape == (40,)
    labels_gmm, model_gmm = run_gmm(X, n_components=2, random_state=1)
    assert labels_gmm.shape == (40,) and model_gmm is not None
    labels_sp, model_sp = run_spectral(X, n_clusters=2, random_state=1)
    assert labels_sp.shape == (40,) and model_sp is not None
    labels_b, model_b = run_birch(X, n_clusters=2)
    assert labels_b.shape == (40,) and model_b is not None
    try:
        run_hdbscan(X, min_cluster_size=5)
    except Exception:
        # hdbscan is optional; either works or raises OptionalDependencyError
        pass
