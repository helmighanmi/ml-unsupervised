import numpy as np
from src.clustering import run_kmeans, run_dbscan, run_hdbscan, run_agglomerative

X = np.random.rand(30, 2)

def test_kmeans():
    labels, centers = run_kmeans(X, n_clusters=2)
    assert len(labels) == len(X)
    assert centers.shape == (2, 2)

def test_dbscan():
    labels = run_dbscan(X, eps=0.5, min_samples=3)
    assert len(labels) == len(X)

def test_hdbscan():
    labels = run_hdbscan(X, min_cluster_size=5)
    assert len(labels) == len(X)

def test_agglomerative():
    labels = run_agglomerative(X, n_clusters=3)
    assert len(labels) == len(X)
