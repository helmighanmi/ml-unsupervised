import numpy as np
from src.clustering import run_kmeans
from src.evaluation import evaluate_clustering

X = np.random.rand(40, 2)

def test_evaluate_valid_clusters():
    labels, _ = run_kmeans(X, n_clusters=2)
    metrics = evaluate_clustering(X, labels)
    assert 'silhouette' in metrics
    assert 'davies_bouldin' in metrics
    assert 'calinski_harabasz' in metrics

def test_evaluate_invalid_clusters():
    labels = np.zeros(40)
    metrics = evaluate_clustering(X, labels)
    assert 'error' in metrics
