"""
Author: GHNAMI Helmi
Date: 2025-09-16
Position: Data-Science
"""

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_clustering(X, labels):
    metrics = {}
    if len(set(labels)) > 1 and -1 not in set(labels):
        metrics['silhouette'] = silhouette_score(X, labels)
        metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
    else:
        metrics['error'] = 'Invalid clustering for metrics (single cluster or noise only)'
    return metrics
