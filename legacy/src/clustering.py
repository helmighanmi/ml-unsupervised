"""
Author: GHNAMI Helmi
Date: 2025-09-16
Position: Data-Science
"""

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import hdbscan

def run_kmeans(X, n_clusters=3, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return labels, model.cluster_centers_

def run_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels

def run_hdbscan(X, min_cluster_size=10):
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(X)
    return labels

def run_agglomerative(X, n_clusters=3):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return labels

from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering, Birch

def run_gmm(X, n_components=3, random_state=42):
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = model.fit_predict(X)
    return labels, model

def run_spectral(X, n_clusters=3, random_state=42):
    model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=random_state)
    labels = model.fit_predict(X)
    return labels, model

def run_birch(X, n_clusters=3):
    model = Birch(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return labels, model
