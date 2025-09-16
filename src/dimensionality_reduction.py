"""
Author: GHNAMI Helmi
Date: 2025-09-16
Position: Data-Science
"""

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def run_pca(X, n_components=2):
    model = PCA(n_components=n_components)
    reduced = model.fit_transform(X)
    return reduced, model

def run_tsne(X, n_components=2, perplexity=30, random_state=42):
    model = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    reduced = model.fit_transform(X)
    return reduced

def run_umap(X, n_components=2, random_state=42):
    model = umap.UMAP(n_components=n_components, random_state=random_state)
    reduced = model.fit_transform(X)
    return reduced
