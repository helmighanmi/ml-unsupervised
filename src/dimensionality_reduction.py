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

from sklearn.decomposition import KernelPCA, FastICA, NMF, FactorAnalysis

def run_kernel_pca(X, n_components=2, kernel='rbf'):
    model = KernelPCA(n_components=n_components, kernel=kernel)
    reduced = model.fit_transform(X)
    return reduced, model

def run_ica(X, n_components=2, random_state=42):
    model = FastICA(n_components=n_components, random_state=random_state)
    reduced = model.fit_transform(X)
    return reduced, model

def run_nmf(X, n_components=2, random_state=42):
    model = NMF(n_components=n_components, random_state=random_state)
    reduced = model.fit_transform(X)
    return reduced, model

def run_factor_analysis(X, n_components=2, random_state=42):
    model = FactorAnalysis(n_components=n_components, random_state=random_state)
    reduced = model.fit_transform(X)
    return reduced, model
