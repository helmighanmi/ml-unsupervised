import numpy as np
from src.dimensionality_reduction import run_pca, run_tsne, run_umap

X = np.random.rand(50, 10)

def test_pca_2d():
    reduced, model = run_pca(X, n_components=2)
    assert reduced.shape == (50, 2)

def test_pca_3d():
    reduced, model = run_pca(X, n_components=3)
    assert reduced.shape == (50, 3)

def test_tsne():
    reduced = run_tsne(X, n_components=2, perplexity=5, random_state=42)
    assert reduced.shape == (50, 2)

def test_umap():
    reduced = run_umap(X, n_components=2, random_state=42)
    assert reduced.shape == (50, 2)
