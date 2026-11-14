# Path: tests/unit/test_visualization.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-helmi

import numpy as np
import pytest

from ml_unsupervised.visualization import plot_clusters, plot_embedding


def test_plot_clusters_returns_figure() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    labels = np.array([0, 0, 1])
    fig = plot_clusters(X, labels, title="t")
    assert fig.layout.title.text == "t"


def test_plot_embedding_2d_and_3d() -> None:
    X2 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    fig2 = plot_embedding(X2, labels=[0, 1, 1])
    assert fig2.data
    X3 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    fig3 = plot_embedding(X3)
    assert fig3.data


def test_plot_clusters_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        plot_clusters(np.ones((3, 3)), [0, 1, 2])
    with pytest.raises(ValueError):
        plot_clusters(np.ones((3, 2)), [0, 1])
