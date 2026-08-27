# Path: src/ml_unsupervised/visualization.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Plotly visualization helpers kept outside model-training code."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
from numpy.typing import ArrayLike

from .utils import as_2d_float_array


def plot_clusters(X: ArrayLike, labels: ArrayLike, title: str = "Clustering"):
    values = as_2d_float_array(X)
    if values.shape[1] != 2:
        raise ValueError("plot_clusters requires exactly two dimensions.")
    label_array = np.asarray(labels)
    if len(label_array) != len(values):
        raise ValueError("labels must contain one value per sample.")
    frame = pd.DataFrame(values, columns=["x", "y"])
    frame["cluster"] = label_array.astype(str)
    return px.scatter(frame, x="x", y="y", color="cluster", title=title, opacity=0.8)


def plot_embedding(X: ArrayLike, labels: ArrayLike | None = None, title: str = "Dimensionality Reduction"):
    values = as_2d_float_array(X)
    if values.shape[1] not in {2, 3}:
        raise ValueError("Embedding must have 2 or 3 dimensions.")
    label_values = np.asarray(labels).astype(str) if labels is not None else np.repeat("all", len(values))
    if len(label_values) != len(values):
        raise ValueError("labels must contain one value per sample.")
    columns = ["x", "y"] if values.shape[1] == 2 else ["x", "y", "z"]
    frame = pd.DataFrame(values, columns=columns)
    frame["label"] = label_values
    if values.shape[1] == 2:
        return px.scatter(frame, x="x", y="y", color="label", title=title, opacity=0.8)
    return px.scatter_3d(frame, x="x", y="y", z="z", color="label", title=title, opacity=0.75)
