"""
Author: GHNAMI Helmi
Date: 2025-09-16
Position: Data-Science
"""

import plotly.express as px
import pandas as pd

def plot_clusters(X, labels, title='Clustering'):
    df = pd.DataFrame(X, columns=['x', 'y'])
    df['cluster'] = labels.astype(str)
    fig = px.scatter(df, x='x', y='y', color='cluster', title=title,
                     opacity=0.8, color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(marker=dict(size=8, line=dict(width=0)))
    return fig

def plot_embedding(X, labels=None, title='Dimensionality Reduction'):
    cols = ['x', 'y'] if X.shape[1] == 2 else ['x', 'y', 'z']
    df = pd.DataFrame(X, columns=cols)
    df['label'] = labels.astype(str) if labels is not None else 'all'
    if X.shape[1] == 2:
        fig = px.scatter(df, x='x', y='y', color='label', title=title,
                         opacity=0.8, color_discrete_sequence=px.colors.qualitative.Set2)
    elif X.shape[1] == 3:
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', title=title,
                            opacity=0.7, color_discrete_sequence=px.colors.qualitative.Set2)
    else:
        raise ValueError('Embedding must have 2 or 3 dimensions')
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    return fig
