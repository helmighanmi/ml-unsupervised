<!--
Path: docs/migration.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Migration from the original project

## Import namespace

Before:

```python
from src.clustering import run_kmeans
```

After:

```python
from ml_unsupervised.clustering import run_kmeans
```

Recommended production API:

```python
from ml_unsupervised import ClusteringEngine

labels = ClusteringEngine(
    "kmeans",
    {"n_clusters": 3, "random_state": 42},
).fit_predict(X)
```

## Notebook-owned logic

The original Streamlit application instantiated scikit-learn anomaly detectors and reduction models directly, and the text-feedback notebook instantiated `SentenceTransformer` directly. Those responsibilities now live in `AnomalyDetector`, `DimensionalityReducer`, and `TextFeedbackClusterer`.

Notebooks remain useful for analysis and visualization, but they are no longer an execution dependency.

## Configuration

The default YAML configuration moved from `config.yaml` to `configs/default.yaml`.

## Packaging

The legacy `setup.py` and duplicated dependency declarations were removed. `pyproject.toml` is now the authoritative package definition.

## CLI

Production workflows can now be invoked without Jupyter:

```bash
ml-unsupervised cluster ...
ml-unsupervised anomaly ...
ml-unsupervised reduce ...
```
