<!--
Path: docs/architecture.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Architecture

## Objective

The architecture separates **reusable ML behavior** from **interactive analysis and presentation**. A notebook must never be required to execute a production workflow.

## Layers

### Core package — `src/ml_unsupervised/`

Owns validation, preprocessing, clustering, dimensionality reduction, anomaly detection, evaluation, persistence, text-feedback embedding orchestration and the CLI.

### Presentation — `streamlit_app/`

Owns Streamlit widgets and rendering. It imports the same core classes available to every other consumer.

### Analysis — `notebooks/`

Owns exploration, plots, comparison and interpretation. Notebooks may assemble package APIs but should not define reusable production classes, model-loading logic or duplicated algorithms.

### Verification — `tests/`

Unit tests validate individual contracts. Integration tests validate cross-module behavior such as CLI output, persistence and notebook/package boundaries.

## Important boundaries

- `ClusteringEngine` owns estimator construction and prediction capability checks.
- `DimensionalityReducer` owns reduction estimator construction.
- `AnomalyDetector` normalizes detector output to `0=inlier`, `1=outlier`.
- `ClusteringPipeline` composes preprocessing, optional reduction and clustering.
- `TextFeedbackClusterer` owns embedding-model loading and text-to-cluster orchestration.
- `ClusteringEvaluator` handles noise-aware internal clustering metrics.

## Inference capability

Not every unsupervised estimator supports inference for unseen observations. The public API exposes this difference explicitly. A fitted KMeans pipeline can predict new samples. Standard DBSCAN, AgglomerativeClustering and SpectralClustering cannot use the same `predict()` contract, so the package raises a clear error.
