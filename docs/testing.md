<!--
Path: docs/testing.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Testing strategy

The project distinguishes software correctness from model-quality exploration.

## Unit tests

Cover validation, estimator facades, normalized anomaly labels, noise-aware evaluation, preprocessing and persistence contracts.

## Integration tests

Cover CLI execution and the architectural rule that notebooks must not import the legacy `src.*` namespace.

## Notebook policy

CI validates notebook structure and package usage but does not execute every notebook. This avoids model downloads, long UMAP/t-SNE runs and environment-specific rendering in pull-request checks.

## ML quality

Unsupervised metrics are context-dependent. Silhouette, Davies-Bouldin and Calinski-Harabasz are useful diagnostics, not universal business-quality guarantees. Domain validation and stability checks should be added for production datasets.
