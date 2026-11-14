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

Cover validation, estimator facades, normalized anomaly labels, noise-aware evaluation, preprocessing and persistence contracts. Tests should also lock in **behavioral regressions** where practical (e.g. deterministic clustering on blobs, injected outliers detected, correct error types for unsupported inference/optional deps).

## Integration tests

Cover CLI execution (cluster/anomaly/reduce/demo) and the architectural rule that notebooks must not import the legacy `src.*` namespace.

## Benchmarks

`benchmarks/run_benchmarks.py` provides reproducible method comparisons (metrics, runtime, stability) on synthetic/sample data. Benchmarks are evidence for documentation and method selection, not required on every PR.

## Notebook policy

CI validates notebook structure and package usage but does not execute every notebook. This avoids model downloads, long UMAP/t-SNE runs and environment-specific rendering in pull-request checks.

## Quality gates

CI runs linting, formatting checks, type checking, and tests with coverage. A minimum coverage threshold is enforced to prevent regression in tested behavior.

## ML quality

Unsupervised metrics are context-dependent. Silhouette, Davies-Bouldin and Calinski-Harabasz are useful diagnostics, not universal business-quality guarantees. Domain validation and stability checks should be added for production datasets.
