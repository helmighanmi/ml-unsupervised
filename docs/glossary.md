<!--
Path: docs/glossary.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Glossary

- **Clustering:** grouping samples so that items in the same group are more similar to each other than to those in other groups, without using labels.
- **Dimensionality reduction:** transforming high-dimensional data into fewer dimensions while preserving useful structure (for compression, denoising, or visualization).
- **Anomaly / outlier:** a sample that deviates significantly from the expected pattern (fraud, faults, rare events). In this toolkit: `0 = inlier`, `1 = outlier`.
- **Noise (label `-1`):** in density-based clustering (DBSCAN/HDBSCAN), points not assigned to any cluster.
- **Inference / out-of-sample prediction:** assigning a cluster or label to *new* samples using a fitted model. Not all clustering algorithms support this (e.g. standard DBSCAN/Agglomerative do not in this API).
- **Silhouette score:** measure of how similar a sample is to its own cluster vs the nearest other cluster (higher is better).
- **Davies–Bouldin / Calinski–Harabasz:** internal clustering validity indices (lower / higher is better respectively).
- **PCA (Principal Component Analysis):** linear projection onto directions of maximum variance.
- **t-SNE / UMAP:** nonlinear manifold-learning methods commonly used for 2D/3D visualization.
- **Contamination / nu:** expected proportion or bound related to anomalies, used by detectors as a parameter/prior.
- **Embedding:** a vector representation of data (numeric features or encoded text) in a continuous space.
- **Pipeline:** a composed workflow (preprocessing → optional reduction → model) that can be fit, evaluated, and persisted consistently.
- **Stability:** robustness of clustering results to perturbations (different seeds, subsamples); often assessed with agreement measures like ARI.
