<!--
Path: docs/metrics.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Metrics and interpretation

Unsupervised metrics are **diagnostics, not ground truth**. Always combine them with visual inspection, stability checks, and domain validation.

## Internal clustering metrics (no labels required)

The toolkit reports these via `ClusteringEvaluator` / `ClusteringMetrics`:

| Metric | Range / direction | Intuition | Caveats |
|---|---|---|---|
| **Silhouette** | [-1, 1], higher better | Separation vs cohesion of clusters | Favors convex clusters; can be misleading for density/manifold shapes |
| **Davies–Bouldin** | [0, ∞), lower better | Average cluster similarity | Sensitive to cluster structure |
| **Calinski–Harabasz** | [0, ∞), higher better | Ratio of between/within dispersion | Tends to favor more convex/balanced partitions |

### Noise-aware evaluation
For density methods (e.g. DBSCAN/HDBSCAN), label `-1` denotes noise. With `ignore_noise=True` (default in evaluation paths where used), metrics are computed on non-noise samples and `ignored_noise_samples` is reported. A result with many noise points can have “good” metrics on the remainder but poor overall coverage—always report noise rate.

### When metrics are undefined
Metrics require at least 2 clusters and enough samples; single-cluster or degenerate cases return no metrics (pipelines handle this gracefully).

## Choosing `k` / comparing methods
- Compare methods on the **same preprocessed data**.
- Sweep `k` (e.g. 2–10) and inspect metric curves, not just the maximum.
- Prefer **stable** solutions: rerun with different seeds/subsamples and check consistency (e.g. Adjusted Rand Index between runs).
- Use **cluster profiles** (feature means per cluster) to ensure segments are interpretable and actionable.

## Anomaly detection
- Toolkit normalizes to `0=inlier, 1=outlier`; `outlier_ratio` is the observed rate.
- `contamination`/`nu` are **priors/targets**, not guaranteed rates.
- Validate with known incidents, spot checks, or downstream impact—internal scores alone are insufficient.

## Dimensionality reduction
- PCA explained variance (from the underlying estimator) indicates information retained.
- t-SNE/UMAP plots are for **exploration**: distances/clusters in the plot can be artifacts; do not treat as quantitative proof.

## Recommended reporting
When presenting results, report: algorithm + key params, preprocessing, sample count, number of clusters/noise rate, metrics, stability (if computed), and a short interpretation/limitations note.
