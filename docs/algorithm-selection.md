<!--
Path: docs/algorithm-selection.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Algorithm selection guide

Use this guide to choose a starting point quickly. In unsupervised learning there is no single “best” model; validate with metrics, stability checks, and domain judgment.

## Clustering

| Algorithm | Choose when | Strengths | Limitations | Inference on new data? |
|---|---|---|---|---|
| **KMeans** | Roughly spherical clusters, known or testable `k`, scalable baseline | Fast, simple, easy to explain | Assumes convex clusters; sensitive to scale and outliers | **Yes** |
| **Gaussian Mixture (GMM)** | Soft/overlapping clusters, probabilistic membership useful | Soft assignments, elliptical clusters | Sensitive to initialization/assumptions; choose components | **Yes** |
| **Birch** | Large datasets, streaming-like summaries, need inference | Efficient, hierarchical summary | Sensitive to threshold/structure | **Yes** |
| **DBSCAN** | Arbitrary shapes and explicit noise detection; `k` unknown | Finds noise (`-1`), no fixed `k` | Sensitive to `eps`/`min_samples`; struggles with varying density | **No** (standard API) |
| **HDBSCAN** | Varying densities, want robust density clustering + noise | Handles varying density better than DBSCAN | Optional dependency; parameter intuition needed | **No** (standard API) |
| **Agglomerative** | Want hierarchy/dendrogram intuition, small-medium data | Flexible linkage strategies | No standard out-of-sample predict; can be slow | **No** |
| **Spectral** | Non-convex/manifold structure (e.g. moons) on small data | Captures graph/manifold structure | Memory/compute heavy; sensitive params | Implementation-dependent; not exposed as inference here |

**Practical default:** start with **KMeans** (after scaling, optionally PCA) as a baseline, then compare with **HDBSCAN/DBSCAN** if you suspect noise or non-convex structure.

## Dimensionality reduction

| Method | Choose when | Notes |
|---|---|---|
| **PCA** | Linear structure, speed, interpretability, preprocessing for clustering | Deterministic given data; supports transform/inverse intuition |
| **t-SNE** | 2D/3D visualization of local neighborhoods | Visualization-focused; no out-of-sample transform in this API; perplexity must fit sample size |
| **UMAP** | Visualization or nonlinear embedding (optional dep) | Preserves more global structure than t-SNE in many cases; stochastic without fixed seed/settings |
| **Kernel PCA** | Nonlinear projection with kernel trick | Can be memory intensive |
| **ICA** | Independent latent sources/components | Assumptions matter |
| **NMF** | Non-negative data and additive parts-based factors | Requires non-negative inputs |
| **Factor Analysis** | Latent factor modeling with noise | Statistical assumptions apply |

**Practical default:** **PCA** for preprocessing/analysis; **t-SNE/UMAP** for visualization only (don’t over-interpret distances).

## Anomaly detection

| Method | Choose when | Notes |
|---|---|---|
| **Isolation Forest** | General-purpose tabular outliers, good default | Fast, handles high dimensions reasonably; set contamination/random_state |
| **Local Outlier Factor (LOF)** | Local density deviations matter | This toolkit enables `novelty=True` for inference consistency |
| **One-Class SVM** | Boundary-based novelty detection, smaller/cleaner normal data | Sensitive to kernel/nu/gamma and scaling |

**Contract:** toolkit outputs `0 = inlier`, `1 = outlier` (and higher anomaly scores indicate more anomalous).

## Text feedback clustering

Use `TextFeedbackClusterer` when inputs are short texts (reviews, feedback, tickets). It embeds text (SentenceTransformers, lazy-loaded) then clusters embeddings. Start with **KMeans** for a fixed number of themes or **HDBSCAN** to discover themes and noise (requires `[heavy]` + `[nlp]` extras as applicable).
