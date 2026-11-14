<!--
Path: docs/tutorial.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Tutorial: from data to interpreted results

This tutorial shows the recommended workflow using the package API. It uses the bundled sample dataset (or any similar CSV with numeric features).

## 1. Install and run the demo

```bash
python -m pip install -e ".[dev]"
ml-unsupervised demo
```

You should see a JSON summary with row/cluster counts and clustering metrics.

## 2. Cluster a dataset (Python)

```python
from ml_unsupervised import ClusteringPipeline
from ml_unsupervised.data import load_tabular_dataset

dataset = load_tabular_dataset(
    "data/sample_customers.csv",
    feature_columns=["age", "annual_income", "spending_score"],
)

pipeline = ClusteringPipeline(
    scaler="robust",
    reduction_method="pca",
    n_components=2,
    clustering_method="kmeans",
    clustering_params={"n_clusters": 3, "random_state": 42, "n_init": 10},
)

result = pipeline.fit_predict(dataset.X)
print(result.labels[:10])
print(result.metrics.to_dict() if result.metrics else "metrics unavailable")
```

**Interpretation tips**
- Inspect `result.metrics` (silhouette higher is better, Davies–Bouldin lower is better).
- Profile clusters by joining `result.labels` back to `dataset.frame` and computing per-cluster means.
- If you need to assign **new** customers later, keep using an inference-capable method (e.g. KMeans/GMM/Birch) and call `pipeline.predict(new_X)` / `pipeline.save(...)`.

## 3. Try density clustering (noise-aware)

```python
from ml_unsupervised import ClusteringEngine

labels = ClusteringEngine("dbscan", {"eps": 0.7, "min_samples": 5}).fit_predict(dataset.X)
# label -1 means noise
```

DBSCAN/HDBSCAN do not support out-of-sample `predict()` in this API; use `fit_predict()` on the full dataset.

## 4. Detect anomalies

```python
from ml_unsupervised import AnomalyDetector

det = AnomalyDetector("isolation_forest", {"contamination": 0.05, "random_state": 42})
res = det.fit_predict(dataset.X)
print(res.outlier_ratio, res.labels[:10])  # 0=inlier, 1=outlier
```

Validate flagged points against domain expectations; `contamination` is a target/prior.

## 5. Visualize

```python
from ml_unsupervised.visualization import plot_embedding

fig = plot_embedding(result.transformed, labels=result.labels, title="Clusters (PCA)")
fig.show()
```

## 6. CLI (no notebook)

```bash
ml-unsupervised cluster data/sample_customers.csv \
  --features age annual_income spending_score \
  --method kmeans --param n_clusters=3 --param random_state=42 \
  --reduce pca --components 2 \
  --output outputs/clustered.csv
```

## 7. Next steps
- Read [Algorithm selection](algorithm-selection.md) and [Metrics](metrics.md).
- For text feedback, install `.[nlp]` and use `TextFeedbackClusterer` (see README/examples).
- For HDBSCAN/UMAP, install `.[heavy]`.
- See [Model card](model-card.md) for intended use and limitations.
