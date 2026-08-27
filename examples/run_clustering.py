# Path: examples/run_clustering.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Run the production clustering API directly from Python."""

from sklearn.datasets import load_iris

from ml_unsupervised import ClusteringPipeline

X = load_iris().data
pipeline = ClusteringPipeline(
    scaler="robust",
    reduction_method="pca",
    n_components=2,
    clustering_method="kmeans",
    clustering_params={"n_clusters": 3, "random_state": 42, "n_init": 10},
)
result = pipeline.fit_predict(X)

print("First labels:", result.labels[:10].tolist())
print("Metrics:", result.metrics.to_dict() if result.metrics else None)
pipeline.save("artifacts/iris_clustering.joblib")
