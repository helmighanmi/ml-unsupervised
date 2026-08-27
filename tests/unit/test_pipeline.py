# Path: tests/unit/test_pipeline.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from pathlib import Path

from sklearn.datasets import make_blobs

from ml_unsupervised import ClusteringPipeline


def test_pipeline_fit_predict_predict_and_persistence(tmp_path: Path) -> None:
    X, _ = make_blobs(n_samples=80, centers=3, n_features=4, random_state=42)
    pipeline = ClusteringPipeline(
        scaler="standard",
        reduction_method="pca",
        n_components=2,
        clustering_method="kmeans",
        clustering_params={"n_clusters": 3, "random_state": 42, "n_init": 10},
    )
    result = pipeline.fit_predict(X)
    assert result.transformed.shape == (80, 2)
    assert result.labels.shape == (80,)
    assert pipeline.predict(X[:5]).shape == (5,)

    artifact = pipeline.save(tmp_path / "pipeline.joblib")
    loaded = ClusteringPipeline.load(artifact)
    assert loaded.predict(X[:5]).shape == (5,)
