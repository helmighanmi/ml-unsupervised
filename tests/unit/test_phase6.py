# Path: tests/unit/test_phase6.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-helmi

import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_blobs

from ml_unsupervised.diagnostics import embedding_trustworthiness, hopkins_statistic
from ml_unsupervised.registry import ModelRegistry
from ml_unsupervised.reporting import build_markdown_report, write_report


def test_hopkins_and_trustworthiness() -> None:
    X, _ = make_blobs(n_samples=80, centers=3, n_features=4, random_state=42)
    h = hopkins_statistic(X, sample_size=20, random_state=42)
    assert 0.0 <= h <= 1.0
    # Use first 2 dims as a trivial embedding for API/metric sanity
    t = embedding_trustworthiness(X, X[:, :2], n_neighbors=5)
    assert 0.0 <= t <= 1.0


def test_reporting_and_registry(tmp_path: Path) -> None:
    summary = {"clusters": 3, "rows": 100, "metrics": {"silhouette": 0.5}}
    profile = pd.DataFrame([{"cluster": 0, "count": 50}, {"cluster": 1, "count": 50}])
    md = build_markdown_report(title="T", summary=summary, profile=profile, notes=["n1"])
    assert "# T" in md and "silhouette" in md
    out_md = write_report(tmp_path / "r.md", title="T", summary=summary, profile=profile)
    out_html = write_report(tmp_path / "r.html", title="T", summary=summary, profile=profile)
    assert out_md.exists() and out_html.exists()
    assert "<html" in out_html.read_text(encoding="utf-8").lower()

    artifact = tmp_path / "model.joblib"
    artifact.write_text("dummy", encoding="utf-8")
    reg = ModelRegistry(tmp_path / "reg")
    entry = reg.register("demo", artifact, version="v1", metadata={"k": 3})
    assert reg.latest("demo").version == "v1"
    assert entry.metadata["k"] == 3


def test_report_cli(tmp_path: Path) -> None:
    from ml_unsupervised.cli import main

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({"rows": 10, "clusters": 2}), encoding="utf-8")
    profile_path = tmp_path / "profile.csv"
    pd.DataFrame([{"cluster": 0, "count": 5}, {"cluster": 1, "count": 5}]).to_csv(profile_path, index=False)
    out = tmp_path / "report.md"
    code = main(
        ["report", "--summary", str(summary_path), "--profile", str(profile_path), "--output", str(out)]
    )
    assert code == 0 and out.exists()


def test_api_create_app_health_and_cluster() -> None:
    from fastapi.testclient import TestClient
    from ml_unsupervised.api import create_app

    app = create_app()
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    X = [[0.0, 0.1], [0.2, 0.0], [5.0, 5.1], [5.2, 5.0]]
    resp = client.post(
        "/cluster",
        json={"X": X, "method": "kmeans", "params": {"n_clusters": 2, "random_state": 42, "n_init": 10}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["labels"]) == 4
