# Path: benchmarks/run_benchmarks.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-helmi

"""Reproducible benchmark comparing clustering methods on synthetic/sample data.

Outputs a Markdown table to stdout and optionally writes results to a file.
Benchmarks are intended for documentation and method-selection evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from ml_unsupervised.diagnostics import bootstrap_stability, compare_methods
from ml_unsupervised.data import load_tabular_dataset


def _run_synthetic(random_state: int = 42) -> pd.DataFrame:
    X, _ = make_blobs(n_samples=300, centers=4, n_features=5, cluster_std=1.2, random_state=random_state)
    methods = [
        ("kmeans", {"n_clusters": 4, "random_state": random_state, "n_init": 10}),
        ("gmm", {"n_components": 4, "random_state": random_state}),
        ("agglomerative", {"n_clusters": 4}),
        ("dbscan", {"eps": 1.1, "min_samples": 5}),
    ]
    rows = []
    for method, params in methods:
        t0 = time.perf_counter()
        res = compare_methods(X, [(method, params)])[0]
        elapsed = time.perf_counter() - t0
        stab = (
            bootstrap_stability(X, method=method, params=params, n_runs=5, random_state=random_state)
            if method in {"kmeans", "gmm"}
            else {"mean_ari": np.nan}
        )
        rows.append(
            {
                "dataset": "synthetic_blobs",
                "method": method,
                "n_clusters": res.n_clusters,
                "noise_ratio": round(res.noise_ratio, 3),
                "silhouette": round(res.metrics.silhouette, 4) if res.metrics else None,
                "davies_bouldin": round(res.metrics.davies_bouldin, 4) if res.metrics else None,
                "calinski_harabasz": round(res.metrics.calinski_harabasz, 2) if res.metrics else None,
                "stability_ari": None
                if np.isnan(stab.get("mean_ari", np.nan))
                else round(float(stab["mean_ari"]), 4),
                "seconds": round(elapsed, 4),
            }
        )
    return pd.DataFrame(rows)


def _run_sample() -> pd.DataFrame:
    path = Path(__file__).resolve().parents[1] / "data" / "sample_customers.csv"
    if not path.exists():
        return pd.DataFrame()
    dataset = load_tabular_dataset(path, feature_columns=["age", "annual_income", "spending_score"])
    X = dataset.X
    methods = [
        ("kmeans", {"n_clusters": 3, "random_state": 42, "n_init": 10}),
        ("gmm", {"n_components": 3, "random_state": 42}),
        ("agglomerative", {"n_clusters": 3}),
    ]
    rows = []
    for method, params in methods:
        t0 = time.perf_counter()
        res = compare_methods(X, [(method, params)])[0]
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "dataset": "sample_customers",
                "method": method,
                "n_clusters": res.n_clusters,
                "noise_ratio": round(res.noise_ratio, 3),
                "silhouette": round(res.metrics.silhouette, 4) if res.metrics else None,
                "davies_bouldin": round(res.metrics.davies_bouldin, 4) if res.metrics else None,
                "calinski_harabasz": round(res.metrics.calinski_harabasz, 2) if res.metrics else None,
                "stability_ari": None,
                "seconds": round(elapsed, 4),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run clustering benchmarks.")
    parser.add_argument("--output", type=Path, help="Optional path to write Markdown results.")
    parser.add_argument("--json", type=Path, help="Optional path to write JSON results.")
    args = parser.parse_args()

    frames = [_run_synthetic(), _run_sample()]
    frames = [f for f in frames if not f.empty]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        md = "No benchmark results."
    else:
        try:
            md = df.to_markdown(index=False)
        except Exception:
            headers = list(df.columns)
            lines = [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
            for _, row in df.iterrows():
                lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
            md = "\n".join(lines)
    print(md)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md + "\n", encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
