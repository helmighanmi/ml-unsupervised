# Path: tests/integration/test_cli.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from pathlib import Path

import pandas as pd

from ml_unsupervised.cli import main


def test_cluster_cli_writes_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    profile_path = tmp_path / "profile.csv"
    pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.2, 10.0, 10.1, 10.2],
            "y": [0.0, 0.2, 0.1, 10.0, 10.2, 10.1],
        }
    ).to_csv(input_path, index=False)

    exit_code = main(
        [
            "cluster",
            str(input_path),
            "--output",
            str(output_path),
            "--method",
            "kmeans",
            "--param",
            "n_clusters=2",
            "--param",
            "random_state=42",
            "--profile-out",
            str(profile_path),
        ]
    )
    assert exit_code == 0
    result = pd.read_csv(output_path)
    assert "cluster" in result.columns
    assert len(result) == 6
    assert profile_path.exists()


def test_anomaly_and_reduce_cli(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"x": [0, 0.1, 0.2, 10.0, 0.15, 9.8], "y": [0, 0.1, 0.2, 10.0, 0.12, 9.9]}).to_csv(
        input_path, index=False
    )
    anom_out = tmp_path / "anom.csv"
    summary_out = tmp_path / "summary.json"
    assert (
        main(
            [
                "anomaly",
                str(input_path),
                "--output",
                str(anom_out),
                "--param",
                "contamination=0.2",
                "--param",
                "random_state=42",
                "--summary-out",
                str(summary_out),
            ]
        )
        == 0
    )
    assert "is_outlier" in pd.read_csv(anom_out).columns
    assert summary_out.exists()

    red_out = tmp_path / "red.csv"
    assert (
        main(["reduce", str(input_path), "--output", str(red_out), "--method", "pca", "--components", "2"])
        == 0
    )
    red = pd.read_csv(red_out)
    assert "component_1" in red.columns and "component_2" in red.columns


def test_demo_cli_runs() -> None:
    # Uses bundled repo sample data; should succeed in project environment.
    exit_code = main(["demo"])
    assert exit_code == 0
