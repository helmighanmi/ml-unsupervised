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
        ]
    )
    assert exit_code == 0
    result = pd.read_csv(output_path)
    assert "cluster" in result.columns
    assert len(result) == 6
