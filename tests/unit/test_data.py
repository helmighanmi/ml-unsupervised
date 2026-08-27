# Path: tests/unit/test_data.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from pathlib import Path

import pandas as pd
import pytest

from ml_unsupervised.data import load_tabular_dataset


def test_loader_defaults_to_numeric_features(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"age": [20, 30], "income": [10.0, 20.0], "segment": ["a", "b"]}).to_csv(path, index=False)
    dataset = load_tabular_dataset(path)
    assert dataset.feature_frame.columns.tolist() == ["age", "income"]
    assert dataset.X.shape == (2, 2)


def test_loader_rejects_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1.0, None], "y": [2.0, 3.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing values"):
        load_tabular_dataset(path)
