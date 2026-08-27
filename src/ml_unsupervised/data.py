# Path: src/ml_unsupervised/data.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Tabular dataset loading and validation shared by CLI and UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class TabularDataset:
    frame: pd.DataFrame
    feature_frame: pd.DataFrame

    @property
    def X(self) -> NDArray[np.float64]:
        return self.feature_frame.to_numpy(dtype=float)


SUPPORTED_TABULAR_SUFFIXES = {".csv", ".xlsx"}


def load_tabular_dataset(
    path: str | Path,
    *,
    feature_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
) -> TabularDataset:
    """Load numeric features from CSV/XLSX and fail clearly on unsafe input."""
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_TABULAR_SUFFIXES:
        raise ValueError(f"Unsupported dataset format {suffix!r}; use CSV or XLSX.")

    frame = pd.read_csv(source) if suffix == ".csv" else pd.read_excel(source)
    if frame.empty:
        raise ValueError("Dataset is empty.")

    if feature_columns:
        missing = sorted(set(feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Unknown feature columns: {missing}")
        features = frame.loc[:, feature_columns].copy()
    else:
        features = frame.drop(columns=exclude_columns or [], errors="ignore").select_dtypes(include="number")

    if features.shape[1] == 0:
        raise ValueError("No numeric feature columns are available for modeling.")
    if features.isna().any().any():
        missing_columns = features.columns[features.isna().any()].tolist()
        raise ValueError(
            "Missing values detected in modeling features. Impute or clean these columns first: "
            f"{missing_columns}"
        )
    return TabularDataset(frame=frame, feature_frame=features)
