# Path: src/ml_unsupervised/data/loading.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-helmi

"""Tabular dataset loading and validation shared by CLI and UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..core.exceptions import OptionalDependencyError


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

    if suffix == ".csv":
        frame = pd.read_csv(source)
    else:
        # XLSX support requires the optional 'openpyxl' dependency.
        # The package lazily imports it, so we check early and give a clear message.
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:
            raise OptionalDependencyError(
                "XLSX support requires the 'openpyxl' package. Install it with: python -m pip install '.[ui]'"
            ) from exc
        frame = pd.read_excel(source)

    excluded = set(exclude_columns or [])
    if feature_columns is not None:
        missing = [column for column in feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Requested feature columns are missing from dataset: {missing}")
        feature_frame = frame[feature_columns].copy()
    else:
        feature_frame = frame.drop(columns=[c for c in frame.columns if c in excluded], errors="ignore")
        feature_frame = feature_frame.select_dtypes(include=[np.number]).copy()

    if feature_frame.empty:
        raise ValueError("No numeric feature columns available after filtering.")

    non_numeric = [c for c in feature_frame.columns if not pd.api.types.is_numeric_dtype(feature_frame[c])]
    if non_numeric:
        raise ValueError(f"All modeling features must be numeric; non-numeric columns: {non_numeric}")
    if feature_frame.isna().any().any():
        raise ValueError(
            "Missing values detected in modeling features. "
            "Impute or remove them before clustering or anomaly detection."
        )

    return TabularDataset(frame=frame, feature_frame=feature_frame)
