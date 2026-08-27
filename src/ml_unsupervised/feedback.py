# Path: src/ml_unsupervised/feedback.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Text-feedback embedding and clustering without notebook-owned model logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .clustering import ClusteringEngine
from .dimensionality_reduction import DimensionalityReducer
from .exceptions import OptionalDependencyError


class TextEncoder(Protocol):
    """Minimal interface allowing production encoders to be dependency-injected."""

    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class FeedbackClusteringResult:
    texts: list[str]
    embeddings: NDArray[np.float64]
    labels: NDArray[np.int_]
    projection: NDArray[np.float64] | None = None


@dataclass(slots=True)
class TextFeedbackClusterer:
    """Embed text and cluster the embeddings through reusable Python code.

    ``encoder`` can be injected in tests or services. If omitted, SentenceTransformer
    is loaded lazily, so importing the core package never downloads a model.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    clustering_method: str = "hdbscan"
    clustering_params: dict[str, Any] = field(default_factory=lambda: {"min_cluster_size": 2})
    encoder: TextEncoder | None = None
    normalize_embeddings: bool = True

    def _get_encoder(self) -> TextEncoder:
        if self.encoder is not None:
            return self.encoder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise OptionalDependencyError(
                "Text feedback clustering requires the 'nlp' extra: pip install -e '.[nlp]'"
            ) from exc
        self.encoder = SentenceTransformer(self.model_name)
        return self.encoder

    def embed(self, texts: list[str]) -> NDArray[np.float64]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain at least one non-empty string.")
        encoder = self._get_encoder()
        embeddings = encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        array = np.asarray(embeddings, dtype=float)
        if array.ndim != 2 or len(array) != len(texts):
            raise ValueError("Encoder returned an invalid embedding matrix.")
        return array

    def fit_predict(
        self,
        texts: list[str],
        *,
        projection_method: str | None = "pca",
        n_components: int = 2,
    ) -> FeedbackClusteringResult:
        embeddings = self.embed(texts)
        labels = ClusteringEngine(self.clustering_method, dict(self.clustering_params)).fit_predict(embeddings)
        projection = None
        if projection_method:
            projection = DimensionalityReducer(projection_method, n_components=n_components).fit_transform(embeddings)
        return FeedbackClusteringResult(
            texts=list(texts), embeddings=embeddings, labels=labels, projection=projection
        )
