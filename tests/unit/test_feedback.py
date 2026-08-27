# Path: tests/unit/test_feedback.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

import numpy as np

from ml_unsupervised.feedback import TextFeedbackClusterer


class FakeEncoder:
    def encode(self, sentences, **kwargs):
        return np.array([[float(i), float(i % 2), 1.0] for i, _ in enumerate(sentences)])


def test_feedback_service_is_testable_without_downloading_model() -> None:
    texts = ["a", "b", "c", "d"]
    service = TextFeedbackClusterer(
        encoder=FakeEncoder(),
        clustering_method="kmeans",
        clustering_params={"n_clusters": 2, "random_state": 42, "n_init": 10},
    )
    result = service.fit_predict(texts, projection_method="pca", n_components=2)
    assert result.embeddings.shape == (4, 3)
    assert result.labels.shape == (4,)
    assert result.projection is not None and result.projection.shape == (4, 2)
