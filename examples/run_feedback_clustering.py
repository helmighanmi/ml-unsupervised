# Path: examples/run_feedback_clustering.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Cluster text feedback through the reusable NLP service."""

from ml_unsupervised.feedback import TextFeedbackClusterer

feedback = [
    "Checkout is slow on mobile",
    "Payment page takes too long",
    "I love the new dashboard",
    "The analytics view is excellent",
    "Search results are inaccurate",
    "Search often misses the right item",
]

service = TextFeedbackClusterer(
    clustering_method="kmeans",
    clustering_params={"n_clusters": 3, "random_state": 42, "n_init": 10},
)
result = service.fit_predict(feedback)
print(list(zip(result.texts, result.labels.tolist(), strict=True)))
