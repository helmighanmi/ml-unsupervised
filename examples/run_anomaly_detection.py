# Path: examples/run_anomaly_detection.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Run anomaly detection directly from Python without Streamlit or Jupyter."""

from sklearn.datasets import load_iris

from ml_unsupervised import AnomalyDetector

X = load_iris().data
result = AnomalyDetector(
    "isolation_forest",
    {"contamination": 0.1, "random_state": 42},
).fit_predict(X)

print("Outlier ratio:", result.outlier_ratio)
print("First normalized labels (0=inlier, 1=outlier):", result.labels[:10].tolist())
