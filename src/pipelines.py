"""
Author: GHNAMI Helmi
Date: 2025-09-16
Position: Data-Science
"""

from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from src.preprocessing import get_scaler
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def clustering_pipeline(n_clusters=3):
    config = load_config()
    scaler_name = config.get("preprocessing", {}).get("scaler", "robust")
    scaler = get_scaler(scaler_name)

    pipe = Pipeline([
        ("scaler", scaler),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
    ])
    return pipe
