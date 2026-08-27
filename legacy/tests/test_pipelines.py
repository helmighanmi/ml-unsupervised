import numpy as np
from src.pipelines import clustering_pipeline

X = np.random.rand(20, 5)

def test_pipeline_fit_predict():
    pipe = clustering_pipeline(n_clusters=2)
    pipe.fit(X)
    labels = pipe.predict(X)
    assert len(labels) == len(X)
