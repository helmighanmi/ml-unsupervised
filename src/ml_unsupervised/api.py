# Path: src/ml_unsupervised/api.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-helmi

"""Minimal FastAPI service exposing core unsupervised workflows.

This module keeps FastAPI as an optional dependency: import/use only when the
`api` extra is installed.
"""

from typing import Any

import numpy as np


def create_app():
    try:
        from fastapi import Body, FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("FastAPI service requires the 'api' extra: pip install -e '.[api]'") from exc

    from .anomaly_detection import AnomalyDetector
    from .dimensionality_reduction import DimensionalityReducer
    from .exceptions import MLUnsupervisedError
    from .pipelines import ClusteringPipeline

    app = FastAPI(title="ML Unsupervised Toolkit API", version="1.0.0")

    class MatrixInput(BaseModel):
        X: list[list[float]] = Field(..., description="2D numeric feature matrix")

    class ClusterInput(MatrixInput):
        method: str = "kmeans"
        scaler: str = "robust"
        reduction_method: str | None = None
        n_components: int = 2
        params: dict[str, Any] = Field(default_factory=dict)

    class AnomalyInput(MatrixInput):
        method: str = "isolation_forest"
        params: dict[str, Any] = Field(default_factory=dict)

    class ReduceInput(MatrixInput):
        method: str = "pca"
        n_components: int = 2
        params: dict[str, Any] = Field(default_factory=dict)

    for model in (MatrixInput, ClusterInput, AnomalyInput, ReduceInput):
        model.model_rebuild()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/cluster")
    def cluster(payload: ClusterInput = Body(...)) -> dict[str, Any]:  # noqa: B008
        try:
            pipeline = ClusteringPipeline(
                scaler=payload.scaler,
                clustering_method=payload.method,
                clustering_params=payload.params,
                reduction_method=payload.reduction_method,
                n_components=payload.n_components,
            )
            result = pipeline.fit_predict(np.asarray(payload.X, dtype=float))
            return {
                "labels": result.labels.tolist(),
                "metrics": result.metrics.to_dict() if result.metrics else None,
            }
        except (MLUnsupervisedError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/anomaly")
    def anomaly(payload: AnomalyInput = Body(...)) -> dict[str, Any]:  # noqa: B008
        try:
            det = AnomalyDetector(payload.method, payload.params)
            result = det.fit_predict(np.asarray(payload.X, dtype=float))
            return {
                "labels": result.labels.tolist(),
                "scores": result.scores.tolist() if result.scores is not None else None,
                "outlier_ratio": result.outlier_ratio,
            }
        except (MLUnsupervisedError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/reduce")
    def reduce(payload: ReduceInput = Body(...)) -> dict[str, Any]:  # noqa: B008
        try:
            reducer = DimensionalityReducer(payload.method, payload.n_components, payload.params)
            emb = reducer.fit_transform(np.asarray(payload.X, dtype=float))
            return {"embedding": emb.tolist()}
        except (MLUnsupervisedError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


# Convenience for `uvicorn ml_unsupervised.api:app`
try:  # pragma: no cover - depends on optional extra
    app = create_app()
except Exception:
    app = None
