<!--
Path: docs/architecture.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Architecture

## Objective

Separate **reusable ML behavior** from **interfaces and presentation**, with explicit dependency direction. A notebook must never be required to execute a production workflow.

## Layered package layout (`src/ml_unsupervised/`)

```text
core/            # exceptions, shared validation
data/            # loading, preprocessing
algorithms/      # clustering, reduction, anomaly, feedback
workflows/       # composed pipelines
evaluation/      # metrics, diagnostics
reporting/       # plots, markdown/html reports
infrastructure/  # config, registry
interfaces/      # CLI, FastAPI
```

### Dependency rules

| Layer | May import | Must not import |
|---|---|---|
| `core` | stdlib, NumPy | other package layers |
| `data` | `core` | algorithms, workflows, interfaces |
| `algorithms` | `core` | data loaders, workflows, reporting, interfaces |
| `evaluation` | `core`, `algorithms` | workflows, reporting, interfaces |
| `workflows` | `core`, `data`, `algorithms`, `evaluation` | reporting, interfaces |
| `reporting` | `core` (+ result types) | algorithms, workflows, interfaces |
| `infrastructure` | `core` | algorithms, interfaces |
| `interfaces` | all layers | — |

Rules are guarded by `tests/test_architecture.py`.

## Consumers

- **`streamlit_app/`** — thin presentation client.
- **`notebooks/`** — analysis only (no production-owned logic).
- **CLI/API** — adapters in `interfaces/`.

## Compatibility

Legacy flat module paths (e.g. `ml_unsupervised.clustering`, `ml_unsupervised.pipelines`) are retained as **re-export shims** for incremental migration (see ADR `004-layered-package-layout.md`). Prefer canonical layered imports for new code.

## Important boundaries

- `algorithms.clustering.ClusteringEngine` owns estimator construction and prediction capability checks.
- `algorithms.reduction.DimensionalityReducer` owns reduction estimators.
- `algorithms.anomaly.AnomalyDetector` normalizes output to `0=inlier`, `1=outlier`.
- `workflows.pipeline.ClusteringPipeline` composes preprocessing, optional reduction, clustering, and evaluation.
- `algorithms.feedback.TextFeedbackClusterer` owns embedding-model loading and orchestration.
- `evaluation.metrics.ClusteringEvaluator` handles noise-aware internal clustering metrics.

## Inference capability

Not every unsupervised estimator supports inference for unseen observations. The public API exposes this explicitly: estimators without out-of-sample `predict()` raise a clear error instead of returning misleading results.
