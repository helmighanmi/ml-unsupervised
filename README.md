<!--
Path: README.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# ML Unsupervised Toolkit

A production-oriented Python toolkit for **clustering, dimensionality reduction, anomaly detection, text-feedback clustering, evaluation, and reusable unsupervised-learning pipelines**.

The repository is intentionally **package first**:

> Production logic lives in `src/ml_unsupervised/`. Notebooks and Streamlit are clients of that package, never the owner of model logic.

That makes the same implementation usable from a Python service, scheduled job, CLI, Streamlit application, automated test, Docker container, or notebook.

## Engineering scope

- Clustering: KMeans, DBSCAN, HDBSCAN, Agglomerative, GMM, Spectral, Birch
- Dimensionality reduction: PCA, t-SNE, UMAP, Kernel PCA, ICA, NMF, Factor Analysis
- Anomaly detection: Isolation Forest, One-Class SVM, Local Outlier Factor
- Text feedback: SentenceTransformer embeddings + reusable clustering service
- Noise-aware clustering evaluation
- Stateful preprocessing and clustering pipelines
- Model/pipeline persistence with Joblib
- CLI execution without Jupyter
- Thin Streamlit presentation layer
- Unit/integration tests, linting, type checking, CI, security audit, CodeQL, Dependabot
- Dockerized CLI/UI runtime

## Architecture

```text
CSV/XLSX / NumPy / service input
              │
              ▼
      ml_unsupervised.data
              │
              ▼
   FeaturePreprocessor
              │
        optional reduction
              │
              ▼
      ClusteringEngine ─────► ClusteringEvaluator
              │
              ▼
        labels / model

Other production APIs:

AnomalyDetector
TextFeedbackClusterer
DimensionalityReducer
ClusteringPipeline

Consumers:
├── CLI
├── Streamlit
├── Python applications
├── tests
└── notebooks (analysis only)
```

See [`docs/architecture.md`](docs/architecture.md) for design details.

## Python version

The reference runtime is **Python 3.11**. CI validates Python 3.11 and 3.12.

## Quickstart

### Linux / macOS / GitHub Codespaces

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

Install additional interfaces only when needed:

```bash
# Streamlit + Excel input
python -m pip install -e ".[ui]"

# SentenceTransformer feedback clustering
python -m pip install -e ".[nlp]"

# Jupyter analysis environment
python -m pip install -e ".[notebooks]"
```

## Use the Python classes directly

### End-to-end clustering

```python
from sklearn.datasets import load_iris
from ml_unsupervised import ClusteringPipeline

X = load_iris().data

pipeline = ClusteringPipeline(
    scaler="standard",
    reduction_method="pca",
    n_components=2,
    clustering_method="kmeans",
    clustering_params={
        "n_clusters": 3,
        "random_state": 42,
        "n_init": 10,
    },
)

result = pipeline.fit_predict(X)
print(result.labels)
print(result.metrics.to_dict() if result.metrics else None)

# Persist a fitted inference-capable workflow.
pipeline.save("artifacts/iris.joblib")
```

Load it later:

```python
from ml_unsupervised import ClusteringPipeline

pipeline = ClusteringPipeline.load("artifacts/iris.joblib")
labels = pipeline.predict(new_samples)
```

Out-of-sample `predict()` is intentionally available only for algorithms whose estimator supports it. For example, KMeans supports prediction; Agglomerative clustering does not.

### Direct clustering engine

```python
from ml_unsupervised import ClusteringEngine

labels = ClusteringEngine(
    "kmeans",
    {"n_clusters": 5, "random_state": 42},
).fit_predict(X)
```

### Dimensionality reduction

```python
from ml_unsupervised import DimensionalityReducer

embedding = DimensionalityReducer(
    "pca",
    n_components=2,
).fit_transform(X)
```

### Anomaly detection

```python
from ml_unsupervised import AnomalyDetector

result = AnomalyDetector(
    "isolation_forest",
    {"contamination": 0.05, "random_state": 42},
).fit_predict(X)

# Public contract: 0 = inlier, 1 = outlier
print(result.labels)
print(result.outlier_ratio)
```

### Text-feedback clustering

```python
from ml_unsupervised.feedback import TextFeedbackClusterer

feedback = [
    "Checkout is slow",
    "Payment fails on mobile",
    "The dashboard looks great",
]

service = TextFeedbackClusterer(
    clustering_method="kmeans",
    clustering_params={"n_clusters": 2, "random_state": 42},
)
result = service.fit_predict(feedback)
```

The SentenceTransformer model is loaded lazily. Importing the core package does not download an embedding model.

## Convenience functions

Small scripts can still use function-style APIs:

```python
from ml_unsupervised.clustering import run_kmeans
from ml_unsupervised.dimensionality_reduction import run_pca

X_pca, pca = run_pca(X, n_components=2)
labels, centers = run_kmeans(X_pca, n_clusters=3)
```

These wrappers make migration from older notebooks straightforward while the class APIs remain the recommended production interface.

## Run without a notebook: CLI

After installation:

```bash
ml-unsupervised --help
```

Cluster a CSV:

```bash
ml-unsupervised cluster data/customers.csv \
  --features age annual_income spending_score \
  --method kmeans \
  --scaler standard \
  --reduce pca \
  --components 2 \
  --param n_clusters=5 \
  --param random_state=42 \
  --output outputs/customers_clustered.csv \
  --model-out artifacts/customer_pipeline.joblib
```

Detect anomalies:

```bash
ml-unsupervised anomaly data/customers.csv \
  --features age annual_income spending_score \
  --method isolation_forest \
  --param contamination=0.05 \
  --output outputs/anomalies.csv
```

Create a low-dimensional embedding:

```bash
ml-unsupervised reduce data/customers.csv \
  --features age annual_income spending_score \
  --method pca \
  --components 2 \
  --output outputs/embedding.csv
```

The repository includes a deterministic sample dataset:

```bash
ml-unsupervised cluster data/sample_customers.csv \
  --features age annual_income spending_score \
  --method kmeans \
  --param n_clusters=5 \
  --output outputs/sample_clusters.csv
```

## Notebooks: analysis, not production runtime

The notebooks intentionally contain exploration, visualization and interpretation only. They import the same package used by scripts and applications.

Start with:

```text
notebooks/00_package_quickstart.ipynb
```

Then explore:

```text
01_clustering_basics.ipynb
02_dimensionality_reduction.ipynb
03_clustering_with_dim_reduction.ipynb
04_pipeline_and_inference.ipynb
05_realworld_customer_clustering.ipynb
06_using_config.ipynb
07_llm_feedback_clustering.ipynb
08_anomaly_detection.ipynb
09_additional_methods.ipynb
```

Run Jupyter only when you want analysis:

```bash
python -m pip install -e ".[notebooks]"
jupyter lab
```

## Streamlit

The Streamlit app delegates model work to `ml_unsupervised`; it does not redefine the algorithms.

```bash
python -m pip install -e ".[ui]"
streamlit run streamlit_app/app.py
```

## Docker

Build:

```bash
docker build -t ml-unsupervised:local .
```

The container defaults to the CLI:

```bash
docker run --rm ml-unsupervised:local --help
```

Run the UI:

```bash
docker compose up --build
```

Then open port `8501` in your environment.

## Quality gates

```bash
make lint
make typecheck
make test
make quality
```

CI runs quality checks on Python 3.11 and 3.12 and builds the Docker image. A separate workflow runs dependency auditing and CodeQL.

## Repository layout

```text
.
├── .github/
│   └── workflows/
├── configs/
├── data/
├── docs/
│   └── decisions/
├── examples/
├── notebooks/            # exploration and analysis only
├── src/
│   └── ml_unsupervised/  # production code
├── streamlit_app/        # presentation layer
├── tests/
│   ├── unit/
│   └── integration/
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## Design principles

1. **One implementation of each ML behavior.** UI and notebooks call package code.
2. **Explicit algorithm limitations.** Unsupported inference fails clearly instead of pretending all clustering methods can predict new samples.
3. **Reproducible defaults.** Randomized algorithms use explicit random states where supported.
4. **Optional heavyweight features stay lazy.** Importing the package does not download NLP models.
5. **Evaluation matches unsupervised reality.** Density-clustering noise can be excluded from internal metrics.
6. **Portable execution.** The same package runs from Python, CLI, Docker, Streamlit, tests and notebooks.

## License

The project source is licensed under the MIT License. Third-party dependencies retain their respective licenses.
