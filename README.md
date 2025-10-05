# 🧠 ML Unsupervised Toolkit

![CI](https://github.com/<your-username>/ml-unsupervised/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/<your-username>/ml-unsupervised/branch/main/graph/badge.svg)](https://codecov.io/gh/<your-username>/ml-unsupervised)

A clean, professional repository for **clustering, dimensionality reduction, evaluation, and pipelines** in scikit-learn + Plotly.

## Features
- ✅ Clustering: KMeans, DBSCAN, HDBSCAN, Agglomerative
- ✅ Dimensionality Reduction: PCA, UMAP, t-SNE
- ✅ Evaluation: Silhouette, Davies-Bouldin, Calinski-Harabasz
- ✅ Pipelines for inference
- ✅ Interactive Plotly visualizations
- ✅ Unit tests + CI + Coverage

## Quickstart
```bash
git clone https://github.com/<your-username>/ml-unsupervised.git
cd ml-unsupervised
pip install -r requirements.txt
pytest
```

## Notebooks
- `01_clustering_basics.ipynb`
- `02_dimensionality_reduction.ipynb`
- `03_clustering_with_dim_reduction.ipynb`
- `04_pipeline_and_inference.ipynb`

## 📉 Anomaly Detection

This repo also includes unsupervised **anomaly detection** methods:

- Isolation Forest
- One-Class SVM
- Local Outlier Factor (LOF)

📓 Notebook: `08_anomaly_detection.ipynb`

## 📊 Summary of Modules

| Category                | Methods / Features                                   | Notebook(s)                                   |
|--------------------------|------------------------------------------------------|-----------------------------------------------|
| **Clustering**           | KMeans, DBSCAN, HDBSCAN, Agglomerative               | `01_clustering_basics.ipynb`, `03_clustering_with_dim_reduction.ipynb` |
| **Dimensionality Reduction** | PCA, t-SNE, UMAP                                 | `02_dimensionality_reduction.ipynb`, `03_clustering_with_dim_reduction.ipynb` |
| **Pipelines**            | sklearn Pipelines with scaling + clustering          | `04_pipeline_and_inference.ipynb`             |
| **Real-world Apps**      | Customer segmentation, LLM embeddings clustering     | `05_realworld_customer_clustering.ipynb`, `07_llm_feedback_clustering.ipynb` |
| **Config Management**    | YAML-based configs with `config.yaml` + loader        | `06_using_config.ipynb`                       |
| **Anomaly Detection**    | Isolation Forest, One-Class SVM, Local Outlier Factor| `08_anomaly_detection.ipynb`                  |

| **Advanced Clustering**  | Gaussian Mixture Models (GMM), Spectral Clustering, Birch | `09_additional_methods.ipynb` |
| **Advanced Dimensionality Reduction** | Kernel PCA, ICA, NMF, Factor Analysis | `09_additional_methods.ipynb` |


## 🐳 Run with Docker

You can run this project inside a Docker container for full reproducibility.

### Build the Docker image
```bash
docker build -t ml-unsupervised .
```

### Run tests inside Docker
```bash
docker run --rm ml-unsupervised
```

### Run Jupyter notebooks inside Docker
```bash
docker run -it -p 8888:8888 ml-unsupervised \
  jupyter notebook --ip=0.0.0.0 --allow-root
```
Then open [http://localhost:8888](http://localhost:8888) in your browser.

### Run with docker-compose (simpler)
```bash
docker-compose up
```
This will start Jupyter Notebook at [http://localhost:8888](http://localhost:8888).

## ⚡ Using Makefile

For convenience, common commands are included in a `Makefile`:

```bash
make docker-dev      # Build Docker image
make test        # Run tests inside Docker
make notebook    # Launch Jupyter Notebook (http://localhost:8888)
make compose     # Start Jupyter with docker-compose
make run-prod    # Build & run prod
```

## Nexts 
```bash
Play with real and more comlex datasets. Then add metrics and edge cases of each related datasets.
```
