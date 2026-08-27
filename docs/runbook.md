<!--
Path: docs/runbook.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Operational runbook

## CLI fails on missing values

The tabular loader fails fast when modeling columns contain missing values. Impute or clean upstream; do not silently replace values in production.

## `predict()` is unavailable

Some clustering algorithms are transductive and do not support prediction on unseen data. Use an inference-capable algorithm such as KMeans/GMM/Birch, or redesign the assignment strategy explicitly.

## HDBSCAN/UMAP import error

Install the project dependencies in the active environment. These libraries are lazily imported so unrelated features remain importable.

## SentenceTransformer error or model download

Install the NLP extra with `pip install -e ".[nlp]"`. The first real encoder use may download the selected model. Production environments should control their model cache and artifact provenance.

## Pipeline artifact compatibility

Joblib/scikit-learn artifacts are not a stable cross-version interchange format. Record the package environment used to train artifacts and rebuild them when upgrading core ML dependencies.
