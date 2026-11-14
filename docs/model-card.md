<!--
Path: docs/model-card.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Model card

## Overview
**ML Unsupervised Toolkit** provides reusable implementations of classical unsupervised methods: clustering, dimensionality reduction, anomaly detection, and text-feedback clustering. It is a **library/toolkit**, not a single pretrained model (except optional third-party encoders such as SentenceTransformers used on demand).

## Intended use
- Exploratory data analysis and segmentation on tabular numeric data.
- Outlier/novelty detection with explicit review.
- Low-dimensional visualization and preprocessing.
- Grouping short text feedback into themes (with appropriate NLP extras).
- Education and prototyping of unsupervised workflows with production-oriented structure (validation, persistence, CLI, tests).

## Out-of-scope / not intended for
- Fully automated, high-stakes decisions without human review and domain validation.
- Causal inference, fairness certification, or safety guarantees.
- Direct use on PII/sensitive data without governance, minimization, and access controls.
- Treating 2D/3D embeddings (t-SNE/UMAP) as quantitative proof of structure.

## Data considerations
- Tabular loaders expect **finite numeric** modeling features; missing values must be handled upstream.
- Results depend heavily on **feature selection, scaling, and preprocessing**.
- Text clustering quality depends on language, domain, and the chosen embedding model; embeddings may reflect biases present in pretraining data.

## Evaluation and validation
- Internal metrics (silhouette, Davies–Bouldin, Calinski–Harabasz) are **diagnostics**, not business truth.
- Recommended: compare multiple algorithms/params, check stability across seeds/subsamples, inspect cluster profiles and sample members, and validate with domain experts.
- For anomaly detection, validate flagged cases against known outcomes/incidents.

## Limitations
- Different algorithms make different assumptions (cluster shape, density, scale); poor fit yields misleading groupings.
- Some estimators do not support out-of-sample prediction; the API raises explicit errors in those cases.
- Density-clustering metrics can look good on non-noise points while excluding many samples as noise—always report noise rates.
- Reproducibility depends on random seeds, library versions, and environment; persisted artifacts are not guaranteed portable across major dependency upgrades.

## Ethical considerations
- Segmentation/anomaly labels can influence access, pricing, or scrutiny; monitor for disparate impact and feedback loops.
- Communicate uncertainty: clusters are mathematical groupings, not inherent truths about individuals.
- When clustering user feedback, avoid exposing sensitive content and respect data-retention/privacy policies.

## Operational notes
- Optional heavy/NLP dependencies are imported lazily; install required extras (`[heavy]`, `[nlp]`, `[ui]`).
- Use the CLI/pipeline persistence for repeatable runs; record parameters, data versions, and package versions for auditability.
