<!--
Path: CHANGELOG.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Changelog

## 1.0.0

- Replaced the legacy `src` import namespace with the installable `ml_unsupervised` package.
- Added class-based clustering, dimensionality-reduction, anomaly-detection and pipeline APIs.
- Moved SentenceTransformer feedback logic from the notebook into `TextFeedbackClusterer`.
- Made Streamlit a thin presentation layer over package APIs.
- Added a notebook-free CLI for clustering, anomaly detection and dimensionality reduction.
- Added pipeline persistence for inference-capable workflows.
- Improved input validation and explicit algorithm-capability errors.
- Made clustering evaluation noise-aware for DBSCAN/HDBSCAN outputs.
- Normalized production anomaly labels to `0=inlier`, `1=outlier` while preserving legacy helper functions.
- Added deterministic sample customer data and a package-first quickstart notebook.
- Removed the duplicate configuration-demo notebook and the unrelated sample spreadsheet test artifact.
- Added CI, Docker, security scanning, Dependabot, ADRs and development/runbook documentation.
