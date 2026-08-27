<!--
Path: docs/decisions/001-package-first-notebook-boundary.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# ADR-001: Package-first, notebook-second architecture

## Status

Accepted.

## Context

The original project mixed reusable modules with notebook-centric execution and duplicated anomaly/reduction logic inside the Streamlit app. That makes production integration, testing and reuse harder.

## Decision

All reusable behavior lives in `src/ml_unsupervised/`. Notebooks and Streamlit import those APIs and remain analysis/presentation layers.

## Consequences

- The project can run without Jupyter.
- Tests exercise the same code users execute.
- Streamlit cannot silently drift from the package implementation.
- Notebook cells become shorter and focus on interpretation rather than infrastructure.
