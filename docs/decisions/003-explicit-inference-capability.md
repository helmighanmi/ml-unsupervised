<!--
Path: docs/decisions/003-explicit-inference-capability.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# ADR-003: Do not pretend every clustering method supports inference

## Status

Accepted.

## Context

A generic clustering API can easily imply that every fitted estimator can label future samples. This is false for common algorithms including AgglomerativeClustering and standard DBSCAN.

## Decision

Expose `predict()` only through runtime capability checks. If the chosen estimator has no valid prediction operation, raise an explicit domain error and direct callers to `fit_predict()`.

## Consequences

The API is less superficially uniform but more correct and safer for production use.
