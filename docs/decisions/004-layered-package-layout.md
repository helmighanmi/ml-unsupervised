<!--
Path: docs/decisions/004-layered-package-layout.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-helmi
-->

# ADR 004: Layered package layout with compatibility shims

## Status
Accepted

## Context
The toolkit grew from a small flat package into a broader system (algorithms, workflows, evaluation/diagnostics, reporting, infrastructure, interfaces). A flat module layout made responsibilities and dependency direction implicit, increasing the risk of boundary erosion.

## Decision
Adopt an incremental migration to an 8-subpackage layered layout under `src/ml_unsupervised/`:

- `core` — exceptions and shared validation
- `data` — loading and preprocessing
- `algorithms` — estimator/method implementations
- `workflows` — composed pipelines
- `evaluation` — metrics and diagnostics
- `reporting` — plots and report generation
- `infrastructure` — configuration and registry
- `interfaces` — CLI and API adapters

Dependency direction is constrained by layer (see `docs/architecture.md`) and guarded by `tests/test_architecture.py`. To preserve backward compatibility, legacy module paths remain as re-export shims during the migration period.

## Consequences
- **Positive:** clearer ownership, safer refactoring, easier onboarding, enforceable boundaries.
- **Positive:** existing imports continue to work via shims (migration path A).
- **Negative:** temporary duplication/shim maintenance and indirection.
- **Follow-up:** remove deprecated shims in a future major release and update consumers to canonical paths.
