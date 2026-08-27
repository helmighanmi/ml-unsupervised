<!--
Path: docs/decisions/002-python-311-baseline.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# ADR-002: Python 3.11 reference runtime

## Status

Accepted.

## Decision

Use Python 3.11 as the documented runtime and validate Python 3.11/3.12 in CI.

## Rationale

The current scientific stack has broad support on Python 3.11 while the latest scikit-learn line requires Python 3.11 or newer. The choice balances modern language/runtime support with binary-wheel availability for the scientific dependencies used here.
