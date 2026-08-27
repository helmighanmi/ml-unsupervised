<!--
Path: docs/development.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Development

## Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install optional surfaces only when working on them:

```bash
python -m pip install -e ".[ui]"
python -m pip install -e ".[nlp]"
python -m pip install -e ".[notebooks]"
```

## Local checks

```bash
ruff check .
ruff format --check .
mypy src/ml_unsupervised
pytest
```

## Adding an algorithm

1. Add estimator construction to the appropriate engine/reducer.
2. Document whether it supports out-of-sample inference.
3. Add deterministic unit coverage when the estimator supports a random state.
4. Add or update a notebook only as an analysis/demo consumer.
5. Keep optional heavy imports lazy when possible.
