<!--
Path: docs/benchmarks.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-helmi
-->

# Benchmarks

This project includes a reproducible benchmark script to compare clustering methods on synthetic and sample data.

## Run

```bash
python benchmarks/run_benchmarks.py
# or
make benchmark
```

Optionally write outputs:

```bash
python benchmarks/run_benchmarks.py --output outputs/benchmark.md --json outputs/benchmark.json
```

## What is measured

- **Internal metrics:** silhouette (higher better), Davies–Bouldin (lower better), Calinski–Harabasz (higher better)
- **Structure:** number of clusters found, noise ratio (for density methods)
- **Stability:** subsample ARI agreement for selected methods (higher/more consistent is better)
- **Runtime:** wall-clock seconds for fit+evaluate on the benchmark dataset

## Interpretation guidelines

- Prefer methods that are **good enough and stable**, not just best on one metric.
- Compare on **preprocessed data** representative of your real workload.
- Density methods may report strong metrics on non-noise points while leaving many samples as noise—always inspect noise ratio.
- Treat results as **evidence**, not proof; validate with domain knowledge and downstream outcomes.

## Caveats

- Results depend on dataset shape, scaling, parameter choices, random seeds, and library versions.
- Synthetic benchmarks demonstrate behavior under controlled assumptions and may not reflect your production data.
- For rigorous evaluation, extend the script with your own datasets and fixed experimental protocol.
