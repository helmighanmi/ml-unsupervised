# Path: src/ml_unsupervised/interfaces/cli.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Command-line interface for notebook-free execution."""

from __future__ import annotations

import argparse
import json
import logging
from importlib.resources import as_file, files
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..algorithms.anomaly import AnomalyDetector
from ..data.loading import load_tabular_dataset
from ..evaluation.diagnostics import anomaly_summary, cluster_profile
from ..algorithms.reduction import DimensionalityReducer
from ..core.exceptions import MLUnsupervisedError
from ..workflows.pipeline import ClusteringPipeline
from ..reporting.reports import write_report


def coerce_scalar(value: str):
    """Convert a CLI key=value value to bool/int/float/None/string."""
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_key_value_params(items: list[str] | None) -> dict[str, object]:
    """Parse repeated CLI parameters formatted as ``key=value``."""
    params: dict[str, object] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid parameter {item!r}; expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid parameter {item!r}; key cannot be empty.")
        params[key] = coerce_scalar(value)
    return params


LOGGER = logging.getLogger(__name__)


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="Input CSV or XLSX file.")
    parser.add_argument(
        "--features", nargs="+", help="Explicit feature columns. Defaults to all numeric columns."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-unsupervised",
        description="Run unsupervised ML workflows without opening a notebook.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable informational logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cluster = subparsers.add_parser(
        "cluster", help="Scale, optionally reduce, and cluster a tabular dataset."
    )
    _add_data_args(cluster)
    cluster.add_argument("--method", default="kmeans")
    cluster.add_argument("--scaler", default="robust", choices=["robust", "standard", "minmax"])
    cluster.add_argument("--reduce", dest="reduction_method", default=None)
    cluster.add_argument("--components", type=int, default=2)
    cluster.add_argument(
        "--param", action="append", help="Clustering parameter as key=value; may be repeated."
    )
    cluster.add_argument(
        "--model-out", type=Path, help="Optional joblib path for a fitted inference-capable pipeline."
    )
    cluster.add_argument(
        "--profile-out", type=Path, help="Optional CSV path for per-cluster feature profile."
    )

    anomaly = subparsers.add_parser("anomaly", help="Run anomaly detection on a tabular dataset.")
    _add_data_args(anomaly)
    anomaly.add_argument("--method", default="isolation_forest")
    anomaly.add_argument("--param", action="append", help="Detector parameter as key=value; may be repeated.")
    anomaly.add_argument(
        "--summary-out", type=Path, help="Optional JSON path for anomaly summary diagnostics."
    )

    reduce = subparsers.add_parser("reduce", help="Generate a lower-dimensional embedding.")
    _add_data_args(reduce)
    reduce.add_argument("--method", default="pca")
    reduce.add_argument("--components", type=int, default=2)
    reduce.add_argument("--param", action="append", help="Reducer parameter as key=value; may be repeated.")

    demo = subparsers.add_parser("demo", help="Run a quick end-to-end demo on the bundled sample dataset.")
    demo.add_argument("--output", type=Path, help="Optional CSV path for labeled demo output.")

    report = subparsers.add_parser(
        "report", help="Generate a Markdown/HTML report from summary/profile artifacts."
    )
    report.add_argument("--summary", type=Path, required=True, help="JSON summary file.")
    report.add_argument("--profile", type=Path, help="Optional CSV profile/diagnostics table.")
    report.add_argument("--title", default="Unsupervised analysis report", help="Report title.")
    report.add_argument("--output", type=Path, required=True, help="Output report path (.md or .html).")

    serve = subparsers.add_parser("serve", help="Run the FastAPI service (requires 'api' extra).")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _write_output(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    LOGGER.info("Wrote %s rows to %s", len(frame), output)


def _run_cluster(args: argparse.Namespace) -> dict[str, object]:
    dataset = load_tabular_dataset(args.input, feature_columns=args.features)
    params = parse_key_value_params(args.param)
    pipeline = ClusteringPipeline(
        scaler=args.scaler,
        clustering_method=args.method,
        clustering_params=params,
        reduction_method=args.reduction_method,
        n_components=args.components,
    )
    result = pipeline.fit_predict(dataset.X)
    if args.model_out:
        # Check both the reducer and clusterer before writing any outputs.
        pipeline.predict(dataset.X[:1])
    output_frame = dataset.frame.copy()
    output_frame["cluster"] = result.labels
    _write_output(output_frame, args.output)
    profile_path = None
    if args.profile_out:
        profile = cluster_profile(dataset.feature_frame, result.labels)
        args.profile_out.parent.mkdir(parents=True, exist_ok=True)
        profile.to_csv(args.profile_out, index=False)
        profile_path = str(args.profile_out)
    if args.model_out:
        pipeline.save(args.model_out)
    return {
        "rows": len(output_frame),
        "clusters": len(set(result.labels.tolist()) - {-1}),
        "metrics": result.metrics.to_dict() if result.metrics else None,
        "output": str(args.output),
        "model": str(args.model_out) if args.model_out else None,
        "profile_output": profile_path,
    }


def _run_anomaly(args: argparse.Namespace) -> dict[str, object]:
    dataset = load_tabular_dataset(args.input, feature_columns=args.features)
    detector = AnomalyDetector(args.method, parse_key_value_params(args.param))
    result = detector.fit_predict(dataset.X)
    output_frame = dataset.frame.copy()
    output_frame["is_outlier"] = result.labels
    if result.scores is not None:
        output_frame["anomaly_score"] = result.scores
    _write_output(output_frame, args.output)
    summary_path = None
    if args.summary_out:
        summary = anomaly_summary(dataset.feature_frame, result.labels)
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        summary_path = str(args.summary_out)
    return {
        "rows": len(output_frame),
        "outlier_ratio": result.outlier_ratio,
        "output": str(args.output),
        "summary_output": summary_path,
    }


def _run_reduce(args: argparse.Namespace) -> dict[str, object]:
    dataset = load_tabular_dataset(args.input, feature_columns=args.features)
    reducer = DimensionalityReducer(args.method, args.components, parse_key_value_params(args.param))
    embedding = reducer.fit_transform(dataset.X)
    output_frame = dataset.frame.copy()
    for index in range(embedding.shape[1]):
        output_frame[f"component_{index + 1}"] = embedding[:, index]
    _write_output(output_frame, args.output)
    return {"rows": len(output_frame), "components": embedding.shape[1], "output": str(args.output)}


def _run_demo(args: argparse.Namespace) -> dict[str, object]:
    """Run a fast, reproducible clustering demo on bundled sample data."""
    with as_file(files("ml_unsupervised.data").joinpath("sample_customers.csv")) as sample:
        dataset = load_tabular_dataset(sample, feature_columns=["age", "annual_income", "spending_score"])
    pipeline = ClusteringPipeline(
        scaler="robust",
        reduction_method="pca",
        n_components=2,
        clustering_method="kmeans",
        clustering_params={"n_clusters": 3, "random_state": 42, "n_init": 10},
    )
    result = pipeline.fit_predict(dataset.X)
    summary: dict[str, object] = {
        "demo": "customer clustering",
        "rows": len(dataset.frame),
        "clusters": int(len(set(result.labels.tolist()))),
        "metrics": result.metrics.to_dict() if result.metrics else None,
        "features": dataset.feature_frame.columns.tolist(),
    }
    if args.output:
        output_frame = dataset.frame.copy()
        output_frame["cluster"] = result.labels
        for index in range(result.transformed.shape[1]):
            output_frame[f"component_{index + 1}"] = result.transformed[:, index]
        _write_output(output_frame, args.output)
        summary["output"] = str(args.output)
    return summary


def _run_report(args: argparse.Namespace) -> dict[str, object]:
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("Summary JSON must contain an object.")
    profile = pd.read_csv(args.profile) if args.profile else None
    notes = [
        "Unsupervised metrics are diagnostics, not ground truth; validate with domain context.",
        "Results depend on preprocessing, parameters, data quality, and library versions.",
    ]
    out = write_report(args.output, title=args.title, summary=summary, profile=profile, notes=notes)
    return {"report": str(out), "format": out.suffix.lstrip(".") or "md"}


def _run_serve(args: argparse.Namespace) -> dict[str, object]:
    try:
        import uvicorn

        from .api import create_app
    except ImportError as exc:
        raise MLUnsupervisedError("Serve command requires the 'api' extra: pip install -e '.[api]'") from exc
    app = create_app()
    # Run blocking server; CLI returns when server stops.
    uvicorn.run(app, host=args.host, port=args.port)
    return {"status": "stopped"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    try:
        if args.command == "cluster":
            summary = _run_cluster(args)
        elif args.command == "anomaly":
            summary = _run_anomaly(args)
        elif args.command == "reduce":
            summary = _run_reduce(args)
        elif args.command == "demo":
            summary = _run_demo(args)
        elif args.command == "report":
            summary = _run_report(args)
        else:
            summary = _run_serve(args)
    except (FileNotFoundError, MLUnsupervisedError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
