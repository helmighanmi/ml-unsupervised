# Path: src/ml_unsupervised/cli.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Command-line interface for notebook-free execution."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from .anomaly_detection import AnomalyDetector
from .data import load_tabular_dataset
from .dimensionality_reduction import DimensionalityReducer
from .pipelines import ClusteringPipeline
from .utils import parse_key_value_params

LOGGER = logging.getLogger(__name__)


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="Input CSV or XLSX file.")
    parser.add_argument("--features", nargs="+", help="Explicit feature columns. Defaults to all numeric columns.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-unsupervised",
        description="Run unsupervised ML workflows without opening a notebook.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable informational logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cluster = subparsers.add_parser("cluster", help="Scale, optionally reduce, and cluster a tabular dataset.")
    _add_data_args(cluster)
    cluster.add_argument("--method", default="kmeans")
    cluster.add_argument("--scaler", default="robust", choices=["robust", "standard", "minmax"])
    cluster.add_argument("--reduce", dest="reduction_method", default=None)
    cluster.add_argument("--components", type=int, default=2)
    cluster.add_argument("--param", action="append", help="Clustering parameter as key=value; may be repeated.")
    cluster.add_argument("--model-out", type=Path, help="Optional joblib path for a fitted inference-capable pipeline.")

    anomaly = subparsers.add_parser("anomaly", help="Run anomaly detection on a tabular dataset.")
    _add_data_args(anomaly)
    anomaly.add_argument("--method", default="isolation_forest")
    anomaly.add_argument("--param", action="append", help="Detector parameter as key=value; may be repeated.")

    reduce = subparsers.add_parser("reduce", help="Generate a lower-dimensional embedding.")
    _add_data_args(reduce)
    reduce.add_argument("--method", default="pca")
    reduce.add_argument("--components", type=int, default=2)
    reduce.add_argument("--param", action="append", help="Reducer parameter as key=value; may be repeated.")
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
    output_frame = dataset.frame.copy()
    output_frame["cluster"] = result.labels
    _write_output(output_frame, args.output)
    if args.model_out:
        pipeline.save(args.model_out)
    return {
        "rows": len(output_frame),
        "clusters": int(len(set(result.labels.tolist()))),
        "metrics": result.metrics.to_dict() if result.metrics else None,
        "output": str(args.output),
        "model": str(args.model_out) if args.model_out else None,
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
    return {"rows": len(output_frame), "outlier_ratio": result.outlier_ratio, "output": str(args.output)}


def _run_reduce(args: argparse.Namespace) -> dict[str, object]:
    dataset = load_tabular_dataset(args.input, feature_columns=args.features)
    reducer = DimensionalityReducer(args.method, args.components, parse_key_value_params(args.param))
    embedding = reducer.fit_transform(dataset.X)
    output_frame = dataset.frame.copy()
    for index in range(embedding.shape[1]):
        output_frame[f"component_{index + 1}"] = embedding[:, index]
    _write_output(output_frame, args.output)
    return {"rows": len(output_frame), "components": embedding.shape[1], "output": str(args.output)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        if args.command == "cluster":
            summary = _run_cluster(args)
        elif args.command == "anomaly":
            summary = _run_anomaly(args)
        else:
            summary = _run_reduce(args)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
