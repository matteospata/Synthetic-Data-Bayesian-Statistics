from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from synthetic_data_platform.bayesian import compare_posteriors
from synthetic_data_platform.evaluation import compare_tables
from synthetic_data_platform.pipeline import PipelineConfig, SyntheticDataPipeline
from synthetic_data_platform.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic tabular data and run Bayesian analysis.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Train a generator and create an evaluated synthetic dataset.")
    run.add_argument("--input", required=True)
    run.add_argument("--output", default="artifacts")
    run.add_argument("--model", choices=["vae", "gan"], default="vae")
    run.add_argument("--epochs", type=int, default=120)
    run.add_argument("--samples", type=int, default=500)
    run.add_argument("--use-spark", action="store_true", help="Run the distributed Spark profiling stage before model training.")
    profile = subparsers.add_parser("profile", help="Show a compact profile of a CSV dataset.")
    profile.add_argument("--input", required=True)
    evaluate = subparsers.add_parser("evaluate", help="Compare real and synthetic datasets.")
    evaluate.add_argument("--real", required=True)
    evaluate.add_argument("--synthetic", required=True)
    bayes = subparsers.add_parser("bayes", help="Compare Bayesian posterior summaries.")
    bayes.add_argument("--real", required=True)
    bayes.add_argument("--synthetic", required=True)
    spark_profile = subparsers.add_parser("spark-profile", help="Profile a CSV with PySpark aggregations.")
    spark_profile.add_argument("--input", required=True)
    spark_profile.add_argument("--output")
    spark_materialize = subparsers.add_parser("spark-materialize", help="Convert a CSV into Parquet with PySpark.")
    spark_materialize.add_argument("--input", required=True)
    spark_materialize.add_argument("--output", required=True)
    spark_evaluate = subparsers.add_parser("spark-evaluate", help="Compare real and synthetic CSVs with PySpark.")
    spark_evaluate.add_argument("--real", required=True)
    spark_evaluate.add_argument("--synthetic", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        result = SyntheticDataPipeline(PipelineConfig(model=args.model, epochs=args.epochs, samples=args.samples, use_spark=args.use_spark)).run(args.input, args.output)
        print(json.dumps({"run_id": result["run_id"], "artifact_dir": result["artifact_dir"], "quality_score": result["quality"]["quality_score"]}, indent=2))
    elif args.command == "profile":
        frame = pd.read_csv(args.input)
        print(json.dumps({"rows": len(frame), "columns": list(frame.columns), "missing_values": frame.isna().sum().to_dict(), "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()}}, indent=2, default=str))
    elif args.command in {"evaluate", "bayes"}:
        real, synthetic = pd.read_csv(args.real), pd.read_csv(args.synthetic)
        report = compare_tables(real, synthetic) if args.command == "evaluate" else compare_posteriors(real, synthetic)
        print(json.dumps(report, indent=2, default=str))
    else:
        from synthetic_data_platform import spark_engine
        try:
            if args.command == "spark-profile":
                report = spark_engine.profile_csv(args.input)
                if args.output:
                    write_json(report, args.output)
                print(json.dumps(report, indent=2, default=str))
            elif args.command == "spark-materialize":
                report = spark_engine.materialize_parquet(args.input, args.output)
                print(json.dumps(report, indent=2, default=str))
            else:
                report = spark_engine.evaluate_csvs(args.real, args.synthetic)
                print(json.dumps(report, indent=2, default=str))
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
