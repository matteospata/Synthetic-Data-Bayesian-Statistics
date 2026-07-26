from __future__ import annotations

from pathlib import Path
from typing import Any


def _spark_imports():
    try:
        from pyspark.sql import DataFrame, SparkSession, functions as F
        from pyspark.sql.types import (
            ByteType,
            DecimalType,
            DoubleType,
            FloatType,
            IntegerType,
            LongType,
            ShortType,
        )
    except ImportError as exc:
        raise RuntimeError("PySpark is optional. Install it with: pip install -e '.[spark]'") from exc
    return DataFrame, SparkSession, F, (ByteType, DecimalType, DoubleType, FloatType, IntegerType, LongType, ShortType)


def create_spark_session(app_name: str = "SyntheticDataPlatform", master: str | None = None):
    """Create a small local Spark session suitable for development and CI."""
    _, SparkSession, _, _ = _spark_imports()
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.bindAddress", "127.0.0.1")
    )
    if master:
        builder = builder.master(master)
    elif not SparkSession.getActiveSession():
        builder = builder.master("local[2]")
    return builder.getOrCreate()


def read_csv(spark, path: str | Path):
    """Read a CSV through Spark with a header and inferred schema."""
    source = str(path)
    if not Path(source).exists():
        raise FileNotFoundError(source)
    return spark.read.option("header", True).option("inferSchema", True).option("mode", "FAILFAST").csv(source)


def _numeric_columns(frame, numeric_types: tuple) -> list[str]:
    return [field.name for field in frame.schema.fields if isinstance(field.dataType, numeric_types)]


def profile_dataframe(frame) -> dict[str, Any]:
    """Compute a distributed data-quality profile without converting rows to pandas."""
    _, _, F, numeric_types = _spark_imports()
    row_count = frame.count()
    duplicate_count = row_count - frame.dropDuplicates().count()
    missing_exprs = [F.sum(F.col(column).isNull().cast("long")).alias(column) for column in frame.columns]
    missing_row = frame.agg(*missing_exprs).first() if missing_exprs else None
    missing_values = {column: int(missing_row[column] or 0) for column in frame.columns} if missing_row else {}
    numeric_columns = _numeric_columns(frame, numeric_types)
    numeric_stats: dict[str, dict[str, float | None]] = {}
    if numeric_columns:
        aggregations = []
        for column in numeric_columns:
            aggregations.extend([F.mean(F.col(column)).alias(f"{column}__mean"), F.stddev_pop(F.col(column)).alias(f"{column}__std")])
        stats_row = frame.agg(*aggregations).first()
        for column in numeric_columns:
            numeric_stats[column] = {"mean": float(stats_row[f"{column}__mean"]) if stats_row[f"{column}__mean"] is not None else None, "std": float(stats_row[f"{column}__std"]) if stats_row[f"{column}__std"] is not None else None}
    issues = []
    for column, missing in missing_values.items():
        rate = missing / row_count if row_count else 0.0
        if rate > 0.05:
            issues.append({"column": column, "type": "missing_values", "rate": rate, "severity": "warning"})
    if duplicate_count:
        issues.append({"type": "duplicate_rows", "count": duplicate_count, "severity": "warning"})
    return {"engine": "pyspark", "rows": row_count, "columns": list(frame.columns), "schema": {field.name: field.dataType.simpleString() for field in frame.schema.fields}, "missing_values": missing_values, "duplicate_rows": duplicate_count, "numeric_stats": numeric_stats, "quality_issues": issues}


def profile_csv(path: str | Path, spark=None) -> dict[str, Any]:
    owns_session = spark is None
    spark = spark or create_spark_session()
    try:
        return profile_dataframe(read_csv(spark, path))
    finally:
        if owns_session:
            spark.stop()


def materialize_parquet(input_path: str | Path, output_path: str | Path, spark=None) -> dict[str, Any]:
    """Materialize a validated CSV as Parquet for downstream Spark jobs."""
    owns_session = spark is None
    spark = spark or create_spark_session()
    try:
        frame = read_csv(spark, input_path)
        destination = Path(output_path)
        frame.write.mode("overwrite").parquet(str(destination))
        return {"engine": "pyspark", "input": str(input_path), "output": str(destination), "rows": frame.count(), "format": "parquet"}
    finally:
        if owns_session:
            spark.stop()


def evaluate_csvs(real_path: str | Path, synthetic_path: str | Path, spark=None) -> dict[str, Any]:
    """Compare real and synthetic CSVs with Spark aggregations."""
    _, _, F, numeric_types = _spark_imports()
    owns_session = spark is None
    spark = spark or create_spark_session()
    try:
        real, synthetic = read_csv(spark, real_path), read_csv(spark, synthetic_path)
        if real.columns != synthetic.columns:
            raise ValueError("Real and synthetic tables must have the same columns in the same order.")
        numeric_columns = [column for column in real.columns if isinstance(real.schema[column].dataType, numeric_types) and isinstance(synthetic.schema[column].dataType, numeric_types)]
        numeric: dict[str, dict[str, float]] = {}
        scores: list[float] = []
        for column in numeric_columns:
            real_stats = real.agg(F.mean(column).alias("mean"), F.stddev_pop(column).alias("std")).first()
            synthetic_stats = synthetic.agg(F.mean(column).alias("mean"), F.stddev_pop(column).alias("std")).first()
            real_mean, synthetic_mean = float(real_stats["mean"]), float(synthetic_stats["mean"])
            real_std = float(real_stats["std"] or 0.0)
            synthetic_std = float(synthetic_stats["std"] or 0.0)
            scale = real_std or 1.0
            score = max(0.0, 1.0 - min(1.0, 0.5 * abs(real_mean - synthetic_mean) / scale + 0.5 * abs(real_std - synthetic_std) / scale))
            scores.append(score)
            numeric[column] = {"real_mean": real_mean, "synthetic_mean": synthetic_mean, "real_std": real_std, "synthetic_std": synthetic_std, "score": score}
        categorical: dict[str, dict[str, float]] = {}
        for column in [column for column in real.columns if column not in numeric_columns]:
            real_count = real.groupBy(column).count().withColumnRenamed("count", "real_count")
            synthetic_count = synthetic.groupBy(column).count().withColumnRenamed("count", "synthetic_count")
            total_real, total_synthetic = real.count(), synthetic.count()
            joined = real_count.join(synthetic_count, on=column, how="full").fillna(0)
            distance = joined.select(F.sum(F.abs(F.col("real_count") / total_real - F.col("synthetic_count") / total_synthetic)).alias("distance")).first()["distance"] or 0.0
            total_variation = float(distance) / 2.0
            score = max(0.0, 1.0 - min(1.0, total_variation))
            scores.append(score)
            categorical[column] = {"total_variation_distance": total_variation, "score": score}
        return {"engine": "pyspark", "rows": {"real": real.count(), "synthetic": synthetic.count()}, "numeric": numeric, "categorical": categorical, "quality_score": sum(scores) / len(scores) if scores else 0.0}
    finally:
        if owns_session:
            spark.stop()

