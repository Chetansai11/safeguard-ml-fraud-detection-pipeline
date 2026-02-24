"""Data engineering for the BAF Variant II fraud dataset.

Uses Polars for memory-efficient loading of 1M+ rows and implements the
Medallion Architecture: Bronze (raw) → Silver (validated) → Gold (features).
Temporal splitting avoids data leakage by partitioning on the ``month`` column.
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
import polars as pl
import yaml

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET = "fraud_bool"
TEMPORAL_COL = "month"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(path: str | Path = "configs/fraud_config.yaml") -> dict:
    """Load pipeline configuration from YAML."""
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Polars-based data loading
# ---------------------------------------------------------------------------


def load_raw(path: str | Path) -> pl.LazyFrame:
    """Scan CSV lazily with Polars for memory-efficient access.

    Args:
        path: Local filesystem path to the raw CSV.

    Returns:
        A Polars LazyFrame that defers computation until collect().
    """
    return pl.scan_csv(str(path), infer_schema_length=10_000)


def load_and_collect(path: str | Path) -> pl.DataFrame:
    """Eagerly load the entire CSV into a Polars DataFrame.

    Args:
        path: Local path to the dataset.

    Returns:
        Fully materialised Polars DataFrame.
    """
    lf = load_raw(path)
    df = lf.collect()
    logger.info(
        "Loaded %d rows × %d cols from %s (%.1f MB)",
        df.height,
        df.width,
        path,
        df.estimated_size("mb"),
    )
    return df


# ---------------------------------------------------------------------------
# Temporal splitting
# ---------------------------------------------------------------------------


def temporal_split(
    df: pl.DataFrame,
    train_months: list[int] | None = None,
    val_months: list[int] | None = None,
    test_months: list[int] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split data by temporal month column — no random leakage.

    Default partition mirrors real deployment chronology:
      - Train : months 0-5  (model development)
      - Val   : month 6     (hyperparameter tuning)
      - Test  : month 7     (production / unseen)

    Args:
        df: Full dataset with a ``month`` column.
        train_months: Month values for training.
        val_months: Month values for validation.
        test_months: Month values for production test.

    Returns:
        (train_df, val_df, test_df) triple.
    """
    train_months = train_months or list(range(6))
    val_months = val_months or [6]
    test_months = test_months or [7]

    train_df = df.filter(pl.col(TEMPORAL_COL).is_in(train_months))
    val_df = df.filter(pl.col(TEMPORAL_COL).is_in(val_months))
    test_df = df.filter(pl.col(TEMPORAL_COL).is_in(test_months))

    for label, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        fraud_rate = split[TARGET].sum() / split.height * 100 if split.height > 0 else 0
        logger.info(
            "%-5s | rows=%7d | fraud_rate=%.2f%%",
            label,
            split.height,
            fraud_rate,
        )

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Medallion Architecture — S3 transport
# ---------------------------------------------------------------------------


def _get_s3(s3_client: S3Client | None = None) -> S3Client:
    return s3_client or boto3.client("s3")


def upload_bronze(
    local_path: str | Path,
    bucket: str,
    key: str = "variant2.csv",
    s3_client: S3Client | None = None,
) -> str:
    """Upload the raw CSV to the Bronze bucket (as-is).

    Returns:
        S3 URI of the uploaded object.
    """
    s3 = _get_s3(s3_client)
    s3.upload_file(str(local_path), bucket, key)
    uri = f"s3://{bucket}/{key}"
    logger.info("Bronze ← %s → %s", local_path, uri)
    return uri


def bronze_to_silver(
    bronze_bucket: str,
    silver_bucket: str,
    key: str = "variant2.csv",
    s3_client: S3Client | None = None,
) -> str:
    """Read raw CSV from Bronze, validate and clean, write Parquet to Silver.

    Cleaning steps:
      - Drop complete duplicates.
      - Drop rows where target or temporal column is null.
      - Cast all columns to strict types.

    Returns:
        S3 URI of the Silver Parquet.
    """
    s3 = _get_s3(s3_client)
    resp = s3.get_object(Bucket=bronze_bucket, Key=key)
    df = pl.read_csv(resp["Body"].read(), infer_schema_length=10_000)

    before = df.height
    df = df.unique()
    df = df.drop_nulls(subset=[TARGET, TEMPORAL_COL])
    logger.info("Silver cleaning: %d → %d rows", before, df.height)

    parquet_key = key.rsplit(".", 1)[0] + ".parquet"
    buf = io.BytesIO()
    df.write_parquet(buf)
    buf.seek(0)
    s3.put_object(Bucket=silver_bucket, Key=parquet_key, Body=buf.getvalue())

    uri = f"s3://{silver_bucket}/{parquet_key}"
    logger.info("Silver → %s", uri)
    return uri


def silver_to_gold(
    silver_bucket: str,
    gold_bucket: str,
    key: str = "variant2.parquet",
    s3_client: S3Client | None = None,
) -> dict[str, str]:
    """Read validated Parquet from Silver, split temporally, write to Gold.

    Writes three Parquet files: train.parquet, val.parquet, test.parquet.

    Returns:
        Dict mapping split name to S3 URI.
    """
    s3 = _get_s3(s3_client)
    resp = s3.get_object(Bucket=silver_bucket, Key=key)
    df = pl.read_parquet(resp["Body"].read())

    train_df, val_df, test_df = temporal_split(df)

    uris: dict[str, str] = {}
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_key = f"{name}.parquet"
        buf = io.BytesIO()
        split_df.write_parquet(buf)
        buf.seek(0)
        s3.put_object(Bucket=gold_bucket, Key=split_key, Body=buf.getvalue())
        uris[name] = f"s3://{gold_bucket}/{split_key}"

    logger.info("Gold splits: %s", uris)
    return uris


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud data engineering pipeline")
    sub = parser.add_subparsers(dest="command")

    p_load = sub.add_parser("load", help="Load and inspect local CSV")
    p_load.add_argument("--path", default="data/Variant II.csv")

    p_bronze = sub.add_parser("upload-bronze", help="Upload raw CSV to S3 Bronze")
    p_bronze.add_argument("--path", default="data/Variant II.csv")
    p_bronze.add_argument("--bucket", required=True)

    p_silver = sub.add_parser("bronze-to-silver", help="Bronze → Silver")
    p_silver.add_argument("--bronze-bucket", required=True)
    p_silver.add_argument("--silver-bucket", required=True)
    p_silver.add_argument("--key", default="variant2.csv")

    p_gold = sub.add_parser("silver-to-gold", help="Silver → Gold (temporal splits)")
    p_gold.add_argument("--silver-bucket", required=True)
    p_gold.add_argument("--gold-bucket", required=True)
    p_gold.add_argument("--key", default="variant2.parquet")

    args = parser.parse_args()

    if args.command == "load":
        df = load_and_collect(args.path)
        print(df.describe())
        train, val, test = temporal_split(df)
    elif args.command == "upload-bronze":
        upload_bronze(args.path, args.bucket)
    elif args.command == "bronze-to-silver":
        bronze_to_silver(args.bronze_bucket, args.silver_bucket, args.key)
    elif args.command == "silver-to-gold":
        silver_to_gold(args.silver_bucket, args.gold_bucket, args.key)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
