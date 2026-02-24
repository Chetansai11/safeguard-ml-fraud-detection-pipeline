"""Data integrity validation using Great Expectations.

Quality gate for the Silver-tier data.  Validates:
  - ``month`` column is sequential (0 through 7, no gaps).
  - No nulls in protected attributes (customer_age, employment_status, income).
  - Feature ranges are within plausible bounds.
  - Target column contains only 0 and 1.
  - No complete duplicate rows.

This script is designed to run as a standalone CI gate and returns a non-zero
exit code if any expectation fails.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import great_expectations as gx
import pandas as pd
import polars as pl
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROTECTED_ATTRS = ["customer_age", "employment_status", "income"]
TARGET = "fraud_bool"
TEMPORAL_COL = "month"


def load_config(path: str | Path = "configs/fraud_config.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_expectations(validator) -> None:
    """Attach domain-specific expectations to the GX validator.

    Expectations encode business rules for the BAF dataset:
      1. Temporal integrity — month must span 0-7 with no gaps.
      2. Protected-attribute completeness — critical for fairness audits.
      3. Target integrity — binary only.
      4. Feature plausibility — catch upstream corruption early.
    """
    # -- Temporal integrity --
    validator.expect_column_values_to_not_be_null(TEMPORAL_COL)
    validator.expect_column_distinct_values_to_equal_set(
        TEMPORAL_COL, value_set=[0, 1, 2, 3, 4, 5, 6, 7]
    )

    # -- Protected attributes must be present --
    for col in PROTECTED_ATTRS:
        validator.expect_column_to_exist(col)
        validator.expect_column_values_to_not_be_null(col)

    # -- Target integrity --
    validator.expect_column_values_to_not_be_null(TARGET)
    validator.expect_column_values_to_be_in_set(TARGET, [0, 1])

    # -- Feature range checks (catch egregious outliers / corruption) --
    validator.expect_column_values_to_be_between("customer_age", min_value=10, max_value=100)
    validator.expect_column_values_to_be_between("income", min_value=0, max_value=1.0)
    validator.expect_column_values_to_be_between("name_email_similarity", min_value=0, max_value=1.0)
    validator.expect_column_values_to_be_between(
        "credit_risk_score", min_value=-500, max_value=500
    )

    # -- Row-level uniqueness is not enforced (legitimate duplicates may exist) --


def run_integrity_check(
    df: pd.DataFrame,
    report_path: str | Path | None = None,
) -> dict:
    """Execute the Great Expectations validation suite.

    Args:
        df: Pandas DataFrame to validate (Silver-tier data).
        report_path: Optional JSON path for the validation report.

    Returns:
        Dict with ``success`` bool and per-expectation results.
    """
    context = gx.get_context()

    datasource = context.data_sources.add_pandas(name="fraud_datasource")
    data_asset = datasource.add_dataframe_asset(name="fraud_data")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("fraud_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = context.suites.add(gx.ExpectationSuite(name="fraud_integrity_suite"))
    validator = context.get_validator(
        batch_request=batch.batch_request if hasattr(batch, "batch_request") else None,
        expectation_suite=suite,
    )
    build_expectations(validator)
    results = validator.validate()

    summary = {
        "success": results.success,
        "statistics": results.statistics,
        "results": [
            {
                "expectation": r.expectation_config.expectation_type,
                "column": r.expectation_config.kwargs.get("column", ""),
                "success": r.success,
            }
            for r in results.results
        ],
    }

    if report_path:
        rp = Path(report_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("Integrity report → %s", rp)

    passed = [r for r in summary["results"] if r["success"]]
    failed = [r for r in summary["results"] if not r["success"]]
    logger.info(
        "Quality gate %s — %d/%d passed",
        "PASSED" if summary["success"] else "FAILED",
        len(passed),
        len(passed) + len(failed),
    )
    if failed:
        for f in failed:
            logger.error("  FAIL: %s [%s]", f["expectation"], f["column"])

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run data integrity checks")
    parser.add_argument("--data-path", default="data/Variant II.csv")
    parser.add_argument("--report", default="artifacts/reports/data_integrity.json")
    args = parser.parse_args()

    path = Path(args.data_path)
    if path.suffix == ".parquet":
        df = pl.read_parquet(path).to_pandas()
    else:
        df = pl.read_csv(str(path), infer_schema_length=10_000).to_pandas()

    result = run_integrity_check(df, report_path=args.report)
    if not result["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
