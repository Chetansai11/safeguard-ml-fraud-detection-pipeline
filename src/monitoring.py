"""Distribution-shift monitoring with Evidently AI.

Compares feature distributions between the training period (months 0-5) and
the production test period (month 7) to detect data drift that may degrade
model performance.  Generates both an HTML report (for human review) and a
JSON summary (for CI/CD gating).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import polars as pl
import yaml
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.legacy.report import Report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET = "fraud_bool"
TEMPORAL_COL = "month"


def load_config(path: str | Path = "configs/fraud_config.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_column_mapping(config: dict) -> ColumnMapping:
    """Construct Evidently ColumnMapping from the project config."""
    features = config["features"]
    return ColumnMapping(
        target=TARGET,
        numerical_features=features["numerical"] + features["protected"],
        categorical_features=features["categorical"] + features["binary"],
    )


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    config: dict,
    report_dir: str | Path = "artifacts/reports",
) -> dict:
    """Generate an Evidently drift report comparing two temporal windows.

    Args:
        reference_df: Training-period data (months 0-5).
        current_df: Production-period data (month 7).
        config: Project configuration dict.
        report_dir: Output directory for HTML and JSON reports.

    Returns:
        Drift summary dict with per-column and dataset-level drift flags.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    column_mapping = build_column_mapping(config)

    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    report.save_html(str(report_dir / "drift_report.html"))
    report.save_json(str(report_dir / "drift_report.json"))

    report_dict = report.as_dict()

    drift_result = report_dict["metrics"][0]["result"]
    dataset_drift = drift_result.get("dataset_drift", False)
    drift_share = drift_result.get("drift_share", 0.0)
    n_drifted = drift_result.get("number_of_drifted_columns", 0)
    n_columns = drift_result.get("number_of_columns", 0)

    drifted_columns = []
    for col_name, col_data in drift_result.get("drift_by_columns", {}).items():
        if col_data.get("drift_detected", False):
            drifted_columns.append(
                {
                    "column": col_name,
                    "stattest": col_data.get("stattest_name", ""),
                    "p_value": col_data.get("p_value"),
                    "drift_score": col_data.get("drift_score"),
                }
            )

    summary = {
        "dataset_drift_detected": dataset_drift,
        "drift_share": round(drift_share, 4),
        "drifted_columns": n_drifted,
        "total_columns": n_columns,
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
        "drifted_column_details": drifted_columns,
    }

    (report_dir / "drift_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info(
        "Drift: %s | %.0f%% features drifted (%d/%d) | %d drifted columns",
        "DETECTED" if dataset_drift else "NOT detected",
        drift_share * 100,
        n_drifted,
        n_columns,
        len(drifted_columns),
    )

    return summary


def generate_markdown_summary(summary: dict) -> str:
    """Render the drift summary as Markdown (suitable for CML PR comments)."""
    lines = [
        "## Evidently AI — Distribution Shift Report",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Dataset Drift Detected | **{'YES' if summary['dataset_drift_detected'] else 'NO'}** |",
        f"| Drifted Features | {summary['drifted_columns']}/{summary['total_columns']} ({summary['drift_share']:.1%}) |",
        f"| Reference Samples (train) | {summary['reference_rows']:,} |",
        f"| Current Samples (month 7) | {summary['current_rows']:,} |",
        "",
    ]
    if summary["drifted_column_details"]:
        lines.append("### Drifted Columns")
        lines.append("")
        lines.append("| Column | Test | p-value |")
        lines.append("|--------|------|---------|")
        for col in summary["drifted_column_details"]:
            pval = f"{col['p_value']:.4f}" if col["p_value"] is not None else "N/A"
            lines.append(f"| {col['column']} | {col['stattest']} | {pval} |")
        lines.append("")

    if summary["dataset_drift_detected"]:
        lines.append(
            "> **Warning**: Significant distribution shift detected. "
            "Investigate before relying on month-7 predictions."
        )
    else:
        lines.append(
            "> **OK**: No significant distribution shift between training and production data."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Evidently drift report")
    parser.add_argument("--data-path", default="data/Variant II.csv")
    parser.add_argument("--config", default="configs/fraud_config.yaml")
    parser.add_argument("--report-dir", default="artifacts/reports")
    args = parser.parse_args()

    config = load_config(args.config)

    path = Path(args.data_path)
    if path.suffix == ".parquet":
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(str(path), infer_schema_length=10_000)

    train_months = config["data"]["train_months"]
    test_months = config["data"]["test_months"]

    ref_df = df.filter(pl.col(TEMPORAL_COL).is_in(train_months)).to_pandas()
    cur_df = df.filter(pl.col(TEMPORAL_COL).is_in(test_months)).to_pandas()

    logger.info("Reference (months %s): %d rows", train_months, len(ref_df))
    logger.info("Current  (months %s): %d rows", test_months, len(cur_df))

    summary = generate_drift_report(ref_df, cur_df, config, report_dir=args.report_dir)

    md = generate_markdown_summary(summary)
    md_path = Path(args.report_dir) / "drift_summary.md"
    md_path.write_text(md)
    logger.info("Markdown summary → %s", md_path)


if __name__ == "__main__":
    main()
