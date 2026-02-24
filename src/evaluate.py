"""Comprehensive model evaluation for fraud detection.

Evaluates the trained XGBoost model on the held-out production test set
(month 7) with fraud-specific metrics, threshold analysis, and fairness
metrics across protected groups.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str | Path = "configs/fraud_config.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    booster: xgb.Booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    loss_strategy: str = "scale_pos_weight",
) -> dict:
    """Run full evaluation suite on the production test set.

    Args:
        booster: Trained XGBoost Booster.
        X_test: Preprocessed test features.
        y_test: Binary test labels.
        feature_names: Feature names for the DMatrix.
        loss_strategy: ``"scale_pos_weight"`` or ``"focal"`` (affects sigmoid).

    Returns:
        Dict with metrics, threshold analysis, confusion matrix, and report.
    """
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    y_score = booster.predict(dtest)
    if loss_strategy == "focal":
        y_score = 1.0 / (1.0 + np.exp(-y_score))

    roc_auc = roc_auc_score(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    fpr, tpr, roc_thresholds = roc_curve(y_test, y_score)
    idx_1pct = np.searchsorted(fpr, 0.01, side="right") - 1
    recall_at_1pct = float(tpr[max(idx_1pct, 0)])

    idx_5pct = np.searchsorted(fpr, 0.05, side="right") - 1
    recall_at_5pct = float(tpr[max(idx_5pct, 0)])

    # Optimal threshold by F1
    prec_arr, rec_arr, pr_thresholds = precision_recall_curve(y_test, y_score)
    f1_scores = np.where(
        (prec_arr[:-1] + rec_arr[:-1]) > 0,
        2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1]),
        0.0,
    )
    best_idx = np.argmax(f1_scores)
    best_threshold = float(pr_thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    y_pred = (y_score >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "metrics": {
            "roc_auc": round(roc_auc, 5),
            "pr_auc": round(pr_auc, 5),
            "recall_at_1pct_fpr": round(recall_at_1pct, 5),
            "recall_at_5pct_fpr": round(recall_at_5pct, 5),
            "best_f1": round(best_f1, 5),
            "optimal_threshold": round(best_threshold, 5),
        },
        "confusion_matrix": cm,
        "classification_report": report,
        "y_score": y_score,
    }


# ---------------------------------------------------------------------------
# Fairness audit
# ---------------------------------------------------------------------------

def fairness_audit(
    y_true: np.ndarray,
    y_score: np.ndarray,
    protected_df: pd.DataFrame,
    threshold: float,
) -> dict:
    """Compute group-level metrics across protected attributes.

    For each protected attribute group, computes:
      - Positive prediction rate (selection rate).
      - True positive rate (recall).
      - False positive rate.

    Args:
        y_true: Binary ground-truth.
        y_score: Model scores.
        protected_df: DataFrame with protected attribute columns.
        threshold: Decision threshold for binarising scores.

    Returns:
        Nested dict: ``{attribute: {group: {metric: value}}}``.
    """
    y_pred = (y_score >= threshold).astype(int)
    results: dict[str, dict] = {}

    for col in protected_df.columns:
        col_results: dict[str, dict] = {}
        for group in protected_df[col].dropna().unique():
            mask = (protected_df[col] == group).values
            if mask.sum() < 10:
                continue
            yt, yp, ys = y_true[mask], y_pred[mask], y_score[mask]
            tp = ((yt == 1) & (yp == 1)).sum()
            fp = ((yt == 0) & (yp == 1)).sum()
            fn = ((yt == 1) & (yp == 0)).sum()
            tn = ((yt == 0) & (yp == 0)).sum()
            col_results[str(group)] = {
                "n_samples": int(mask.sum()),
                "selection_rate": round(yp.mean(), 5),
                "tpr": round(tp / max(tp + fn, 1), 5),
                "fpr": round(fp / max(fp + tn, 1), 5),
                "roc_auc": round(roc_auc_score(yt, ys), 5) if yt.sum() > 0 else None,
            }
        results[col] = col_results

    return results


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save_evaluation_plots(
    y_true: np.ndarray,
    y_score: np.ndarray,
    report_dir: Path,
) -> None:
    """Generate and save PR curve, ROC curve, and score distribution plots."""
    report_dir.mkdir(parents=True, exist_ok=True)

    # PR curve
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(rec, prec, where="post", linewidth=1.5, color="#2563eb")
    ax.fill_between(rec, prec, step="post", alpha=0.15, color="#2563eb")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Production Test (Month 7)")
    ax.grid(alpha=0.3)
    fig.savefig(report_dir / "pr_curve_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ROC curve with 1 % FPR line
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=1.5, color="#2563eb")
    ax.axvline(0.01, color="red", linestyle="--", alpha=0.7, label="1% FPR")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — Production Test (Month 7)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(report_dir / "roc_curve_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Score distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_score[y_true == 0], bins=100, alpha=0.6, label="Legit", color="#64748b")
    ax.hist(y_score[y_true == 1], bins=100, alpha=0.6, label="Fraud", color="#dc2626")
    ax.set_xlabel("Predicted Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()
    ax.set_yscale("log")
    fig.savefig(report_dir / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Evaluation plots saved to %s", report_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    from src.data_engineering import load_and_collect, temporal_split
    from src.feature_pipeline import prepare_splits

    parser = argparse.ArgumentParser(description="Evaluate fraud model on month-7 test set")
    parser.add_argument("--model-path", default="artifacts/model.xgb")
    parser.add_argument("--data-path", default="data/Variant II.csv")
    parser.add_argument("--config", default="configs/fraud_config.yaml")
    parser.add_argument("--report-dir", default="artifacts/reports")
    args = parser.parse_args()

    config = load_config(args.config)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    booster = xgb.Booster()
    booster.load_model(args.model_path)

    df = load_and_collect(args.data_path)
    train_df, val_df, test_df = temporal_split(df)
    splits = prepare_splits(
        train_df, val_df, test_df,
        fairness_mode=config["fairness"]["mode"],
        artifact_dir="artifacts",
    )

    result = evaluate_model(
        booster,
        splits["X_test"],
        splits["y_test"],
        splits["feature_names"],
        loss_strategy=config["model"]["loss_strategy"],
    )

    threshold = result["metrics"]["optimal_threshold"]
    fairness = fairness_audit(
        splits["y_test"],
        result["y_score"],
        splits["protected_test"],
        threshold=threshold,
    )
    result["fairness_audit"] = fairness

    save_evaluation_plots(splits["y_test"], result["y_score"], report_dir)

    serialisable = {k: v for k, v in result.items() if k != "y_score"}
    (report_dir / "evaluation.json").write_text(json.dumps(serialisable, indent=2, default=str))

    thresholds = config["evaluation"]["thresholds"]
    gate_passed = (
        result["metrics"]["recall_at_1pct_fpr"] >= thresholds["recall_at_1pct_fpr"]
        and result["metrics"]["pr_auc"] >= thresholds["pr_auc"]
    )

    logger.info("Metrics: %s", result["metrics"])
    logger.info("Quality gate: %s", "PASSED" if gate_passed else "FAILED")

    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
