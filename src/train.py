"""XGBoost fraud classifier with MLflow experiment tracking.

Supports two imbalance-handling strategies:
  - **scale_pos_weight**: Standard reweighting (default, production-proven).
  - **focal loss**: Custom objective that down-weights easy negatives.

MLflow logs include precision-recall curves, Recall @ 1 % FPR (the
banking-industry standard), SHAP feature-importance plots, and the
serialised model artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import shap
import xgboost as xgb
import yaml
from sklearn.metrics import (
    average_precision_score,
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
# Focal loss — custom XGBoost objective
# ---------------------------------------------------------------------------


def focal_binary_objective(
    predt: np.ndarray,
    dtrain: xgb.DMatrix,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary focal loss objective for XGBoost.

    Down-weights well-classified (easy) examples so the model focuses on
    hard positives (missed fraud).  ``gamma`` controls how aggressively
    easy examples are suppressed; ``alpha`` balances pos/neg classes.

    Args:
        predt: Raw predictions (logits) from the current boosting round.
        dtrain: XGBoost DMatrix with labels.
        gamma: Focusing parameter (higher → more focus on hard examples).
        alpha: Class-balance weight for the positive class.

    Returns:
        (gradient, hessian) arrays of shape ``(n_samples,)``.
    """
    y = dtrain.get_label()
    p = 1.0 / (1.0 + np.exp(-predt))  # sigmoid
    p = np.clip(p, 1e-7, 1.0 - 1e-7)

    # alpha weights: alpha for positives, (1-alpha) for negatives
    alpha_t = y * alpha + (1.0 - y) * (1.0 - alpha)
    # p_t: model confidence for the true class
    p_t = y * p + (1.0 - y) * (1.0 - p)
    focal_weight = alpha_t * (1.0 - p_t) ** gamma

    # Standard log-loss gradient/hessian, then reweight
    grad = focal_weight * (p - y)
    hess = focal_weight * p * (1.0 - p)
    hess = np.maximum(hess, 1e-7)

    return grad, hess


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Fraud-oriented evaluation metrics.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted probabilities for the positive class.

    Returns:
        Dictionary with ROC-AUC, PR-AUC, Recall @ 1 % FPR, and F1.
    """
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx_1pct = np.searchsorted(fpr, 0.01, side="right") - 1
    recall_at_1pct_fpr = float(tpr[max(idx_1pct, 0)])

    best_f1 = 0.0
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    for prec, rec in zip(precisions, recalls):
        denom = prec + rec
        if denom > 0:
            best_f1 = max(best_f1, 2 * prec * rec / denom)

    return {
        "roc_auc": round(roc_auc, 5),
        "pr_auc": round(pr_auc, 5),
        "recall_at_1pct_fpr": round(recall_at_1pct_fpr, 5),
        "best_f1": round(best_f1, 5),
    }


# ---------------------------------------------------------------------------
# MLflow artifact helpers
# ---------------------------------------------------------------------------


def _log_pr_curve(y_true: np.ndarray, y_score: np.ndarray, prefix: str, out_dir: Path) -> None:
    """Plot and log a precision-recall curve to MLflow."""
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(rec, prec, where="post", linewidth=1.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve ({prefix})")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    path = out_dir / f"pr_curve_{prefix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(str(path))


def _log_shap(booster: xgb.Booster, X: np.ndarray, feature_names: list[str], out_dir: Path) -> None:
    """Compute and log SHAP feature-importance bar plot.

    Falls back to XGBoost built-in gain importance if SHAP is incompatible
    with the installed XGBoost version.
    """
    shap_values = None
    try:
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X[:5000])
    except (ValueError, TypeError, Exception) as exc:
        logger.warning("SHAP TreeExplainer failed (%s); falling back to gain importance", exc)

    if shap_values is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X[:5000], feature_names=feature_names, show=False, plot_type="bar"
        )
        path = out_dir / "shap_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        mlflow.log_artifact(str(path))

        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = dict(sorted(zip(feature_names, mean_abs.tolist()), key=lambda x: -x[1]))
    else:
        raw_scores = booster.get_score(importance_type="gain")
        fname_map = {f"f{i}": name for i, name in enumerate(feature_names)}
        importance = {}
        for k, v in raw_scores.items():
            importance[fname_map.get(k, k)] = v
        importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

        top_n = list(importance.items())[:20]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh([n for n, _ in reversed(top_n)], [v for _, v in reversed(top_n)])
        ax.set_xlabel("Mean Gain")
        ax.set_title("Feature Importance (XGBoost Gain)")
        path = out_dir / "shap_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(path))

    imp_path = out_dir / "shap_importance.json"
    imp_path.write_text(json.dumps(importance, indent=2))
    mlflow.log_artifact(str(imp_path))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    config_path: str | Path = "configs/fraud_config.yaml",
    output_dir: str | Path = "artifacts",
    tracking_uri: str | None = None,
) -> tuple[xgb.Booster, dict[str, float]]:
    """Train an XGBoost fraud classifier with full MLflow instrumentation.

    Args:
        X_train: Preprocessed training features.
        y_train: Binary training labels.
        X_val: Preprocessed validation features.
        y_val: Binary validation labels.
        feature_names: Human-readable feature names (post-encoding).
        config_path: Path to the YAML configuration.
        output_dir: Local directory for saving artifacts.
        tracking_uri: MLflow server URI (None for local file store).

    Returns:
        (booster, metrics_dict) from the validation set evaluation.
    """
    config = load_config(config_path)
    hp = config["model"]["hyperparameters"]
    loss_strategy = config["model"]["loss_strategy"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    # Compute class imbalance ratio
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    imbalance_ratio = n_neg / max(n_pos, 1)
    logger.info("Class ratio: neg=%d, pos=%d, imbalance=%.1f:1", n_neg, n_pos, imbalance_ratio)

    xgb_params: dict[str, Any] = {
        "max_depth": hp["max_depth"],
        "learning_rate": hp["learning_rate"],
        "subsample": hp["subsample"],
        "colsample_bytree": hp["colsample_bytree"],
        "min_child_weight": hp["min_child_weight"],
        "gamma": hp["gamma"],
        "reg_alpha": hp["reg_alpha"],
        "reg_lambda": hp["reg_lambda"],
        "eval_metric": hp["eval_metric"],
        "seed": hp["seed"],
        "tree_method": "hist",
    }

    custom_obj = None
    if loss_strategy == "focal":
        gamma_focal = config["model"]["focal_gamma"]
        alpha_focal = config["model"]["focal_alpha"]
        custom_obj = partial(focal_binary_objective, gamma=gamma_focal, alpha=alpha_focal)
        logger.info("Using focal loss (gamma=%.1f, alpha=%.2f)", gamma_focal, alpha_focal)
    else:
        xgb_params["objective"] = config["model"]["objective"]
        xgb_params["scale_pos_weight"] = imbalance_ratio
        logger.info("Using scale_pos_weight=%.1f", imbalance_ratio)

    with mlflow.start_run(run_name=f"xgb-fraud-{loss_strategy}") as run:
        mlflow.log_params(xgb_params)
        mlflow.log_param("loss_strategy", loss_strategy)
        mlflow.log_param("n_estimators", hp["n_estimators"])
        mlflow.log_param("imbalance_ratio", round(imbalance_ratio, 2))
        mlflow.log_param("train_rows", len(y_train))
        mlflow.log_param("val_rows", len(y_val))

        booster = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=hp["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=hp["early_stopping_rounds"],
            obj=custom_obj,
            verbose_eval=100,
        )

        # Score validation set
        y_score = booster.predict(dval)
        if loss_strategy == "focal":
            y_score = 1.0 / (1.0 + np.exp(-y_score))  # sigmoid for focal logits

        metrics = compute_metrics(y_val, y_score)
        mlflow.log_metrics(metrics)
        logger.info("Validation metrics: %s", metrics)

        # -- Artifacts --
        model_path = output_dir / "model.ubj"
        booster.save_model(str(model_path))
        mlflow.log_artifact(str(model_path))
        mlflow.xgboost.log_model(booster, name="xgboost-model")

        _log_pr_curve(y_val, y_score, "val", output_dir)
        _log_shap(booster, X_val, feature_names, output_dir)

        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        mlflow.log_artifact(str(metrics_path))

        logger.info("MLflow run: %s", run.info.run_id)

    return booster, metrics


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Train from pre-split local files (for development / Makefile target)."""
    from src.data_engineering import load_and_collect, temporal_split
    from src.feature_pipeline import prepare_splits

    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data-path", default="data/Variant II.csv")
    parser.add_argument("--config", default="configs/fraud_config.yaml")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_and_collect(args.data_path)
    train_df, val_df, test_df = temporal_split(df)

    splits = prepare_splits(
        train_df,
        val_df,
        test_df,
        fairness_mode=config["fairness"]["mode"],
        artifact_dir=args.output_dir,
    )

    train_model(
        X_train=splits["X_train"],
        y_train=splits["y_train"],
        X_val=splits["X_val"],
        y_val=splits["y_val"],
        feature_names=splits["feature_names"],
        config_path=args.config,
        output_dir=args.output_dir,
        tracking_uri=args.tracking_uri,
    )


if __name__ == "__main__":
    main()
