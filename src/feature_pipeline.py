"""Feature engineering pipeline with fairness-aware preprocessing.

Builds a scikit-learn ColumnTransformer that applies:
  - OneHotEncoding for categorical features (payment_type, housing_status, …).
  - RobustScaling for numerical features (velocity_*, income, …).
  - Passthrough for binary indicator columns.

Protected attributes (customer_age, employment_status, income) are stripped
from model features in "unaware" fairness mode but preserved in a sidecar
DataFrame for post-hoc fairness auditing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions — derived from BAF Variant II schema
# ---------------------------------------------------------------------------

ALL_CATEGORICAL = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
ALL_NUMERICAL = [
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "proposed_credit_limit",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
    "device_fraud_count",
    "bank_months_count",
]
ALL_BINARY = [
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "has_other_cards",
    "foreign_request",
    "keep_alive_session",
]
PROTECTED_ATTRS = ["customer_age", "employment_status", "income"]
TARGET = "fraud_bool"
TEMPORAL_COL = "month"


def load_config(path: str | Path = "configs/fraud_config.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Column resolver
# ---------------------------------------------------------------------------

def resolve_feature_columns(
    fairness_mode: str = "unaware",
) -> tuple[list[str], list[str], list[str]]:
    """Return (categorical, numerical, binary) column lists respecting fairness mode.

    In *unaware* mode the three protected attributes are excluded from model
    input but remain available for downstream auditing.  In *aware* mode all
    columns are used.

    Args:
        fairness_mode: ``"unaware"`` or ``"aware"``.

    Returns:
        Three-tuple of column name lists.
    """
    if fairness_mode == "unaware":
        cat = [c for c in ALL_CATEGORICAL if c not in PROTECTED_ATTRS]
        num = [c for c in ALL_NUMERICAL if c not in PROTECTED_ATTRS]
    else:
        cat = list(ALL_CATEGORICAL)
        num = list(ALL_NUMERICAL)
    return cat, num, list(ALL_BINARY)


# ---------------------------------------------------------------------------
# ColumnTransformer builder
# ---------------------------------------------------------------------------

def build_preprocessor(
    categorical_cols: list[str],
    numerical_cols: list[str],
    binary_cols: list[str],
) -> ColumnTransformer:
    """Construct a scikit-learn ColumnTransformer.

    Transformers:
      - ``cat``: OneHotEncoder (sparse output, handle_unknown='ignore').
      - ``num``: RobustScaler (resistant to outlier-heavy fraud features).
      - ``bin``: Passthrough (already 0/1).

    Args:
        categorical_cols: Columns to one-hot encode.
        numerical_cols: Columns to robust-scale.
        binary_cols: Columns to pass through unchanged.

    Returns:
        Fitted-ready ColumnTransformer.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"),
                categorical_cols,
            ),
            ("num", RobustScaler(), numerical_cols),
            ("bin", "passthrough", binary_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def prepare_splits(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    fairness_mode: str = "unaware",
    artifact_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    """Fit the preprocessor on training data and transform all splits.

    Also extracts protected attributes into separate arrays for fairness
    auditing and derives an ``age_group`` column from ``customer_age``.

    Args:
        train_df: Training split (months 0-5).
        val_df: Validation split (month 6).
        test_df: Production test split (month 7).
        fairness_mode: ``"unaware"`` or ``"aware"``.
        artifact_dir: Where to persist the fitted preprocessor.

    Returns:
        Dictionary with keys: ``X_train``, ``X_val``, ``X_test``,
        ``y_train``, ``y_val``, ``y_test``, ``protected_train``,
        ``protected_val``, ``protected_test``, ``preprocessor``,
        ``feature_names``.
    """
    cat_cols, num_cols, bin_cols = resolve_feature_columns(fairness_mode)
    preprocessor = build_preprocessor(cat_cols, num_cols, bin_cols)

    train_pd = train_df.to_pandas()
    val_pd = val_df.to_pandas()
    test_pd = test_df.to_pandas()

    feature_cols = cat_cols + num_cols + bin_cols
    _validate_columns(train_pd, feature_cols)

    X_train = preprocessor.fit_transform(train_pd[feature_cols])
    X_val = preprocessor.transform(val_pd[feature_cols])
    X_test = preprocessor.transform(test_pd[feature_cols])

    y_train = train_pd[TARGET].values.astype(np.int8)
    y_val = val_pd[TARGET].values.astype(np.int8)
    y_test = test_pd[TARGET].values.astype(np.int8)

    feature_names = list(preprocessor.get_feature_names_out())

    def _extract_protected(pdf: pd.DataFrame) -> pd.DataFrame:
        prot = pdf[PROTECTED_ATTRS].copy()
        if "customer_age" in prot.columns:
            prot["age_group"] = pd.cut(
                prot["customer_age"],
                bins=[0, 25, 35, 45, 55, 65, 120],
                labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
            )
        return prot

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, artifact_dir / "preprocessor.joblib")
    logger.info("Preprocessor saved to %s", artifact_dir / "preprocessor.joblib")

    logger.info(
        "Features: %d total (%d cat-encoded, %d num-scaled, %d binary passthrough)",
        len(feature_names),
        len([f for f in feature_names if f.startswith("cat__")]),
        len(num_cols),
        len(bin_cols),
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "protected_train": _extract_protected(train_pd),
        "protected_val": _extract_protected(val_pd),
        "protected_test": _extract_protected(test_pd),
        "preprocessor": preprocessor,
        "feature_names": feature_names,
    }


def _validate_columns(df: pd.DataFrame, expected: list[str]) -> None:
    """Raise early if required columns are missing."""
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {sorted(missing)}")
