"""Tests for feature pipeline module."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.data_engineering import temporal_split
from src.feature_pipeline import (
    ALL_BINARY,
    PROTECTED_ATTRS,
    build_preprocessor,
    prepare_splits,
    resolve_feature_columns,
)


class TestResolveColumns:
    def test_unaware_excludes_protected(self):
        cat, num, binary = resolve_feature_columns("unaware")
        for prot in PROTECTED_ATTRS:
            assert prot not in cat
            assert prot not in num

    def test_aware_includes_all(self):
        cat, num, binary = resolve_feature_columns("aware")
        assert "employment_status" in cat
        assert "income" in num
        assert "customer_age" in num

    def test_binary_unchanged_by_mode(self):
        _, _, bin_unaware = resolve_feature_columns("unaware")
        _, _, bin_aware = resolve_feature_columns("aware")
        assert bin_unaware == bin_aware == ALL_BINARY


class TestBuildPreprocessor:
    def test_transformer_has_expected_steps(self):
        cat, num, binary = resolve_feature_columns("unaware")
        ct = build_preprocessor(cat, num, binary)
        names = [name for name, _, _ in ct.transformers]
        assert names == ["cat", "num", "bin"]


class TestPrepareSplits:
    def test_output_shapes(self, synthetic_fraud_df: pl.DataFrame, tmp_path):
        train, val, test = temporal_split(synthetic_fraud_df)
        splits = prepare_splits(
            train, val, test, fairness_mode="unaware", artifact_dir=str(tmp_path)
        )

        assert splits["X_train"].shape[0] == train.height
        assert splits["X_val"].shape[0] == val.height
        assert splits["X_test"].shape[0] == test.height
        assert splits["X_train"].shape[1] == splits["X_val"].shape[1]

    def test_labels_are_binary(self, synthetic_fraud_df: pl.DataFrame, tmp_path):
        train, val, test = temporal_split(synthetic_fraud_df)
        splits = prepare_splits(train, val, test, artifact_dir=str(tmp_path))
        for key in ["y_train", "y_val", "y_test"]:
            assert set(np.unique(splits[key])).issubset({0, 1})

    def test_protected_attrs_preserved(self, synthetic_fraud_df: pl.DataFrame, tmp_path):
        train, val, test = temporal_split(synthetic_fraud_df)
        splits = prepare_splits(train, val, test, artifact_dir=str(tmp_path))
        for key in ["protected_train", "protected_val", "protected_test"]:
            assert "customer_age" in splits[key].columns
            assert "employment_status" in splits[key].columns
            assert "income" in splits[key].columns
            assert "age_group" in splits[key].columns

    def test_preprocessor_saved(self, synthetic_fraud_df: pl.DataFrame, tmp_path):
        train, val, test = temporal_split(synthetic_fraud_df)
        prepare_splits(train, val, test, artifact_dir=str(tmp_path))
        assert (tmp_path / "preprocessor.joblib").exists()

    def test_feature_names_present(self, synthetic_fraud_df: pl.DataFrame, tmp_path):
        train, val, test = temporal_split(synthetic_fraud_df)
        splits = prepare_splits(train, val, test, artifact_dir=str(tmp_path))
        assert len(splits["feature_names"]) > 0
        assert splits["X_train"].shape[1] == len(splits["feature_names"])
