"""Tests for data engineering module."""

from __future__ import annotations

import polars as pl
import pytest

from src.data_engineering import load_and_collect, temporal_split


class TestTemporalSplit:
    def test_default_month_ranges(self, synthetic_fraud_df: pl.DataFrame):
        train, val, test = temporal_split(synthetic_fraud_df)
        assert train.height > 0
        assert val.height > 0
        assert test.height > 0

    def test_no_overlap(self, synthetic_fraud_df: pl.DataFrame):
        train, val, test = temporal_split(synthetic_fraud_df)
        train_months = set(train["month"].unique().to_list())
        val_months = set(val["month"].unique().to_list())
        test_months = set(test["month"].unique().to_list())
        assert train_months & val_months == set()
        assert train_months & test_months == set()
        assert val_months & test_months == set()

    def test_train_months_are_0_through_5(self, synthetic_fraud_df: pl.DataFrame):
        train, _, _ = temporal_split(synthetic_fraud_df)
        assert set(train["month"].unique().to_list()) == {0, 1, 2, 3, 4, 5}

    def test_val_is_month_6(self, synthetic_fraud_df: pl.DataFrame):
        _, val, _ = temporal_split(synthetic_fraud_df)
        assert set(val["month"].unique().to_list()) == {6}

    def test_test_is_month_7(self, synthetic_fraud_df: pl.DataFrame):
        _, _, test = temporal_split(synthetic_fraud_df)
        assert set(test["month"].unique().to_list()) == {7}

    def test_total_rows_preserved(self, synthetic_fraud_df: pl.DataFrame):
        train, val, test = temporal_split(synthetic_fraud_df)
        assert train.height + val.height + test.height == synthetic_fraud_df.height

    def test_custom_months(self, synthetic_fraud_df: pl.DataFrame):
        train, val, test = temporal_split(
            synthetic_fraud_df, train_months=[0, 1], val_months=[2], test_months=[3]
        )
        assert set(train["month"].unique().to_list()) == {0, 1}
        assert set(val["month"].unique().to_list()) == {2}
        assert set(test["month"].unique().to_list()) == {3}


class TestLoadAndCollect:
    def test_load_csv(self, synthetic_csv: str):
        df = load_and_collect(synthetic_csv)
        assert isinstance(df, pl.DataFrame)
        assert df.height == 800
        assert "fraud_bool" in df.columns
