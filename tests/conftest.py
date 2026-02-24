"""Shared fixtures for fraud detection tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def synthetic_fraud_df() -> pl.DataFrame:
    """Small deterministic dataset mimicking BAF Variant II schema.

    Contains 800 rows across months 0-7 with ~5 % fraud rate.
    """
    rng = np.random.default_rng(42)
    n = 800

    payment_types = ["AA", "AB", "AC", "AD", "AE"]
    housing = ["BA", "BB", "BC", "BD", "BE", "BF", "BG"]
    sources = ["INTERNET", "TELEAPP"]
    devices = ["windows", "macintosh", "linux", "x11", "other"]
    emp_status = ["CA", "CB", "CC", "CD", "CE", "CF", "CG"]

    data = {
        "fraud_bool": (rng.random(n) < 0.05).astype(int),
        "month": np.repeat(np.arange(8), n // 8),
        "income": rng.uniform(0, 1, n).round(4),
        "name_email_similarity": rng.uniform(0, 1, n).round(4),
        "prev_address_months_count": rng.integers(0, 400, n),
        "current_address_months_count": rng.integers(0, 500, n),
        "customer_age": rng.integers(18, 80, n),
        "days_since_request": rng.uniform(0, 30, n).round(2),
        "intended_balcon_amount": rng.uniform(-10, 100, n).round(2),
        "payment_type": rng.choice(payment_types, n),
        "zip_count_4w": rng.integers(0, 5000, n),
        "velocity_6h": rng.uniform(0, 50, n).round(2),
        "velocity_24h": rng.uniform(0, 100, n).round(2),
        "velocity_4w": rng.uniform(0, 1000, n).round(2),
        "bank_branch_count_8w": rng.integers(0, 30, n),
        "date_of_birth_distinct_emails_4w": rng.integers(0, 15, n),
        "employment_status": rng.choice(emp_status, n),
        "credit_risk_score": rng.integers(-200, 300, n),
        "email_is_free": rng.integers(0, 2, n),
        "housing_status": rng.choice(housing, n),
        "phone_home_valid": rng.integers(0, 2, n),
        "phone_mobile_valid": rng.integers(0, 2, n),
        "bank_months_count": rng.integers(0, 400, n),
        "has_other_cards": rng.integers(0, 2, n),
        "proposed_credit_limit": rng.uniform(200, 5000, n).round(0),
        "foreign_request": rng.integers(0, 2, n),
        "source": rng.choice(sources, n),
        "session_length_in_minutes": rng.uniform(0, 60, n).round(2),
        "device_os": rng.choice(devices, n),
        "keep_alive_session": rng.integers(0, 2, n),
        "device_distinct_emails_8w": rng.integers(0, 10, n),
        "device_fraud_count": rng.integers(0, 5, n),
    }
    return pl.DataFrame(data)


@pytest.fixture
def synthetic_csv(synthetic_fraud_df: pl.DataFrame, tmp_path) -> str:
    """Write synthetic data to a temp CSV and return the path."""
    path = tmp_path / "test_fraud.csv"
    synthetic_fraud_df.write_csv(str(path))
    return str(path)
