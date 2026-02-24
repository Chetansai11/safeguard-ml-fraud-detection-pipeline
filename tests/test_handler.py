"""Tests for the FastAPI inference handler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from app.handler import FraudRequest, app

CATEGORICAL = ["payment_type", "housing_status", "source", "device_os"]
NUMERICAL = [
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
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
BINARY = [
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "has_other_cards",
    "foreign_request",
    "keep_alive_session",
]


def _sample_payload() -> dict:
    return {
        "payment_type": "AA",
        "housing_status": "BA",
        "source": "INTERNET",
        "device_os": "windows",
        "email_is_free": 1,
        "phone_home_valid": 1,
        "phone_mobile_valid": 0,
        "has_other_cards": 0,
        "foreign_request": 0,
        "keep_alive_session": 1,
        "name_email_similarity": 0.45,
        "prev_address_months_count": 24.0,
        "current_address_months_count": 60.0,
        "days_since_request": 0.5,
        "intended_balcon_amount": 50.0,
        "zip_count_4w": 1200.0,
        "velocity_6h": 5.0,
        "velocity_24h": 15.0,
        "velocity_4w": 200.0,
        "bank_branch_count_8w": 3.0,
        "date_of_birth_distinct_emails_4w": 2.0,
        "credit_risk_score": 100.0,
        "proposed_credit_limit": 1500.0,
        "session_length_in_minutes": 10.0,
        "device_distinct_emails_8w": 1.0,
        "device_fraud_count": 0.0,
        "bank_months_count": 36.0,
    }


def _build_mock_preprocessor_and_model():
    """Build a fitted preprocessor and a matching XGBoost model.

    The preprocessor is fit on a small synthetic frame so that it can
    transform real request payloads. The model is then trained on the
    preprocessor's output dimensionality.
    """
    rng = np.random.default_rng(42)
    n = 200

    df = pd.DataFrame(
        {
            "payment_type": rng.choice(["AA", "AB", "AC"], n),
            "housing_status": rng.choice(["BA", "BB", "BC"], n),
            "source": rng.choice(["INTERNET", "TELEAPP"], n),
            "device_os": rng.choice(["windows", "macintosh", "linux"], n),
            **{col: rng.uniform(0, 100, n) for col in NUMERICAL},
            **{col: rng.integers(0, 2, n) for col in BINARY},
        }
    )

    feature_cols = CATEGORICAL + NUMERICAL + BINARY
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"),
                CATEGORICAL,
            ),
            ("num", RobustScaler(), NUMERICAL),
            ("bin", "passthrough", BINARY),
        ],
        remainder="drop",
    )
    X = preprocessor.fit_transform(df[feature_cols])

    y = (rng.random(n) > 0.95).astype(int)
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "seed": 42,
        "scale_pos_weight": 19,
    }
    booster = xgb.train(params, dtrain, num_boost_round=10)

    return preprocessor, booster


@pytest.fixture
def client(monkeypatch):
    """FastAPI test client with mock model and preprocessor injected."""
    import app.handler as handler_module

    preprocessor, booster = _build_mock_preprocessor_and_model()
    monkeypatch.setattr(handler_module, "_booster", booster)
    monkeypatch.setattr(handler_module, "_preprocessor", preprocessor)
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["model_loaded"] is True
        assert resp.json()["preprocessor_loaded"] is True


class TestPredictEndpoint:
    def test_valid_prediction(self, client):
        resp = client.post("/predict", json=_sample_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert isinstance(data["is_fraud"], bool)
        assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]

    def test_missing_field(self, client):
        payload = {"payment_type": "AA"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


class TestBatchEndpoint:
    def test_batch(self, client):
        resp = client.post("/predict/batch", json=[_sample_payload(), _sample_payload()])
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        for item in data:
            assert 0.0 <= item["fraud_probability"] <= 1.0


class TestFraudRequestValidation:
    def test_valid_request(self):
        req = FraudRequest(**_sample_payload())
        assert req.payment_type == "AA"

    def test_invalid_binary_field(self):
        payload = _sample_payload()
        payload["email_is_free"] = 5
        with pytest.raises(Exception):
            FraudRequest(**payload)
