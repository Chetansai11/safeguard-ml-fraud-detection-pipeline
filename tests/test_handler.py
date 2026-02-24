"""Tests for the FastAPI inference handler."""

from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient

from app.handler import FraudRequest, app


@pytest.fixture
def mock_booster():
    """Train a tiny XGBoost model for handler testing."""
    rng = np.random.default_rng(42)
    n = 300
    X = rng.random((n, 27))
    y = (rng.random(n) > 0.95).astype(int)
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "seed": 42,
        "scale_pos_weight": 19,
    }
    return xgb.train(params, dtrain, num_boost_round=10)


@pytest.fixture
def client(mock_booster, monkeypatch):
    """FastAPI test client with mock model injected."""
    import app.handler as handler_module

    monkeypatch.setattr(handler_module, "_booster", mock_booster)
    monkeypatch.setattr(handler_module, "_preprocessor", None)
    return TestClient(app)


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


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["model_loaded"] is True


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
        assert len(resp.json()) == 2


class TestFraudRequestValidation:
    def test_valid_request(self):
        req = FraudRequest(**_sample_payload())
        assert req.payment_type == "AA"

    def test_invalid_binary_field(self):
        payload = _sample_payload()
        payload["email_is_free"] = 5
        with pytest.raises(Exception):
            FraudRequest(**payload)
