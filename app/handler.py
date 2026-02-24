"""AWS Lambda handler for real-time fraud inference.

Wraps a FastAPI application with Mangum for Lambda compatibility.
Loads the latest approved model from the SageMaker Model Registry and
the fitted preprocessor at cold-start, then serves fraud predictions
through a REST API.
"""

from __future__ import annotations

import logging
import os
import tarfile
import tempfile
from contextlib import asynccontextmanager
from functools import lru_cache

import boto3
import joblib
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from mangum import Mangum
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PACKAGE_GROUP = os.getenv("MODEL_PACKAGE_GROUP", "fraud-detection-baf")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
PREPROCESSOR_S3_URI = os.getenv("PREPROCESSOR_S3_URI", "")
LOSS_STRATEGY = os.getenv("LOSS_STRATEGY", "scale_pos_weight")

_booster: xgb.Booster | None = None
_preprocessor = None


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class FraudRequest(BaseModel):
    """Input payload mirroring the BAF Variant II feature set."""

    payment_type: str = Field(..., description="Payment method code")
    housing_status: str = Field(..., description="Housing status category")
    source: str = Field(..., description="Application source channel")
    device_os: str = Field(..., description="Operating system of the device")
    email_is_free: int = Field(..., ge=0, le=1)
    phone_home_valid: int = Field(..., ge=0, le=1)
    phone_mobile_valid: int = Field(..., ge=0, le=1)
    has_other_cards: int = Field(..., ge=0, le=1)
    foreign_request: int = Field(..., ge=0, le=1)
    keep_alive_session: int = Field(..., ge=0, le=1)
    name_email_similarity: float = Field(..., ge=0, le=1)
    prev_address_months_count: float = Field(...)
    current_address_months_count: float = Field(...)
    days_since_request: float = Field(...)
    intended_balcon_amount: float = Field(...)
    zip_count_4w: float = Field(...)
    velocity_6h: float = Field(...)
    velocity_24h: float = Field(...)
    velocity_4w: float = Field(...)
    bank_branch_count_8w: float = Field(...)
    date_of_birth_distinct_emails_4w: float = Field(...)
    credit_risk_score: float = Field(...)
    proposed_credit_limit: float = Field(...)
    session_length_in_minutes: float = Field(...)
    device_distinct_emails_8w: float = Field(...)
    device_fraud_count: float = Field(...)
    bank_months_count: float = Field(...)


class FraudResponse(BaseModel):
    """Prediction output."""

    fraud_probability: float
    is_fraud: bool
    risk_tier: str
    threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _sm_client():
    return boto3.client("sagemaker", region_name=AWS_REGION)


@lru_cache(maxsize=1)
def _s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def _resolve_model_uri() -> str:
    """Get S3 URI of the latest approved model from the registry."""
    sm = _sm_client()
    resp = sm.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    packages = resp.get("ModelPackageSummaryList", [])
    if not packages:
        raise RuntimeError(f"No approved model in group '{MODEL_PACKAGE_GROUP}'")
    details = sm.describe_model_package(ModelPackageName=packages[0]["ModelPackageArn"])
    return details["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]


def _download_from_s3(s3_uri: str, local_dir: str) -> str:
    """Download and optionally extract a tar.gz from S3."""
    s3 = _s3_client()
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    local_path = os.path.join(local_dir, os.path.basename(key))
    s3.download_file(bucket, key, local_path)
    if local_path.endswith(".tar.gz"):
        with tarfile.open(local_path) as tar:
            tar.extractall(local_dir)
    return local_dir


def load_artifacts() -> None:
    """Cold-start: download model and preprocessor, load into memory."""
    global _booster, _preprocessor

    if _booster is None:
        model_uri = _resolve_model_uri()
        with tempfile.TemporaryDirectory() as tmpdir:
            _download_from_s3(model_uri, tmpdir)
            model_file = None
            for candidate in ["xgboost-model", "model.xgb"]:
                fp = os.path.join(tmpdir, candidate)
                if os.path.exists(fp):
                    model_file = fp
                    break
            if model_file is None:
                for f in os.listdir(tmpdir):
                    if f.endswith((".xgb", ".model", ".json")):
                        model_file = os.path.join(tmpdir, f)
                        break
            if model_file is None:
                raise FileNotFoundError(f"No model file found in {model_uri}")
            _booster = xgb.Booster()
            _booster.load_model(model_file)
        logger.info("Booster loaded from %s", model_uri)

    if _preprocessor is None and PREPROCESSOR_S3_URI:
        with tempfile.TemporaryDirectory() as tmpdir:
            _download_from_s3(PREPROCESSOR_S3_URI, tmpdir)
            joblib_files = [f for f in os.listdir(tmpdir) if f.endswith(".joblib")]
            if joblib_files:
                _preprocessor = joblib.load(os.path.join(tmpdir, joblib_files[0]))
                logger.info("Preprocessor loaded from %s", PREPROCESSOR_S3_URI)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

RISK_THRESHOLDS = {"high": 0.5, "medium": 0.15, "low": 0.0}
DEFAULT_THRESHOLD = 0.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_artifacts()
    except Exception:
        logger.warning("Artifact pre-load failed; will retry on first request.", exc_info=True)
    yield


app = FastAPI(
    title="Fraud Detection — Real-Time Inference",
    description="BAF Variant II fraud scoring API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=_booster is not None,
        preprocessor_loaded=_preprocessor is not None,
    )


def _classify_risk(prob: float) -> str:
    if prob >= RISK_THRESHOLDS["high"]:
        return "HIGH"
    if prob >= RISK_THRESHOLDS["medium"]:
        return "MEDIUM"
    return "LOW"


def _predict_single(request: FraudRequest) -> FraudResponse:
    """Score a single transaction."""
    try:
        load_artifacts()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc

    import pandas as pd

    row = pd.DataFrame([request.model_dump()])

    if _preprocessor is not None:
        X = _preprocessor.transform(row)
    else:
        X = row.values.astype(np.float32)

    dmatrix = xgb.DMatrix(X)
    score = float(_booster.predict(dmatrix)[0])

    if LOSS_STRATEGY == "focal":
        score = 1.0 / (1.0 + np.exp(-score))

    return FraudResponse(
        fraud_probability=round(score, 5),
        is_fraud=score >= DEFAULT_THRESHOLD,
        risk_tier=_classify_risk(score),
        threshold=DEFAULT_THRESHOLD,
    )


@app.post("/predict", response_model=FraudResponse)
async def predict(request: FraudRequest):
    """Score a single transaction for fraud."""
    return _predict_single(request)


@app.post("/predict/batch", response_model=list[FraudResponse])
async def predict_batch(requests: list[FraudRequest]):
    """Score up to 200 transactions in one call."""
    if len(requests) > 200:
        raise HTTPException(status_code=400, detail="Batch limit is 200.")
    return [_predict_single(r) for r in requests]


handler = Mangum(app, lifespan="on")
