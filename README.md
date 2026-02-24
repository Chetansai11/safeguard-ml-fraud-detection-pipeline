# Fraud Detection System — BAF Suite Variant II

Production-grade fraud detection pipeline built on AWS, using the [Bank Account Fraud (BAF) Dataset — Variant II](https://arxiv.org/abs/2211.13358) (NeurIPS 2022, ~1M transactions). Features temporal-aware training, fairness-aware preprocessing, and full MLOps lifecycle.

## Architecture

```
data/Variant II.csv
       │
       ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  S3 Bronze  │────▶│  S3 Silver   │────▶│  S3 Gold    │
│  (Raw CSV)  │     │ (Validated)  │     │ (Splits)    │
└─────────────┘     └──────┬───────┘     └──────┬──────┘
                           │                     │
                  Great Expectations      Temporal Split
                   Quality Gate           mo 0-5 / 6 / 7
                                                 │
                                     ┌───────────┴───────────┐
                                     │  ColumnTransformer     │
                                     │  (OHE + RobustScaler)  │
                                     │  Fairness-aware        │
                                     └───────────┬───────────┘
                                                 │
                                          ┌──────▼──────┐
                                          │  XGBoost    │
                                          │  + MLflow   │
                                          └──────┬──────┘
                                                 │
                                          ┌──────▼──────┐
                                          │  Evidently  │
                                          │  Drift Mon. │
                                          └──────┬──────┘
                                                 │
                                          ┌──────▼──────┐
                                          │  Lambda     │
                                          │  (FastAPI)  │
                                          └─────────────┘
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Polars** for loading | Memory-efficient lazy evaluation for 1M+ rows |
| **Temporal split** (not random) | Prevents data leakage; mirrors real deployment chronology |
| **scale_pos_weight / focal loss** | Handles extreme class imbalance (~1-2% fraud rate) |
| **Recall @ 1% FPR** as primary metric | Banking industry standard — minimise false accusations |
| **Fairness "unaware" mode** | Protected attributes excluded from model, preserved for auditing |
| **RobustScaler** (not StandardScaler) | Resistant to outlier-heavy fraud features |

## Project Structure

```
aws_project/
├── .github/workflows/pipeline.yml   # CI/CD: lint, test, drift report, deploy
├── app/
│   └── handler.py                   # FastAPI + Mangum Lambda handler
├── configs/
│   └── fraud_config.yaml            # Features, hyperparameters, thresholds
├── data/
│   └── Variant II.csv               # BAF Variant II dataset (not committed)
├── src/
│   ├── data_engineering.py          # Polars loader, temporal split, Medallion
│   ├── feature_pipeline.py          # ColumnTransformer, fairness preprocessing
│   ├── train.py                     # XGBoost + focal loss + MLflow + SHAP
│   ├── evaluate.py                  # PR curves, Recall@1%FPR, fairness audit
│   ├── data_integrity.py            # Great Expectations quality gate
│   └── monitoring.py                # Evidently drift report (month 7 vs train)
├── terraform/                       # S3, ECR, MLflow, IAM (least-privilege)
├── tests/                           # pytest suite with synthetic fixtures
├── Dockerfile                       # Lambda-compatible container
├── Makefile                         # Automation targets
├── pyproject.toml                   # Dependencies and tooling config
└── requirements.txt                 # Pinned production dependencies
```

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Place the BAF Variant II dataset at data/Variant II.csv

# 3. Run quality gate
make integrity

# 4. Train model (logs to MLflow)
make train

# 5. Evaluate on month-7 test set
make evaluate

# 6. Generate drift monitoring report
make monitor

# 7. Run tests
make test

# 8. Lint
make lint
```

## Dataset — BAF Variant II

30 features across 1M transactions with ~1-2% fraud rate.

| Feature Type | Columns | Preprocessing |
|-------------|---------|---------------|
| **Categorical** | `payment_type`, `housing_status`, `source`, `device_os` | OneHotEncoder |
| **Numerical** | `velocity_6h`, `velocity_24h`, `credit_risk_score`, … (17 cols) | RobustScaler |
| **Binary** | `email_is_free`, `phone_home_valid`, `foreign_request`, … (6 cols) | Passthrough |
| **Protected** | `customer_age`, `employment_status`, `income` | Excluded (unaware mode) |

### Temporal Split

| Split | Months | Purpose |
|-------|--------|---------|
| Train | 0-5 | Model development |
| Validation | 6 | Hyperparameter tuning, early stopping |
| Test | 7 | Production / unseen evaluation |

## Metrics

| Metric | Why It Matters |
|--------|----------------|
| **Recall @ 1% FPR** | Industry standard — detect fraud while limiting false accusations to 1% |
| **PR-AUC** | Better than ROC-AUC for imbalanced datasets |
| **Best F1** | Optimal threshold calibration |
| **Fairness audit** | Per-group TPR, FPR, selection rate across protected attributes |

## MLflow Logged Artifacts

- Hyperparameters and class imbalance ratio
- Precision-Recall curve (validation)
- SHAP feature importance (bar plot + JSON)
- Model binary (`model.ubj`)
- Full metrics JSON

## Monitoring

The monitoring module generates an Evidently AI drift report comparing training months (0-5) against the production test month (7). Outputs:

- `drift_report.html` — Interactive visual report
- `drift_summary.json` — Machine-readable summary
- `drift_summary.md` — Markdown for CML PR comments

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check + model/preprocessor status |
| `POST` | `/predict` | Single-transaction fraud scoring |
| `POST` | `/predict/batch` | Batch scoring (up to 200) |

### Example Response

```json
{
  "fraud_probability": 0.0342,
  "is_fraud": false,
  "risk_tier": "LOW",
  "threshold": 0.5
}
```

## Infrastructure (Terraform)

Provisions via `make tf-apply`:
- 3 S3 Buckets (Bronze / Silver / Gold) with encryption + versioning
- 1 ECR Repository with scan-on-push
- 1 SageMaker MLflow Tracking Server
- 1 IAM Role with least-privilege policies

## CI/CD Pipeline

| Stage | Trigger | Actions |
|-------|---------|---------|
| Lint & Test | All pushes/PRs | `ruff`, `pytest` |
| Drift Report | PRs only | Evidently report posted via CML |
| Train & Evaluate | Merge to `main` | Integrity gate → Train → Evaluate |
| Build & Push | Merge to `main` | Docker build → ECR push |
