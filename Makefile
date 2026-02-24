.PHONY: help install lint format test train evaluate monitor integrity clean

PYTHON     := python
PIP        := pip
AWS_REGION := us-east-1
PROJECT    := fraud-detection
CONFIG     := configs/fraud_config.yaml
DATA       := data/Variant\ II.csv

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

install: ## Install all dependencies (prod + dev)
	$(PIP) install -e ".[dev]"

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint: ## Run ruff linter and format check
	ruff check src/ app/ tests/
	ruff format --check src/ app/ tests/

format: ## Auto-format code
	ruff check --fix src/ app/ tests/
	ruff format src/ app/ tests/

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test: ## Run pytest suite
	pytest tests/ -v --tb=short --cov=src --cov=app --cov-report=term-missing

# ---------------------------------------------------------------------------
# ML workflow
# ---------------------------------------------------------------------------

train: ## Train XGBoost fraud model (MLflow tracked)
	$(PYTHON) -m src.train \
		--data-path $(DATA) \
		--config $(CONFIG) \
		--output-dir artifacts/

evaluate: ## Evaluate model on month-7 test set
	$(PYTHON) -m src.evaluate \
		--model-path artifacts/model.ubj \
		--data-path $(DATA) \
		--config $(CONFIG) \
		--report-dir artifacts/reports/

# ---------------------------------------------------------------------------
# Data quality & monitoring
# ---------------------------------------------------------------------------

integrity: ## Run Great Expectations quality gate
	$(PYTHON) -m src.data_integrity \
		--data-path $(DATA) \
		--report artifacts/reports/data_integrity.json

monitor: ## Generate Evidently drift report (month 7 vs training)
	$(PYTHON) -m src.monitoring \
		--data-path $(DATA) \
		--config $(CONFIG) \
		--report-dir artifacts/reports/

# ---------------------------------------------------------------------------
# Medallion (S3)
# ---------------------------------------------------------------------------

upload-bronze: ## Upload raw CSV to S3 Bronze
	$(PYTHON) -m src.data_engineering upload-bronze \
		--path $(DATA) --bucket $(PROJECT)-bronze

bronze-to-silver: ## Bronze → Silver (validate + Parquet)
	$(PYTHON) -m src.data_engineering bronze-to-silver \
		--bronze-bucket $(PROJECT)-bronze \
		--silver-bucket $(PROJECT)-silver

silver-to-gold: ## Silver → Gold (temporal splits)
	$(PYTHON) -m src.data_engineering silver-to-gold \
		--silver-bucket $(PROJECT)-silver \
		--gold-bucket $(PROJECT)-gold

# ---------------------------------------------------------------------------
# Docker / deploy
# ---------------------------------------------------------------------------

docker-build: ## Build Lambda container image
	docker build -t $(PROJECT)-inference:latest .

docker-run: ## Run Lambda container locally
	docker run -p 9000:8080 $(PROJECT)-inference:latest

# ---------------------------------------------------------------------------
# Terraform
# ---------------------------------------------------------------------------

tf-init: ## Initialize Terraform
	cd terraform && terraform init

tf-plan: ## Plan Terraform changes
	cd terraform && terraform plan -var-file=terraform.tfvars

tf-apply: ## Apply Terraform infrastructure
	cd terraform && terraform apply -var-file=terraform.tfvars -auto-approve

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean: ## Remove local artifacts and caches
	rm -rf artifacts/ .pytest_cache __pycache__ .ruff_cache mlruns/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
