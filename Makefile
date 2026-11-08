# Single entry point for every task. If a command is not here, it is not supported.
#
# Windows note: `make` is not installed by default. The equivalent commands are
# documented in README.md under "Windows / PowerShell"; nothing here is magic.

SHELL := /bin/bash
API_DIR := services/api
WEB_DIR := apps/web
PY ?= python

# The shared `devanagari_text` package is a path-mounted module rather than an
# installed distribution, so both the API and the ML tooling see one copy. This
# is the same mechanism the Docker image uses (ENV PYTHONPATH=/app:/app/packages),
# which is why "works locally" and "works in the image" mean the same thing.
export PYTHONPATH := $(CURDIR):$(CURDIR)/packages

.PHONY: help install dev dev-api dev-web test test-api test-web test-ml lint \
        lint-api lint-web typecheck typecheck-api typecheck-web build ci eval \
        export-onnx verify-model docker docker-up deploy-check smoke clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---- setup ----------------------------------------------------------------
install: ## Install web + api dependencies
	pnpm install
	$(PY) -m pip install -r $(API_DIR)/requirements-dev.txt

dev: ## Run api and web together
	$(MAKE) -j2 dev-api dev-web

dev-api: ## Run FastAPI with autoreload on :8000
	cd $(API_DIR) && $(PY) -m uvicorn app.main:app --reload --port 8000

dev-web: ## Run Next.js on :3000
	pnpm --filter @varnam/web dev

# ---- quality --------------------------------------------------------------
test: test-api test-web ## Run all tests

test-api: ## pytest with a coverage gate
	cd $(API_DIR) && $(PY) -m pytest -q --cov=app --cov-report=term-missing --cov-fail-under=75

test-web: ## vitest
	pnpm --filter @varnam/web test

lint: lint-api lint-web ## Lint everything

lint-api: ## ruff
	cd $(API_DIR) && $(PY) -m ruff check .

lint-web: ## eslint
	pnpm --filter @varnam/web lint

typecheck: typecheck-api typecheck-web ## Type-check everything

typecheck-api: ## mypy
	cd $(API_DIR) && $(PY) -m mypy app

typecheck-web: ## tsc --noEmit
	pnpm --filter @varnam/web typecheck

build: ## Production build of the web app
	pnpm --filter @varnam/web build

ci: lint typecheck test build ## Everything CI runs
	@echo "CI OK"

# ---- model ----------------------------------------------------------------
eval: ## Measure CER/WER on the held-out slice (writes docs/EVALUATION.md)
	$(PY) -m ml.eval --manifest data/eval_manifest.csv

export-onnx: ## Export + INT8-quantize the recognition model
	$(PY) -m ml.export_onnx --checkpoint artifacts/best.pt --charset data/charset.txt \
		--out $(API_DIR)/models

verify-model: ## Check exported artifacts are internally consistent
	$(PY) -m ml.verify --model $(API_DIR)/models/rec.onnx --charset $(API_DIR)/models/charset.txt

# ---- deploy ---------------------------------------------------------------
docker: ## Build the API image
	docker build -f $(API_DIR)/Dockerfile -t varnam-api:local .

docker-up: ## api + postgres locally
	docker compose up --build

deploy-check: ## Verify production config before shipping
	$(PY) scripts/deploy_check.py

smoke: ## Hit a live deployment end to end
	bash scripts/smoke_test.sh $${API_URL:-http://127.0.0.1:8000}

clean: ## Remove build artifacts
	rm -rf apps/web/.next $(API_DIR)/.pytest_cache $(API_DIR)/.ruff_cache $(API_DIR)/.mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
