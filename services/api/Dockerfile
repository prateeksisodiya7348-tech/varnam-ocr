# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Multi-stage image for the Varnam OCR API.
#
# Why multi-stage: the wheel cache and build headers needed to compile Pillow's
# optional deps are ~400MB and irrelevant at runtime. A slim final stage keeps
# the image under 200MB, which matters because Render's free tier pulls the
# image on every cold start after a deploy.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY services/api/requirements.txt ./requirements.txt

# Build into an isolated venv that we copy wholesale into the runtime stage.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt


FROM python:3.12-slim AS runtime

# PYTHONUNBUFFERED keeps logs streaming (critical: a buffered container looks
# hung in the Render dashboard). PYTHONDONTWRITEBYTECODE avoids writing .pyc
# files into a read-only-ish layer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:/app/packages" \
    APP_ENV=production \
    PORT=8000

# libgomp is required by onnxruntime's CPU kernels; curl is used by HEALTHCHECK.
# Both are removed from the layer cache only; apt lists are dropped explicitly.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Shared normalization package first: both training and serving import it, and
# a single copy is what prevents train/serve skew.
COPY packages/ /app/packages/
COPY services/api/app/ /app/app/
COPY services/api/models/ /app/models/

# Run as an unprivileged user. A container that only needs to read its own code
# and write nothing should never be root - if the OCR pipeline is ever exploited
# through a malformed image, the blast radius is a nobody-owned process.
RUN useradd --create-home --uid 10001 varnam \
    && chown -R varnam:varnam /app
USER varnam

EXPOSE 8000

# Readiness (not liveness): the platform should not send traffic to a container
# whose model has not finished loading.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health/ready || exit 1

# Single worker on purpose: two uvicorn workers would each load the model,
# doubling RSS past the 512MB ceiling of a free instance.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "30"]
