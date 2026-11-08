# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Single image that serves BOTH the API and the built frontend.
#
# Why one service: a free instance can only run one thing comfortably, and a
# split deployment adds a second hostname plus a CORS dependency that can (and
# did) fail independently. Serving the SPA from the same origin removes that
# whole class of problem.
#
# Stages:
#   web     - builds the Next.js static export (discarded after copying `out/`)
#   builder - installs the Python runtime into a self-contained venv
#   runtime - slim image with the venv, the app, the model, and the static UI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------- web build
FROM node:20-slim AS web

RUN npm install --global pnpm@9
WORKDIR /web

# Only the workspace manifests and the web app are needed. Copying them first
# keeps this layer cacheable when only Python sources change.
COPY package.json pnpm-workspace.yaml ./
COPY apps/web ./apps/web

# NEXT_PUBLIC_API_BASE_URL is intentionally EMPTY: the client then calls
# relative URLs (/v1/ocr) and hits whatever origin served the page. That is what
# makes the same build work on Render and on localhost.
ENV NEXT_OUTPUT=export \
    NEXT_PUBLIC_API_BASE_URL=""
RUN pnpm install --filter @varnam/web... --ignore-scripts \
    && pnpm --filter @varnam/web build


# ------------------------------------------------------------- python build
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY services/api/requirements.txt ./requirements.txt

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt


# ------------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime

# PYTHONUNBUFFERED keeps logs streaming (a buffered container looks hung in the
# Render dashboard). PYTHONDONTWRITEBYTECODE avoids writing .pyc into the layer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:/app/packages" \
    APP_ENV=production \
    STATIC_DIR=/app/static \
    PORT=8000

# libgomp is required by onnxruntime's CPU kernels; curl is used by HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Shared normalization package first: both training and serving import it, and a
# single copy is what prevents train/serve skew.
COPY packages/ /app/packages/
COPY services/api/app/ /app/app/
COPY services/api/models/ /app/models/
# The built SPA. app.main mounts this directory at "/" when it exists.
COPY --from=web /web/apps/web/out/ /app/static/

# Run as an unprivileged user: if the pipeline is ever exploited through a
# malformed image, the blast radius is a nobody-owned process.
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
