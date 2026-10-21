# syntax=docker/dockerfile:1

# ── Build stage: install dependencies into a virtualenv and bake in the embedding model ──
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Download the ONNX model at build time so containers start fast and need no network at runtime.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ENV FASTEMBED_CACHE_PATH=/opt/models
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBEDDING_MODEL}')"

# ── Runtime stage ──
FROM python:3.12-slim

ENV PATH=/opt/venv/bin:$PATH \
    FASTEMBED_CACHE_PATH=/opt/models \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

RUN useradd --create-home --uid 10001 membrain
COPY --from=build /opt/venv /opt/venv
COPY --from=build --chown=membrain /opt/models /opt/models

WORKDIR /app/backend
COPY --chown=membrain backend/ /app/backend/
COPY --chown=membrain frontend/ /app/frontend/
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh

USER membrain
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

ENTRYPOINT ["entrypoint.sh"]
CMD ["api"]
