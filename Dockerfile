# ── HuggingFace Spaces — Cricket QA API ──────────────────────────────────────
# Deterministic path only (no torch/transformers).
# Default /infer endpoint (use_llm=false) answers in < 1ms from DuckDB + StatsEngine.
# use_llm=true returns HTTP 503 — ML model not loaded in this build.

FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-hf.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements-hf.txt


FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /install /usr/local

# Application code
COPY agents/    ./agents/
COPY service/   ./service/
COPY data/sample_match.json ./data/sample_match.json

# Pre-built DuckDB — all 74 IPL 2022 matches, 15 598 deliveries
COPY ipl2022.duckdb ./

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEFAULT_MATCH_FILE=data/sample_match.json

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "7860"]
