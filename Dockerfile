# ── HuggingFace Spaces — Cricket QA (API + Frontend) ─────────────────────────
# Stage 1  : build the React/Vite frontend
# Stage 2  : install Python deps
# Stage 3  : lightweight runtime with API + static files
#
# /infer endpoint (use_llm=false) answers in < 1ms from DuckDB + StatsEngine.
# use_llm=true returns HTTP 503 — ML model not loaded in this build.


# ── Stage 1: build frontend ───────────────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /frontend

COPY Frontend/package.json Frontend/package-lock.json ./
RUN npm ci --prefer-offline

COPY Frontend/ ./
RUN npm run build


# ── Stage 2: install Python deps ─────────────────────────────────────────────
FROM python:3.11-slim AS py-builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-hf.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements-hf.txt


# ── Stage 3: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Python packages
COPY --from=py-builder /install /usr/local

# Application code
COPY agents/    ./agents/
COPY service/   ./service/
COPY data/sample_match.json ./data/sample_match.json

# Pre-built DuckDB — all 74 IPL 2022 matches, 15 598 deliveries
COPY ipl2022.duckdb ./

# Built frontend (served as static files by FastAPI)
COPY --from=frontend-builder /frontend/dist ./static/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEFAULT_MATCH_FILE=data/sample_match.json

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "7860"]
