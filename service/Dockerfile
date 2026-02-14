FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y git

# Copy only required files
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY data ./data
COPY agents ./agents
COPY service ./service

# HuggingFace cache
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]

