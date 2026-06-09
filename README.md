# Cricket Intelligence — AI Analytics System 🏏

A deterministic, grounded cricket Q&A system built on a multi-agent pipeline. Questions about IPL 2022 are answered directly from a DuckDB database — no LLM hallucination on numeric facts. An optional LoRA-fine-tuned Qwen 0.5B model handles natural-language formatting when enabled.

[![CI](https://github.com/Contributer00001/cricket-qa-system/actions/workflows/ci.yml/badge.svg)](https://github.com/Contributer00001/cricket-qa-system/actions/workflows/ci.yml)
[![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.3-yellow.svg)](https://duckdb.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20Space-Live-orange.svg)](https://huggingface.co/spaces/namankhatak/cricket-qa-system)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Live demo → [huggingface.co/spaces/namankhatak/cricket-qa-system](https://huggingface.co/spaces/namankhatak/cricket-qa-system)**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset — IPL 2022](#dataset--ipl-2022)
- [Evaluation Results](#evaluation-results)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Running Evaluations](#running-evaluations)
- [Training Your Own Model](#training-your-own-model)
- [CI/CD](#cicd)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)

---

## Overview

Most LLM-based sports QA systems hallucinate numbers. This system avoids that by using a **deterministic-first** design:

1. Every question is classified by a rule-based `QueryRouter` into a typed `Intent`
2. A `StatsEngine` or `DuckDBEngine` computes the answer directly from data
3. An optional `AnalystAgent` (fine-tuned Qwen 0.5B) formats the answer — but can never change the number

The result is **100% exact-match accuracy** on a 953-fixture gold set covering all 74 IPL 2022 matches, with sub-millisecond latency.

### What it can answer

| Category | Examples |
|---|---|
| **Tournament leaders** | Who scored the most runs in IPL 2022? |
| **Records** | What was the highest team score? Best bowling figures? |
| **Player stats** | How many sixes did MS Dhoni hit? What was Kohli's average? |
| **Standings** | Which team finished top of the points table? |
| **Per-match stats** | Powerplay score in match 1? Wickets in the death overs? |
| **Phase analysis** | Runs in overs 3–5? Last 2 overs summary? |
| **Live commentary** | Upload ball-by-ball data and ask any question in real time |

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────┐
│  CricketOrchestrator                            │
│  ┌───────────────────┐  ┌────────────────────┐  │
│  │  QueryRouter      │  │  Intent classifier │  │
│  │  (rule-based)     │  │  < 1ms, no LLM     │  │
│  └────────┬──────────┘  └────────────────────┘  │
│           │                                      │
│    ┌──────┴────────────────────────┐             │
│    │  Tournament question?         │             │
│    │  (contains "ipl", "2022" …)  │             │
│    └──────┬───────────────┬────────┘             │
│           │ YES            │ NO                  │
│           ▼                ▼                     │
│  ┌──────────────┐  ┌─────────────────────┐      │
│  │ DuckDBEngine │  │ StatsEngine         │      │
│  │ IPL 2022 DB  │  │ (per-match)         │      │
│  │ 74 matches   │  │ commentary data     │      │
│  │ 15,598 balls │  │ < 1ms               │      │
│  └──────┬───────┘  └──────────┬──────────┘      │
│         └──────────┬──────────┘                  │
│                    │                             │
│                    ▼  (optional)                 │
│         ┌─────────────────────┐                 │
│         │  AnalystAgent       │                 │
│         │  Qwen 0.5B + LoRA   │                 │
│         │  use_llm=true only  │                 │
│         └─────────────────────┘                 │
└─────────────────────────────────────────────────┘
      │
      ▼
Grounded Answer (sourced from data, not generated)
```

### Agent Modules

| Module | Role |
|---|---|
| `query_router.py` | Classifies question into 20+ typed intents in < 1ms |
| `stats_engine.py` | Computes per-match stats (runs, wickets, phases, overs) from commentary JSON |
| `duckdb_engine.py` | Queries the IPL 2022 DuckDB for tournament-wide stats and records |
| `orchestrator.py` | Routes question through the correct pipeline and assembles the response |
| `multi_agent.py` | `StatsTool`, `RetrieverAgent`, `AnalystAgent` — LLM path only |
| `planner_agent.py` | Optional query planning layer |
| `tool_router.py` | Tool dispatch abstraction |

---

## Dataset — IPL 2022

The system ships with a complete IPL 2022 DuckDB database built from ball-by-ball delivery data.

| Metric | Value |
|---|---|
| Matches | 74 |
| Deliveries | 15,598 |
| Players | 247 |
| Teams | 10 |
| Database size | 2.9 MB |

### Database Tables

| Table | Description |
|---|---|
| `matches` | Match metadata, results, venues |
| `deliveries` | Ball-by-ball data — runs, wickets, extras |
| `innings` | Innings totals and scores |
| `batting_performances` | Individual batting scorecards per match |
| `bowling_performances` | Individual bowling figures per match |
| `tournament_batting` | Aggregated batting stats across the full season |
| `tournament_bowling` | Aggregated bowling stats across the full season |
| `standings` | Final IPL 2022 points table |

Build the database locally:
```bash
python -m database.build_ipl_db
```

---

## Evaluation Results

The system is evaluated on two complementary benchmarks.

### 1. Gold Set Evaluation — 953 Fixtures

Ground-truth fixtures reconstructed directly from the DuckDB database, covering all 74 IPL 2022 matches.

#### Overall (953 fixtures)

| Metric | Score |
|---|---|
| **Exact Match** | **100.0%** |
| Contains Match | 100.0% |
| Numeric Match | 53.0% |
| Latency p50 | 0.109 ms |
| Latency p95 | 0.360 ms |

> **Note on numeric match**: Scores below 100% here are expected — numeric match only triggers when both the prediction and expected value are pure floats. Questions like *"Who won Match 34?"* or *"Who was the top scorer?"* return strings (`"Gujarat Titans"`, `"Jos Buttler (863 runs)"`) that cannot be parsed as float, so they count as 0 for numeric match while still being 100% exact match. This is a metric limitation, not a system error.

#### Per-Match Breakdown (941 fixtures — all 74 matches)

| Category | Fixtures | Exact Match | Contains Match |
|---|---|---|---|
| Innings Stats (boundaries, dot balls) | 142 | **100%** | 100% |
| Innings Total (score, wickets, full) | 444 | **100%** | 100% |
| Match Result (winner, margin) | 74 | **100%** | 100% |
| Phase (powerplay, death overs) | 134 | **100%** | 100% |
| Top Performer (scorer, bowler) | 147 | **100%** | 100% |

#### Tournament Breakdown (12 fixtures)

| Category | Fixtures | Exact Match |
|---|---|---|
| Tournament Batting (leaders, averages) | 4 | 100% |
| Tournament Bowling (wickets, economy) | 2 | 100% |
| Tournament Records (highest score, best figures) | 4 | 100% |
| Standings (points table) | 1 | 100% |
| Phase Analysis (powerplay averages by team) | 1 | 100% |

---

### 2. RAGAS Evaluation — 33 Fixtures (Per-Match Path)

RAG quality metrics on the deterministic per-match pipeline.

| Metric | Score |
|---|---|
| **Faithfulness** | **100%** |
| **Answer Correctness** | **100%** |
| **Context Recall** | **100%** |
| **Answer Semantic Similarity** | **100%** |
| Context Precision | 91.4% |
| Intent Accuracy | 100% |
| Latency p50 | 0.71 ms |
| Latency p95 | 0.87 ms |
| Latency p99 | 1.86 ms |

#### RAGAS by Category

| Category | Fixtures | Faithfulness | Correctness |
|---|---|---|---|
| Basic (runs, wickets) | 7 | 100% | 100% |
| Over Range | 13 | 100% | 100% |
| Edge Cases | 5 | 100% | 100% |
| Ambiguous Questions | 4 | 100% | 100% |
| Phase (powerplay, death) | 2 | 100% | 100% |
| Boundaries | 1 | 100% | 100% |
| Advanced | 1 | 100% | 100% |

---

### 3. CI Benchmark — 33 Fixtures

Run automatically on every push via GitHub Actions (Python 3.10 and 3.11).

| Metric | Score | CI Threshold |
|---|---|---|
| Exact Match | **100%** | ≥ 70% |
| Numeric Match | 100% | — |
| Intent Accuracy | 100% | — |
| Avg Latency | < 1 ms | — |

---

### Key Takeaway

> The system achieves **zero hallucination on numeric facts** because answers are computed from data, not generated by a language model. The optional LLM path (`use_llm=true`) formats the answer — but cannot change the number.

---

## Quick Start

### Prerequisites

- Python 3.10 or 3.11
- Docker (optional)
- HuggingFace token — only required for the LLM path (`use_llm=true`)

### Option 1: HuggingFace Space (No setup)

Visit: **[huggingface.co/spaces/namankhatak/cricket-qa-system](https://huggingface.co/spaces/namankhatak/cricket-qa-system)**

### Option 2: Docker

```bash
git clone https://github.com/Contributer00001/cricket-qa-system.git
cd cricket-qa-system

# Deterministic path — no HF token needed
docker build -t cricket-qa -f Dockerfile .
docker run -p 7860:7860 cricket-qa

# LLM path — requires token
docker run -p 7860:7860 -e HF_TOKEN="hf_..." cricket-qa
```

### Option 3: Local Development

```bash
git clone https://github.com/Contributer00001/cricket-qa-system.git
cd cricket-qa-system

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Build the IPL 2022 database (required for tournament queries)
python -m database.build_ipl_db

# Start the API
uvicorn service.app:app --host 0.0.0.0 --port 7860 --reload
```

Then open `http://localhost:7860` — the React frontend loads automatically.

---

## API Reference

Base URL (local): `http://localhost:7860`  
Base URL (HF Space): `https://namankhatak-cricket-qa-system.hf.space`

### `GET /healthz` — Liveness

```bash
curl http://localhost:7860/healthz
# {"status": "ok"}
```

### `GET /readyz` — Readiness

```bash
curl http://localhost:7860/readyz
# {"ready": false}   ← true only after LLM loads (use_llm path)
```

### `GET /ipl/stats` — Tournament Summary

```bash
curl http://localhost:7860/ipl/stats
```
```json
{
  "matches": 74,
  "deliveries": 15598,
  "players": 247,
  "top_scorer": {"player": "Jos Buttler", "runs": 863},
  "top_wicket_taker": {"player": "Yuzvendra Chahal", "wickets": 27},
  "db_loaded": true
}
```

### `POST /infer` — Question Answering

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | Natural language question (1–500 chars) |
| `use_llm` | bool | `false` | Enable LLM formatting (requires HF_TOKEN) |
| `max_tokens` | int | `32` | Max LLM output tokens (1–256) |
| `commentary` | object | `null` | Ball-by-ball commentary for per-match queries |

**Example — Tournament query (no commentary needed):**

```bash
curl -X POST http://localhost:7860/infer \
  -H "Content-Type: application/json" \
  -d '{"question": "Who scored the most runs in IPL 2022?"}'
```
```json
{
  "answer": "Jos Buttler (863 runs)",
  "intent": "top_run_scorers",
  "confidence": 1.0,
  "context_used": "Source: duckdb\nIntent: top_run_scorers",
  "llm_used": false,
  "status": "success",
  "data": [
    {"player": "Jos Buttler", "team": "RR", "runs": 863},
    ...
  ]
}
```

**Example — Per-match query (with commentary):**

```bash
curl -X POST http://localhost:7860/infer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many runs were scored in the powerplay?",
    "commentary": {
      "commentaries": [
        {"event": "ball", "over": 1, "run": 4, "four": true,  "six": false},
        {"event": "ball", "over": 1, "run": 0, "four": false, "six": false},
        {"event": "ball", "over": 6, "run": 6, "four": false, "six": true}
      ]
    }
  }'
```
```json
{
  "answer": "10",
  "intent": "powerplay",
  "confidence": 1.0,
  "llm_used": false,
  "status": "success"
}
```

**Commentary event schema:**

```json
{
  "event":  "ball" | "wicket",
  "over":   1,
  "run":    4,
  "four":   true,
  "six":    false,
  "batsman": "Player Name",   // optional
  "bowler":  "Player Name"    // optional
}
```

**Rate limit:** 60 requests/minute per IP (via slowapi).  
**Auth:** Set `X-API-Key` header when `API_KEY` env var is configured.

---

## Project Structure

```
cricket-qa-system/
│
├── agents/                         # Core pipeline agents
│   ├── query_router.py             # Intent classification (20+ intents, rule-based)
│   ├── stats_engine.py             # Deterministic per-match computation
│   ├── duckdb_engine.py            # IPL 2022 tournament queries
│   ├── orchestrator.py             # Pipeline coordinator & routing logic
│   ├── multi_agent.py              # LLM path: StatsTool, RetrieverAgent, AnalystAgent
│   ├── planner_agent.py            # Optional query planning
│   └── tool_router.py              # Tool dispatch abstraction
│
├── service/
│   └── app.py                      # FastAPI app (rate limiting, auth, static serving)
│
├── Frontend/                       # React + Vite + TypeScript UI
│   ├── src/
│   │   ├── app/components/         # UI components (IPL2022QA, VisionProCricketPlatform)
│   │   └── app/api/client.ts       # Typed API client
│   └── dist/                       # Pre-built static assets
│
├── database/
│   └── build_ipl_db.py             # Builds ipl2022.duckdb from raw CSV data
│
├── evaluation/
│   ├── benchmark.py                # 33-fixture CI benchmark (deterministic path)
│   ├── dataset.py                  # Ground-truth benchmark fixtures
│   ├── gold_set_eval.py            # 953-fixture gold set evaluation
│   ├── ragas_eval.py               # RAGAS faithfulness/recall evaluation
│   ├── Generate_gold_set.py        # Generates gold fixtures from DuckDB
│   ├── gold_sets/
│   │   ├── per_match_gold.json     # 941 per-match ground-truth fixtures
│   │   └── tournament_gold.json    # 12 tournament ground-truth fixtures
│   ├── gold_set_results.json       # Latest gold set evaluation results
│   └── ragas_results.json          # Latest RAGAS evaluation results
│
├── scripts/
│   ├── train_sft.py                # LoRA fine-tuning (SFT) script
│   ├── prepare_dataset.py          # Training data preparation
│   └── generate_qa.py              # Q&A pair generation from match data
│
├── tests/
│   ├── test_stats_engine.py        # StatsEngine unit tests
│   ├── test_query_router.py        # QueryRouter unit tests
│   └── test_inference.py           # FastAPI integration tests (62 total)
│
├── observability/
│   └── metrics.py                  # Prometheus metrics definitions
│
├── data/
│   └── sample_match.json           # 5-over demo match (47 runs, 3 wkts)
│
├── ipl2022.duckdb                  # Pre-built IPL 2022 database (2.9 MB)
├── Dockerfile                      # HF Spaces container (no torch, port 7860)
├── docker-compose.yml              # Local development stack
├── requirements.txt                # Full dependencies (includes torch, peft)
├── requirements-hf.txt             # Lightweight deps for HF Space deployment
└── conftest.py                     # pytest sys.path setup for CI
```

---

## Model Details

| Property | Value |
|---|---|
| Base model | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| Fine-tuning method | LoRA (PEFT) |
| Adapter repo | [Conqueror00001/qwen-ipl-lora](https://huggingface.co/Conqueror00001/qwen-ipl-lora) |
| Training | Supervised fine-tuning on cricket Q&A pairs |
| Grounding | Regex-based — model can only return values present in the context |
| Device | CPU / MPS (Apple Silicon) |
| Activation | `use_llm=true` in request body only |

The LLM is **not** used by default. The deterministic path (`use_llm=false`) answers 100% of benchmark questions correctly and is the default on the HuggingFace Space.

---

## Running Evaluations

### CI Benchmark (33 fixtures)
```bash
python -m evaluation.benchmark --data-file data/sample_match.json
```

### Gold Set Evaluation (953 fixtures)
```bash
# Requires ipl2022.duckdb
python -m evaluation.gold_set_eval \
  --db ipl2022.duckdb \
  --gold-dir evaluation/gold_sets/ \
  --output-json evaluation/gold_set_results.json
```

### RAGAS Evaluation (33 fixtures)
```bash
python -m evaluation.ragas_eval
```

### Run All Tests
```bash
pytest tests/ -v --cov=agents --cov=service --cov-report=term-missing
```

### Reproduce Results

The `gold_set_results.json` and `ragas_results.json` checked into this repo were produced with:
- Python 3.12.3
- DuckDB 1.3.0
- `ipl2022.duckdb` built from the Cricsheet IPL 2022 ball-by-ball dataset

---

## Training Your Own Model

```bash
# 1. Prepare training data from your match JSONs
python scripts/prepare_dataset.py

# 2. Generate Q&A pairs
python scripts/generate_qa.py

# 3. Fine-tune with LoRA (requires GPU or MPS)
python scripts/train_sft.py

# 4. Upload adapter to HuggingFace (optional)
#    Then set LORA_REPO=your-hf-username/your-adapter-repo
```

Training data format: ball-by-ball commentary JSON (same as the `/infer` request body).

---

## CI/CD

GitHub Actions runs on every push to `main` and `improve-rag`, and on all PRs to `main`.

### Pipeline

```
push / PR
   │
   ├─ test (Python 3.10)
   │   ├─ ruff check (lint)
   │   └─ pytest (62 tests)
   │
   ├─ test (Python 3.11)
   │   ├─ ruff check (lint)
   │   └─ pytest (62 tests)
   │
   └─ benchmark (Python 3.10, after tests pass)
       └─ exact-match ≥ 70% assertion
```

All tests run **without downloading any model** — `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` are set in CI to prevent accidental downloads.

### Test Coverage

| Test File | What it covers |
|---|---|
| `test_stats_engine.py` | All `StatsEngine` computations against sample match |
| `test_query_router.py` | Intent classification for 20+ intent types |
| `test_inference.py` | FastAPI endpoints (health, readiness, infer, edge cases) |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'agents'`
pytest needs the project root on `sys.path`. The `conftest.py` at the root handles this automatically. Make sure you run pytest from the project root:
```bash
cd cricket-qa-system
pytest tests/
```

### `PydanticUndefinedAnnotation: name 'Query' is not defined` (Python 3.11)
This was caused by `from __future__ import annotations` in `service/app.py` combined with slowapi's decorator wrapping. Fixed — the future import has been removed.

### HF Space build error: `/Frontend not found`
The Dockerfile uses `COPY static/ ./static/` — the pre-built frontend. The `Frontend/` source directory is not uploaded to the Space. If you need to rebuild the frontend: `cd Frontend && npm ci && npm run build`, then re-upload `Frontend/dist/` as `static/`.

### First LLM request takes 2–5 minutes
The model (~500 MB) is downloaded from HuggingFace on the first `use_llm=true` request. Subsequent requests are instant. Check `GET /readyz` to see when it's ready.

### Rate limit exceeded (429)
The API allows 60 requests/minute per IP. Spread your requests or remove `slowapi` from `requirements-hf.txt` for development.

---

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Contributer00001">
        <img src="https://github.com/Contributer00001.png" width="100px;" alt="Parth Giri"/>
        <br />
        <sub><b>Parth Giri</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/namankhatakdotcpp">
        <img src="https://github.com/namankhatakdotcpp.png" width="100px;" alt="Naman Khatak"/>
        <br />
        <sub><b>Naman Khatak</b></sub>
      </a>
    </td>
  </tr>
</table>

**Made with ❤️ by Parth Giri and Naman Khatak**

---

## Acknowledgments

- **Base Model**: [Qwen 2.5 by Alibaba Cloud](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Fine-tuning**: [PEFT by HuggingFace](https://github.com/huggingface/peft)
- **API Framework**: [FastAPI by Tiangolo](https://fastapi.tiangolo.com/)
- **Analytics Engine**: [DuckDB](https://duckdb.org/)
- **Data**: [Cricsheet](https://cricsheet.org/) IPL 2022 ball-by-ball data
- **Evaluation**: [RAGAS](https://github.com/explodinggradients/ragas)

## License

MIT License — see [LICENSE](LICENSE) for details.
