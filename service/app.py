# service/app.py
import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from agents.orchestrator import CricketOrchestrator

# ── Config ─────────────────────────────────────────────────────────────────

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LORA_REPO  = os.environ.get("LORA_REPO",  "Conqueror00001/qwen-ipl-lora")
API_KEY    = os.environ.get("API_KEY")       # None → auth disabled (dev mode)

# ── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Rate limiter (optional — degrades gracefully if slowapi not installed) ─

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address)
    _RATE_LIMIT_AVAILABLE = True
except ImportError:
    logger.warning("slowapi not installed — rate limiting disabled")

    class _NoopLimiter:  # type: ignore[no-redef]
        def limit(self, *_a, **_kw):
            def decorator(f):
                return f
            return decorator

    limiter = _NoopLimiter()  # type: ignore[assignment]
    _RATE_LIMIT_AVAILABLE = False

# ── API-key auth ───────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(x_api_key: Optional[str] = Depends(_api_key_header)) -> None:
    """
    If API_KEY is configured, every /infer request must carry the matching
    X-API-Key header.  When API_KEY is unset, auth is bypassed (dev mode).
    """
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("service starting — no-LLM path ready immediately")
    yield
    logger.info("service shutting down — draining in-flight requests")


# ── /api prefix middleware ─────────────────────────────────────────────────
# The frontend build calls /api/infer, /api/ipl/stats, etc.
# This middleware strips the /api prefix so FastAPI routes (/infer, etc.) match.

class StripApiPrefixMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/"):
            request.scope["path"] = request.url.path[4:]  # /api/infer → /infer
        return await call_next(request)


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(title="Cricket QA API", version="2.0", lifespan=lifespan)

app.add_middleware(StripApiPrefixMiddleware)

if _RATE_LIMIT_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed — /metrics unavailable")

# ── Request / response models ──────────────────────────────────────────────

class Commentary(BaseModel):
    commentaries: list = []

class Query(BaseModel):
    question:  str = Field(..., min_length=1, max_length=500)
    max_tokens: int = Field(default=32, ge=1, le=256)
    use_llm:   bool = False
    commentary: Optional[Commentary] = None

# ── Global state ───────────────────────────────────────────────────────────

_lock = threading.Lock()
_state: dict = {
    "orchestrator":      None,
    "default_commentary": None,
    "loaded":            False,
}

# ── Model loading ──────────────────────────────────────────────────────────

def _load() -> None:
    """Load model, LoRA adapter, and default commentary. Called once."""
    # Heavy imports deferred so test collection and HF Spaces work without torch/peft.
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "LLM dependencies (torch, transformers, peft) not installed. "
            "This build supports the deterministic path only (use_llm=false)."
        ) from exc

    from agents.multi_agent import AnalystAgent

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    hf_token = os.environ.get("HF_TOKEN")
    logger.info("HF_TOKEN: %s", "set" if hf_token else "not set")

    logger.info("Loading tokenizer from %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)

    logger.info("Loading base model from %s (device=%s)", BASE_MODEL, device)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        token=hf_token,
    ).to(device)

    logger.info("Loading LoRA adapter from %s", LORA_REPO)
    model = PeftModel.from_pretrained(base, LORA_REPO, token=hf_token)
    model.eval()

    analyst = AnalystAgent(model=model, tokenizer=tokenizer, device=device)
    _state["orchestrator"] = CricketOrchestrator(analyst=analyst)

    match_file = os.environ.get("DEFAULT_MATCH_FILE", "data/sample_match.json")
    if os.path.exists(match_file):
        with open(match_file) as f:
            _state["default_commentary"] = json.load(f)
        logger.info("Default commentary loaded from %s", match_file)
    else:
        logger.warning("Default commentary file not found: %s", match_file)
        _state["default_commentary"] = {"commentaries": []}

    _state["loaded"] = True
    logger.info("Model ready")


def ensure_loaded() -> None:
    """Double-checked locking — safe under concurrent threads."""
    if _state["loaded"]:
        return
    with _lock:
        if _state["loaded"]:
            return
        t0 = time.time()
        _load()
        logger.info("Model loaded in %.1fs", time.time() - t0)


def _get_no_llm_orchestrator() -> CricketOrchestrator:
    return CricketOrchestrator(analyst=None)  # db_path defaults to "ipl2022.duckdb"


# ── Helpers ────────────────────────────────────────────────────────────────

def _load_default_commentary() -> dict:
    match_file = os.environ.get("DEFAULT_MATCH_FILE", "data/sample_match.json")
    if os.path.exists(match_file):
        with open(match_file) as f:
            return json.load(f)
    return {"commentaries": []}


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/ipl/stats")
def ipl_stats():
    """
    Returns a summary of the IPL 2022 DuckDB dataset.
    Powers the frontend live-stats header badge.
    Returns db_loaded: false with zeroes if ipl2022.duckdb does not exist.
    """
    db_path = "ipl2022.duckdb"
    empty = {
        "matches": 0,
        "deliveries": 0,
        "players": 0,
        "top_scorer": {"player": "", "runs": 0},
        "top_wicket_taker": {"player": "", "wickets": 0},
        "db_loaded": False,
    }
    if not os.path.exists(db_path):
        return empty
    try:
        import duckdb as _duckdb
        con = _duckdb.connect(db_path, read_only=True)
        matches = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        deliveries = con.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
        players = con.execute("SELECT COUNT(*) FROM tournament_batting").fetchone()[0]
        top_bat = con.execute(
            "SELECT player_name, runs FROM tournament_batting ORDER BY runs DESC LIMIT 1"
        ).fetchone()
        top_bowl = con.execute(
            "SELECT player_name, wickets FROM tournament_bowling ORDER BY wickets DESC LIMIT 1"
        ).fetchone()
        con.close()
        return {
            "matches": int(matches),
            "deliveries": int(deliveries),
            "players": int(players),
            "top_scorer": {
                "player": top_bat[0] if top_bat else "",
                "runs": int(top_bat[1]) if top_bat else 0,
            },
            "top_wicket_taker": {
                "player": top_bowl[0] if top_bowl else "",
                "wickets": int(top_bowl[1]) if top_bowl else 0,
            },
            "db_loaded": True,
        }
    except Exception as exc:
        logger.exception("DuckDB stats query failed: %s", exc)
        return empty


@app.get("/")
def root():
    """Serve the frontend if built, otherwise return API info."""
    index = os.path.join("static", "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {
        "service": "Cricket QA API",
        "version": "2.0",
        "endpoints": {
            "GET  /healthz": "liveness",
            "GET  /health":  "liveness (alias)",
            "GET  /readyz":  "readiness",
            "POST /infer":   "question answering",
            "GET  /metrics": "Prometheus metrics",
        },
    }


@app.get("/healthz")
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/readyz")
def ready():
    return {"ready": _state["loaded"]}


@app.post("/infer")
@limiter.limit("60/minute")
async def infer(
    request: Request,
    q: Query,
    _: None = Depends(verify_api_key),
):
    """
    Main inference endpoint.

    - No-LLM path (default, use_llm=false): deterministic, < 5ms.
    - LLM path (use_llm=true): loads model lazily, 30s timeout enforced.
    Rate limit: 60 requests / minute per IP.
    Auth: X-API-Key header required when API_KEY env var is set.
    """
    commentary_data: Optional[dict] = (
        q.commentary.model_dump()
        if q.commentary and q.commentary.commentaries
        else None
    )

    if q.use_llm:
        # ── LLM path: run load + inference in a thread (non-blocking) ──
        def _llm_pipeline() -> dict:
            ensure_loaded()
            orch = _state["orchestrator"]
            if orch is None:
                raise RuntimeError("Model failed to load")
            data = commentary_data if commentary_data is not None else (
                _state["default_commentary"] or _load_default_commentary()
            )
            if not data or not data.get("commentaries"):
                raise ValueError("No commentary data available")
            return orch.answer(q.question, data)

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_llm_pipeline),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("LLM inference timed out question=%r", q.question)
            return {
                "answer":      "Request timed out",
                "intent":      "timeout",
                "confidence":  0.0,
                "context_used": "",
                "llm_used":    True,
                "status":      "timeout",
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("LLM pipeline error: %s", exc)
            raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc

    else:
        # ── Deterministic path: run synchronously, no timeout needed ──
        data = commentary_data if commentary_data is not None else _load_default_commentary()
        if not data or not data.get("commentaries"):
            raise HTTPException(status_code=400, detail="No commentary data available")

        orchestrator = _get_no_llm_orchestrator()
        try:
            result = orchestrator.answer(q.question, data)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Inference error: %s", exc)
            raise HTTPException(status_code=500, detail="Inference failed") from exc

    return {
        "answer":       result["answer"],
        "intent":       result["intent"],
        "confidence":   result["confidence"],
        "context_used": result["context_used"],
        "llm_used":     result["llm_used"],
        "data":         result.get("data"),
        "status":       "success",
    }


# ── Static frontend (SPA) ──────────────────────────────────────────────────
# Mounted last so API routes above take priority.
# /assets/* → static/assets/* (JS, CSS, images)
# Any other unmatched path → index.html (React Router handles client-side routing)

_STATIC_DIR = "static"

if os.path.isdir(_STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(_STATIC_DIR, "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        """Fall through to index.html for all unmatched routes (SPA routing)."""
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"))
