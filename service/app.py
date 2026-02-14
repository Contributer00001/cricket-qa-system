# service/app.py
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from agents.multi_agent import StatsTool, RetrieverAgent, AnalystAgent
import json
import logging
import time

# ================= CONFIG =================
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_REPO = "Conqueror00001/qwen-ipl-lora"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================= FASTAPI APP =================
app = FastAPI(title="Cricket LoRA Inference API", version="1.0")

# ================= REQUEST MODELS =================
class Commentary(BaseModel):
    commentaries: list = []

class Query(BaseModel):
    question: str  # Standardized field name (not "prompt")
    max_tokens: int = 100
    commentary: Commentary | None = None

# ================= GLOBAL STATE =================
_model_state = {
    "tokenizer": None,
    "model": None,
    "analyst_agent": None,
    "commentary_data": None,
    "loaded": False,
    "loading": False
}

# ================= LAZY LOAD HELPER =================
def ensure_model_loaded():
    """
    Lazy load model on first /infer request (not at startup).
    This prevents Docker from downloading model when container starts.
    """
    
    if _model_state["loaded"]:
        logger.info("Model already loaded, skipping")
        return  # Already loaded
    
    if _model_state["loading"]:
        # Another request is loading, wait for it
        logger.info("Model loading in progress, waiting...")
        for i in range(60):  # Wait up to 60 seconds
            time.sleep(1)
            if _model_state["loaded"]:
                logger.info("Model loaded by another request")
                return
        raise RuntimeError("Model loading timeout after 60 seconds")
    
    # Mark as loading to prevent concurrent loads
    _model_state["loading"] = True
    
    try:
        logger.info("=" * 60)
        logger.info("🔄 LAZY LOADING MODEL (first request)")
        logger.info("=" * 60)
        start_time = time.time()
        
        # Get HF token from environment
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            logger.info(f"✓ Using HF_TOKEN: {hf_token[:10]}...")
        else:
            logger.warning("⚠ HF_TOKEN not set - loading from public repo only")
        
        # Load tokenizer
        logger.info(f"📥 Loading tokenizer from {BASE_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            token=hf_token
        )
        logger.info("✓ Tokenizer loaded")
        
        # Load base model
        logger.info(f"📥 Loading base model from {BASE_MODEL}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            token=hf_token
        ).to(DEVICE)
        logger.info(f"✓ Base model loaded on device: {DEVICE}")
        
        # Load LoRA adapter
        logger.info(f"📥 Loading LoRA adapter from {LORA_REPO}...")
        model = PeftModel.from_pretrained(
            base_model,
            LORA_REPO,
            token=hf_token
        )
        model.eval()
        logger.info("✓ LoRA adapter loaded")
        
        # Create analyst agent
        analyst_agent = AnalystAgent(
            model=model,
            tokenizer=tokenizer,
            device=DEVICE
        )
        logger.info("✓ AnalystAgent created")
        
        # Load default commentary
        # Allow data path to be configured via environment variable
        MATCH_FILE = os.environ.get(
            "DEFAULT_MATCH_FILE",
            "data/sample_match.json"  # Uses sample data by default
        )        
        
        if os.path.exists(MATCH_FILE):
            with open(MATCH_FILE) as f:
                commentary_data = json.load(f)
            logger.info(f"✓ Default commentary loaded from {MATCH_FILE}")
        else:
            logger.warning(f"⚠ Default commentary file not found: {MATCH_FILE}")
            commentary_data = {"commentaries": []}
        
        # Update global state
        _model_state["tokenizer"] = tokenizer
        _model_state["model"] = model
        _model_state["analyst_agent"] = analyst_agent
        _model_state["commentary_data"] = commentary_data
        _model_state["loaded"] = True
        
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"✅ MODEL LOADED in {elapsed:.2f}s")
        logger.info("=" * 60)
        
    except Exception as e:
        _model_state["loading"] = False
        logger.error("=" * 60)
        logger.error(f"❌ MODEL LOAD FAILED: {e}")
        logger.error("=" * 60)
        raise
    finally:
        _model_state["loading"] = False

# ================= ENDPOINTS =================

@app.get("/")
def root():
    """Root endpoint with API info"""
    return {
        "service": "Cricket LoRA Inference API",
        "version": "1.0",
        "endpoints": {
            "health": "/healthz",
            "ready": "/readyz",
            "infer": "/infer (POST)"
        }
    }

@app.get("/healthz")
def health():
    """
    Basic health check - always returns OK.
    Does NOT require model to be loaded.
    """
    return {"status": "ok"}

@app.get("/readyz")
def ready():
    """
    Readiness check - returns true only after model is loaded.
    Used by Kubernetes/Docker to know when service is ready.
    """
    return {
        "ready": _model_state["loaded"],
        "loading": _model_state["loading"]
    }

@app.post("/infer")
def infer(q: Query):
    """
    Main inference endpoint.
    Loads model lazily on first call.
    
    Request body:
    {
        "question": "How many runs in the last 5 overs?",
        "max_tokens": 100,  // optional
        "commentary": {...}  // optional
    }
    """
    try:
        # Lazy load model if not loaded yet
        ensure_model_loaded()
        
        analyst_agent = _model_state["analyst_agent"]
        default_commentary = _model_state["commentary_data"]
        
        if analyst_agent is None:
            raise HTTPException(
                status_code=503,
                detail="Model failed to load"
            )
        
        # Use request commentary if provided, otherwise use default
        if q.commentary and q.commentary.commentaries:
            commentary_data = q.commentary.dict()
            logger.info("Using commentary from request")
        else:
            commentary_data = default_commentary
            logger.info("Using default commentary")
        
        if not commentary_data or not commentary_data.get("commentaries"):
            raise HTTPException(
                status_code=400,
                detail="No commentary data available"
            )
        
        # 1️⃣ Build deterministic stats
        logger.info(f"Question: {q.question}")
        stats_tool = StatsTool(commentary_data)
        
        # 2️⃣ Generate context
        retriever = RetrieverAgent(stats_tool)
        context = retriever.act(q.question)
        logger.info(f"Context: {context.strip()}")
        
        # 3️⃣ Get grounded answer
        answer = analyst_agent.act(question=q.question, context=context)
        logger.info(f"Answer: {answer}")
        
        return {
            "answer": answer,
            "context_used": context,
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )