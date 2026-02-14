import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from agents.multi_agent import StatsTool, RetrieverAgent, AnalystAgent
import json


# ================= CONFIG =================
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_REPO = "Conqueror00001/qwen-ipl-lora"   # HF repo
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ================= FASTAPI APP =================
app = FastAPI(title="Cricket LoRA Inference API", version="1.0")

# ================= REQUEST MODEL =================
class Commentary(BaseModel):
    commentaries: list = []

class Query(BaseModel):
    question: str
    max_tokens: int = 100
    commentary: Commentary | None = None


# ================= GLOBAL MODEL STATE =================
tokenizer = None
model = None
analyst_agent = None
commentary_data_global = None


# ================= LOAD MODEL AT STARTUP =================
@app.on_event("startup")
def load_model():
    global tokenizer, model, analyst_agent, commentary_data_global
    print("🚀 Loading model at startup...")

    try:
        # Load tokenizer + base model
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32
        ).to(DEVICE)

        model = PeftModel.from_pretrained(base_model, LORA_REPO)
        model.eval()

        analyst_agent = AnalystAgent(model=model, tokenizer=tokenizer, device=DEVICE)

        # ⭐ LOAD IPL COMMENTARY FILE
        MATCH_FILE = "data/Indian_Premier_League_2022-03-26/match_innings_commentary/innings_1_Chennai_Super_Kings_vs_Kolkata_Knight_Riders_Match_1_match_innings_1_commentary.json"

        with open(MATCH_FILE) as f:
            commentary_data_global = json.load(f)

        print("✅ Model + Commentary loaded successfully!")

    except Exception as e:
        print("❌ MODEL LOAD FAILED:", e)



# ================= HEALTH =================
@app.get("/healthz")
def health():
    return {"status": "ok"}

# ================= READY =================
@app.get("/readyz")
def ready():
    if model is None:
        return {"ready": False}
    return {"ready": True}

# ================= INFERENCE =================
@app.post("/infer")
def infer(q: Query):
    global commentary_data_global

    if analyst_agent is None:
        return {"error": "Model not loaded"}

    if commentary_data_global is None:
        return {"error": "Commentary not loaded"}

    # 1️⃣ Deterministic Stats Tool
    stats_tool = StatsTool(commentary_data_global)

    # 2️⃣ Retriever Agent
    retriever = RetrieverAgent(stats_tool)
    context = retriever.act(q.question)

    # 3️⃣ Analyst Agent
    answer = analyst_agent.act(question=q.question, context=context)

    return {
        "answer": answer,
        "context_used": context
    }


