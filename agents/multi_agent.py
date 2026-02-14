#!/usr/bin/env python3
# agents/multi_agent.py

from __future__ import annotations
from typing import Dict, List, Any
import torch
import logging
import re

# ============================================================
# Logging (production friendly)
# ============================================================

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================
# StatsTool (Deterministic Tool Layer)
# ============================================================

class StatsTool:
    """
    Deterministic cricket statistics tool.
    No ML, no hallucination. Pure computation.
    """

    def __init__(self, commentary_data: Dict[str, Any]):
        self.commentaries: List[dict] = commentary_data.get("commentaries", [])

        # Keep all relevant events
        self.events = [
            c for c in self.commentaries
            if c.get("event") in {"ball", "wicket"}
        ]

    def _max_over(self) -> int:
        if not self.events:
            return 0
        return max(int(c.get("over", 0)) for c in self.events)

    def runs_last_n_overs(self, n: int) -> int:
        max_over = self._max_over()
        return sum(
            int(c.get("run", 0))
            for c in self.events
            if int(c.get("over", 0)) >= max_over - n + 1
        )

    def total_runs(self) -> int:
        return sum(int(c.get("run", 0)) for c in self.events)

    def total_wickets(self) -> int:
        return sum(1 for c in self.events if c.get("event") == "wicket")

    def total_fours(self) -> int:
        return sum(1 for c in self.events if c.get("four") is True)

    def total_sixes(self) -> int:
        return sum(1 for c in self.events if c.get("six") is True)


# ============================================================
# RetrieverAgent (Structured Context Generator)
# ============================================================

class RetrieverAgent:
    """
    Converts StatsTool outputs into LLM-readable factual context.
    """

    def __init__(self, tool: StatsTool):
        self.tool = tool

    def act(self, question: str) -> str:
        # Question currently unused, but kept for extensibility
        context = (
            f"Context:\n"
            f"- Total runs: {self.tool.total_runs()}\n"
            f"- Runs in last 5 overs: {self.tool.runs_last_n_overs(5)}\n"
            f"- Total wickets: {self.tool.total_wickets()}\n"
            f"- Fours: {self.tool.total_fours()}\n"
            f"- Sixes: {self.tool.total_sixes()}\n"
        )

        logger.info("Retriever context generated")
        return context


# ============================================================
# AnalystAgent (LLM Reasoning Layer)
# ============================================================

class AnalystAgent:
    """
    LLM-backed grounded reasoning agent.

    STRICT DESIGN:
    - No hallucinations
    - Uses only provided context
    - Deterministic decoding
    """

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def act(self, question: str, context: str) -> str:
        """
        Generate grounded answer from structured context.
        
        STRICT GROUNDING:
        - Only accepts numbers present in context
        - No hallucinations allowed
        - Deterministic extraction
        """
        
        prompt = f"""You are a cricket statistics assistant.

RULES (MANDATORY):
1. Use ONLY numbers from the context below.
2. Output ONLY the numeric answer. No explanations.
3. If the answer is not in the context, output exactly: Not available

Question: {question}

Context:
{context}

Answer (number only):""".strip()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                num_beams=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract answer after prompt
        if "Answer (number only):" in decoded:
            raw_answer = decoded.split("Answer (number only):")[-1].strip()
        else:
            # Fallback: take everything after prompt
            raw_answer = decoded[len(prompt):].strip()
        
        # Clean up (take first line only)
        raw_answer = raw_answer.split("\n")[0].strip()
        
        # Extract numbers from model output
        model_numbers = re.findall(r"\d+", raw_answer)
        
        # Extract numbers from context (for grounding)
        context_numbers = re.findall(r"\d+", context)
        
        # STRICT GROUNDING: only accept numbers present in context
        if model_numbers:
            for num in model_numbers:
                if num in context_numbers:
                    logger.info(f"✓ Question: {question}")
                    logger.info(f"✓ Answer: {num} (grounded in context)")
                    return num
            logger.warning(f"⚠ Model output '{raw_answer}' contains numbers not in context")
        
        # Check if model explicitly said unavailable
        if "not available" in raw_answer.lower():
            logger.info(f"✓ Question: {question}")
            logger.info(f"✓ Answer: Not available (model stated explicitly)")
            return "The information is not available in the provided data."
        
        # Default fallback
        logger.warning(f"⚠ Could not extract grounded answer from: {raw_answer}")
        logger.info(f"✓ Question: {question}")
        logger.info(f"✓ Answer: Not available (fallback)")
        return "The information is not available in the provided data."