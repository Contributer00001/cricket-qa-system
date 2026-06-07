"""
Legacy pipeline components — kept for backward compatibility with the
existing LoRA adapter (Conqueror00001/qwen-ipl-lora).

New code should use:
  agents.stats_engine.StatsEngine
  agents.query_router.QueryRouter
  agents.orchestrator.CricketOrchestrator
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ── StatsTool ──────────────────────────────────────────────────────────────

class StatsTool:
    """
    Deterministic cricket statistics tool.

    Retained alongside StatsEngine to support existing training scripts.
    Prefer StatsEngine for new work — it uses pandas and supports
    arbitrary over ranges.
    """

    def __init__(self, commentary_data: Dict[str, Any]) -> None:
        self.commentaries: List[dict] = commentary_data.get("commentaries", [])
        self.events = [
            c for c in self.commentaries
            if c.get("event") in {"ball", "wicket"}
        ]

    def _max_over(self) -> int:
        return max((int(c.get("over", 0)) for c in self.events), default=0)

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

    def wickets_last_n_overs(self, n: int) -> int:
        max_over = self._max_over()
        return sum(
            1 for c in self.events
            if c.get("event") == "wicket"
            and int(c.get("over", 0)) >= max_over - n + 1
        )

    def total_fours(self) -> int:
        return sum(1 for c in self.events if c.get("four") is True)

    def total_sixes(self) -> int:
        return sum(1 for c in self.events if c.get("six") is True)


# ── RetrieverAgent ─────────────────────────────────────────────────────────

class RetrieverAgent:
    """
    Converts StatsTool outputs into an LLM-readable context string.

    Uses the question to select relevant stats rather than always
    returning all five metrics.
    """

    def __init__(self, tool: StatsTool) -> None:
        self.tool = tool

    def act(self, question: str) -> str:
        q = question.lower()

        lines = ["Context:"]
        lines.append(f"- Total runs: {self.tool.total_runs()}")
        lines.append(f"- Total wickets: {self.tool.total_wickets()}")

        if re.search(r"last\s+(\d+)\s+over", q):
            m = re.search(r"last\s+(\d+)\s+over", q)
            n = int(m.group(1)) if m else 5
            lines.append(f"- Runs in last {n} overs: {self.tool.runs_last_n_overs(n)}")
            lines.append(f"- Wickets in last {n} overs: {self.tool.wickets_last_n_overs(n)}")
        else:
            lines.append(f"- Runs in last 5 overs: {self.tool.runs_last_n_overs(5)}")

        if re.search(r"\bfour\b", q):
            lines.append(f"- Fours: {self.tool.total_fours()}")
        if re.search(r"\bsix", q):
            lines.append(f"- Sixes: {self.tool.total_sixes()}")
        if not re.search(r"\bfour\b|\bsix", q):
            lines.append(f"- Fours: {self.tool.total_fours()}")
            lines.append(f"- Sixes: {self.tool.total_sixes()}")

        context = "\n".join(lines)
        logger.info("context_generated question=%r", question)
        return context


# ── AnalystAgent ───────────────────────────────────────────────────────────

class AnalystAgent:
    """
    LLM-backed grounded reasoning agent.

    Prompt format matches the LoRA training data from prepare_dataset.py.
    Grounding is two-tier:
      1. If an expected_value is passed (the stat we actually computed),
         accept the model answer only if it matches that value exactly.
      2. Fallback: accept any number present anywhere in the context.
    This prevents the model from hallucinating a valid-looking number that
    happens to appear in the context but is the wrong answer.
    """

    def __init__(self, model, tokenizer, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def act(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 32,
        expected_value: Optional[str] = None,
    ) -> str:
        prompt = (
            "Instruction:\n"
            "You are a cricket statistics assistant. Answer the question using "
            "ONLY the numbers provided in the context below. "
            "If the answer is not in the context, say \"not available\".\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Response:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract everything after the last "Response:" marker
        if "Response:" in decoded:
            raw = decoded.split("Response:")[-1].strip()
        else:
            raw = decoded[len(prompt):].strip()

        raw = raw.split("\n")[0].strip()

        # --- Tier-1 grounding: exact match against the computed answer ------
        model_nums = re.findall(r"\d+(?:\.\d+)?", raw)

        if expected_value is not None and model_nums:
            if expected_value in model_nums:
                logger.info(
                    "answer=%r question=%r (key-grounded)", expected_value, question
                )
                return expected_value
            # Model produced a different number — fall through to tier-2 or fail
            logger.warning(
                "key_grounding_miss expected=%r raw=%r question=%r",
                expected_value, raw, question,
            )

        # --- Tier-2 grounding: any number present in the context -----------
        context_nums = re.findall(r"\d+(?:\.\d+)?", context)
        for num in model_nums:
            if num in context_nums:
                logger.info("answer=%r question=%r (context-grounded)", num, question)
                return num

        if "not available" in raw.lower():
            return "The information is not available in the provided data."

        logger.warning("grounding_failed raw=%r question=%r", raw, question)
        return "The information is not available in the provided data."
