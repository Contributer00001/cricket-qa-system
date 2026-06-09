"""
RAGAS Evaluation for the Cricket QA System
===========================================

This module evaluates the Cricket QA pipeline using RAGAS metrics.

Two evaluation modes
--------------------
1. Non-LLM mode (always runs):
   Custom implementations of RAGAS-equivalent metrics that need no
   external API:
   - faithfulness_score    : answer number present in computed context
   - context_recall_score  : ground-truth number present in context
   - context_precision_score: relevant context lines / total lines
   - answer_correctness    : exact-match + numeric tolerance
   - answer_semantic_sim   : numeric proximity to ground truth

2. LLM mode (optional, activates when GOOGLE_API_KEY is set):
   Full ragas.evaluate() pipeline backed by Gemini as judge:
   - Faithfulness       : LLM verifies answer is grounded in context
   - Context Precision  : LLM judges if each context chunk is relevant
   - Context Recall     : LLM checks if context covers ground truth

Usage
-----
    python3.12 -m evaluation.ragas_eval
    python3.12 -m evaluation.ragas_eval --data-file data/sample_match.json
    python3.12 -m evaluation.ragas_eval --output-json evaluation/ragas_results.json

    # With LLM judge (requires GOOGLE_API_KEY in .env or env):
    source .env && python3.12 -m evaluation.ragas_eval --llm
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from agents.orchestrator import CricketOrchestrator
from evaluation.dataset import BENCHMARK, BenchmarkItem

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)  # suppress noisy ragas logs


# ── Non-LLM metric helpers ────────────────────────────────────────────────

def _nums(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def faithfulness_score(answer: str, context: str) -> float:
    """
    Faithfulness: are all numbers in the answer present in the context?
    For a grounded numeric QA system this should always be 1.0.
    """
    ans_nums = _nums(answer)
    if not ans_nums:
        return 1.0  # non-numeric "not available" answers are acceptable
    ctx_nums = _nums(context)
    return len(ans_nums & ctx_nums) / len(ans_nums)


def context_recall_score(context: str, ground_truth: str) -> float:
    """
    Context Recall: does the context contain every number in the ground truth?
    """
    gt_nums = _nums(ground_truth)
    if not gt_nums:
        return 1.0
    ctx_nums = _nums(context)
    return len(gt_nums & ctx_nums) / len(gt_nums)


def context_precision_score(context: str, question: str, ground_truth: str) -> float:
    """
    Context Precision: fraction of context lines that are directly relevant
    to answering the question.

    A line is "relevant" if it contains at least one ground-truth number or
    matches a keyword from the question.
    """
    lines = [line for line in context.splitlines() if line.startswith("-")]
    if not lines:
        return 1.0

    gt_nums = _nums(ground_truth)
    q_words = set(re.findall(r"\b\w{4,}\b", question.lower()))

    relevant = 0
    for line in lines:
        line_nums = _nums(line)
        line_words = set(re.findall(r"\b\w{4,}\b", line.lower()))
        if (gt_nums & line_nums) or (q_words & line_words):
            relevant += 1

    return relevant / len(lines)


def answer_correctness_score(answer: str, ground_truth: str, tol: float = 0.05) -> float:
    """
    Combined correctness: exact string match (1.0) or numeric tolerance
    match (0.8) or no match (0.0).
    """
    if answer.strip() == ground_truth.strip():
        return 1.0
    try:
        if abs(float(answer) - float(ground_truth)) <= tol:
            return 0.8
    except (ValueError, TypeError):
        pass
    return 0.0


def answer_semantic_sim(answer: str, ground_truth: str) -> float:
    """
    Numeric proximity: 1 - normalised absolute error.
    Falls back to 0 for non-numeric answers that differ from ground truth.
    """
    if answer.strip() == ground_truth.strip():
        return 1.0
    try:
        a, g = float(answer), float(ground_truth)
        denom = max(abs(a), abs(g), 1e-9)
        return max(0.0, 1.0 - abs(a - g) / denom)
    except (ValueError, TypeError):
        return 0.0


# ── Per-sample dataclass ──────────────────────────────────────────────────

@dataclass
class RAGASResult:
    question: str
    category: str
    expected_intent: str
    predicted_intent: str
    ground_truth: str
    answer: str
    context: str
    latency_ms: float
    # non-LLM metrics
    faithfulness: float
    context_recall: float
    context_precision: float
    answer_correctness: float
    answer_semantic_sim: float
    # LLM metrics (None when not available)
    llm_faithfulness: Optional[float] = None
    llm_answer_relevancy: Optional[float] = None
    llm_context_precision: Optional[float] = None
    llm_context_recall: Optional[float] = None
    llm_answer_correctness: Optional[float] = None


# ── Non-LLM evaluation ────────────────────────────────────────────────────

def run_non_llm_evaluation(
    commentary_data: dict,
    items: Optional[list[BenchmarkItem]] = None,
) -> list[RAGASResult]:
    orchestrator = CricketOrchestrator(analyst=None)
    items = items or BENCHMARK
    results: list[RAGASResult] = []

    for item in items:
        t0 = time.perf_counter()
        out = orchestrator.answer(item.question, commentary_data)
        elapsed = (time.perf_counter() - t0) * 1000

        answer  = str(out["answer"]).strip()
        context = out["context_used"]

        results.append(RAGASResult(
            question=item.question,
            category=item.category,
            expected_intent=item.intent,
            predicted_intent=out["intent"],
            ground_truth=item.expected_answer,
            answer=answer,
            context=context,
            latency_ms=round(elapsed, 2),
            faithfulness=faithfulness_score(answer, context),
            context_recall=context_recall_score(context, item.expected_answer),
            context_precision=context_precision_score(context, item.question, item.expected_answer),
            answer_correctness=answer_correctness_score(answer, item.expected_answer),
            answer_semantic_sim=answer_semantic_sim(answer, item.expected_answer),
        ))

    return results


# ── LLM evaluation (Claude as judge) ─────────────────────────────────────

def run_llm_evaluation(results: list[RAGASResult]) -> list[RAGASResult]:
    """
    Augments non-LLM results with RAGAS LLM-based metrics using Google
    Gemini (via Google AI Studio) as the judge.

    Metrics used (all LLM-only, no OpenAI embeddings needed):
      - faithfulness      : answer claims are grounded in context
      - context_precision : retrieved context is relevant to the question
      - context_recall    : context covers all ground-truth information

    answer_relevancy and answer_correctness are excluded because they
    require vector embeddings for cosine similarity — those are already
    covered by our deterministic answer_correctness and answer_semantic_sim.

    Requires: GOOGLE_API_KEY environment variable (Google AI Studio key).
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.info("GOOGLE_API_KEY not set — skipping LLM evaluation")
        return results

    try:
        from datasets import Dataset
        from langchain_community.embeddings import FakeEmbeddings
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas import RunConfig, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as e:
        logger.warning("LLM evaluation unavailable — missing package: %s", e)
        return results

    print("\n[RAGAS] Running LLM evaluation with gemini-2.0-flash-lite (Google AI Studio)...")
    print("        Metrics: faithfulness · context_precision · context_recall")
    print("        (sequential, rate-limit safe for free-tier keys)")

    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=api_key,
            temperature=0,
            request_timeout=120,
        )
    )
    # FakeEmbeddings satisfies RAGAS's embeddings initialisation without
    # requiring any paid embeddings API. The three LLM-only metrics we use
    # never actually invoke the embeddings object.
    embeddings = LangchainEmbeddingsWrapper(FakeEmbeddings(size=384))

    # Lower concurrency + higher timeout — essential for free-tier rate limits
    run_cfg = RunConfig(timeout=120, max_retries=3, max_wait=30, max_workers=1)

    dataset = Dataset.from_list([
        {
            "question":     r.question,
            "contexts":     [r.context],
            "answer":       r.answer,
            "ground_truth": r.ground_truth,
        }
        for r in results
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = evaluate(
            dataset=dataset,
            metrics=[faithfulness, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            run_config=run_cfg,
            raise_exceptions=False,
            show_progress=True,
        )

    scores_df = scores.to_pandas()

    for i, r in enumerate(results):
        row = scores_df.iloc[i]
        r.llm_faithfulness       = float(row.get("faithfulness",      float("nan")))
        r.llm_context_precision  = float(row.get("context_precision", float("nan")))
        r.llm_context_recall     = float(row.get("context_recall",    float("nan")))
        # Not computed (require vector embeddings) — left as None
        r.llm_answer_relevancy   = None
        r.llm_answer_correctness = None

    return results


# ── Aggregation & reporting ───────────────────────────────────────────────

def aggregate(results: list[RAGASResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    def avg(f):
        vals = [v for r in results if (v := f(r)) is not None and v == v]  # skip NaN
        return round(sum(vals) / len(vals), 4) if vals else None

    by_cat: dict[str, dict] = {}
    for cat in sorted({r.category for r in results}):
        sub = [r for r in results if r.category == cat]
        by_cat[cat] = {
            "n":                   len(sub),
            "faithfulness":        avg(lambda r: r.faithfulness),
            "context_recall":      avg(lambda r: r.context_recall),
            "context_precision":   avg(lambda r: r.context_precision),
            "answer_correctness":  avg(lambda r: r.answer_correctness),
        }

    lat = sorted(r.latency_ms for r in results)
    p50 = lat[n // 2]
    p95 = lat[int(n * 0.95)]
    p99 = lat[min(int(n * 0.99), n - 1)]

    base = {
        "n":                      n,
        "faithfulness":           avg(lambda r: r.faithfulness),
        "context_recall":         avg(lambda r: r.context_recall),
        "context_precision":      avg(lambda r: r.context_precision),
        "answer_correctness":     avg(lambda r: r.answer_correctness),
        "answer_semantic_sim":    avg(lambda r: r.answer_semantic_sim),
        "intent_accuracy":        round(sum(r.expected_intent == r.predicted_intent for r in results) / n, 4),
        "latency_p50_ms":         p50,
        "latency_p95_ms":         p95,
        "latency_p99_ms":         p99,
        "by_category":            by_cat,
    }

    # Add LLM metrics if computed
    llm_keys = ("llm_faithfulness", "llm_answer_relevancy", "llm_context_precision",
                "llm_context_recall", "llm_answer_correctness")
    if any(getattr(results[0], k) is not None for k in llm_keys):
        base["llm_faithfulness"]       = avg(lambda r: r.llm_faithfulness)
        base["llm_answer_relevancy"]   = avg(lambda r: r.llm_answer_relevancy)
        base["llm_context_precision"]  = avg(lambda r: r.llm_context_precision)
        base["llm_context_recall"]     = avg(lambda r: r.llm_context_recall)
        base["llm_answer_correctness"] = avg(lambda r: r.llm_answer_correctness)

    return base


def print_report(results: list[RAGASResult], summary: dict) -> None:
    has_llm = summary.get("llm_faithfulness") is not None

    print("\n" + "=" * 78)
    print("CRICKET QA — RAGAS EVALUATION REPORT")
    print("=" * 78)

    hdr = (
        f"{'Question':<48} "
        f"{'Faith':>6} "
        f"{'CtxRec':>7} "
        f"{'CtxPre':>7} "
        f"{'Corr':>6} "
        f"{'ms':>5}"
    )
    print(hdr)
    print("-" * 78)

    for r in results:
        ok = "✓" if r.answer_correctness >= 0.8 else "✗"
        print(
            f"{ok} {r.question[:46]:<47} "
            f"{r.faithfulness:6.2f} "
            f"{r.context_recall:7.2f} "
            f"{r.context_precision:7.2f} "
            f"{r.answer_correctness:6.2f} "
            f"{r.latency_ms:5.1f}"
        )

    print("\n" + "=" * 78)
    print("AGGREGATE SCORES (non-LLM)")
    print(f"  Questions evaluated   : {summary['n']}")
    print(f"  Faithfulness          : {summary['faithfulness']:.4f}   (answer nums in context)")
    print(f"  Context Recall        : {summary['context_recall']:.4f}   (GT nums in context)")
    print(f"  Context Precision     : {summary['context_precision']:.4f}   (relevant context lines)")
    print(f"  Answer Correctness    : {summary['answer_correctness']:.4f}   (exact / near match)")
    print(f"  Answer Semantic Sim   : {summary['answer_semantic_sim']:.4f}   (numeric proximity)")
    print(f"  Intent Accuracy       : {summary['intent_accuracy']:.4f}")
    print(f"  Latency p50/p95/p99   : {summary['latency_p50_ms']:.1f} / "
          f"{summary['latency_p95_ms']:.1f} / {summary['latency_p99_ms']:.1f} ms")

    if has_llm:
        print("\nAGGREGATE SCORES (LLM judge — Claude claude-haiku-4-5)")
        print(f"  Faithfulness          : {summary.get('llm_faithfulness', 'N/A'):.4f}  (answer claims grounded in context)")
        print(f"  Context Precision     : {summary.get('llm_context_precision', 'N/A'):.4f}  (context relevant to question)")
        print(f"  Context Recall        : {summary.get('llm_context_recall', 'N/A'):.4f}  (context covers ground truth)")
        print("  Answer Relevancy      : n/a  (skipped — needs OpenAI embeddings)")
        print("  Answer Correctness    : n/a  (skipped — needs OpenAI embeddings)")
    else:
        print("\n  ℹ️  LLM metrics not computed.")
        print("     Set ANTHROPIC_API_KEY to enable Claude-judged RAGAS metrics.")

    print("\nBY CATEGORY")
    for cat, s in summary["by_category"].items():
        print(f"  {cat:<14}  n={s['n']}  "
              f"faith={s['faithfulness']:.2f}  "
              f"recall={s['context_recall']:.2f}  "
              f"prec={s['context_precision']:.2f}  "
              f"corr={s['answer_correctness']:.2f}")

    print("=" * 78)
    print()


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation for Cricket QA")
    parser.add_argument("--data-file", default="data/sample_match.json")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--llm",
        action="store_true",
        default=False,
        help="Run LLM-based RAGAS metrics (requires ANTHROPIC_API_KEY)",
    )
    args = parser.parse_args()

    with open(args.data_file) as f:
        data = json.load(f)

    print(f"[RAGAS] Evaluating {len(BENCHMARK)} questions on {args.data_file}")
    results = run_non_llm_evaluation(data)

    if args.llm or os.environ.get("GOOGLE_API_KEY"):
        results = run_llm_evaluation(results)

    summary = aggregate(results)
    print_report(results, summary)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {"summary": summary, "results": [asdict(r) for r in results]},
                f, indent=2,
            )
        print(f"Full results written to {out}")


if __name__ == "__main__":
    main()
