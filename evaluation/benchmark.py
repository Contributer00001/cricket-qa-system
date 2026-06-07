"""
Benchmark runner for the Cricket QA system.

Metrics reported:
  - exact_match        : answer == expected (string equality after strip)
  - numeric_match      : float(answer) ≈ float(expected) within 0.05
  - intent_accuracy    : predicted intent == expected intent
  - avg_confidence     : mean QueryPlan.confidence
  - per_category       : above metrics broken down by question category

Usage::

    python -m evaluation.benchmark
    python -m evaluation.benchmark --data-file path/to/custom.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from agents.orchestrator import CricketOrchestrator
from agents.query_router import QueryRouter
from evaluation.dataset import BENCHMARK, BenchmarkItem


@dataclass
class EvalResult:
    question: str
    expected_answer: str
    predicted_answer: str
    expected_intent: str
    predicted_intent: str
    exact_match: bool
    numeric_match: bool
    intent_correct: bool
    confidence: float
    latency_ms: float
    category: str


def _numeric_match(pred: str, expected: str, tol: float = 0.05) -> bool:
    try:
        return abs(float(pred) - float(expected)) <= tol
    except (ValueError, TypeError):
        return False


_MIN_FIXTURES_WARN = 50


def run_benchmark(
    commentary_data: dict,
    items: Optional[list[BenchmarkItem]] = None,
) -> list[EvalResult]:
    """
    Run all benchmark items against the no-LLM deterministic path.

    Parameters
    ----------
    commentary_data : dict
        Commentary JSON (commentaries list).
    items : list[BenchmarkItem], optional
        Defaults to the full BENCHMARK from evaluation.dataset.
    """
    orchestrator = CricketOrchestrator(analyst=None)
    items = items or BENCHMARK

    if len(items) < _MIN_FIXTURES_WARN:
        import warnings
        warnings.warn(
            f"Benchmark has only {len(items)} fixtures (< {_MIN_FIXTURES_WARN}). "
            "Small benchmark sets give overconfident accuracy scores — "
            "add more questions to evaluation/dataset.py.",
            stacklevel=2,
        )
    results: list[EvalResult] = []

    for item in items:
        t0 = time.perf_counter()
        output = orchestrator.answer(item.question, commentary_data)
        latency_ms = (time.perf_counter() - t0) * 1000

        pred = str(output["answer"]).strip()
        exact = pred == item.expected_answer
        numeric = _numeric_match(pred, item.expected_answer)

        results.append(EvalResult(
            question=item.question,
            expected_answer=item.expected_answer,
            predicted_answer=pred,
            expected_intent=item.intent,
            predicted_intent=output["intent"],
            exact_match=exact,
            numeric_match=numeric,
            intent_correct=output["intent"] == item.intent,
            confidence=output["confidence"],
            latency_ms=round(latency_ms, 2),
            category=item.category,
        ))

    return results


def summarise(results: list[EvalResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    exact = sum(r.exact_match for r in results) / n
    numeric = sum(r.numeric_match for r in results) / n
    intent = sum(r.intent_correct for r in results) / n
    avg_conf = sum(r.confidence for r in results) / n
    avg_lat = sum(r.latency_ms for r in results) / n

    by_category: dict[str, dict] = {}
    categories = {r.category for r in results}
    for cat in sorted(categories):
        cat_items = [r for r in results if r.category == cat]
        nc = len(cat_items)
        by_category[cat] = {
            "n": nc,
            "exact_match": round(sum(r.exact_match for r in cat_items) / nc, 3),
            "intent_accuracy": round(sum(r.intent_correct for r in cat_items) / nc, 3),
        }

    return {
        "n_questions": n,
        "exact_match": round(exact, 3),
        "numeric_match": round(numeric, 3),
        "intent_accuracy": round(intent, 3),
        "avg_confidence": round(avg_conf, 3),
        "avg_latency_ms": round(avg_lat, 2),
        "by_category": by_category,
    }


def print_report(results: list[EvalResult], summary: dict) -> None:
    print("\n" + "=" * 70)
    print("CRICKET QA BENCHMARK REPORT")
    print("=" * 70)

    for r in results:
        status = "PASS" if r.exact_match else ("~OK" if r.numeric_match else "FAIL")
        print(f"[{status}] {r.question[:55]:<55} "
              f"exp={r.expected_answer!r:>6} got={r.predicted_answer!r:>6}  "
              f"intent={'OK' if r.intent_correct else 'MISS':<4} "
              f"{r.latency_ms:.1f}ms")

    print("\n" + "-" * 70)
    print(f"Total questions  : {summary['n_questions']}")
    print(f"Exact match      : {summary['exact_match']:.1%}")
    print(f"Numeric match    : {summary['numeric_match']:.1%}")
    print(f"Intent accuracy  : {summary['intent_accuracy']:.1%}")
    print(f"Avg confidence   : {summary['avg_confidence']:.2f}")
    print(f"Avg latency      : {summary['avg_latency_ms']:.1f} ms")
    print("\nBy category:")
    for cat, stats in summary["by_category"].items():
        print(f"  {cat:<15} n={stats['n']}  "
              f"exact={stats['exact_match']:.1%}  "
              f"intent={stats['intent_accuracy']:.1%}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Cricket QA benchmark")
    parser.add_argument(
        "--data-file",
        default="data/sample_match.json",
        help="Path to commentary JSON file",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write full results as JSON",
    )
    args = parser.parse_args()

    with open(args.data_file) as f:
        commentary_data = json.load(f)

    results = run_benchmark(commentary_data)
    summary = summarise(results)
    print_report(results, summary)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {"summary": summary, "results": [asdict(r) for r in results]},
                f, indent=2,
            )
        print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
