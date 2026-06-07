"""
Rule-based query intent classifier for cricket statistics questions.

Two-stage design:
  1. Fast regex/keyword matching covers ~90% of questions in < 1ms.
  2. A structured QueryPlan is returned so the orchestrator knows
     exactly which StatsEngine methods to call and which context
     keys to include in the LLM prompt.

No LLM is required for routing — keeping cold-path latency at zero.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Intent(str, Enum):
    TOTAL_RUNS = "total_runs"
    RUNS_LAST_N_OVERS = "runs_last_n_overs"
    RUNS_IN_OVER_RANGE = "runs_in_over_range"
    TOTAL_WICKETS = "total_wickets"
    WICKETS_LAST_N_OVERS = "wickets_last_n_overs"
    WICKETS_IN_OVER_RANGE = "wickets_in_over_range"
    TOTAL_FOURS = "total_fours"
    TOTAL_SIXES = "total_sixes"
    BOUNDARIES = "boundaries"
    DOT_BALLS = "dot_balls"
    RUN_RATE = "run_rate"
    POWERPLAY = "powerplay"
    DEATH_OVERS = "death_overs"
    TOP_SCORER = "top_scorer"
    PLAYER_RUNS = "player_runs"
    OVER_SUMMARY = "over_summary"
    UNKNOWN = "unknown"


@dataclass
class QueryPlan:
    intent: Intent
    entities: dict = field(default_factory=dict)
    confidence: float = 1.0
    context_keys: list[str] = field(default_factory=list)

    def requires_player_data(self) -> bool:
        return self.intent in {Intent.TOP_SCORER, Intent.PLAYER_RUNS}


# ── Entity extraction helpers ──────────────────────────────────────────────

_WORD_TO_INT: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _extract_n(text: str) -> Optional[int]:
    """Return N from 'last N overs' (digits or words)."""
    m = re.search(r"last\s+(\d+)\s+over", text)
    if m:
        return int(m.group(1))
    for word, val in _WORD_TO_INT.items():
        if re.search(rf"last\s+{word}\s+over", text):
            return val
    m = re.search(r"(\d+)\s+over", text)
    if m:
        return int(m.group(1))
    return None


def _extract_over_range(text: str) -> Optional[tuple[int, int]]:
    """Return (start, end) from 'overs X to Y', 'overs X-Y', 'overs X–Y'."""
    m = re.search(r"over[s]?\s+(\d+)\s*(?:to|[-–]|through)\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _extract_player(original: str) -> Optional[str]:
    """Return a player name when one is mentioned (capitalised-word heuristic)."""
    patterns = [
        r"did\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+score",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+scored",
        r"runs?\s+(?:by|for|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'s\s+(?:runs?|score|total)",
        r"how\s+(?:many\s+)?runs?\s+did\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, original)
        if m:
            return m.group(1)
    return None


# ── Shorthand predicates ───────────────────────────────────────────────────

def _has_runs(q: str) -> bool:
    return bool(re.search(r"\bruns?\b|scored", q))

def _has_wickets(q: str) -> bool:
    return bool(re.search(r"\bwickets?\b", q))

def _has_score(q: str) -> bool:
    return bool(re.search(r"\bscores?\b|\btotal\b", q))

def _is_last_n(q: str) -> bool:
    return bool(re.search(r"last\s+\d+\s+over|last\s+(?:few|some)\s+over", q))


# ── Classifier ─────────────────────────────────────────────────────────────

class QueryRouter:
    """
    Maps a natural-language cricket question to a QueryPlan.

    Usage::

        plan = QueryRouter.route("How many runs in the last 3 overs?")
        # plan.intent  → Intent.RUNS_LAST_N_OVERS
        # plan.entities → {"n": 3}
    """

    @staticmethod
    def route(question: str) -> QueryPlan:
        q = question.lower().strip()
        return QueryRouter._classify(q, question)

    @staticmethod
    def _classify(q: str, original: str) -> QueryPlan:  # noqa: C901
        # ── Powerplay ──────────────────────────────────────────────────────
        if re.search(r"powerplay|power\s+play|first\s+6\s+over", q):
            return QueryPlan(
                intent=Intent.POWERPLAY,
                context_keys=["powerplay"],
                confidence=0.95,
            )

        # ── Death overs ────────────────────────────────────────────────────
        # "death over" / "final over" always → DEATH_OVERS.
        # "last (four|4) overs" → DEATH_OVERS only when the question does NOT
        # explicitly request a run count (e.g. "runs in the last 4 overs"
        # should route to RUNS_LAST_N_OVERS instead).
        _death_explicit = re.search(r"death\s+over|final\s+over", q)
        _death_last4    = re.search(r"last\s+(?:four|4)\s+over", q) \
                          and not (_has_runs(q) or _has_score(q))
        if _death_explicit or _death_last4:
            return QueryPlan(
                intent=Intent.DEATH_OVERS,
                context_keys=["death_overs"],
                confidence=0.95,
            )

        # ── Player-specific ────────────────────────────────────────────────
        # Guard: skip player extraction when the question is about overs
        # (prevents "Runs scored" from being misread as player "Runs")
        player = None if (_is_last_n(q) or _extract_over_range(q)) else _extract_player(original)
        if player and (_has_runs(q) or _has_score(q)):
            return QueryPlan(
                intent=Intent.PLAYER_RUNS,
                entities={"player": player},
                context_keys=["total_runs"],
                confidence=0.80,
            )
        if re.search(r"top\s+scor|highest\s+scor|most\s+runs?", q):
            return QueryPlan(
                intent=Intent.TOP_SCORER,
                context_keys=[],
                confidence=0.90,
            )

        # ── Over-range runs (must come before last-N and total checks) ─────
        over_range = _extract_over_range(q)
        if over_range and (_has_runs(q) or _has_score(q)):
            s, e = over_range
            return QueryPlan(
                intent=Intent.RUNS_IN_OVER_RANGE,
                entities={"start_over": s, "end_over": e},
                context_keys=["total_runs"],
                confidence=0.90,
            )

        # ── Last N overs — runs (must come before generic total_runs) ──────
        if _is_last_n(q) and (_has_runs(q) or _has_score(q)):
            n = _extract_n(q)
            key = f"last_{n}_overs" if n in (3, 5) else "last_5_overs"
            return QueryPlan(
                intent=Intent.RUNS_LAST_N_OVERS,
                entities={"n": n or 5},
                context_keys=[key],
                confidence=0.95 if n else 0.70,
            )

        # ── Last N overs — wickets ─────────────────────────────────────────
        if _is_last_n(q) and _has_wickets(q):
            n = _extract_n(q)
            return QueryPlan(
                intent=Intent.WICKETS_LAST_N_OVERS,
                entities={"n": n or 5},
                context_keys=["total_wickets"],
                confidence=0.95 if n else 0.70,
            )

        # ── Over-range wickets ─────────────────────────────────────────────
        if over_range and _has_wickets(q):
            s, e = over_range
            return QueryPlan(
                intent=Intent.WICKETS_IN_OVER_RANGE,
                entities={"start_over": s, "end_over": e},
                context_keys=["total_wickets"],
                confidence=0.90,
            )

        # ── Total wickets ──────────────────────────────────────────────────
        if _has_wickets(q):
            return QueryPlan(
                intent=Intent.TOTAL_WICKETS,
                context_keys=["total_wickets"],
                confidence=0.90,
            )

        # ── Total runs / innings total (explicit keywords) ─────────────────
        if re.search(r"total\s+runs?|final\s+score|innings\s+total|overall\s+score", q):
            return QueryPlan(
                intent=Intent.TOTAL_RUNS,
                context_keys=["total_runs", "total_wickets"],
                confidence=0.90,
            )

        # ── Fours ──────────────────────────────────────────────────────────
        if re.search(r"\bfours?\b", q) and not re.search(r"last\s+fours?\s+over", q):
            return QueryPlan(
                intent=Intent.TOTAL_FOURS,
                context_keys=["total_fours"],
                confidence=0.85,
            )

        # ── Sixes ──────────────────────────────────────────────────────────
        if re.search(r"\bsixes?\b", q) and not re.search(r"last\s+six\s+over", q):
            return QueryPlan(
                intent=Intent.TOTAL_SIXES,
                context_keys=["total_sixes"],
                confidence=0.85,
            )

        # ── Boundaries (fours + sixes) ─────────────────────────────────────
        if re.search(r"\bboundar", q):
            return QueryPlan(
                intent=Intent.BOUNDARIES,
                context_keys=["total_fours", "total_sixes"],
                confidence=0.90,
            )

        # ── Dot balls ──────────────────────────────────────────────────────
        if re.search(r"dot\s*balls?|maiden|no[\s-]runs?", q):
            return QueryPlan(
                intent=Intent.DOT_BALLS,
                context_keys=["dot_balls"],
                confidence=0.90,
            )

        # ── Run rate ───────────────────────────────────────────────────────
        if re.search(r"run[\s-]?rate|economy|\brr\b", q):
            return QueryPlan(
                intent=Intent.RUN_RATE,
                context_keys=["run_rate"],
                confidence=0.90,
            )

        # ── Over-by-over summary ───────────────────────────────────────────
        if re.search(r"over[\s-]by[\s-]over|each\s+over|every\s+over|per\s+over", q):
            return QueryPlan(
                intent=Intent.OVER_SUMMARY,
                context_keys=[],
                confidence=0.85,
            )

        # ── Generic runs fallback (low confidence) ─────────────────────────
        if _has_runs(q) or re.search(r"\bscore\b", q):
            return QueryPlan(
                intent=Intent.TOTAL_RUNS,
                context_keys=["total_runs", "last_5_overs"],
                confidence=0.60,
            )

        return QueryPlan(
            intent=Intent.UNKNOWN,
            context_keys=[],
            confidence=0.0,
        )
