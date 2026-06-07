"""
Ground-truth benchmark dataset for the Cricket QA system.

All expected answers are verified against data/sample_match.json:
  Over 1: 7 runs  (0,4,1,0,2,0)  — 0 wickets, 1 four, 0 sixes
  Over 2: 11 runs (6,0,1,0,4,0)  — 1 wicket,  1 four, 1 six
  Over 3: 7 runs  (2,0,1,4,0,0)  — 1 wicket,  1 four, 0 sixes
  Over 4: 13 runs (6,1,0,0,2,4)  — 0 wickets, 1 four, 1 six
  Over 5: 9 runs  (0,1,0,0,6,2)  — 1 wicket,  0 fours, 1 six

  Totals : 47 runs / 3 wickets / 4 fours / 3 sixes / 11 dot balls
  Run rate: 9.4 runs/over

Derived over-range values (used in new fixtures below):
  Overs 1–2 : 18 runs, 1 wicket
  Overs 1–3 : 25 runs, 2 wickets
  Overs 1–4 : 38 runs, 2 wickets
  Overs 2–4 : 31 runs, 2 wickets
  Overs 3–5 : 29 runs, 2 wickets
  Overs 4–5 : 22 runs, 1 wicket
  Last 2     : 22 runs  (overs 4–5)
  Last 4     : 40 runs  (overs 2–5)
  Death overs (overs 2–5): 40 runs, 3 wickets → "40/3"
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchmarkItem:
    question: str
    expected_answer: str       # exact string the system must return
    intent: str
    category: str
    description: str = ""


# ── Original 15 fixtures ───────────────────────────────────────────────────

BENCHMARK: list[BenchmarkItem] = [
    # Basic counts
    BenchmarkItem(
        question="How many total runs were scored?",
        expected_answer="47",
        intent="total_runs",
        category="basic",
    ),
    BenchmarkItem(
        question="What is the final score?",
        expected_answer="47",
        intent="total_runs",
        category="basic",
    ),
    BenchmarkItem(
        question="How many wickets fell in the innings?",
        expected_answer="3",
        intent="total_wickets",
        category="basic",
    ),
    BenchmarkItem(
        question="How many wickets were taken?",
        expected_answer="3",
        intent="total_wickets",
        category="basic",
    ),
    BenchmarkItem(
        question="How many fours were hit?",
        expected_answer="4",
        intent="total_fours",
        category="basic",
    ),
    BenchmarkItem(
        question="How many sixes were hit?",
        expected_answer="3",
        intent="total_sixes",
        category="basic",
    ),
    BenchmarkItem(
        question="How many dot balls were bowled?",
        expected_answer="11",
        intent="dot_balls",
        category="basic",
    ),
    # Last N overs — runs
    BenchmarkItem(
        question="How many runs were scored in the last 5 overs?",
        expected_answer="47",
        intent="runs_last_n_overs",
        category="over_range",
        description="max_over=5 so last 5 = all overs",
    ),
    BenchmarkItem(
        question="How many runs in the last 3 overs?",
        expected_answer="29",
        intent="runs_last_n_overs",
        category="over_range",
        description="overs 3,4,5: 7+13+9=29",
    ),
    BenchmarkItem(
        question="Runs scored in the last 1 overs?",
        expected_answer="9",
        intent="runs_last_n_overs",
        category="over_range",
        description="over 5 only: 9 runs",
    ),
    # Last N overs — wickets
    BenchmarkItem(
        question="How many wickets in the last 3 overs?",
        expected_answer="2",
        intent="wickets_last_n_overs",
        category="over_range",
        description="overs 3,4,5: over 3 + over 5 = 2",
    ),
    BenchmarkItem(
        question="Wickets fallen in the last 2 overs?",
        expected_answer="1",
        intent="wickets_last_n_overs",
        category="over_range",
        description="overs 4,5: only over 5 has a wicket",
    ),
    # Phases
    BenchmarkItem(
        question="What happened in the powerplay?",
        expected_answer="47/3 (RR 9.4)",
        intent="powerplay",
        category="phase",
        description="all 5 overs are inside the powerplay window",
    ),
    # Boundaries
    BenchmarkItem(
        question="How many boundaries were hit?",
        expected_answer="7",
        intent="boundaries",
        category="boundaries",
        description="4 fours + 3 sixes = 7",
    ),
    # Run rate
    BenchmarkItem(
        question="What was the overall run rate?",
        expected_answer="9.4",
        intent="run_rate",
        category="advanced",
        description="47 runs / 5 overs",
    ),

    # ── New fixtures: over ranges beyond last-N ────────────────────────────

    BenchmarkItem(
        question="How many runs were scored in overs 1 to 2?",
        expected_answer="18",
        intent="runs_in_over_range",
        category="over_range",
        description="overs 1,2: 7+11=18",
    ),
    BenchmarkItem(
        question="How many runs in overs 3 to 5?",
        expected_answer="29",
        intent="runs_in_over_range",
        category="over_range",
        description="overs 3,4,5: 7+13+9=29",
    ),
    BenchmarkItem(
        question="Runs in overs 1 to 3?",
        expected_answer="25",
        intent="runs_in_over_range",
        category="over_range",
        description="overs 1,2,3: 7+11+7=25",
    ),
    BenchmarkItem(
        question="Runs in overs 4 to 5?",
        expected_answer="22",
        intent="runs_in_over_range",
        category="over_range",
        description="overs 4,5: 13+9=22",
    ),
    BenchmarkItem(
        question="Wickets in overs 1 to 3?",
        expected_answer="2",
        intent="wickets_in_over_range",
        category="over_range",
        description="overs 1,2,3: 0+1+1=2",
    ),
    BenchmarkItem(
        question="How many wickets fell in overs 3 to 5?",
        expected_answer="2",
        intent="wickets_in_over_range",
        category="over_range",
        description="overs 3,4,5: 1+0+1=2",
    ),

    # ── New fixtures: last-N beyond what was covered ───────────────────────

    BenchmarkItem(
        question="How many runs in the last 2 overs?",
        expected_answer="22",
        intent="runs_last_n_overs",
        category="over_range",
        description="overs 4,5: 13+9=22",
    ),
    BenchmarkItem(
        question="Runs in the last 4 overs?",
        expected_answer="40",
        intent="runs_last_n_overs",
        category="over_range",
        description="overs 2,3,4,5: 11+7+13+9=40",
    ),

    # ── New fixtures: phases ───────────────────────────────────────────────

    BenchmarkItem(
        question="Death overs performance?",
        expected_answer="40/3",
        intent="death_overs",
        category="phase",
        description="overs 2–5: 40 runs, 3 wickets",
    ),

    # ── New fixtures: ambiguous / rephrased total-runs queries ─────────────

    BenchmarkItem(
        question="How many runs did the team score?",
        expected_answer="47",
        intent="total_runs",
        category="ambiguous",
        description="generic 'runs' fallback → total_runs",
    ),
    BenchmarkItem(
        question="What was the run tally?",
        expected_answer="47",
        intent="total_runs",
        category="ambiguous",
        description="'run' word match → total_runs fallback",
    ),
    BenchmarkItem(
        question="Total score?",
        expected_answer="47",
        intent="total_runs",
        category="ambiguous",
        description="'score' word match → total_runs fallback",
    ),
    BenchmarkItem(
        question="What's the run rate for the innings?",
        expected_answer="9.4",
        intent="run_rate",
        category="ambiguous",
    ),

    # ── New fixtures: edge-case phrasing ───────────────────────────────────

    BenchmarkItem(
        question="How many balls went for no runs?",
        expected_answer="11",
        intent="dot_balls",
        category="edge",
        description="'no runs' matches dot_balls pattern",
    ),
    BenchmarkItem(
        question="How many times did a wicket fall?",
        expected_answer="3",
        intent="total_wickets",
        category="edge",
        description="'wicket' word match → total_wickets",
    ),
    BenchmarkItem(
        question="How many boundaries were there?",
        expected_answer="7",
        intent="boundaries",
        category="edge",
        description="4 fours + 3 sixes = 7",
    ),
    BenchmarkItem(
        question="What's the total number of fours hit?",
        expected_answer="4",
        intent="total_fours",
        category="edge",
    ),
    BenchmarkItem(
        question="How many sixes were struck?",
        expected_answer="3",
        intent="total_sixes",
        category="edge",
    ),
]
