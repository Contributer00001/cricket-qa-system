"""
Unit tests for QueryRouter intent classification.
"""
import pytest

from agents.query_router import Intent, QueryRouter


def route(q: str):
    return QueryRouter.route(q)


# ── Basic intents ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("question", [
    "How many total runs were scored?",
    "What is the final score?",
    "What was the innings total?",
])
def test_total_runs(question):
    assert route(question).intent == Intent.TOTAL_RUNS


@pytest.mark.parametrize("question", [
    "How many wickets fell?",
    "Total wickets taken?",
    "How many wickets were taken in the innings?",
])
def test_total_wickets(question):
    assert route(question).intent == Intent.TOTAL_WICKETS


@pytest.mark.parametrize("question", [
    "How many fours were hit?",
    "Total fours in the innings?",
])
def test_total_fours(question):
    assert route(question).intent == Intent.TOTAL_FOURS


@pytest.mark.parametrize("question", [
    "How many sixes were hit?",
    "Total sixes in the match?",
])
def test_total_sixes(question):
    assert route(question).intent == Intent.TOTAL_SIXES


# ── Last N overs ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("question,expected_n", [
    ("How many runs in the last 3 overs?", 3),
    ("How many runs were scored in the last 5 overs?", 5),
    ("Runs in the last 10 overs?", 10),
])
def test_runs_last_n_overs(question, expected_n):
    plan = route(question)
    assert plan.intent == Intent.RUNS_LAST_N_OVERS
    assert plan.entities["n"] == expected_n


@pytest.mark.parametrize("question,expected_n", [
    ("How many wickets in the last 3 overs?", 3),
    ("Wickets fallen in the last 5 overs?", 5),
])
def test_wickets_last_n_overs(question, expected_n):
    plan = route(question)
    assert plan.intent == Intent.WICKETS_LAST_N_OVERS
    assert plan.entities["n"] == expected_n


# ── Over ranges ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("question,start,end", [
    ("How many runs in overs 10 to 15?", 10, 15),
    ("Runs in overs 1-6?", 1, 6),
])
def test_runs_in_over_range(question, start, end):
    plan = route(question)
    assert plan.intent == Intent.RUNS_IN_OVER_RANGE
    assert plan.entities["start_over"] == start
    assert plan.entities["end_over"] == end


# ── Phases ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("question", [
    "What happened in the powerplay?",
    "Powerplay score?",
    "First 6 overs performance?",
])
def test_powerplay(question):
    assert route(question).intent == Intent.POWERPLAY


@pytest.mark.parametrize("question", [
    "How did the team perform in the death overs?",
    "Death overs score?",
])
def test_death_overs(question):
    assert route(question).intent == Intent.DEATH_OVERS


# ── Other intents ──────────────────────────────────────────────────────────

def test_dot_balls():
    assert route("How many dot balls were bowled?").intent == Intent.DOT_BALLS

def test_run_rate():
    assert route("What was the run rate?").intent == Intent.RUN_RATE

def test_boundaries():
    assert route("How many boundaries were hit?").intent == Intent.BOUNDARIES

def test_over_summary():
    assert route("Give me an over-by-over breakdown.").intent == Intent.OVER_SUMMARY


# ── Confidence ─────────────────────────────────────────────────────────────

def test_high_confidence_explicit_n():
    plan = route("How many runs in the last 5 overs?")
    assert plan.confidence >= 0.90

def test_lower_confidence_vague():
    plan = route("How many runs?")
    assert plan.confidence <= 0.65

def test_unknown_intent():
    plan = route("What is the weather like today?")
    assert plan.intent == Intent.UNKNOWN
    assert plan.confidence == 0.0
