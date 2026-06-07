"""
Unit tests for StatsEngine.

All expected values are hand-verified against data/sample_match.json.
"""
import pytest

from agents.stats_engine import OverRangeStats, PlayerStats, StatsEngine

# ── Fixtures ───────────────────────────────────────────────────────────────

SAMPLE = {
    "commentaries": [
        # Over 1 — 7 runs, 0 wickets, 1 four
        {"event": "ball",   "over": 1, "run": 0, "four": False, "six": False},
        {"event": "ball",   "over": 1, "run": 4, "four": True,  "six": False},
        {"event": "ball",   "over": 1, "run": 1, "four": False, "six": False},
        {"event": "ball",   "over": 1, "run": 0, "four": False, "six": False},
        {"event": "ball",   "over": 1, "run": 2, "four": False, "six": False},
        {"event": "ball",   "over": 1, "run": 0, "four": False, "six": False},
        # Over 2 — 11 runs, 1 wicket, 1 four, 1 six
        {"event": "ball",   "over": 2, "run": 6, "four": False, "six": True},
        {"event": "ball",   "over": 2, "run": 0, "four": False, "six": False},
        {"event": "ball",   "over": 2, "run": 1, "four": False, "six": False},
        {"event": "wicket", "over": 2, "run": 0, "four": False, "six": False},
        {"event": "ball",   "over": 2, "run": 4, "four": True,  "six": False},
        {"event": "ball",   "over": 2, "run": 0, "four": False, "six": False},
        # Over 3 — 7 runs, 1 wicket, 1 four
        {"event": "ball",   "over": 3, "run": 2, "four": False, "six": False},
        {"event": "ball",   "over": 3, "run": 0, "four": False, "six": False},
        {"event": "ball",   "over": 3, "run": 1, "four": False, "six": False},
        {"event": "ball",   "over": 3, "run": 4, "four": True,  "six": False},
        {"event": "wicket", "over": 3, "run": 0, "four": False, "six": False},
        {"event": "ball",   "over": 3, "run": 0, "four": False, "six": False},
        # Over 4 — 13 runs, 0 wickets, 1 four, 1 six
        {"event": "ball",   "over": 4, "run": 6, "four": False, "six": True},
        {"event": "ball",   "over": 4, "run": 1, "four": False, "six": False},
        {"event": "ball",   "over": 4, "run": 0, "four": False, "six": False},
        {"event": "ball",   "over": 4, "run": 0, "four": False, "six": False},
        {"event": "ball",   "over": 4, "run": 2, "four": False, "six": False},
        {"event": "ball",   "over": 4, "run": 4, "four": True,  "six": False},
        # Over 5 — 9 runs, 1 wicket, 1 six
        {"event": "ball",   "over": 5, "run": 0, "four": False, "six": False},
        {"event": "ball",   "over": 5, "run": 1, "four": False, "six": False},
        {"event": "ball",   "over": 5, "run": 0, "four": False, "six": False},
        {"event": "wicket", "over": 5, "run": 0, "four": False, "six": False},
        {"event": "ball",   "over": 5, "run": 6, "four": False, "six": True},
        {"event": "ball",   "over": 5, "run": 2, "four": False, "six": False},
    ]
}

PLAYER_DATA = {
    "commentaries": [
        {"event": "ball",   "over": 1, "run": 4, "four": True,  "six": False, "batsman": "Kohli"},
        {"event": "ball",   "over": 1, "run": 6, "four": False, "six": True,  "batsman": "Kohli"},
        {"event": "ball",   "over": 1, "run": 1, "four": False, "six": False, "batsman": "Dhoni"},
        {"event": "wicket", "over": 2, "run": 0, "four": False, "six": False, "batsman": "Kohli"},
        {"event": "ball",   "over": 2, "run": 4, "four": True,  "six": False, "batsman": "Dhoni"},
    ]
}


@pytest.fixture
def engine():
    return StatsEngine(SAMPLE)


@pytest.fixture
def player_engine():
    return StatsEngine(PLAYER_DATA)


# ── Basic aggregations ─────────────────────────────────────────────────────

def test_total_runs(engine):
    assert engine.total_runs() == 47

def test_total_wickets(engine):
    assert engine.total_wickets() == 3

def test_total_fours(engine):
    assert engine.total_fours() == 4

def test_total_sixes(engine):
    assert engine.total_sixes() == 3

def test_dot_balls(engine):
    # wicket deliveries are excluded from dot-ball count
    assert engine.dot_balls() == 11

def test_run_rate(engine):
    # 47 runs in 30 balls = 5 overs → 9.4
    assert engine.run_rate() == pytest.approx(9.4, abs=0.05)

def test_max_over(engine):
    assert engine.max_over() == 5


# ── Over-range queries ─────────────────────────────────────────────────────

def test_runs_last_5_overs(engine):
    assert engine.runs_last_n_overs(5) == 47

def test_runs_last_3_overs(engine):
    # Overs 3,4,5 = 7+13+9 = 29
    assert engine.runs_last_n_overs(3) == 29

def test_runs_in_overs_range(engine):
    assert engine.runs_in_overs(1, 2) == 18

def test_wickets_last_n_overs(engine):
    assert engine.wickets_last_n_overs(3) == 2

def test_wickets_in_over_range(engine):
    assert engine.wickets_in_overs(1, 3) == 2


# ── Phase breakdowns ───────────────────────────────────────────────────────

def test_powerplay_stats(engine):
    pp = engine.powerplay_stats()
    assert isinstance(pp, OverRangeStats)
    assert pp.runs == 47
    assert pp.wickets == 3

def test_death_overs_stats(engine):
    do = engine.death_overs_stats()
    # max_over=5, start=max(1,5-3)=2 → overs 2-5 = 11+7+13+9=40
    assert do.runs == 40
    assert do.wickets == 3


# ── Over-by-over summary ───────────────────────────────────────────────────

def test_over_by_over(engine):
    summary = engine.over_by_over()
    assert len(summary) == 5
    over1 = next(o for o in summary if o["over"] == 1)
    assert over1["runs"] == 7
    assert over1["wickets"] == 0


# ── Player analytics ───────────────────────────────────────────────────────

def test_player_runs(player_engine):
    assert player_engine.player_runs("Kohli") == 10

def test_player_runs_case_insensitive(player_engine):
    assert player_engine.player_runs("kohli") == 10

def test_player_runs_no_data(engine):
    assert engine.player_runs("Kohli") is None

def test_top_scorer(player_engine):
    ts = player_engine.top_scorer()
    assert isinstance(ts, PlayerStats)
    assert ts.name == "Kohli"
    assert ts.runs == 10

def test_top_scorer_no_data(engine):
    assert engine.top_scorer() is None


# ── Context builder ────────────────────────────────────────────────────────

def test_build_context_default(engine):
    ctx = engine.build_context()
    assert "Total runs: 47" in ctx
    assert "Total wickets: 3" in ctx

def test_build_context_subset(engine):
    ctx = engine.build_context(keys=["total_runs"])
    assert "Total runs: 47" in ctx
    assert "Fours" not in ctx


# ── Error handling ─────────────────────────────────────────────────────────

def test_empty_commentary_raises():
    with pytest.raises(ValueError, match="no deliveries"):
        StatsEngine({"commentaries": []})

def test_missing_required_field_raises():
    with pytest.raises(ValueError, match="missing required fields"):
        StatsEngine({"commentaries": [{"event": "ball", "over": 1}]})
