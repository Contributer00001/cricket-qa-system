"""
Stage 2: IPL 2022 Benchmark Generator
Generates ground-truth Q&A pairs from DuckDB, scaling to all 74 matches.
Run: python -m evaluation.generate_gold_set --db ipl2022.duckdb --out evaluation/gold_sets/
"""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
import duckdb


def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path, read_only=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-match fixtures (auto-generated from scorecard data)
# ─────────────────────────────────────────────────────────────────────────────

def generate_per_match(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Generate ~10 Q&A pairs per match = ~740 fixtures total."""
    fixtures = []
    matches = con.execute("SELECT match_id, match_number, title, short_title, date FROM matches ORDER BY match_number").fetchall()

    for (match_id, match_num, title, short_title, date) in matches:
        # Get innings scores
        innings = con.execute(
            "SELECT inning_number, scores, scores_full, name FROM innings WHERE match_id=? ORDER BY inning_number",
            [match_id]
        ).fetchall()
        if not innings:
            continue

        for (inn_num, scores, scores_full, inn_name) in innings:
            team = inn_name.replace(" Innings", "").strip()
            runs_wickets = scores.split("/") if "/" in scores else [scores, "10"]
            runs = runs_wickets[0]
            wickets = runs_wickets[1].split("(")[0].strip() if len(runs_wickets) > 1 else "10"
            overs_match = re.search(r'\((\d+\.?\d*)\s*ov', scores_full or "")
            overs = overs_match.group(1) if overs_match else "20"

            # Q1: total runs
            fixtures.append({
                "question": f"How many runs did {team} score in match {match_num}?",
                "answer": runs,
                "match_id": match_id,
                "match_number": match_num,
                "inning": inn_num,
                "category": "innings_total",
                "source": "scorecard",
            })
            # Q2: wickets
            fixtures.append({
                "question": f"How many wickets did {team} lose in match {match_num}?",
                "answer": wickets,
                "match_id": match_id,
                "match_number": match_num,
                "inning": inn_num,
                "category": "innings_total",
                "source": "scorecard",
            })
            # Q3: full scorecard line
            fixtures.append({
                "question": f"What was {team}'s score in match {match_num}?",
                "answer": scores_full,
                "match_id": match_id,
                "match_number": match_num,
                "inning": inn_num,
                "category": "innings_total",
                "source": "scorecard",
            })

        # Q4: match result
        result = con.execute("SELECT result FROM matches WHERE match_id=?", [match_id]).fetchone()
        if result and result[0]:
            fixtures.append({
                "question": f"Who won match {match_num} of IPL 2022?",
                "answer": result[0],
                "match_id": match_id,
                "match_number": match_num,
                "inning": None,
                "category": "match_result",
                "source": "matches",
            })

        # Q5: highest scorer per match
        top_bat = con.execute("""
            SELECT player_name, runs FROM batting_performances
            WHERE match_id=? AND inning=1
            ORDER BY runs DESC LIMIT 1
        """, [match_id]).fetchone()
        if top_bat and top_bat[1] > 0:
            fixtures.append({
                "question": f"Who scored the most runs in innings 1 of match {match_num}?",
                "answer": f"{top_bat[0]} ({top_bat[1]} runs)",
                "match_id": match_id,
                "match_number": match_num,
                "inning": 1,
                "category": "top_performer",
                "source": "scorecard",
            })

        # Q6: best bowler per match
        top_bowl = con.execute("""
            SELECT player_name, wickets, runs_conceded FROM bowling_performances
            WHERE match_id=? AND inning=1
            ORDER BY wickets DESC, runs_conceded ASC LIMIT 1
        """, [match_id]).fetchone()
        if top_bowl and top_bowl[1] > 0:
            fixtures.append({
                "question": f"Who took the most wickets bowling in innings 1 of match {match_num}?",
                "answer": f"{top_bowl[0]} ({top_bowl[1]}W/{top_bowl[2]}R)",
                "match_id": match_id,
                "match_number": match_num,
                "inning": 1,
                "category": "top_performer",
                "source": "scorecard",
            })

        # Q7: powerplay score (overs 1-6) from deliveries
        pp = con.execute("""
            SELECT SUM(run) as pp_runs, SUM(wicket) as pp_wkts
            FROM deliveries WHERE match_id=? AND inning=1 AND over_num <= 6
        """, [match_id]).fetchone()
        if pp and pp[0] is not None:
            fixtures.append({
                "question": f"What was the powerplay score for the first innings of match {match_num}?",
                "answer": f"{int(pp[0])}/{int(pp[1])}",
                "match_id": match_id,
                "match_number": match_num,
                "inning": 1,
                "category": "phase",
                "source": "deliveries",
            })

        # Q8: death overs score (overs 17-20) from deliveries
        death = con.execute("""
            SELECT SUM(run) as d_runs, SUM(wicket) as d_wkts
            FROM deliveries WHERE match_id=? AND inning=1 AND over_num >= 17
        """, [match_id]).fetchone()
        if death and death[0] is not None:
            fixtures.append({
                "question": f"How many runs were scored in the death overs of innings 1 in match {match_num}?",
                "answer": str(int(death[0])),
                "match_id": match_id,
                "match_number": match_num,
                "inning": 1,
                "category": "phase",
                "source": "deliveries",
            })

        # Q9: boundaries
        bounds = con.execute("""
            SELECT SUM(four)+SUM(six) as total, SUM(four) as fours, SUM(six) as sixes
            FROM deliveries WHERE match_id=? AND inning=1
        """, [match_id]).fetchone()
        if bounds and bounds[0] is not None:
            fixtures.append({
                "question": f"How many boundaries were hit in innings 1 of match {match_num}?",
                "answer": str(int(bounds[0])),
                "match_id": match_id,
                "match_number": match_num,
                "inning": 1,
                "category": "innings_stats",
                "source": "deliveries",
            })

        # Q10: dot balls
        dots = con.execute("""
            SELECT COUNT(*) FROM deliveries
            WHERE match_id=? AND inning=1 AND run=0 AND noball=0 AND wide=0
        """, [match_id]).fetchone()
        if dots:
            fixtures.append({
                "question": f"How many dot balls were bowled in innings 1 of match {match_num}?",
                "answer": str(int(dots[0])),
                "match_id": match_id,
                "match_number": match_num,
                "inning": 1,
                "category": "innings_stats",
                "source": "deliveries",
            })

    return fixtures


# ─────────────────────────────────────────────────────────────────────────────
# Tournament-wide fixtures (cross-match analytics)
# ─────────────────────────────────────────────────────────────────────────────

def generate_tournament(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Generate cross-match tournament analytics Q&A."""
    fixtures = []

    # Top run scorer
    r = con.execute("SELECT player_name, runs FROM tournament_batting ORDER BY runs DESC LIMIT 1").fetchone()
    if r:
        fixtures.append({"question": "Who scored the most runs in IPL 2022?",
                         "answer": f"{r[0]} ({r[1]} runs)", "category": "tournament_batting", "source": "tournament_batting"})

    # Top wicket taker
    r = con.execute("SELECT player_name, wickets FROM tournament_bowling ORDER BY wickets DESC LIMIT 1").fetchone()
    if r:
        fixtures.append({"question": "Who took the most wickets in IPL 2022?",
                         "answer": f"{r[0]} ({r[1]} wickets)", "category": "tournament_bowling", "source": "tournament_bowling"})

    # Best average
    r = con.execute("SELECT player_name, ROUND(average,2) as avg FROM tournament_batting WHERE innings >= 8 ORDER BY average DESC LIMIT 1").fetchone()
    if r:
        fixtures.append({"question": "Which batsman had the best batting average in IPL 2022?",
                         "answer": f"{r[0]} (avg {r[1]})", "category": "tournament_batting", "source": "tournament_batting"})

    # Best economy
    r = con.execute("SELECT player_name, ROUND(economy,2) as eco FROM tournament_bowling WHERE overs >= 20 ORDER BY economy ASC LIMIT 1").fetchone()
    if r:
        fixtures.append({"question": "Which bowler had the best economy rate in IPL 2022?",
                         "answer": f"{r[0]} (economy {r[1]})", "category": "tournament_bowling", "source": "tournament_bowling"})

    # Most sixes
    r = con.execute("SELECT player_name, sixes FROM tournament_batting ORDER BY sixes DESC LIMIT 1").fetchone()
    if r:
        fixtures.append({"question": "Who hit the most sixes in IPL 2022?",
                         "answer": f"{r[0]} ({r[1]} sixes)", "category": "tournament_batting", "source": "tournament_batting"})

    # Most fours
    r = con.execute("SELECT player_name, fours FROM tournament_batting ORDER BY fours DESC LIMIT 1").fetchone()
    if r:
        fixtures.append({"question": "Who hit the most fours in IPL 2022?",
                         "answer": f"{r[0]} ({r[1]} fours)", "category": "tournament_batting", "source": "tournament_batting"})

    # Highest team score
    r = con.execute("""
        SELECT m.short_title, i.scores_full, m.match_number
        FROM innings i JOIN matches m ON i.match_id = m.match_id
        ORDER BY CAST(SPLIT_PART(i.scores,'/',1) AS INTEGER) DESC LIMIT 1
    """).fetchone()
    if r:
        fixtures.append({"question": "What is the highest team score in IPL 2022?",
                         "answer": f"{r[0]}: {r[1]} (Match {r[2]})", "category": "tournament_records", "source": "innings"})

    # Table standings
    r = con.execute("SELECT team_name, points, wins FROM standings ORDER BY rank ASC LIMIT 1").fetchone()
    if r:
        fixtures.append({"question": "Which team finished top of the IPL 2022 points table?",
                         "answer": f"{r[0]} ({r[1]} points, {r[2]} wins)", "category": "standings", "source": "standings"})

    # Total sixes in tournament
    r = con.execute("SELECT SUM(six) FROM deliveries").fetchone()
    if r:
        fixtures.append({"question": "How many sixes were hit in IPL 2022?",
                         "answer": str(int(r[0])), "category": "tournament_records", "source": "deliveries"})

    # Player with highest individual score
    r = con.execute("""
        SELECT bp.player_name, bp.runs, m.short_title
        FROM batting_performances bp JOIN matches m ON bp.match_id = m.match_id
        ORDER BY bp.runs DESC LIMIT 1
    """).fetchone()
    if r:
        fixtures.append({"question": "What is the highest individual score in IPL 2022?",
                         "answer": f"{r[0]}: {r[1]} runs ({r[2]})", "category": "tournament_records", "source": "scorecard"})

    # Best bowling figures
    r = con.execute("""
        SELECT bp.player_name, bp.wickets, bp.runs_conceded, m.short_title
        FROM bowling_performances bp JOIN matches m ON bp.match_id = m.match_id
        ORDER BY bp.wickets DESC, bp.runs_conceded ASC LIMIT 1
    """).fetchone()
    if r:
        fixtures.append({"question": "What are the best bowling figures in a single innings in IPL 2022?",
                         "answer": f"{r[0]}: {r[1]}/{r[2]} ({r[3]})", "category": "tournament_records", "source": "scorecard"})

    # Avg powerplay score per team
    r = con.execute("""
        SELECT m.team1, ROUND(AVG(pp.pp_runs),1) as avg_pp
        FROM (
            SELECT match_id, SUM(run) as pp_runs FROM deliveries
            WHERE inning=1 AND over_num<=6 GROUP BY match_id
        ) pp JOIN matches m ON pp.match_id = m.match_id
        GROUP BY m.team1 ORDER BY avg_pp DESC LIMIT 1
    """).fetchone()
    if r:
        fixtures.append({"question": "Which team scored the most runs on average in powerplay overs in IPL 2022?",
                         "answer": f"{r[0]} (avg {r[1]} per match)", "category": "phase_analysis", "source": "deliveries"})

    return fixtures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="ipl2022.duckdb", help="DuckDB file path")
    parser.add_argument("--out", default="evaluation/gold_sets", help="Output directory")
    parser.add_argument("--per-match", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    con = connect(args.db)

    # Per-match gold set
    per_match = generate_per_match(con)
    with open(f"{args.out}/per_match_gold.json", "w") as f:
        json.dump(per_match, f, indent=2)
    print(f"Per-match fixtures: {len(per_match)}")

    # Tournament gold set
    tournament = generate_tournament(con)
    with open(f"{args.out}/tournament_gold.json", "w") as f:
        json.dump(tournament, f, indent=2)
    print(f"Tournament fixtures: {len(tournament)}")

    total = len(per_match) + len(tournament)
    print(f"Total: {total} ground-truth fixtures")

    # Category summary
    from collections import Counter
    cats = Counter(f['category'] for f in per_match + tournament)
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")

    con.close()


if __name__ == "__main__":
    main()