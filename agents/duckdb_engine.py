"""
agents/duckdb_engine.py
Cross-match IPL analytics powered by DuckDB.
Answers tournament-wide questions that the per-match StatsEngine cannot.
"""
from __future__ import annotations
import os, re, logging
from typing import Optional
import duckdb

logger = logging.getLogger(__name__)

DEFAULT_DB = os.environ.get("IPL_DB_PATH", "ipl2022.duckdb")


class DuckDBEngine:
    """Answers IPL-wide analytical questions from a DuckDB database."""

    def __init__(self, db_path: str = DEFAULT_DB):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"DuckDB not found: {db_path}. Run scripts/build_ipl_db.py first.")
        self._db_path = db_path
        self._con: Optional[duckdb.DuckDBPyConnection] = None

    def _conn(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(self._db_path, read_only=True)
        return self._con

    def close(self):
        if self._con:
            self._con.close()
            self._con = None

    # ── Tournament-wide batting ───────────────────────────────────────────

    def top_run_scorers(self, n: int = 5) -> list[dict]:
        rows = self._conn().execute("""
            SELECT player_name, team, matches, runs, highest, average, strike_rate,
                   hundreds, fifties, fours, sixes
            FROM tournament_batting ORDER BY runs DESC LIMIT ?
        """, [n]).fetchall()
        cols = ["player","team","matches","runs","highest","average","strike_rate",
                "hundreds","fifties","fours","sixes"]
        return [dict(zip(cols, r)) for r in rows]

    def top_wicket_takers(self, n: int = 5) -> list[dict]:
        rows = self._conn().execute("""
            SELECT player_name, team, matches, wickets, overs, economy, average, strike_rate
            FROM tournament_bowling ORDER BY wickets DESC LIMIT ?
        """, [n]).fetchall()
        cols = ["player","team","matches","wickets","overs","economy","average","strike_rate"]
        return [dict(zip(cols, r)) for r in rows]

    def best_batting_average(self, min_innings: int = 8) -> list[dict]:
        rows = self._conn().execute("""
            SELECT player_name, team, innings, runs, ROUND(average,2) as average
            FROM tournament_batting WHERE innings >= ?
            ORDER BY average DESC LIMIT 5
        """, [min_innings]).fetchall()
        return [{"player": r[0], "team": r[1], "innings": r[2], "runs": r[3], "average": r[4]} for r in rows]

    def best_economy_rates(self, min_overs: float = 20.0) -> list[dict]:
        rows = self._conn().execute("""
            SELECT player_name, team, overs, wickets, ROUND(economy,2) as economy
            FROM tournament_bowling WHERE overs >= ?
            ORDER BY economy ASC LIMIT 5
        """, [min_overs]).fetchall()
        return [{"player": r[0], "team": r[1], "overs": r[2], "wickets": r[3], "economy": r[4]} for r in rows]

    def most_sixes(self, n: int = 5) -> list[dict]:
        rows = self._conn().execute("""
            SELECT player_name, team, sixes, fours, runs
            FROM tournament_batting ORDER BY sixes DESC LIMIT ?
        """, [n]).fetchall()
        return [{"player": r[0], "team": r[1], "sixes": r[2], "fours": r[3], "runs": r[4]} for r in rows]

    def most_fours(self, n: int = 5) -> list[dict]:
        rows = self._conn().execute("""
            SELECT player_name, team, fours, sixes, runs
            FROM tournament_batting ORDER BY fours DESC LIMIT ?
        """, [n]).fetchall()
        return [{"player": r[0], "team": r[1], "fours": r[2], "sixes": r[3], "runs": r[4]} for r in rows]

    # ── Records ──────────────────────────────────────────────────────────

    def highest_team_score(self) -> dict:
        r = self._conn().execute("""
            SELECT m.short_title, i.scores_full, m.match_number, m.venue
            FROM innings i JOIN matches m ON i.match_id = m.match_id
            ORDER BY CAST(SPLIT_PART(i.scores,'/',1) AS INTEGER) DESC LIMIT 1
        """).fetchone()
        if r:
            return {"match": r[0], "score": r[1], "match_number": r[2], "venue": r[3]}
        return {}

    def lowest_team_score(self) -> dict:
        r = self._conn().execute("""
            SELECT m.short_title, i.scores_full, m.match_number
            FROM innings i JOIN matches m ON i.match_id = m.match_id
            WHERE CAST(SPLIT_PART(i.scores,'/',1) AS INTEGER) > 50
            ORDER BY CAST(SPLIT_PART(i.scores,'/',1) AS INTEGER) ASC LIMIT 1
        """).fetchone()
        if r:
            return {"match": r[0], "score": r[1], "match_number": r[2]}
        return {}

    def highest_individual_score(self) -> dict:
        r = self._conn().execute("""
            SELECT bp.player_name, bp.runs, bp.balls_faced, bp.fours, bp.sixes,
                   m.short_title, m.match_number
            FROM batting_performances bp JOIN matches m ON bp.match_id = m.match_id
            ORDER BY bp.runs DESC LIMIT 1
        """).fetchone()
        if r:
            return {"player": r[0], "runs": r[1], "balls": r[2], "fours": r[3],
                    "sixes": r[4], "match": r[5], "match_number": r[6]}
        return {}

    def best_bowling_figures(self) -> dict:
        r = self._conn().execute("""
            SELECT bp.player_name, bp.wickets, bp.runs_conceded,
                   m.short_title, m.match_number
            FROM bowling_performances bp JOIN matches m ON bp.match_id = m.match_id
            ORDER BY bp.wickets DESC, bp.runs_conceded ASC LIMIT 1
        """).fetchone()
        if r:
            return {"player": r[0], "wickets": r[1], "runs": r[2],
                    "match": r[3], "match_number": r[4]}
        return {}

    def tournament_sixes_total(self) -> int:
        r = self._conn().execute("SELECT SUM(six) FROM deliveries").fetchone()
        return int(r[0]) if r and r[0] else 0

    def tournament_fours_total(self) -> int:
        r = self._conn().execute("SELECT SUM(four) FROM deliveries").fetchone()
        return int(r[0]) if r and r[0] else 0

    # ── Standings ─────────────────────────────────────────────────────────

    def points_table(self) -> list[dict]:
        rows = self._conn().execute("""
            SELECT rank, team_name, team_abbr, played, wins, losses, nrr, points
            FROM standings ORDER BY rank ASC
        """).fetchall()
        return [{"rank": r[0], "team": r[1], "abbr": r[2], "played": r[3],
                 "wins": r[4], "losses": r[5], "nrr": round(r[6], 3), "points": r[7]}
                for r in rows]

    # ── Phase analysis ────────────────────────────────────────────────────

    def powerplay_stats_by_team(self) -> list[dict]:
        rows = self._conn().execute("""
            SELECT m.team1 as team,
                   ROUND(AVG(pp.runs),1) as avg_pp_runs,
                   ROUND(AVG(pp.wickets),2) as avg_pp_wkts,
                   COUNT(*) as matches
            FROM (
                SELECT match_id, SUM(run) as runs, SUM(wicket) as wickets
                FROM deliveries WHERE inning=1 AND over_num<=6
                GROUP BY match_id
            ) pp JOIN matches m ON pp.match_id = m.match_id
            GROUP BY m.team1 ORDER BY avg_pp_runs DESC
        """).fetchall()
        return [{"team": r[0], "avg_runs": r[1], "avg_wickets": r[2], "matches": r[3]} for r in rows]

    def death_overs_stats_by_team(self) -> list[dict]:
        rows = self._conn().execute("""
            SELECT m.team1 as team,
                   ROUND(AVG(d.runs),1) as avg_death_runs,
                   COUNT(*) as matches
            FROM (
                SELECT match_id, SUM(run) as runs
                FROM deliveries WHERE inning=1 AND over_num>=17
                GROUP BY match_id
            ) d JOIN matches m ON d.match_id = m.match_id
            GROUP BY m.team1 ORDER BY avg_death_runs DESC
        """).fetchall()
        return [{"team": r[0], "avg_runs": r[1], "matches": r[2]} for r in rows]

    # ── Player-specific ──────────────────────────────────────────────────

    def player_stats(self, player_name: str) -> dict:
        """Get all stats for a named player."""
        # Batting
        bat = self._conn().execute("""
            SELECT player_name, team, matches, runs, highest, average, strike_rate,
                   hundreds, fifties, fours, sixes, not_outs
            FROM tournament_batting
            WHERE LOWER(player_name) LIKE LOWER(?)
            LIMIT 1
        """, [f"%{player_name}%"]).fetchone()

        # Bowling
        bowl = self._conn().execute("""
            SELECT player_name, team, matches, wickets, overs, economy, average, strike_rate
            FROM tournament_bowling
            WHERE LOWER(player_name) LIKE LOWER(?)
            LIMIT 1
        """, [f"%{player_name}%"]).fetchone()

        result = {"player": player_name}
        if bat:
            result["batting"] = {
                "player": bat[0], "team": bat[1], "matches": bat[2], "runs": bat[3],
                "highest": bat[4], "average": bat[5], "strike_rate": bat[6],
                "hundreds": bat[7], "fifties": bat[8], "fours": bat[9],
                "sixes": bat[10], "not_outs": bat[11]
            }
        if bowl:
            result["bowling"] = {
                "player": bowl[0], "team": bowl[1], "matches": bowl[2], "wickets": bowl[3],
                "overs": bowl[4], "economy": bowl[5], "average": bowl[6], "strike_rate": bowl[7]
            }
        return result

    # ── Match lookup ─────────────────────────────────────────────────────

    def match_summary(self, match_number: int) -> dict:
        r = self._conn().execute("""
            SELECT match_id, title, short_title, date, venue, result
            FROM matches WHERE match_number=?
        """, [match_number]).fetchone()
        if not r:
            return {}

        match_id = r[0]
        innings = self._conn().execute(
            "SELECT inning_number, name, scores_full FROM innings WHERE match_id=? ORDER BY inning_number",
            [match_id]
        ).fetchall()

        return {
            "match_id": match_id,
            "title": r[1],
            "short_title": r[2],
            "date": r[3],
            "venue": r[4],
            "result": r[5],
            "innings": [{"number": i[0], "team": i[1], "score": i[2]} for i in innings],
        }

    def all_matches(self) -> list[dict]:
        rows = self._conn().execute(
            "SELECT match_number, short_title, date, result FROM matches ORDER BY match_number"
        ).fetchall()
        return [{"match_number": r[0], "title": r[1], "date": r[2], "result": r[3]} for r in rows]