"""Simple helper to initialize the IPL database using schema.sql.

This script is intentionally minimal — it reads the `schema.sql` file
and prints it so callers can pipe it into a DB client or expand the
script later to execute against a database connection.
"""

from pathlib import Path


def load_schema():
    root = Path(__file__).resolve().parent
    schema_path = root / "schema.sql"
    return schema_path.read_text(encoding="utf-8")


def main():
    print(load_schema())


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
scripts/build_ipl_db.py
Build ipl2022.duckdb from the Indian Premier League 2022 dataset.

Usage:
    python scripts/build_ipl_db.py --data data/Indian_Premier_League_2022-03-26 --out ipl2022.duckdb
"""
import argparse, json, glob, re, os, sys
import duckdb

def build(data_dir: str, out_path: str):
    if os.path.exists(out_path):
        os.remove(out_path)
    con = duckdb.connect(out_path)

    # ── matches ──────────────────────────────────────────────────────────────
    with open(f"{data_dir}/matches/matches.json") as f:
        matches = json.load(f)

    num_to_id = {str(m["match_number"]): m["match_id"] for m in matches}

    con.execute("""CREATE TABLE matches (
        match_id INTEGER PRIMARY KEY, match_number INTEGER,
        title TEXT, short_title TEXT, date TEXT, venue TEXT, result TEXT,
        team1 TEXT, team2 TEXT, team1_abbr TEXT, team2_abbr TEXT)""")

    rows = []
    for m in matches:
        ta = m.get("teama", {}) or {}
        tb = m.get("teamb", {}) or {}
        rows.append((
            m["match_id"], int(m.get("match_number") or 0),
            m.get("title",""), m.get("short_title",""),
            (m.get("date_start_ist") or "")[:10],
            m.get("venue",{}).get("name","") if isinstance(m.get("venue"),dict) else str(m.get("venue","")),
            m.get("status_note",""),
            ta.get("team_name") or ta.get("title",""),
            tb.get("team_name") or tb.get("title",""),
            ta.get("short_name") or ta.get("abbr",""),
            tb.get("short_name") or tb.get("abbr",""),
        ))
    con.executemany("INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    print(f"  matches: {len(rows)}")

    # ── deliveries ────────────────────────────────────────────────────────────
    con.execute("""CREATE TABLE deliveries (
        match_id INTEGER, inning INTEGER, over_num INTEGER, ball INTEGER,
        batsman TEXT, bowler TEXT, run INTEGER, bat_run INTEGER,
        noball INTEGER, wide INTEGER, four INTEGER, six INTEGER, wicket INTEGER)""")

    deliveries = []
    for fpath in sorted(glob.glob(f"{data_dir}/match_innings_commentary/*.json")):
        mn = re.search(r"Match_(\d+)", os.path.basename(fpath))
        if not mn:
            continue
        match_id = num_to_id.get(mn.group(1))
        if not match_id:
            continue
        match_id = int(match_id)

        with open(fpath) as f:
            data = json.load(f)

        inning_num = int(data.get("inning", {}).get("number", 1))
        pid_name = {str(p["pid"]): p.get("title","") for p in data.get("players", [])}

        for ball in data.get("commentaries", []):
            if ball.get("event") != "ball":
                continue
            batsman = pid_name.get(str(ball.get("batsman_id","")), "")
            bowler = pid_name.get(str(ball.get("bowler_id","")), "")
            comm = ball.get("commentary","")
            if not batsman or not bowler:
                parts = comm.split(" to ")
                if len(parts) >= 2:
                    if not bowler: bowler = parts[0].strip()
                    if not batsman: batsman = parts[1].split(",")[0].strip()
            is_wicket = (
                any(w in comm.lower() for w in ["wicket","caught","bowled","lbw","run out","stumped"])
                and not ball.get("noball_dismissal", False)
                and not ball.get("noball", False)
            )
            deliveries.append((
                match_id, inning_num, int(ball.get("over",0))+1, int(ball.get("ball",0)),
                batsman, bowler, int(ball.get("run",0)), int(ball.get("bat_run",0)),
                1 if ball.get("noball") else 0, 1 if ball.get("wideball") else 0,
                1 if ball.get("four") else 0, 1 if ball.get("six") else 0,
                1 if is_wicket else 0,
            ))

    con.executemany("INSERT INTO deliveries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", deliveries)
    print(f"  deliveries: {len(deliveries)}")

    # ── innings + performances ────────────────────────────────────────────────
    con.execute("""CREATE TABLE innings (
        match_id INTEGER, inning_number INTEGER, name TEXT,
        batting_team_id INTEGER, scores TEXT, scores_full TEXT, max_over INTEGER)""")
    con.execute("""CREATE TABLE batting_performances (
        match_id INTEGER, inning INTEGER, player_name TEXT, player_id TEXT,
        runs INTEGER, balls_faced INTEGER, fours INTEGER, sixes INTEGER)""")
    con.execute("""CREATE TABLE bowling_performances (
        match_id INTEGER, inning INTEGER, player_name TEXT, player_id TEXT,
        overs DOUBLE, maidens INTEGER, runs_conceded INTEGER, wickets INTEGER)""")

    inn_rows, bat_rows, bowl_rows = [], [], []
    for fpath in sorted(glob.glob(f"{data_dir}/scorecards/*.json")):
        with open(fpath) as f:
            sc = json.load(f)
        mid = sc.get("match_id")
        for inn in sc.get("innings", []):
            num = int(inn.get("number",1))
            inn_rows.append((mid, num, inn.get("name",""), inn.get("batting_team_id"),
                             inn.get("scores",""), inn.get("scores_full",""),
                             int(float(inn.get("max_over") or 0))))
            for b in inn.get("batsmen",[]):
                bat_rows.append((mid, num, b.get("name",""), b.get("batsman_id",""),
                                 int(b.get("runs") or 0), int(b.get("balls_faced") or 0),
                                 int(b.get("fours") or 0), int(b.get("sixes") or 0)))
            for bw in inn.get("bowlers",[]):
                bowl_rows.append((mid, num, bw.get("name",""), bw.get("bowler_id",""),
                                  float(bw.get("overs") or 0), int(bw.get("maidens") or 0),
                                  int(bw.get("runs_conceded") or 0), int(bw.get("wickets") or 0)))

    con.executemany("INSERT INTO innings VALUES (?,?,?,?,?,?,?)", inn_rows)
    con.executemany("INSERT INTO batting_performances VALUES (?,?,?,?,?,?,?,?)", bat_rows)
    con.executemany("INSERT INTO bowling_performances VALUES (?,?,?,?,?,?,?,?)", bowl_rows)
    print(f"  innings: {len(inn_rows)}, batting: {len(bat_rows)}, bowling: {len(bowl_rows)}")

    # ── tournament stats ───────────────────────────────────────────────────────
    con.execute("""CREATE TABLE tournament_batting (
        player_id TEXT, player_name TEXT, team TEXT, matches INTEGER, innings INTEGER,
        runs INTEGER, highest INTEGER, average DOUBLE, strike_rate DOUBLE,
        hundreds INTEGER, fifties INTEGER, fours INTEGER, sixes INTEGER, not_outs INTEGER)""")
    con.execute("""CREATE TABLE tournament_bowling (
        player_id TEXT, player_name TEXT, team TEXT, matches INTEGER, innings INTEGER,
        overs DOUBLE, runs_conceded INTEGER, wickets INTEGER,
        economy DOUBLE, average DOUBLE, strike_rate DOUBLE)""")

    with open(f"{data_dir}/batting_stats/batting_most_runs.json") as f:
        br = json.load(f)
    bat_t = []
    for s in br["response"]["stats"]:
        p = s.get("player",{}); team = s.get("team",{})
        bat_t.append((str(p.get("pid","")), p.get("title",""), team.get("abbr",""),
                      int(s.get("matches") or 0), int(s.get("innings") or 0),
                      int(s.get("runs") or 0), int(s.get("highest") or 0),
                      float(s.get("average") or 0), float(s.get("strike") or 0),
                      int(s.get("run100") or 0), int(s.get("run50") or 0),
                      int(s.get("run4") or 0), int(s.get("run6") or 0),
                      int(s.get("notout") or 0)))
    con.executemany("INSERT INTO tournament_batting VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", bat_t)

    with open(f"{data_dir}/bowling_stats/bowling_top_wicket_takers.json") as f:
        bw = json.load(f)
    bowl_t = []
    for s in bw["response"]["stats"]:
        p = s.get("player",{}); team = s.get("team",{})
        bowl_t.append((str(p.get("pid","")), p.get("title",""), team.get("abbr",""),
                       int(s.get("matches") or 0), int(s.get("innings") or 0),
                       float(s.get("overs") or 0), int(s.get("runs") or 0),
                       int(s.get("wickets") or 0), float(s.get("economy") or 0),
                       float(s.get("average") or 0), float(s.get("strike_rate") or 0)))
    con.executemany("INSERT INTO tournament_bowling VALUES (?,?,?,?,?,?,?,?,?,?,?)", bowl_t)
    print(f"  tournament_batting: {len(bat_t)}, tournament_bowling: {len(bowl_t)}")

    # ── standings ─────────────────────────────────────────────────────────────
    con.execute("""CREATE TABLE standings (
        team_id TEXT, team_name TEXT, team_abbr TEXT,
        played INTEGER, wins INTEGER, losses INTEGER, tied INTEGER,
        no_result INTEGER, nrr DOUBLE, points INTEGER, rank INTEGER)""")
    with open(f"{data_dir}/standings/standings.json") as f:
        st = json.load(f)
    stand_rows = []
    for grp in st.get("standings", []):
        for row in grp.get("standings", []):
            team = row.get("team", {})
            stand_rows.append((str(row.get("team_id","")),
                               team.get("title","") if isinstance(team,dict) else str(team),
                               team.get("abbr","") if isinstance(team,dict) else "",
                               int(row.get("played") or 0), int(row.get("win") or 0),
                               int(row.get("los") or 0), int(row.get("tied") or 0),
                               int(row.get("nr") or 0), float(row.get("nrr") or 0),
                               int(row.get("points") or 0), int(row.get("rank") or 0)))
    con.executemany("INSERT INTO standings VALUES (?,?,?,?,?,?,?,?,?,?,?)", stand_rows)
    print(f"  standings: {len(stand_rows)}")

    # ── verify ────────────────────────────────────────────────────────────────
    total_d = con.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]
    top_bat = con.execute("SELECT player_name, runs FROM tournament_batting ORDER BY runs DESC LIMIT 1").fetchone()
    print(f"\nTotal deliveries: {total_d}")
    print(f"Top scorer: {top_bat[0]} - {top_bat[1]} runs")
    con.close()
    print(f"\nDuckDB written to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Indian_Premier_League_2022-03-26")
    parser.add_argument("--out", default="ipl2022.duckdb")
    args = parser.parse_args()
    if not os.path.exists(args.data):
        print(f"ERROR: data directory not found: {args.data}")
        sys.exit(1)
    build(args.data, args.out)


if __name__ == "__main__":
    main()