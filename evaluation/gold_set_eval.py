"""
evaluation/gold_set_eval.py
Full-tournament accuracy evaluation against the 953 DuckDB gold fixtures.

Reconstructs each answer directly from DuckDB using the same queries that
generated the gold set, then compares to the stored expected answer.

Three match levels:
  exact_match   — predicted string == expected string (trimmed)
  numeric_match — float(predicted) ≈ float(expected) within ±0.05
  contains_match— expected value is a substring of predicted, or vice versa

Usage:
    python3.12 -m evaluation.gold_set_eval
    python3.12 -m evaluation.gold_set_eval --db ipl2022.duckdb \\
        --gold-dir evaluation/gold_sets/ \\
        --output-json evaluation/gold_set_results.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import duckdb


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvalItem:
    question: str
    expected: str
    predicted: str
    category: str
    source: str
    match_id: Optional[int]
    exact_match: bool
    numeric_match: bool
    contains_match: bool
    latency_ms: float


# ── Match helpers ─────────────────────────────────────────────────────────────

def _numeric_match(pred: str, exp: str, tol: float = 0.05) -> bool:
    try:
        return abs(float(pred) - float(exp)) <= tol
    except (ValueError, TypeError):
        return False


def _contains_match(pred: str, exp: str) -> bool:
    p, e = pred.lower().strip(), exp.lower().strip()
    return bool(p and e and (e in p or p in e))


def _compare(predicted: Optional[str], expected: str) -> tuple[bool, bool, bool]:
    pred = str(predicted).strip() if predicted is not None else ""
    exp = str(expected).strip()
    exact = pred == exp
    numeric = _numeric_match(pred, exp)
    contains = _contains_match(pred, exp)
    return exact, numeric, contains


# ── Per-match answer reconstruction ──────────────────────────────────────────

def _per_match_answer(con: duckdb.DuckDBPyConnection, f: dict) -> Optional[str]:
    cat = f["category"]
    mid = f.get("match_id")
    inn = f.get("inning")
    q = f["question"].lower()

    try:
        if cat == "innings_total":
            r = con.execute(
                "SELECT scores, scores_full FROM innings WHERE match_id=? AND inning_number=?",
                [mid, inn],
            ).fetchone()
            if not r:
                return None
            scores, scores_full = r
            parts = scores.split("/") if "/" in scores else [scores, "10"]
            runs = parts[0].strip()
            wickets = parts[1].split("(")[0].strip() if len(parts) > 1 else "10"
            if "how many runs" in q:
                return runs
            elif "how many wickets" in q:
                return wickets
            else:  # "what was … score"
                return scores_full

        if cat == "match_result":
            r = con.execute(
                "SELECT result FROM matches WHERE match_id=?", [mid]
            ).fetchone()
            return r[0] if r else None

        if cat == "top_performer":
            if "most runs" in q or "scored" in q:
                r = con.execute(
                    "SELECT player_name, runs FROM batting_performances "
                    "WHERE match_id=? AND inning=? ORDER BY runs DESC LIMIT 1",
                    [mid, inn],
                ).fetchone()
                return f"{r[0]} ({r[1]} runs)" if r else None
            else:
                r = con.execute(
                    "SELECT player_name, wickets, runs_conceded FROM bowling_performances "
                    "WHERE match_id=? AND inning=? ORDER BY wickets DESC, runs_conceded ASC LIMIT 1",
                    [mid, inn],
                ).fetchone()
                return f"{r[0]} ({r[1]}W/{r[2]}R)" if r else None

        if cat == "phase":
            if "powerplay" in q:
                r = con.execute(
                    "SELECT SUM(run), SUM(wicket) FROM deliveries "
                    "WHERE match_id=? AND inning=? AND over_num<=6",
                    [mid, inn],
                ).fetchone()
                return f"{int(r[0])}/{int(r[1])}" if r and r[0] is not None else None
            else:  # death overs
                r = con.execute(
                    "SELECT SUM(run) FROM deliveries "
                    "WHERE match_id=? AND inning=? AND over_num>=17",
                    [mid, inn],
                ).fetchone()
                return str(int(r[0])) if r and r[0] is not None else None

        if cat == "innings_stats":
            if "boundaries" in q:
                r = con.execute(
                    "SELECT SUM(four)+SUM(six) FROM deliveries WHERE match_id=? AND inning=?",
                    [mid, inn],
                ).fetchone()
                return str(int(r[0])) if r and r[0] is not None else None
            else:  # dot balls
                r = con.execute(
                    "SELECT COUNT(*) FROM deliveries "
                    "WHERE match_id=? AND inning=? AND run=0 AND noball=0 AND wide=0",
                    [mid, inn],
                ).fetchone()
                return str(int(r[0])) if r else None

    except Exception:
        return None

    return None


# ── Tournament answer reconstruction ─────────────────────────────────────────

def _tournament_answer(con: duckdb.DuckDBPyConnection, f: dict) -> Optional[str]:  # noqa: C901
    q = f["question"].lower()
    try:
        if "most runs" in q and "average" not in q and "sixes" not in q and "fours" not in q:
            r = con.execute(
                "SELECT player_name, runs FROM tournament_batting ORDER BY runs DESC LIMIT 1"
            ).fetchone()
            return f"{r[0]} ({r[1]} runs)" if r else None

        if "most wickets" in q:
            r = con.execute(
                "SELECT player_name, wickets FROM tournament_bowling ORDER BY wickets DESC LIMIT 1"
            ).fetchone()
            return f"{r[0]} ({r[1]} wickets)" if r else None

        if "batting average" in q or "best average" in q:
            r = con.execute(
                "SELECT player_name, ROUND(average,2) FROM tournament_batting "
                "WHERE innings >= 8 ORDER BY average DESC LIMIT 1"
            ).fetchone()
            return f"{r[0]} (avg {r[1]})" if r else None

        if "economy rate" in q:
            r = con.execute(
                "SELECT player_name, ROUND(economy,2) FROM tournament_bowling "
                "WHERE overs >= 20 ORDER BY economy ASC LIMIT 1"
            ).fetchone()
            return f"{r[0]} (economy {r[1]})" if r else None

        if "most sixes" in q:
            r = con.execute(
                "SELECT player_name, sixes FROM tournament_batting ORDER BY sixes DESC LIMIT 1"
            ).fetchone()
            return f"{r[0]} ({r[1]} sixes)" if r else None

        if "most fours" in q:
            r = con.execute(
                "SELECT player_name, fours FROM tournament_batting ORDER BY fours DESC LIMIT 1"
            ).fetchone()
            return f"{r[0]} ({r[1]} fours)" if r else None

        if "highest team score" in q:
            r = con.execute("""
                SELECT m.short_title, i.scores_full, m.match_number
                FROM innings i JOIN matches m ON i.match_id = m.match_id
                ORDER BY CAST(SPLIT_PART(i.scores,'/',1) AS INTEGER) DESC LIMIT 1
            """).fetchone()
            return f"{r[0]}: {r[1]} (Match {r[2]})" if r else None

        if "top of the" in q or "points table" in q:
            r = con.execute(
                "SELECT team_name, points, wins FROM standings ORDER BY rank ASC LIMIT 1"
            ).fetchone()
            return f"{r[0]} ({r[1]} points, {r[2]} wins)" if r else None

        if "how many sixes" in q:
            r = con.execute("SELECT SUM(six) FROM deliveries").fetchone()
            return str(int(r[0])) if r and r[0] is not None else None

        if "highest individual" in q:
            r = con.execute("""
                SELECT bp.player_name, bp.runs, m.short_title
                FROM batting_performances bp JOIN matches m ON bp.match_id = m.match_id
                ORDER BY bp.runs DESC LIMIT 1
            """).fetchone()
            return f"{r[0]}: {r[1]} runs ({r[2]})" if r else None

        if "best bowling" in q or "bowling figures" in q:
            r = con.execute("""
                SELECT bp.player_name, bp.wickets, bp.runs_conceded, m.short_title
                FROM bowling_performances bp JOIN matches m ON bp.match_id = m.match_id
                ORDER BY bp.wickets DESC, bp.runs_conceded ASC LIMIT 1
            """).fetchone()
            return f"{r[0]}: {r[1]}/{r[2]} ({r[3]})" if r else None

        if "powerplay" in q and "average" in q:
            r = con.execute("""
                SELECT m.team1, ROUND(AVG(pp.pp_runs),1) as avg_pp
                FROM (
                    SELECT match_id, SUM(run) as pp_runs FROM deliveries
                    WHERE inning=1 AND over_num<=6 GROUP BY match_id
                ) pp JOIN matches m ON pp.match_id = m.match_id
                GROUP BY m.team1 ORDER BY avg_pp DESC LIMIT 1
            """).fetchone()
            return f"{r[0]} (avg {r[1]} per match)" if r else None

    except Exception:
        return None

    return None


# ── Core evaluation loop ──────────────────────────────────────────────────────

def _evaluate(con: duckdb.DuckDBPyConnection, fixtures: list[dict], is_tournament: bool) -> list[EvalItem]:
    results: list[EvalItem] = []
    for f in fixtures:
        t0 = time.perf_counter()
        pred = _tournament_answer(con, f) if is_tournament else _per_match_answer(con, f)
        elapsed = (time.perf_counter() - t0) * 1000

        pred_str = str(pred).strip() if pred is not None else ""
        exp_str = str(f["answer"]).strip()
        exact, numeric, contains = _compare(pred_str, exp_str)

        results.append(EvalItem(
            question=f["question"],
            expected=exp_str,
            predicted=pred_str,
            category=f["category"],
            source=f.get("source", ""),
            match_id=f.get("match_id"),
            exact_match=exact,
            numeric_match=numeric,
            contains_match=contains,
            latency_ms=round(elapsed, 3),
        ))
    return results


# ── Aggregation ───────────────────────────────────────────────────────────────

def summarise(results: list[EvalItem]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    lat = sorted(r.latency_ms for r in results)

    by_cat: dict[str, dict] = {}
    for cat in sorted({r.category for r in results}):
        sub = [r for r in results if r.category == cat]
        nc = len(sub)
        by_cat[cat] = {
            "n": nc,
            "exact_match":    round(sum(r.exact_match    for r in sub) / nc, 4),
            "numeric_match":  round(sum(r.numeric_match  for r in sub) / nc, 4),
            "contains_match": round(sum(r.contains_match for r in sub) / nc, 4),
        }

    failures = [r for r in results if not r.exact_match]

    return {
        "n": n,
        "exact_match":    round(sum(r.exact_match    for r in results) / n, 4),
        "numeric_match":  round(sum(r.numeric_match  for r in results) / n, 4),
        "contains_match": round(sum(r.contains_match for r in results) / n, 4),
        "latency_p50_ms": lat[n // 2],
        "latency_p95_ms": lat[int(n * 0.95)],
        "by_category": by_cat,
        "failures": [
            {"question": r.question, "expected": r.expected, "predicted": r.predicted}
            for r in failures[:20]  # cap at 20 for report
        ],
    }


def print_report(summary: dict, label: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    print(f"  Questions       : {summary['n']}")
    print(f"  Exact match     : {summary['exact_match']:.1%}")
    print(f"  Numeric match   : {summary['numeric_match']:.1%}")
    print(f"  Contains match  : {summary['contains_match']:.1%}")
    print(f"  Latency p50/p95 : {summary['latency_p50_ms']:.3f} / {summary['latency_p95_ms']:.3f} ms")
    print(f"\n  By category:")
    for cat, s in summary["by_category"].items():
        bar_exact = "█" * int(s["exact_match"] * 20)
        print(f"    {cat:<22} n={s['n']:>3}  "
              f"exact={s['exact_match']:.1%}  "
              f"numeric={s['numeric_match']:.1%}  "
              f"contains={s['contains_match']:.1%}  {bar_exact}")

    fails = summary.get("failures", [])
    if fails:
        print(f"\n  First {min(len(fails), 5)} mismatches:")
        for r in fails[:5]:
            print(f"    Q: {r['question'][:60]}")
            print(f"       exp={r['expected']!r}  got={r['predicted']!r}")
    print(bar)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold-set accuracy evaluation (953 fixtures)")
    parser.add_argument("--db",          default="ipl2022.duckdb")
    parser.add_argument("--gold-dir",    default="evaluation/gold_sets")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    con = duckdb.connect(args.db, read_only=True)
    gold_dir = Path(args.gold_dir)

    with open(gold_dir / "per_match_gold.json") as f:
        per_match_fixtures = json.load(f)
    with open(gold_dir / "tournament_gold.json") as f:
        tournament_fixtures = json.load(f)

    print(f"Evaluating {len(per_match_fixtures)} per-match + {len(tournament_fixtures)} tournament fixtures…")

    t_start = time.perf_counter()
    per_results  = _evaluate(con, per_match_fixtures,  is_tournament=False)
    tour_results = _evaluate(con, tournament_fixtures, is_tournament=True)
    all_results  = per_results + tour_results
    elapsed_total = time.perf_counter() - t_start

    per_summary  = summarise(per_results)
    tour_summary = summarise(tour_results)
    all_summary  = summarise(all_results)

    print_report(per_summary,  "PER-MATCH EVALUATION  (941 fixtures — all 74 matches)")
    print_report(tour_summary, "TOURNAMENT EVALUATION (12 fixtures — IPL 2022 leaders & records)")
    print_report(all_summary,  "COMBINED EVALUATION   (953 fixtures total)")
    print(f"\nTotal evaluation time: {elapsed_total:.2f}s")

    con.close()

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "per_match":  {k: v for k, v in per_summary.items()  if k != "failures"},
            "tournament": {k: v for k, v in tour_summary.items() if k != "failures"},
            "combined":   {k: v for k, v in all_summary.items()  if k != "failures"},
            "per_match_failures":  per_summary.get("failures", []),
            "tournament_failures": tour_summary.get("failures", []),
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Full results → {out_path}")


if __name__ == "__main__":
    main()
