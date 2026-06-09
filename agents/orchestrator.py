"""
CricketOrchestrator — pipeline coordinator.

Execution order (per-match path):
  1. QueryRouter   → intent + entities (< 1ms, no LLM)
  2. StatsEngine   → deterministic stats (< 5ms)
  3. AnalystAgent  → natural-language formatting (optional, ~500ms–2s)

Tournament path (when ipl2022.duckdb is present):
  Questions about IPL 2022 overall (standings, top scorers, records …)
  are routed directly to DuckDBEngine — no commentary data required.

When no AnalystAgent is provided the orchestrator answers entirely
from deterministic computation, which is more accurate and faster
for all purely numerical queries.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional  # noqa: F401 — Optional used in type hints

from agents.query_router import Intent, QueryPlan, QueryRouter
from agents.stats_engine import StatsEngine

logger = logging.getLogger(__name__)

# Lazy import — gracefully absent if duckdb not installed or file missing
try:
    from agents.duckdb_engine import DuckDBEngine as _DuckDBEngine
    _DUCKDB_AVAILABLE = True
except Exception:  # ImportError or anything else
    _DUCKDB_AVAILABLE = False
    _DuckDBEngine = None  # type: ignore[assignment,misc]

# Keywords that unambiguously indicate a tournament-wide question.
# None of these appear in per-match benchmark questions.
_TOURNAMENT_KEYWORDS = (
    "ipl",
    "2022",
    "tournament",
    "standings",
    "points table",
    "most runs",
    "most wickets",
    "most sixes",
    "most fours",
    "top scorer",
    "top wicket",
    "best economy",
    "best bowling",
    "highest score",
    "highest team",
)

# Maps each intent to the key in the stats dict that holds the primary answer.
_ANSWER_KEY: dict[Intent, str] = {
    Intent.TOTAL_RUNS:            "total_runs",
    Intent.RUNS_LAST_N_OVERS:     "runs_last_n_overs",
    Intent.RUNS_IN_OVER_RANGE:    "runs_in_overs",
    Intent.TOTAL_WICKETS:         "total_wickets",
    Intent.WICKETS_LAST_N_OVERS:  "wickets_last_n_overs",
    Intent.WICKETS_IN_OVER_RANGE: "wickets_in_overs",
    Intent.TOTAL_FOURS:           "total_fours",
    Intent.TOTAL_SIXES:           "total_sixes",
    Intent.BOUNDARIES:            "total_boundaries",
    Intent.DOT_BALLS:             "dot_balls",
    Intent.RUN_RATE:              "run_rate",
}


class CricketOrchestrator:
    """
    Coordinates the full question-answering pipeline.

    Parameters
    ----------
    analyst : optional AnalystAgent
        When provided, used to produce a natural-language answer.
        When absent, answers are formatted deterministically.
    db_path : str
        Path to ipl2022.duckdb.  DuckDBEngine is loaded only if the file
        exists; otherwise self._db is None and tournament queries return
        a graceful error message.
    """

    def __init__(self, analyst=None, db_path: str = "ipl2022.duckdb") -> None:
        self.analyst = analyst
        self._db: Optional[Any] = None
        if _DUCKDB_AVAILABLE and os.path.exists(db_path):
            try:
                self._db = _DuckDBEngine(db_path)  # type: ignore[call-arg]
                logger.info("DuckDBEngine loaded from %s", db_path)
            except Exception as exc:
                logger.warning(
                    "DuckDBEngine init failed (%s) — tournament queries disabled", exc
                )

    # ── Tournament detection ───────────────────────────────────────────────

    @staticmethod
    def _is_tournament_question(q: str) -> bool:
        return any(kw in q for kw in _TOURNAMENT_KEYWORDS)

    # ── Tournament (DuckDB) path ───────────────────────────────────────────

    def answer_tournament(self, question: str) -> dict[str, Any]:
        """Route a tournament-wide IPL question to DuckDBEngine."""
        if self._db is None:
            return {
                "answer": (
                    "Tournament database not available. "
                    "Run scripts/build_ipl_db.py first."
                ),
                "intent": "tournament_error",
                "data": None,
                "source": "duckdb",
            }
        q = question.lower()
        try:
            return self._route_tournament(q, question)
        except Exception as exc:
            logger.exception("DuckDB query failed: %s", exc)
            return {
                "answer": "Could not complete the tournament query.",
                "intent": "tournament_error",
                "data": None,
                "source": "duckdb",
            }

    def _route_tournament(  # noqa: C901
        self, q: str, original: str
    ) -> dict[str, Any]:
        db = self._db

        if any(kw in q for kw in ("most runs", "top scorer", "run scorer")):
            data = db.top_run_scorers(5)
            top = data[0] if data else {}
            return {
                "answer": f"{top.get('player','?')} ({top.get('runs','?')} runs)",
                "intent": "top_run_scorers",
                "data": data,
                "source": "duckdb",
            }

        if any(kw in q for kw in ("most wickets", "top wicket")):
            data = db.top_wicket_takers(5)
            top = data[0] if data else {}
            return {
                "answer": f"{top.get('player','?')} ({top.get('wickets','?')} wickets)",
                "intent": "top_wicket_takers",
                "data": data,
                "source": "duckdb",
            }

        if "most sixes" in q:
            data = db.most_sixes(5)
            top = data[0] if data else {}
            return {
                "answer": f"{top.get('player','?')} ({top.get('sixes','?')} sixes)",
                "intent": "most_sixes",
                "data": data,
                "source": "duckdb",
            }

        if "most fours" in q:
            data = db.most_fours(5)
            top = data[0] if data else {}
            return {
                "answer": f"{top.get('player','?')} ({top.get('fours','?')} fours)",
                "intent": "most_fours",
                "data": data,
                "source": "duckdb",
            }

        if "best economy" in q or "economy rate" in q:
            data = db.best_economy_rates()
            top = data[0] if data else {}
            return {
                "answer": f"{top.get('player','?')} (economy {top.get('economy','?')})",
                "intent": "best_economy_rates",
                "data": data,
                "source": "duckdb",
            }

        if "best average" in q or "batting average" in q:
            data = db.best_batting_average()
            top = data[0] if data else {}
            return {
                "answer": f"{top.get('player','?')} (avg {top.get('average','?')})",
                "intent": "best_batting_average",
                "data": data,
                "source": "duckdb",
            }

        if "highest score" in q or "highest team" in q:
            data = db.highest_team_score()
            return {
                "answer": (
                    f"{data.get('match','?')}: {data.get('score','?')} "
                    f"(Match {data.get('match_number','?')})"
                ),
                "intent": "highest_team_score",
                "data": data,
                "source": "duckdb",
            }

        if "best bowling" in q or "bowling figures" in q:
            data = db.best_bowling_figures()
            return {
                "answer": (
                    f"{data.get('player','?')}: "
                    f"{data.get('wickets','?')}/{data.get('runs','?')} "
                    f"({data.get('match','?')})"
                ),
                "intent": "best_bowling_figures",
                "data": data,
                "source": "duckdb",
            }

        if "total sixes" in q or ("how many sixes" in q and "ipl" in q):
            total = db.tournament_sixes_total()
            return {
                "answer": str(total),
                "intent": "tournament_sixes",
                "data": total,
                "source": "duckdb",
            }

        if "total fours" in q or ("how many fours" in q and "ipl" in q):
            total = db.tournament_fours_total()
            return {
                "answer": str(total),
                "intent": "tournament_fours",
                "data": total,
                "source": "duckdb",
            }

        if "points table" in q or "standings" in q:
            data = db.points_table()
            top = data[0] if data else {}
            return {
                "answer": (
                    f"{top.get('team','?')} top "
                    f"({top.get('points','?')} pts, {top.get('wins','?')} wins)"
                ),
                "intent": "points_table",
                "data": data,
                "source": "duckdb",
            }

        if "powerplay" in q and "team" in q:
            data = db.powerplay_stats_by_team()
            top = data[0] if data else {}
            return {
                "answer": f"{top.get('team','?')} (avg {top.get('avg_runs','?')} PP runs)",
                "intent": "powerplay_stats",
                "data": data,
                "source": "duckdb",
            }

        if "death over" in q and "team" in q:
            data = db.death_overs_stats_by_team()
            top = data[0] if data else {}
            return {
                "answer": (
                    f"{top.get('team','?')} "
                    f"(avg {top.get('avg_runs','?')} death-over runs)"
                ),
                "intent": "death_overs_stats",
                "data": data,
                "source": "duckdb",
            }

        # Player name heuristic — capitalised words
        names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', original)
        if names:
            name = names[0]
            data = db.player_stats(name)
            if data.get("batting") or data.get("bowling"):
                parts: list[str] = []
                if data.get("batting"):
                    parts.append(f"batting: {data['batting'].get('runs', 0)} runs")
                if data.get("bowling"):
                    parts.append(f"bowling: {data['bowling'].get('wickets', 0)} wickets")
                return {
                    "answer": f"{name}: {', '.join(parts)}",
                    "intent": "player_stats",
                    "data": data,
                    "source": "duckdb",
                }

        return {
            "answer": "Could not answer from tournament database.",
            "intent": "tournament_unknown",
            "data": None,
            "source": "duckdb",
        }

    @staticmethod
    def _build_duckdb_context(t: dict[str, Any]) -> str:
        lines = ["Source: duckdb", f"Intent: {t['intent']}"]
        data = t.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                for k, v in list(first.items())[:6]:
                    lines.append(f"- {k.replace('_', ' ').title()}: {v}")
        elif isinstance(data, dict) and data:
            for k, v in list(data.items())[:6]:
                lines.append(f"- {k.replace('_', ' ').title()}: {v}")
        return "\n".join(lines)

    # ── Main entry point ───────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        commentary_data: dict[str, Any],
    ) -> dict[str, Any]:
        q = question.lower()

        # ── Tournament path: DuckDB if question is about IPL overall ──────
        if self._db is not None and self._is_tournament_question(q):
            t = self.answer_tournament(question)
            return {
                "answer":       t["answer"],
                "intent":       t["intent"],
                "confidence":   1.0,
                "context_used": self._build_duckdb_context(t),
                "stats":        {},
                "data":         t.get("data"),
                "llm_used":     False,
            }

        # ── Per-match path (unchanged) ────────────────────────────────────
        engine = StatsEngine(commentary_data)
        plan = QueryRouter.route(question)

        logger.info("intent=%s confidence=%.2f question=%r",
                    plan.intent.value, plan.confidence, question)

        stats = self._execute(plan, engine)
        context = self._build_context(stats)

        if self.analyst is not None:
            expected_value = self._primary_value(plan, stats)
            answer = self.analyst.act(
                question=question,
                context=context,
                expected_value=expected_value,
            )
        else:
            answer = self._format(plan, stats)

        return {
            "answer":       answer,
            "intent":       plan.intent.value,
            "confidence":   round(plan.confidence, 2),
            "context_used": context,
            "stats":        stats,
            "data":         None,
            "llm_used":     self.analyst is not None,
        }

    # ── Plan execution ─────────────────────────────────────────────────────

    def _execute(self, plan: QueryPlan, engine: StatsEngine) -> dict[str, Any]:
        intent = plan.intent
        ent = plan.entities

        try:
            if intent == Intent.TOTAL_RUNS:
                return {
                    "total_runs": engine.total_runs(),
                    "total_wickets": engine.total_wickets(),
                }

            if intent == Intent.RUNS_LAST_N_OVERS:
                n = int(ent.get("n", 5))
                return {"runs_last_n_overs": engine.runs_last_n_overs(n), "n": n}

            if intent == Intent.RUNS_IN_OVER_RANGE:
                s, e = int(ent["start_over"]), int(ent["end_over"])
                return {"runs_in_overs": engine.runs_in_overs(s, e),
                        "start_over": s, "end_over": e}

            if intent == Intent.TOTAL_WICKETS:
                return {"total_wickets": engine.total_wickets()}

            if intent == Intent.WICKETS_LAST_N_OVERS:
                n = int(ent.get("n", 5))
                return {"wickets_last_n_overs": engine.wickets_last_n_overs(n), "n": n}

            if intent == Intent.WICKETS_IN_OVER_RANGE:
                s, e = int(ent["start_over"]), int(ent["end_over"])
                return {"wickets_in_overs": engine.wickets_in_overs(s, e),
                        "start_over": s, "end_over": e}

            if intent == Intent.TOTAL_FOURS:
                return {"total_fours": engine.total_fours()}

            if intent == Intent.TOTAL_SIXES:
                return {"total_sixes": engine.total_sixes()}

            if intent == Intent.BOUNDARIES:
                fours = engine.total_fours()
                sixes = engine.total_sixes()
                return {"total_fours": fours, "total_sixes": sixes,
                        "total_boundaries": fours + sixes}

            if intent == Intent.DOT_BALLS:
                return {"dot_balls": engine.dot_balls()}

            if intent == Intent.RUN_RATE:
                return {"run_rate": engine.run_rate()}

            if intent == Intent.POWERPLAY:
                pp = engine.powerplay_stats()
                return {
                    "powerplay_runs": pp.runs,
                    "powerplay_wickets": pp.wickets,
                    "powerplay_run_rate": pp.run_rate,
                    "powerplay_fours": pp.fours,
                    "powerplay_sixes": pp.sixes,
                    "powerplay_overs": f"1–{pp.end_over}",
                }

            if intent == Intent.DEATH_OVERS:
                do = engine.death_overs_stats()
                return {
                    "death_overs_runs": do.runs,
                    "death_overs_wickets": do.wickets,
                    "death_overs_run_rate": do.run_rate,
                    "death_overs_range": f"{do.start_over}–{do.end_over}",
                }

            if intent == Intent.TOP_SCORER:
                ts = engine.top_scorer()
                if ts:
                    return {
                        "top_scorer": ts.name,
                        "runs": ts.runs,
                        "balls_faced": ts.balls_faced,
                        "strike_rate": ts.strike_rate,
                    }
                return {"error": "player_data_unavailable"}

            if intent == Intent.PLAYER_RUNS:
                player = str(ent.get("player", ""))
                runs = engine.player_runs(player)
                if runs is not None:
                    return {"player": player, "runs": runs}
                return {"error": "player_data_unavailable"}

            if intent == Intent.OVER_SUMMARY:
                return {"over_summary": engine.over_by_over()}

            # UNKNOWN / fallback
            return {
                "total_runs": engine.total_runs(),
                "total_wickets": engine.total_wickets(),
                "total_fours": engine.total_fours(),
                "total_sixes": engine.total_sixes(),
                "dot_balls": engine.dot_balls(),
                "run_rate": engine.run_rate(),
            }

        except Exception:
            logger.exception("Plan execution failed for intent=%s", intent)
            raise

    # ── Primary-value extractor ────────────────────────────────────────────

    @staticmethod
    def _primary_value(plan: QueryPlan, stats: dict[str, Any]) -> Optional[str]:
        if "error" in stats:
            return None
        key = _ANSWER_KEY.get(plan.intent)
        if key and key in stats:
            return str(stats[key])
        return None

    # ── Context builder ────────────────────────────────────────────────────

    @staticmethod
    def _build_context(stats: dict[str, Any]) -> str:
        if "error" in stats:
            return "Context: data unavailable"
        lines = ["Context:"]
        for k, v in stats.items():
            if k not in {"n", "start_over", "end_over"}:
                label = k.replace("_", " ").title()
                lines.append(f"- {label}: {v}")
        return "\n".join(lines)

    # ── Deterministic formatter (no LLM path) ─────────────────────────────

    @staticmethod
    def _format(plan: QueryPlan, stats: dict[str, Any]) -> str:
        if "error" in stats:
            return "The information is not available in the provided data."

        intent = plan.intent

        if intent == Intent.TOTAL_RUNS:
            return str(stats.get("total_runs", "N/A"))

        if intent == Intent.RUNS_LAST_N_OVERS:
            return str(stats.get("runs_last_n_overs", "N/A"))

        if intent == Intent.RUNS_IN_OVER_RANGE:
            return str(stats.get("runs_in_overs", "N/A"))

        if intent == Intent.TOTAL_WICKETS:
            return str(stats.get("total_wickets", "N/A"))

        if intent == Intent.WICKETS_LAST_N_OVERS:
            return str(stats.get("wickets_last_n_overs", "N/A"))

        if intent == Intent.WICKETS_IN_OVER_RANGE:
            return str(stats.get("wickets_in_overs", "N/A"))

        if intent == Intent.TOTAL_FOURS:
            return str(stats.get("total_fours", "N/A"))

        if intent == Intent.TOTAL_SIXES:
            return str(stats.get("total_sixes", "N/A"))

        if intent == Intent.BOUNDARIES:
            return str(stats.get("total_boundaries", "N/A"))

        if intent == Intent.DOT_BALLS:
            return str(stats.get("dot_balls", "N/A"))

        if intent == Intent.RUN_RATE:
            return str(stats.get("run_rate", "N/A"))

        if intent == Intent.POWERPLAY:
            r = stats.get("powerplay_runs")
            w = stats.get("powerplay_wickets")
            rr = stats.get("powerplay_run_rate")
            return f"{r}/{w} (RR {rr})" if r is not None else "N/A"

        if intent == Intent.DEATH_OVERS:
            r = stats.get("death_overs_runs")
            w = stats.get("death_overs_wickets")
            return f"{r}/{w}" if r is not None else "N/A"

        if intent == Intent.TOP_SCORER:
            name = stats.get("top_scorer")
            runs = stats.get("runs")
            return f"{name} ({runs})" if name else "N/A"

        if intent == Intent.PLAYER_RUNS:
            return str(stats.get("runs", "N/A"))

        # UNKNOWN / OVER_SUMMARY
        vals = [f"{k}: {v}" for k, v in stats.items()
                if k not in {"n", "start_over", "end_over", "over_summary"}]
        return "; ".join(vals) if vals else "The information is not available."
