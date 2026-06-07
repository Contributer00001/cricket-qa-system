"""
Pandas-based cricket analytics engine.

Replaces the original StatsTool with a comprehensive engine that supports
arbitrary over ranges, optional player/bowling analytics, and phase breakdowns.
Degrades gracefully when optional fields (batsman, bowler) are absent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OverRangeStats:
    start_over: int
    end_over: int
    runs: int
    wickets: int
    fours: int
    sixes: int
    dot_balls: int
    balls: int

    @property
    def run_rate(self) -> float:
        overs = self.balls / 6.0
        return round(self.runs / overs, 2) if overs > 0 else 0.0

    def to_context(self) -> str:
        return (
            f"Overs {self.start_over}–{self.end_over}: "
            f"{self.runs} runs, {self.wickets} wickets, "
            f"RR {self.run_rate:.2f}, {self.fours} fours, {self.sixes} sixes"
        )


@dataclass
class PlayerStats:
    name: str
    runs: int
    balls_faced: int
    fours: int
    sixes: int
    strike_rate: float
    dismissal: Optional[str] = None


class StatsEngine:
    """
    Comprehensive cricket statistics engine built on pandas.

    Required fields per delivery: event, over, run
    Optional fields: four, six, batsman, bowler, extras, extra_type, wicket_type
    """

    _REQUIRED = {"event", "over", "run"}

    def __init__(self, commentary_data: dict[str, Any]) -> None:
        raw = commentary_data.get("commentaries", [])
        if not raw:
            raise ValueError("commentary_data contains no deliveries")

        self.df = pd.DataFrame(raw)
        self._validate()
        self._normalise()
        self._max_over: int = int(self.df["over"].max())

    # ── Setup ──────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        missing = self._REQUIRED - set(self.df.columns)
        if missing:
            raise ValueError(f"Commentary missing required fields: {missing}")

    def _normalise(self) -> None:
        self.df["over"] = self.df["over"].astype(int)
        self.df["run"] = (
            pd.to_numeric(self.df["run"], errors="coerce").fillna(0).astype(int)
        )
        for col in ("four", "six"):
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(bool)
            else:
                self.df[col] = False

        self.df["is_wicket"] = self.df["event"] == "wicket"
        self.df["is_ball"] = self.df["event"].isin({"ball", "wicket"})
        self.df["is_dot"] = (self.df["run"] == 0) & ~self.df["is_wicket"]

        for col in ("batsman", "bowler"):
            if col not in self.df.columns:
                self.df[col] = None

    # ── Over-range helpers ─────────────────────────────────────────────────

    def _slice(self, start: int, end: int) -> pd.DataFrame:
        return self.df[(self.df["over"] >= start) & (self.df["over"] <= end)]

    def _over_stats(self, start: int, end: int) -> OverRangeStats:
        s = self._slice(start, end)
        return OverRangeStats(
            start_over=start,
            end_over=end,
            runs=int(s["run"].sum()),
            wickets=int(s["is_wicket"].sum()),
            fours=int(s["four"].sum()),
            sixes=int(s["six"].sum()),
            dot_balls=int(s["is_dot"].sum()),
            balls=int(s["is_ball"].sum()),
        )

    # ── Basic aggregations ─────────────────────────────────────────────────

    def total_runs(self) -> int:
        return int(self.df["run"].sum())

    def total_wickets(self) -> int:
        return int(self.df["is_wicket"].sum())

    def total_fours(self) -> int:
        return int(self.df["four"].sum())

    def total_sixes(self) -> int:
        return int(self.df["six"].sum())

    def dot_balls(self) -> int:
        return int(self.df["is_dot"].sum())

    def total_balls(self) -> int:
        return int(self.df["is_ball"].sum())

    def run_rate(self, start: int = 1, end: Optional[int] = None) -> float:
        if end is None:
            end = self._max_over
        return self._over_stats(start, end).run_rate

    def max_over(self) -> int:
        return self._max_over

    # ── Over-range queries ─────────────────────────────────────────────────

    def runs_in_overs(self, start: int, end: int) -> int:
        return int(self._slice(start, end)["run"].sum())

    def runs_last_n_overs(self, n: int) -> int:
        start = max(1, self._max_over - n + 1)
        return self.runs_in_overs(start, self._max_over)

    def wickets_in_overs(self, start: int, end: int) -> int:
        return int(self._slice(start, end)["is_wicket"].sum())

    def wickets_last_n_overs(self, n: int) -> int:
        start = max(1, self._max_over - n + 1)
        return self.wickets_in_overs(start, self._max_over)

    def over_stats(self, start: int, end: int) -> OverRangeStats:
        return self._over_stats(start, end)

    # ── Phase breakdowns ───────────────────────────────────────────────────

    def powerplay_stats(self) -> OverRangeStats:
        end = min(6, self._max_over)
        return self._over_stats(1, end)

    def middle_overs_stats(self) -> OverRangeStats:
        start = min(7, self._max_over)
        end = min(15, self._max_over)
        return self._over_stats(start, end)

    def death_overs_stats(self) -> OverRangeStats:
        start = max(1, self._max_over - 3)
        return self._over_stats(start, self._max_over)

    # ── Per-over summary ───────────────────────────────────────────────────

    def over_by_over(self) -> list[dict]:
        results = []
        for ov in sorted(self.df["over"].unique()):
            s = self._over_stats(int(ov), int(ov))
            results.append({
                "over": int(ov),
                "runs": s.runs,
                "wickets": s.wickets,
                "fours": s.fours,
                "sixes": s.sixes,
                "dot_balls": s.dot_balls,
                "run_rate": s.run_rate,
            })
        return results

    # ── Player analytics (degrades gracefully without batsman field) ───────

    def _has_player_data(self) -> bool:
        return self.df["batsman"].notna().any()

    def player_runs(self, name: str) -> Optional[int]:
        if not self._has_player_data():
            return None
        mask = self.df["batsman"].str.lower() == name.lower()
        return int(self.df.loc[mask, "run"].sum())

    def top_scorer(self) -> Optional[PlayerStats]:
        if not self._has_player_data():
            return None
        grp = (
            self.df.groupby("batsman")
            .agg(runs=("run", "sum"), balls=("is_ball", "sum"),
                 fours=("four", "sum"), sixes=("six", "sum"))
            .reset_index()
        )
        if grp.empty:
            return None
        best = grp.loc[grp["runs"].idxmax()]
        sr = round(best["runs"] / best["balls"] * 100, 1) if best["balls"] > 0 else 0.0
        return PlayerStats(
            name=str(best["batsman"]),
            runs=int(best["runs"]),
            balls_faced=int(best["balls"]),
            fours=int(best["fours"]),
            sixes=int(best["sixes"]),
            strike_rate=sr,
        )

    def all_player_stats(self) -> list[PlayerStats]:
        if not self._has_player_data():
            return []
        grp = (
            self.df.groupby("batsman")
            .agg(runs=("run", "sum"), balls=("is_ball", "sum"),
                 fours=("four", "sum"), sixes=("six", "sum"))
            .reset_index()
        )
        out = []
        for _, row in grp.iterrows():
            sr = round(row["runs"] / row["balls"] * 100, 1) if row["balls"] > 0 else 0.0
            out.append(PlayerStats(
                name=str(row["batsman"]),
                runs=int(row["runs"]),
                balls_faced=int(row["balls"]),
                fours=int(row["fours"]),
                sixes=int(row["sixes"]),
                strike_rate=sr,
            ))
        return sorted(out, key=lambda p: p.runs, reverse=True)

    # ── Bowling analytics (degrades gracefully without bowler field) ────────

    def _has_bowling_data(self) -> bool:
        return self.df["bowler"].notna().any()

    def bowler_stats(self, name: str) -> Optional[dict]:
        if not self._has_bowling_data():
            return None
        mask = self.df["bowler"].str.lower() == name.lower()
        s = self.df[mask]
        balls = int(s["is_ball"].sum())
        runs = int(s["run"].sum())
        wickets = int(s["is_wicket"].sum())
        economy = round(runs / (balls / 6.0), 2) if balls > 0 else 0.0
        return {
            "bowler": name,
            "balls": balls,
            "runs": runs,
            "wickets": wickets,
            "economy": economy,
        }

    # ── Context builder for LLM ────────────────────────────────────────────

    def build_context(self, keys: Optional[list[str]] = None) -> str:
        """
        Produce a structured context string for the LLM.

        keys: subset of stat names to include. Defaults to all.
        Valid keys: total_runs, total_wickets, total_fours, total_sixes,
                    dot_balls, run_rate, last_3_overs, last_5_overs,
                    powerplay, death_overs
        """
        available = {
            "total_runs": f"Total runs: {self.total_runs()}",
            "total_wickets": f"Total wickets: {self.total_wickets()}",
            "total_fours": f"Fours: {self.total_fours()}",
            "total_sixes": f"Sixes: {self.total_sixes()}",
            "dot_balls": f"Dot balls: {self.dot_balls()}",
            "run_rate": f"Overall run rate: {self.run_rate():.2f}",
            "last_3_overs": f"Runs in last 3 overs: {self.runs_last_n_overs(3)}",
            "last_5_overs": f"Runs in last 5 overs: {self.runs_last_n_overs(5)}",
            "powerplay": self.powerplay_stats().to_context(),
            "death_overs": self.death_overs_stats().to_context(),
        }
        chosen = keys if keys else list(available.keys())
        lines = ["Context:"] + [f"- {available[k]}" for k in chosen if k in available]
        return "\n".join(lines)
