# scripts/generate_qa.py
import json
from pathlib import Path

from agents.stats_engine import StatsEngine

DATA_DIR = Path("data/Indian_Premier_League_2022-03-26/match_innings_commentary")
OUTPUT_FILE = Path("data/generated_qa.jsonl")


def match_name_from_filename(path: Path) -> str:
    name = path.stem
    name = name.replace("innings_1_", "")
    name = name.replace("_match_innings_1_commentary", "")
    name = name.replace("_vs_", " vs ")
    name = name.replace("_", " ")
    return name


def generate_qas() -> None:
    qa_pairs = []

    for file in sorted(DATA_DIR.glob("*.json")):
        with open(file, encoding="utf-8") as f:
            data = json.load(f)

        engine = StatsEngine(data)
        match_name = match_name_from_filename(file)

        qa_pairs.extend([
            {
                "prompt": f"Question: How many runs were scored in the last 3 overs of the innings in {match_name}?",
                "response": f"{engine.runs_last_n_overs(3)} runs were scored in the last 3 overs.",
            },
            {
                "prompt": f"Question: How many runs were scored in the last 5 overs of the innings in {match_name}?",
                "response": f"{engine.runs_last_n_overs(5)} runs were scored in the last 5 overs.",
            },
            {
                "prompt": f"Question: What was the total score in the innings in {match_name}?",
                "response": f"The total score was {engine.total_runs()} runs.",
            },
            {
                "prompt": f"Question: How many wickets fell in the last 5 overs of the innings in {match_name}?",
                "response": f"{engine.wickets_last_n_overs(5)} wickets fell in the last 5 overs.",
            },
            {
                "prompt": f"Question: How many wickets fell in the innings in {match_name}?",
                "response": f"{engine.total_wickets()} wickets fell in the innings.",
            },
            {
                "prompt": f"Question: How many fours were hit in the innings in {match_name}?",
                "response": f"{engine.total_fours()} fours were hit.",
            },
            {
                "prompt": f"Question: How many sixes were hit in the innings in {match_name}?",
                "response": f"{engine.total_sixes()} sixes were hit.",
            },
            {
                "prompt": f"Question: How many dot balls were bowled in the innings in {match_name}?",
                "response": f"{engine.dot_balls()} dot balls were bowled.",
            },
            {
                "prompt": f"Question: What was the run rate in the powerplay in {match_name}?",
                "response": f"The powerplay run rate was {engine.powerplay_stats().run_rate}.",
            },
        ])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa) + "\n")

    print(f"Generated {len(qa_pairs)} QA pairs → {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_qas()
