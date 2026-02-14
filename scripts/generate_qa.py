# scripts/generate_qa.py
import json
import os
from pathlib import Path
from agents.multi_agent import StatsTool

DATA_DIR = Path("data/Indian_Premier_League_2022-03-26/match_innings_commentary")
OUTPUT_FILE = Path("data/generated_qa.jsonl")


def match_name_from_filename(path: Path) -> str:
    name = path.stem
    name = name.replace("innings_1_", "")
    name = name.replace("_match_innings_1_commentary", "")
    name = name.replace("_vs_", " vs ")
    name = name.replace("_", " ")
    return name


def generate_qas():
    qa_pairs = []

    for file in DATA_DIR.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        tool = StatsTool(data)
        match_name = match_name_from_filename(file)

        qa_pairs.extend([
            {
                "prompt": f"Question: How many runs were scored in the last 3 overs of the innings in {match_name}?",
                "response": f"{tool.runs_last_n_overs(3)} runs were scored in the last 3 overs."
            },
            {
                "prompt": f"Question: How many runs were scored in the last 5 overs of the innings in {match_name}?",
                "response": f"{tool.runs_last_n_overs(5)} runs were scored in the last 5 overs."
            },
            {
                "prompt": f"Question: What was the total score in the innings in {match_name}?",
                "response": f"The total score was {tool.total_runs()} runs."
            },
            {
                "prompt": f"Question: How many wickets fell in the last 5 overs of the innings in {match_name}?",
                "response": f"{tool.wickets_last_n_overs(5)} wickets fell in the last 5 overs."
            },
            {
                "prompt": f"Question: How many wickets fell in the innings in {match_name}?",
                "response": f"{tool.total_wickets()} wickets fell in the innings."
            },
            {
                "prompt": f"Question: How many fours were hit in the innings in {match_name}?",
                "response": f"{tool.total_fours()} fours were hit."
            },
            {
                "prompt": f"Question: How many sixes were hit in the innings in {match_name}?",
                "response": f"{tool.total_sixes()} sixes were hit."
            }
        ])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa) + "\n")

    print(f"Generated {len(qa_pairs)} QA pairs")


if __name__ == "__main__":
    generate_qas()
