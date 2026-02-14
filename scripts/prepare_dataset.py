import json

INPUT = "data/generated_qa.jsonl"
OUTPUT = "data/train.jsonl"

def format_example(ex):
    return {
        "text": f"""### Instruction:
{ex["prompt"]}

### Response:
{ex["response"]}"""
    }

with open(INPUT) as f, open(OUTPUT, "w") as out:
    for line in f:
        ex = json.loads(line)
        out.write(json.dumps(format_example(ex)) + "\n")

print("Saved to", OUTPUT)
