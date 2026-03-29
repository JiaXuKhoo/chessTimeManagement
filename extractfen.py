import json
import random

INPUT_FILE = "lichess_db_eval.jsonl/lichess_db_eval.jsonl"
OUTPUT_FILE = "15k_sampled_fens.txt"

# Probability of keeping a position
# Adjust depending on how many positions you want
SAMPLE_AMT = 15_000
sample_rate = SAMPLE_AMT / 360_000_000
random.seed(42)  # for reproducibility
# given/360,000,000 = sample_rate

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for i, line in enumerate(fin):

        if random.random() > sample_rate:
            continue

        try:
            data = json.loads(line)
            fen = data["fen"]
            fout.write(fen + "\n")

        except Exception:
            continue

        if i % 1_000_000 == 0:
            print("Processed", i)

print("Finished sampling.")