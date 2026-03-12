import json
import random

INPUT_FILE = "lichess_db_eval.jsonl/lichess_db_eval.jsonl"
OUTPUT_FILE = "7.2k_sampled_fens.txt"

# Probability of keeping a position
# Adjust depending on how many positions you want
SAMPLE_RATE = 0.00002   # 0.01% sample (~36k from 360M)
random.seed(42)  # for reproducibility

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for i, line in enumerate(fin):

        if random.random() > SAMPLE_RATE:
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