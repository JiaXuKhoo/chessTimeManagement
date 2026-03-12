import json, math, random
from collections import Counter

MAX_KNODES = 1600
path = r'lichess_db_eval.jsonl/lichess_db_eval.jsonl'

count_total = 0
count_kept = 0
knodes_hist = Counter()

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        count_total += 1
        obj = json.loads(line)
        ks = [e.get("knodes", 0) for e in obj.get("evals", []) if "knodes" in e]
        if not ks:
            continue

        # record a coarse histogram of max knodes per position
        maxk = max(ks)
        knodes_hist[int(math.log10(maxk+1))] += 1

        # keep only positions that have some eval within your budget
        if any(k <= MAX_KNODES for k in ks):
            count_kept += 1

        if count_total % 100000 == 0:
            print("seen", count_total, "kept", count_kept)

print("Total:", count_total, "Kept:", count_kept)
print("log10(max_knodes+1) histogram:", knodes_hist)
