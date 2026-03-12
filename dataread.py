import json

file_path = r'lichess_db_eval.jsonl/lichess_db_eval.jsonl'

with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        fen = data["fen"]
        evals = data["evals"]
        print(f"====================={i}=====================")
        print(fen)
        print(f"====================={i}=====================")
        print(evals)
        print(f"====================={i}=====================")
        print(f"Number of PVs: {len(evals)}")

        if i == 3:
            break
