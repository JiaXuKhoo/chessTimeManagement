import csv
import time
import multiprocessing as mp
from typing import Optional, Dict, List, Tuple

import chess
import chess.engine


print("TEST")

STOCKFISH_PATH = r"C:\Users\khoo\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
FEN_FILE = "7k_sampled_fens.txt"
OUTPUT_CSV = "parallel_regret_data.csv"

BUCKETS = [50_000, 200_000, 800_000]
REFERENCE_NODES = 1_600_000
MAX_FENS = 1000

REGRET_TOLERANCE_CP = 20
OVERFLOW_LABEL = f">{BUCKETS[-1]}"

MATE_MISS_PENALTY_CP = 1000
MATE_PLY_PENALTY_CP = 50

NUM_WORKERS = 8   # reminder to change to 16


# -------------------------
# Score helpers
# -------------------------

def extract_score(score_obj: chess.engine.Score) -> Optional[Tuple[str, int]]:
    if score_obj is None:
        return None
    if score_obj.is_mate():
        m = score_obj.mate()
        if m is None:
            return None
        return ("mate", m)
    cp = score_obj.score()
    if cp is None:
        return None
    return ("cp", cp)


def compute_hybrid_regret(ref_score, bucket_score):
    if ref_score is None or bucket_score is None:
        return None

    ref_type, ref_val = ref_score
    b_type, b_val = bucket_score

    if ref_type == "cp" and b_type == "cp":
        return max(0, ref_val - b_val)

    if ref_type == "mate" and b_type != "mate":
        return MATE_MISS_PENALTY_CP

    if ref_type == "mate" and b_type == "mate":
        return max(0, (b_val - ref_val) * MATE_PLY_PENALTY_CP)

    if ref_type != "mate" and b_type == "mate":
        return 0

    return None


def score_kind_and_value(score):
    if score is None:
        return None, None
    return score[0], score[1]


# -------------------------
# Engine functions
# -------------------------

def analyse_root_position(engine, fen, nodes):
    board = chess.Board(fen)
    if board.is_game_over(claim_draw=True):
        return None

    info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
    score_obj = info.get("score")
    pv = info.get("pv", [])

    bestmove = pv[0].uci() if pv else None
    score = extract_score(score_obj.white()) if score_obj else None

    return {
        "nodes": nodes,
        "bestmove": bestmove,
        "score": score,
        "depth": info.get("depth"),
        "seldepth": info.get("seldepth"),
    }


def evaluate_move_under_reference(engine, fen, move_uci, ref_nodes):
    board = chess.Board(fen)
    if board.is_game_over(claim_draw=True):
        return None

    original_turn = board.turn
    move = chess.Move.from_uci(move_uci)

    if move not in board.legal_moves:
        return None

    board.push(move)

    info = engine.analyse(board, chess.engine.Limit(nodes=ref_nodes))
    score_obj = info.get("score")

    if score_obj is None:
        return None

    return extract_score(score_obj.pov(original_turn))


# -------------------------
# Worker function
# -------------------------

def process_fen(fen: str):
    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            engine.configure({"Threads": 1, "Hash": 128})

            row = {"fen": fen}

            ref = analyse_root_position(engine, fen, REFERENCE_NODES)
            if ref is None or ref["bestmove"] is None:
                return None

            ref_root_type, ref_root_value = score_kind_and_value(ref["score"])

            row["ref_nodes"] = ref["nodes"]
            row["ref_bestmove"] = ref["bestmove"]
            row["ref_root_score_type"] = ref_root_type
            row["ref_root_score_value"] = ref_root_value

            candidate_moves = {ref["bestmove"]}
            bucket_results = {}

            for b in BUCKETS:
                bucket = analyse_root_position(engine, fen, b)
                bucket_results[b] = bucket

                if bucket is None:
                    continue

                row[f"bucket_{b}_bestmove"] = bucket["bestmove"]

                if bucket["bestmove"]:
                    candidate_moves.add(bucket["bestmove"])

            move_ref_scores = {}
            for move in candidate_moves:
                move_ref_scores[move] = evaluate_move_under_reference(
                    engine, fen, move, REFERENCE_NODES
                )

            ref_best = move_ref_scores.get(ref["bestmove"])
            if ref_best is None:
                return None

            for b in BUCKETS:
                bucket = bucket_results[b]
                if bucket is None or bucket["bestmove"] is None:
                    row[f"bucket_{b}_regret_cp"] = None
                    continue

                move = bucket["bestmove"]
                move_score = move_ref_scores.get(move)

                regret = compute_hybrid_regret(ref_best, move_score)
                row[f"bucket_{b}_regret_cp"] = regret

            row["label_bucket"] = assign_label(row, BUCKETS, REGRET_TOLERANCE_CP)

            return row

    except Exception as e:
        print(f"Error: {e}")
        return None


# -------------------------
# Label assignment
# -------------------------

def assign_label(row, buckets, tolerance_cp):
    for b in buckets:
        regret = row.get(f"bucket_{b}_regret_cp")
        if regret is not None and regret <= tolerance_cp:
            return str(b)
    return OVERFLOW_LABEL


# -------------------------
# Main
# -------------------------

def load_fens(path, max_fens):
    fens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            fen = line.strip()
            if fen:
                fens.append(fen)
            if len(fens) >= max_fens:
                break
    return fens


def main():
    print("Script started")

    fens = load_fens(FEN_FILE, MAX_FENS)
    total = len(fens)
    print(f"Loaded {total} FENs")

    start = time.time()

    results = []
    processed = 0
    kept = 0

    with mp.Pool(NUM_WORKERS) as pool:
        for result in pool.imap_unordered(process_fen, fens):
            processed += 1

            if result is not None:
                results.append(result)
                kept += 1

            if processed % 10 == 0 or processed == total:
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0

                print(
                    f"Processed={processed}/{total} | "
                    f"Kept={kept} | "
                    f"Elapsed={elapsed:.1f}s | "
                    f"Rate={rate:.2f} FEN/s | "
                    f"ETA={eta:.1f}s"
                )

    print(f"Writing {len(results)} rows")

    fieldnames = list(results[0].keys())

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total_elapsed = time.time() - start
    print(f"Done in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()