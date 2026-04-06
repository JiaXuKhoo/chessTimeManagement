import csv
import time
import atexit
import multiprocessing as mp
from typing import Optional, Dict, List, Tuple

import chess
import chess.engine

STOCKFISH_PATH = r"stockfish"
FEN_FILE = "30k_sampled_fens.txt"
OUTPUT_CSV = "final_train_data.csv"

BUCKETS = [25_000, 100_000, 400_000, 1_600_000]
REFERENCE_NODES = 3_200_000

# Hybrid regret constants
MATE_MISS_PENALTY_CP = 1000
MATE_PLY_PENALTY_CP = 50

NUM_WORKERS = 16
HASH_PER_WORKER_MB = 128

ENGINE = None


# -------------------------
# Worker init / cleanup
# -------------------------

def init_worker():
    global ENGINE
    ENGINE = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    ENGINE.configure({
        "Threads": 1,
        "Hash": HASH_PER_WORKER_MB,
    })
    atexit.register(close_engine)


def close_engine():
    global ENGINE
    if ENGINE is not None:
        try:
            ENGINE.quit()
        except Exception:
            pass
        ENGINE = None


# -------------------------
# Score helpers
# -------------------------

def extract_score(score_obj: chess.engine.Score) -> Optional[Tuple[str, int]]:
    """
    Convert python-chess score to:
      ("cp", centipawns)
      ("mate", ply_distance)
    """
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


def score_kind_and_value(score: Optional[Tuple[str, int]]) -> Tuple[Optional[str], Optional[int]]:
    if score is None:
        return None, None
    return score[0], score[1]


def compute_hybrid_regret(
    ref_score: Optional[Tuple[str, int]],
    bucket_score: Optional[Tuple[str, int]]
) -> Optional[int]:
    """
    Piecewise regret, sign-aware for mate values.

    Interpretation:
    - Positive mate value: side-to-move is winning by force
    - Negative mate value: side-to-move is losing by force
    """
    if ref_score is None or bucket_score is None:
        return None

    ref_type, ref_val = ref_score
    b_type, b_val = bucket_score

    # Case 1: both cp
    if ref_type == "cp" and b_type == "cp":
        return max(0, ref_val - b_val)

    # Case 2: reference is mate
    if ref_type == "mate":
        # Ref is winning by force
        if ref_val > 0:
            if b_type != "mate":
                return MATE_MISS_PENALTY_CP
            if b_val <= 0:
                return MATE_MISS_PENALTY_CP
            # both winning mates: slower mate is worse
            return max(0, (b_val - ref_val) * MATE_PLY_PENALTY_CP)

        # Ref is losing by force
        if ref_val < 0:
            if b_type == "mate" and b_val < 0:
                # both losing mates: faster loss is worse
                return max(0, (abs(ref_val) - abs(b_val)) * MATE_PLY_PENALTY_CP)
            if b_type == "mate" and b_val > 0:
                # bucket found a win where ref says loss -> treat as no regret
                return 0
            # bucket cp while ref losing mate: treat as inconclusive at finite depth
            return 0

    # Case 3: reference is cp, bucket is mate
    if ref_type == "cp" and b_type == "mate":
        if b_val > 0:
            # bucket found a forced win -> no regret
            return 0
        else:
            # bucket chose a forced-loss line while ref had a cp-evaluable line
            return MATE_MISS_PENALTY_CP

    return None


# -------------------------
# Engine analysis
# -------------------------

def analyse_root_position(
    engine: chess.engine.SimpleEngine,
    fen: str,
    nodes: int
) -> Optional[Dict]:
    board = chess.Board(fen)

    # Exclude only trivial terminal positions
    if board.is_game_over(claim_draw=True):
        return None

    info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
    score_obj = info.get("score")
    pv = info.get("pv", [])

    bestmove = pv[0].uci() if pv else None
    score = extract_score(score_obj.white()) if score_obj is not None else None

    return {
        "nodes": nodes,
        "bestmove": bestmove,
        "score": score,   # White perspective at root
        "depth": info.get("depth"),
        "seldepth": info.get("seldepth"),
    }


def evaluate_move_under_reference(
    engine: chess.engine.SimpleEngine,
    fen: str,
    move_uci: str,
    ref_nodes: int
) -> Optional[Tuple[str, int]]:
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

    # Reorient to the player who made the root move
    return extract_score(score_obj.pov(original_turn))


# -------------------------
# FEN loading
# -------------------------

def load_fens(path: str) -> List[str]:
    fens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            fen = line.strip()
            if fen:
                fens.append(fen)
    return fens


# -------------------------
# Worker task
# -------------------------

def process_fen(fen: str) -> Optional[Dict]:
    global ENGINE

    try:
        row = {"fen": fen}

        # 1) Reference root search
        ref = analyse_root_position(ENGINE, fen, REFERENCE_NODES)
        if ref is None or ref["bestmove"] is None:
            return None

        ref_root_type, ref_root_value = score_kind_and_value(ref["score"])

        row["ref_nodes"] = ref["nodes"]
        row["ref_bestmove"] = ref["bestmove"]
        row["ref_root_score_type"] = ref_root_type
        row["ref_root_score_value"] = ref_root_value
        row["ref_depth"] = ref["depth"]
        row["ref_seldepth"] = ref["seldepth"]

        # 2) Bucket root searches
        candidate_moves = {ref["bestmove"]}
        bucket_results = {}

        for b in BUCKETS:
            bucket = analyse_root_position(ENGINE, fen, b)
            bucket_results[b] = bucket

            if bucket is None:
                row[f"bucket_{b}_bestmove"] = None
                row[f"bucket_{b}_root_score_type"] = None
                row[f"bucket_{b}_root_score_value"] = None
                row[f"bucket_{b}_depth"] = None
                row[f"bucket_{b}_seldepth"] = None
                row[f"bucket_{b}_same_move"] = None
                continue

            b_root_type, b_root_value = score_kind_and_value(bucket["score"])

            row[f"bucket_{b}_bestmove"] = bucket["bestmove"]
            row[f"bucket_{b}_root_score_type"] = b_root_type
            row[f"bucket_{b}_root_score_value"] = b_root_value
            row[f"bucket_{b}_depth"] = bucket["depth"]
            row[f"bucket_{b}_seldepth"] = bucket["seldepth"]
            row[f"bucket_{b}_same_move"] = (
                bucket["bestmove"] == ref["bestmove"] if bucket["bestmove"] is not None else None
            )

            if bucket["bestmove"] is not None:
                candidate_moves.add(bucket["bestmove"])

        # 3) Re-score candidate moves under reference
        move_ref_scores: Dict[str, Optional[Tuple[str, int]]] = {}
        for move_uci in candidate_moves:
            move_ref_scores[move_uci] = evaluate_move_under_reference(
                ENGINE, fen, move_uci, REFERENCE_NODES
            )

        ref_bestmove_ref_score = move_ref_scores.get(ref["bestmove"])
        if ref_bestmove_ref_score is None:
            return None

        ref_best_type, ref_best_value = score_kind_and_value(ref_bestmove_ref_score)
        row["ref_bestmove_ref_score_type"] = ref_best_type
        row["ref_bestmove_ref_score_value"] = ref_best_value

        # 4) Compute regret for each bucket
        for b in BUCKETS:
            bucket = bucket_results.get(b)
            if bucket is None or bucket["bestmove"] is None:
                row[f"bucket_{b}_move_ref_score_type"] = None
                row[f"bucket_{b}_move_ref_score_value"] = None
                row[f"bucket_{b}_regret_cp"] = None
                continue

            bucket_move = bucket["bestmove"]
            bucket_move_ref_score = move_ref_scores.get(bucket_move)
            bm_type, bm_value = score_kind_and_value(bucket_move_ref_score)

            row[f"bucket_{b}_move_ref_score_type"] = bm_type
            row[f"bucket_{b}_move_ref_score_value"] = bm_value
            row[f"bucket_{b}_regret_cp"] = compute_hybrid_regret(
                ref_bestmove_ref_score,
                bucket_move_ref_score
            )

        return row

    except Exception as e:
        print(f"Error processing FEN: {e}")
        return None


# -------------------------
# Main
# -------------------------

def build_fieldnames() -> List[str]:
    fieldnames = [
        "fen",
        "ref_nodes",
        "ref_bestmove",
        "ref_root_score_type",
        "ref_root_score_value",
        "ref_depth",
        "ref_seldepth",
        "ref_bestmove_ref_score_type",
        "ref_bestmove_ref_score_value",
    ]

    for b in BUCKETS:
        fieldnames += [
            f"bucket_{b}_bestmove",
            f"bucket_{b}_root_score_type",
            f"bucket_{b}_root_score_value",
            f"bucket_{b}_depth",
            f"bucket_{b}_seldepth",
            f"bucket_{b}_same_move",
            f"bucket_{b}_move_ref_score_type",
            f"bucket_{b}_move_ref_score_value",
            f"bucket_{b}_regret_cp",
        ]
    return fieldnames


def main():
    print("Script started")

    fens = load_fens(FEN_FILE)
    total = len(fens)
    print(f"Loaded {total} FENs")

    start = time.time()
    results = []
    processed = 0
    kept = 0

    with mp.Pool(NUM_WORKERS, initializer=init_worker) as pool:
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

    fieldnames = build_fieldnames()
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total_elapsed = time.time() - start
    print(f"Done in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()