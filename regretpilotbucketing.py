import csv
import time
from typing import Optional, Dict, List, Tuple

import chess
import chess.engine


print("TEST")
STOCKFISH_PATH = r"C:\Users\khoo\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
FEN_FILE = "7k_sampled_fens.txt"
OUTPUT_CSV = "pilot_regret_data_hybrid.csv"

BUCKETS = [50_000, 200_000, 800_000]
REFERENCE_NODES = 1_600_000
MAX_FENS = 1000

# Labeling threshold
REGRET_TOLERANCE_CP = 20
OVERFLOW_LABEL = f">{BUCKETS[-1]}"

# Hybrid regret constants
MATE_MISS_PENALTY_CP = 1000      # if ref finds mate and bucket does not
MATE_PLY_PENALTY_CP = 50         # extra cp penalty per ply slower mate


# -------------------------
# Score helpers
# -------------------------

def extract_score(score_obj: chess.engine.Score) -> Optional[Tuple[str, int]]:
    """
    Convert a python-chess score into a structured value:

        ("cp", centipawns)
        ("mate", ply_distance)

    Returns None only if score_obj itself is None.
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


def compute_hybrid_regret(
    ref_score: Optional[Tuple[str, int]],
    bucket_score: Optional[Tuple[str, int]]
) -> Optional[int]:
    """
    Piecewise regret:

    Case 1: both cp
        regret = max(0, ref_cp - bucket_cp)

    Case 2: ref mate, bucket not mate
        regret = MATE_MISS_PENALTY_CP

    Case 3: both mate
        regret = max(0, (bucket_mate - ref_mate) * MATE_PLY_PENALTY_CP)

    Case 4: bucket mate, ref not mate
        regret = 0   (rare search-noise case; safest for pilot)
    """
    if ref_score is None or bucket_score is None:
        return None

    ref_type, ref_val = ref_score
    b_type, b_val = bucket_score

    # both cp
    if ref_type == "cp" and b_type == "cp":
        return max(0, ref_val - b_val)

    # ref mate, bucket not mate
    if ref_type == "mate" and b_type != "mate":
        return MATE_MISS_PENALTY_CP

    # both mate
    if ref_type == "mate" and b_type == "mate":
        # smaller mate distance is better if positive mate-for-side-to-move values
        # this keeps the pilot simple and monotone
        return max(0, (b_val - ref_val) * MATE_PLY_PENALTY_CP)

    # bucket mate, ref not mate
    if ref_type != "mate" and b_type == "mate":
        return 0

    return None


def score_kind_and_value(score: Optional[Tuple[str, int]]) -> Tuple[Optional[str], Optional[int]]:
    if score is None:
        return None, None
    return score[0], score[1]


# -------------------------
# FEN loading
# -------------------------

def load_fens(path: str, max_fens: int) -> List[str]:
    fens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            fen = line.strip()
            if fen:
                fens.append(fen)
            if len(fens) >= max_fens:
                break
    return fens


# -------------------------
# Engine analysis
# -------------------------

def analyse_root_position(
    engine: chess.engine.SimpleEngine,
    fen: str,
    nodes: int
) -> Optional[Dict]:
    """
    Analyse root position at fixed node budget.

    Returns:
        {
            "nodes": int,
            "bestmove": str | None,
            "score": ("cp", x) or ("mate", y) or None,
            "depth": int | None,
            "seldepth": int | None,
        }
    """
    board = chess.Board(fen)

    # Exclude trivial terminal positions only
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
        "score": score,  # from White's perspective at root
        "depth": info.get("depth"),
        "seldepth": info.get("seldepth"),
    }


def evaluate_move_under_reference(
    engine: chess.engine.SimpleEngine,
    fen: str,
    move_uci: str,
    ref_nodes: int
) -> Optional[Tuple[str, int]]:
    """
    Evaluate a specific root move under the reference evaluator:

      1. start from original FEN
      2. make the move
      3. analyse resulting position at ref_nodes
      4. convert score back to ORIGINAL mover's perspective

    Returns:
      ("cp", x) or ("mate", y) or None
    """
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
# Label assignment
# -------------------------

def assign_label(row: Dict, buckets: List[int], tolerance_cp: int) -> str:
    for b in buckets:
        regret = row.get(f"bucket_{b}_regret_cp")
        if regret is not None and regret <= tolerance_cp:
            return str(b)
    return OVERFLOW_LABEL


# -------------------------
# Main pipeline
# -------------------------

def main():
    print("Script started")
    fens = load_fens(FEN_FILE, MAX_FENS)
    print(f"Loaded {len(fens)} FENs")

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
        "label_bucket",
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

    processed = 0
    kept = 0
    skipped = 0
    start = time.time()

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        engine.configure({
            "Threads": 1,
            "Hash": 256,
        })

        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for idx, fen in enumerate(fens, start=1):
                processed += 1

                try:
                    row = {"fen": fen}

                    # 1) Reference root search
                    ref = analyse_root_position(engine, fen, REFERENCE_NODES)
                    if ref is None or ref["bestmove"] is None:
                        skipped += 1
                        continue

                    ref_root_type, ref_root_value = score_kind_and_value(ref["score"])

                    row["ref_nodes"] = ref["nodes"]
                    row["ref_bestmove"] = ref["bestmove"]
                    row["ref_root_score_type"] = ref_root_type
                    row["ref_root_score_value"] = ref_root_value
                    row["ref_depth"] = ref["depth"]
                    row["ref_seldepth"] = ref["seldepth"]

                    # 2) Bucket root searches
                    bucket_results = {}
                    candidate_moves = {ref["bestmove"]}

                    for b in BUCKETS:
                        bucket = analyse_root_position(engine, fen, b)
                        if bucket is None:
                            bucket_results[b] = None
                            continue

                        bucket_results[b] = bucket
                        b_root_type, b_root_value = score_kind_and_value(bucket["score"])

                        row[f"bucket_{b}_bestmove"] = bucket["bestmove"]
                        row[f"bucket_{b}_root_score_type"] = b_root_type
                        row[f"bucket_{b}_root_score_value"] = b_root_value
                        row[f"bucket_{b}_depth"] = bucket["depth"]
                        row[f"bucket_{b}_seldepth"] = bucket["seldepth"]

                        same_move = (
                            bucket["bestmove"] == ref["bestmove"]
                            if bucket["bestmove"] is not None else None
                        )
                        row[f"bucket_{b}_same_move"] = same_move

                        if bucket["bestmove"] is not None:
                            candidate_moves.add(bucket["bestmove"])

                    # 3) Re-score candidate moves under reference
                    move_ref_scores: Dict[str, Optional[Tuple[str, int]]] = {}
                    for move_uci in candidate_moves:
                        move_ref_scores[move_uci] = evaluate_move_under_reference(
                            engine, fen, move_uci, REFERENCE_NODES
                        )

                    ref_bestmove_ref_score = move_ref_scores.get(ref["bestmove"])
                    if ref_bestmove_ref_score is None:
                        skipped += 1
                        continue

                    ref_best_type, ref_best_value = score_kind_and_value(ref_bestmove_ref_score)
                    row["ref_bestmove_ref_score_type"] = ref_best_type
                    row["ref_bestmove_ref_score_value"] = ref_best_value

                    # 4) Compute hybrid regret
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

                        regret_cp = compute_hybrid_regret(
                            ref_bestmove_ref_score,
                            bucket_move_ref_score
                        )
                        row[f"bucket_{b}_regret_cp"] = regret_cp

                    # 5) Assign label
                    row["label_bucket"] = assign_label(
                        row=row,
                        buckets=BUCKETS,
                        tolerance_cp=REGRET_TOLERANCE_CP
                    )

                    writer.writerow(row)
                    kept += 1

                    if idx % 10 == 0:
                        elapsed = time.time() - start
                        avg = elapsed / processed
                        remaining = avg * (len(fens) - processed)
                        print(
                            f"Processed={processed}/{len(fens)} | "
                            f"Kept={kept} | Skipped={skipped} | "
                            f"Elapsed={elapsed:.1f}s | ETA={remaining:.1f}s"
                        )

                except Exception as e:
                    skipped += 1
                    print(f"Skipping FEN {idx} due to error: {e}")

    total_elapsed = time.time() - start
    print(f"\nDone. Output written to {OUTPUT_CSV}")
    print(f"Processed: {processed}, Kept: {kept}, Skipped: {skipped}")
    print(f"Total time: {total_elapsed:.1f}s")


main()