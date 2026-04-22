import csv
import time
import multiprocessing as mp
from typing import Optional, Dict, List, Tuple

import chess
import chess.engine

# -------------------------
# Config
# -------------------------

STOCKFISH_PATH = "stockfish"
FEN_FILE = "30k_sampled_fens.txt"
OUTPUT_CSV = "classifier_train_data.csv"

BUCKETS = [25_000, 100_000, 400_000, 1_600_000]
REFERENCE_NODES = 3_200_000

MATE_MISS_PENALTY_CP = 1000
MATE_PLY_PENALTY_CP = 5

NUM_WORKERS = 16
HASH_PER_WORKER_MB = 128

# Number of FENs handled by one worker task
CHUNK_SIZE = 200


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


def score_kind_and_value(score: Optional[Tuple[str, int]]) -> Tuple[Optional[str], Optional[int]]:
    if score is None:
        return None, None
    return score[0], score[1]


def weighted_cp_regret(ref_cp, bucket_cp, K=100.0):
    raw = max(0, ref_cp - bucket_cp)
    if ref_cp * bucket_cp < 0:
        weight = 1.0
    else:
        avg_magnitude = (abs(ref_cp) + abs(bucket_cp)) / 2
        weight = 1.0 / (1.0 + avg_magnitude / K)
    return int(round(raw * weight))


def compute_hybrid_regret(
    ref_score: Optional[Tuple[str, int]],
    bucket_score: Optional[Tuple[str, int]]
) -> Optional[int]:
    if ref_score is None or bucket_score is None:
        return None

    ref_type, ref_val = ref_score
    b_type, b_val = bucket_score

    # Case 1: both cp
    if ref_type == "cp" and b_type == "cp":
        return weighted_cp_regret(ref_val, b_val, K=100.0)

    # Case 2: reference is mate
    if ref_type == "mate":
        if ref_val >= 0:
            if b_type != "mate":
                return MATE_MISS_PENALTY_CP
            if b_val < 0:
                return MATE_MISS_PENALTY_CP
            return max(0, (b_val - ref_val) * MATE_PLY_PENALTY_CP)

        if ref_val < 0:
            if b_type == "mate" and b_val < 0:
                return max(0, (abs(ref_val) - abs(b_val)) * MATE_PLY_PENALTY_CP)
            if b_type == "mate" and b_val >= 0:
                return 0
            return 0

    # Case 3: reference is cp, bucket is mate
    if ref_type == "cp" and b_type == "mate":
        if b_val >= 0:
            return 0
        return MATE_MISS_PENALTY_CP

    return None


# -------------------------
# Engine lifecycle
# -------------------------

def open_engine() -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({
        "Threads": 1,
        "Hash": HASH_PER_WORKER_MB,
    })
    return engine


def close_engine(engine: Optional[chess.engine.SimpleEngine]) -> None:
    if engine is not None:
        try:
            engine.quit()
        except Exception:
            pass


# -------------------------
# Engine analysis
# -------------------------

def analyse_root_position(
    engine: chess.engine.SimpleEngine,
    fen: str,
    nodes: int
) -> Optional[Dict]:
    board = chess.Board(fen)

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
        "score": score,
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

    return extract_score(score_obj.pov(original_turn))


# -------------------------
# FEN loading / chunking
# -------------------------

def load_fens(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def chunkify(lst: List[str], chunk_size: int) -> List[List[str]]:
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


# -------------------------
# Per-FEN processing
# -------------------------

def process_single_fen(engine: chess.engine.SimpleEngine, fen: str) -> Dict:
    row = {"fen": fen, "status": "ok", "error_msg": None}

    # 1) Reference root search
    ref = analyse_root_position(engine, fen, REFERENCE_NODES)
    if ref is None:
        row["status"] = "skip_terminal_or_no_ref"
        return row

    if ref["bestmove"] is None:
        row["status"] = "skip_no_ref_bestmove"
        return row

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
        bucket = analyse_root_position(engine, fen, b)
        bucket_results[b] = bucket

        if bucket is None:
            row[f"bucket_{b}_bestmove"] = None
            row[f"bucket_{b}_root_score_type"] = None
            row[f"bucket_{b}_root_score_value"] = None
            row[f"bucket_{b}_depth"] = None
            row[f"bucket_{b}_seldepth"] = None
            row[f"bucket_{b}_same_move"] = None
            row[f"bucket_{b}_move_ref_score_type"] = None
            row[f"bucket_{b}_move_ref_score_value"] = None
            row[f"bucket_{b}_regret_cp"] = None
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
            engine, fen, move_uci, REFERENCE_NODES
        )

    ref_bestmove_ref_score = move_ref_scores.get(ref["bestmove"])
    if ref_bestmove_ref_score is None:
        row["status"] = "skip_no_ref_rescore"
        return row

    ref_best_type, ref_best_value = score_kind_and_value(ref_bestmove_ref_score)
    row["ref_bestmove_ref_score_type"] = ref_best_type
    row["ref_bestmove_ref_score_value"] = ref_best_value

    # 4) Compute regret for each bucket
    for b in BUCKETS:
        bucket = bucket_results.get(b)
        if bucket is None or bucket["bestmove"] is None:
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


# -------------------------
# Worker task (chunk-based)
# -------------------------

def process_chunk(fen_chunk: List[str]) -> List[Dict]:
    print(f"[worker] starting chunk of {len(fen_chunk)} FENs", flush=True)
    results: List[Dict] = []
    engine = None

    try:
        engine = open_engine()

        for fen in fen_chunk:
            try:
                if len(results) % 10 == 0:
                    print(f"[worker] processed {len(results)}/{len(fen_chunk)} in this chunk", flush=True)
                row = process_single_fen(engine, fen)
                results.append(row)

            except Exception as e:
                msg = str(e)
                # If engine / event loop died, rebuild engine and retry once
                lowered = msg.lower()
                recoverable = (
                    "event loop is closed" in lowered
                    or "event loop is dead" in lowered
                    or "engine terminated" in lowered
                    or "connection lost" in lowered
                    or "broken pipe" in lowered
                )

                if recoverable:
                    close_engine(engine)
                    engine = open_engine()
                    try:
                        row = process_single_fen(engine, fen)
                        results.append(row)
                        continue
                    except Exception as e2:
                        results.append({
                            "fen": fen,
                            "status": "error",
                            "error_msg": f"retry_failed: {e2}",
                        })
                        continue

                results.append({
                    "fen": fen,
                    "status": "error",
                    "error_msg": msg,
                })

    finally:
        close_engine(engine)

    return results


# -------------------------
# Main
# -------------------------

def build_fieldnames() -> List[str]:
    fieldnames = [
        "fen",
        "status",
        "error_msg",
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

    fen_chunks = chunkify(fens, CHUNK_SIZE)
    total_chunks = len(fen_chunks)

    start = time.time()
    results: List[Dict] = []
    processed_fens = 0

    with mp.Pool(NUM_WORKERS) as pool:
        for chunk_idx, chunk_result in enumerate(pool.imap_unordered(process_chunk, fen_chunks), start=1):
            results.extend(chunk_result)
            processed_fens += len(chunk_result)

            elapsed = time.time() - start
            rate = processed_fens / elapsed if elapsed > 0 else 0
            eta = (total - processed_fens) / rate if rate > 0 else 0

            ok = sum(1 for r in results if r.get("status") == "ok")
            errors = sum(1 for r in results if r.get("status") == "error")
            skipped = processed_fens - ok - errors

            print(
            f"Chunks={chunk_idx}/{total_chunks} | "
            f"Processed={processed_fens}/{total} | "
            f"OK={ok} | Skipped={skipped} | Errors={errors} | "
            f"Elapsed={elapsed:.1f}s | Rate={rate:.2f} FEN/s | ETA={eta:.1f}s",
            flush=True
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
    mp.set_start_method("spawn", force=True)
    main()