import time
from time import perf_counter
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd
import chess

# =========================================================
# Config
# =========================================================

INPUT_CSV = "cleaned_train_data.csv"
TIMED_OUTPUT_CSV = "timed_dataset_static_tol20.csv"
OUTPUT_CSV = "dataset_static_tol20.csv"

BUCKETS = [25_000, 100_000, 400_000, 1_600_000]
TAU_CP = 20

PIECE_NAMES = {
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
}


# =========================================================
# Label extraction
# =========================================================

def choose_bucket_label(row: pd.Series, tau_cp: int = TAU_CP) -> Optional[int]:
    """
    Monotone-smoothed smallest-sufficient-bucket rule.

    Steps:
    1) Read raw regrets for each bucket.
    2) Smooth from largest -> smallest so smaller buckets cannot
       appear better than deeper buckets purely due to noise:
           smoothed[i] = max(raw[i], smoothed[i+1])
    3) Return the smallest bucket whose smoothed regret <= tau_cp.
    4) If none satisfy tolerance, return the largest bucket.
    """
    regrets: List[Optional[float]] = []
    present_any = False

    for b in BUCKETS:
        col = f"bucket_{b}_regret_cp"
        if col in row and pd.notna(row[col]):
            regrets.append(float(row[col]))
            present_any = True
        else:
            regrets.append(None)

    if not present_any:
        return None

    smoothed = regrets[:]

    # Monotone smoothing from largest bucket backwards
    for i in range(len(BUCKETS) - 2, -1, -1):
        if smoothed[i] is None:
            continue

        j = i + 1
        while j < len(BUCKETS) and smoothed[j] is None:
            j += 1

        if j < len(BUCKETS) and smoothed[j] is not None:
            smoothed[i] = max(smoothed[i], smoothed[j])

    for b, r in zip(BUCKETS, smoothed):
        if r is not None and r <= tau_cp:
            return b

    return BUCKETS[-1]


# =========================================================
# Helpers
# =========================================================

def count_legal_move_types(board: chess.Board) -> Tuple[int, int, int]:
    num_captures = 0
    num_checks = 0
    num_promotions = 0

    for move in board.legal_moves:
        if board.is_capture(move):
            num_captures += 1
        if move.promotion is not None:
            num_promotions += 1
        if board.gives_check(move):
            num_checks += 1

    return num_captures, num_checks, num_promotions


def mobility_by_piece_type(board: chess.Board, color: chess.Color) -> Dict[str, int]:
    """
    Counts pseudo-legal attack squares per piece type for side to move only.
    """
    result = {
        "knight_mobility": 0,
        "bishop_mobility": 0,
        "rook_mobility": 0,
        "queen_mobility": 0,
    }

    mapping = {
        chess.KNIGHT: "knight_mobility",
        chess.BISHOP: "bishop_mobility",
        chess.ROOK: "rook_mobility",
        chess.QUEEN: "queen_mobility",
    }

    occupied_by_us = board.occupied_co[color]

    for piece_type, feat_name in mapping.items():
        total = 0
        for sq in board.pieces(piece_type, color):
            attacks = board.attacks(sq)
            total += len(attacks & ~occupied_by_us)
        result[feat_name] = total

    return result


def num_attackers_on_enemy_king(board: chess.Board) -> int:
    stm = board.turn
    enemy = not stm
    enemy_king_sq = board.king(enemy)
    if enemy_king_sq is None:
        return 0
    return len(board.attackers(stm, enemy_king_sq))


def num_pinned_pieces(board: chess.Board, color: chess.Color) -> int:
    count = 0
    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type != chess.KING:
            if board.is_pinned(color, sq):
                count += 1
    return count


# =========================================================
# Timed feature extraction
# =========================================================

def extract_static_features_timed(fen: str) -> Tuple[Dict, Dict[str, float]]:
    timings = {}
    feats = {}

    # Board construction
    t0 = perf_counter()
    board = chess.Board(fen)
    timings["board_construction_s"] = perf_counter() - t0

    # num_legal_moves
    t0 = perf_counter()
    num_legal_moves = board.legal_moves.count()
    feats["num_legal_moves"] = num_legal_moves
    timings["num_legal_moves_s"] = perf_counter() - t0

    # is_check
    t0 = perf_counter()
    feats["is_check"] = int(board.is_check())
    timings["is_check_s"] = perf_counter() - t0

    # side_to_move_white
    t0 = perf_counter()
    feats["side_to_move_white"] = int(board.turn == chess.WHITE)
    timings["side_to_move_s"] = perf_counter() - t0

    # move-type breakdown + ratios
    t0 = perf_counter()
    num_captures, num_checks, num_promotions = count_legal_move_types(board)
    denom = max(1, num_legal_moves)
    feats["capture_ratio"] = num_captures / denom
    feats["check_ratio"] = num_checks / denom
    feats["num_promotions"] = num_promotions
    timings["move_type_breakdown_s"] = perf_counter() - t0

    # mobility
    t0 = perf_counter()
    mobility = mobility_by_piece_type(board, board.turn)
    feats.update(mobility)
    timings["mobility_by_piece_type_s"] = perf_counter() - t0

    # attackers on king
    t0 = perf_counter()
    feats["num_attackers_on_king"] = num_attackers_on_enemy_king(board)
    timings["num_attackers_on_king_s"] = perf_counter() - t0

    # pinned pieces
    t0 = perf_counter()
    feats["num_pinned_pieces"] = num_pinned_pieces(board, board.turn)
    timings["num_pinned_pieces_s"] = perf_counter() - t0

    timings["total_static_feature_overhead_s"] = sum(timings.values())

    return feats, timings


# =========================================================
# Main dataset build
# =========================================================

def build_static_dataset(
    input_csv: str = INPUT_CSV,
    timed_output_csv: str = TIMED_OUTPUT_CSV,
    output_csv: str = OUTPUT_CSV
) -> None:
    df = pd.read_csv(input_csv)

    rows = []
    timing_sums = defaultdict(float)
    timing_counts = defaultdict(int)

    start_all = perf_counter()

    for i, row in enumerate(df.itertuples(index=False), start=1):
        fen = row.fen

        label_bucket = choose_bucket_label(pd.Series(row._asdict()), tau_cp=TAU_CP)
        if label_bucket is None:
            continue

        feats, timings = extract_static_features_timed(fen)

        out = {
            "fen": fen,
            "label_bucket": label_bucket,
            "tau_cp": TAU_CP,
        }
        out.update(feats)
        out.update(timings)
        rows.append(out)

        for k, v in timings.items():
            timing_sums[k] += v
            timing_counts[k] += 1

        if i % 1000 == 0:
            print(f"Processed {i}/{len(df)} rows")

    total_elapsed = perf_counter() - start_all

    out_df = pd.DataFrame(rows)
    out_df.to_csv(timed_output_csv, index=False)

    out_df_no_timing = out_df.loc[:, ~out_df.columns.str.endswith("_s")]
    out_df_no_timing.to_csv(output_csv, index=False)

    print("Rows:", len(out_df_no_timing))
    print(f"Wall-clock total build time: {total_elapsed:.4f}s")

    avg_timings = {
        k: timing_sums[k] / max(timing_counts[k], 1)
        for k in timing_sums
    }
    sorted_timings = sorted(avg_timings.items(), key=lambda x: x[1], reverse=True)

    print("\n=== Average per-position timing (sorted DESC) ===")
    for k, avg in sorted_timings:
        print(f"{k:35s}: {avg * 1e6:8.2f} µs")

    if timing_counts["total_static_feature_overhead_s"] > 0:
        avg_overhead = (
            timing_sums["total_static_feature_overhead_s"]
            / timing_counts["total_static_feature_overhead_s"]
        )
        print("\n=== Controller overhead summary ===")
        print(f"Average static controller overhead per position: {avg_overhead:.8f}s")
        print(f"Average static controller overhead per 1000 positions: {avg_overhead * 1000:.4f}s")


if __name__ == "__main__":
    build_static_dataset()