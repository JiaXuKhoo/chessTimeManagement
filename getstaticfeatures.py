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

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


# =========================================================
# Label extraction
# =========================================================

def choose_bucket_label(row: pd.Series, tau_cp: int = TAU_CP) -> Optional[int]:
    """
    Smallest bucket whose regret <= tau_cp.
    Returns None if any required regret fields are missing.
    """
    for b in BUCKETS:
        col = f"bucket_{b}_regret_cp"
        if col not in row or pd.isna(row[col]):
            continue
        if row[col] <= tau_cp:
            return b

    # If no bucket satisfies tolerance, use largest bucket
    # only if the regret columns exist at all.
    present_any = any((f"bucket_{b}_regret_cp" in row and pd.notna(row[f"bucket_{b}_regret_cp"])) for b in BUCKETS)
    return BUCKETS[-1] if present_any else None


# =========================================================
# Helpers
# =========================================================

def count_piece_type(board: chess.Board, piece_type: chess.PieceType, color: chess.Color) -> int:
    return len(board.pieces(piece_type, color))


def get_piece_counts(board: chess.Board) -> Dict[str, int]:
    feats = {}
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        name = PIECE_NAMES[piece_type]
        feats[f"white_{name}_count"] = count_piece_type(board, piece_type, chess.WHITE)
        feats[f"black_{name}_count"] = count_piece_type(board, piece_type, chess.BLACK)
    return feats


def get_total_material_and_imbalance(board: chess.Board) -> Tuple[int, int]:
    total_material = 0
    material_imbalance = 0

    for piece_type, value in PIECE_VALUES.items():
        w = len(board.pieces(piece_type, chess.WHITE))
        b = len(board.pieces(piece_type, chess.BLACK))
        total_material += value * (w + b)
        material_imbalance += value * (w - b)

    return total_material, material_imbalance


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
    This is cheap and gives a rough openness / mobility signal.
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
            # Remove squares occupied by own pieces
            total += len(attacks & ~occupied_by_us)
        result[feat_name] = total

    return result


def num_attackers_on_enemy_king(board: chess.Board) -> int:
    """
    Number of side-to-move attackers on the opponent king square.
    Chosen because it reflects attacking pressure relevant to the mover's decision.
    """
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

    # -------------------------
    # Very cheap static
    # -------------------------

    t0 = perf_counter()
    feats["num_legal_moves"] = board.legal_moves.count()
    timings["num_legal_moves_s"] = perf_counter() - t0

    t0 = perf_counter()
    feats["is_check"] = int(board.is_check())
    timings["is_check_s"] = perf_counter() - t0

    t0 = perf_counter()
    piece_counts = get_piece_counts(board)
    feats.update(piece_counts)
    timings["piece_counts_by_type_s"] = perf_counter() - t0

    t0 = perf_counter()
    total_material, material_imbalance = get_total_material_and_imbalance(board)
    feats["total_material"] = total_material
    feats["material_imbalance"] = material_imbalance
    timings["material_values_s"] = perf_counter() - t0

    t0 = perf_counter()
    feats["total_pieces"] = sum(1 for p in board.piece_map().values() if p.piece_type != chess.KING)
    timings["total_pieces_s"] = perf_counter() - t0

    t0 = perf_counter()
    feats["side_to_move_white"] = int(board.turn == chess.WHITE)
    timings["side_to_move_s"] = perf_counter() - t0

    t0 = perf_counter()
    feats["white_can_castle_kingside"] = int(board.has_kingside_castling_rights(chess.WHITE))
    feats["white_can_castle_queenside"] = int(board.has_queenside_castling_rights(chess.WHITE))
    feats["black_can_castle_kingside"] = int(board.has_kingside_castling_rights(chess.BLACK))
    feats["black_can_castle_queenside"] = int(board.has_queenside_castling_rights(chess.BLACK))
    timings["castling_rights_s"] = perf_counter() - t0

    t0 = perf_counter()
    feats["halfmove_clock"] = board.halfmove_clock
    timings["halfmove_clock_s"] = perf_counter() - t0

    # -------------------------
    # More engineered static
    # -------------------------

    t0 = perf_counter()
    num_captures, num_checks, num_promotions = count_legal_move_types(board)
    feats["num_captures"] = num_captures
    feats["num_checks"] = num_checks
    feats["num_promotions"] = num_promotions
    timings["move_type_breakdown_s"] = perf_counter() - t0

    t0 = perf_counter()
    mobility = mobility_by_piece_type(board, board.turn)
    feats.update(mobility)
    timings["mobility_by_piece_type_s"] = perf_counter() - t0

    t0 = perf_counter()
    feats["num_attackers_on_king"] = num_attackers_on_enemy_king(board)
    timings["num_attackers_on_king_s"] = perf_counter() - t0

    t0 = perf_counter()
    feats["num_pinned_pieces"] = num_pinned_pieces(board, board.turn)
    timings["num_pinned_pieces_s"] = perf_counter() - t0

    # Total overhead for controller static feature extraction
    timings["total_static_feature_overhead_s"] = sum(timings.values())

    return feats, timings


# =========================================================
# Main dataset build
# =========================================================

def build_static_dataset(input_csv: str = INPUT_CSV, timed_output_csv: str = TIMED_OUTPUT_CSV, output_csv: str = OUTPUT_CSV) -> None:
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
    out_df = out_df.loc[:, ~out_df.columns.str.endswith('_s')]
    out_df.to_csv(output_csv, index=False)

    print("Rows:", len(out_df))
    print(f"Wall-clock total build time: {total_elapsed:.4f}s")


        # Compute averages first
    avg_timings = {
        k: timing_sums[k] / max(timing_counts[k], 1)
        for k in timing_sums
    }
    # Sort by value (descending)
    sorted_timings = sorted(avg_timings.items(), key=lambda x: x[1], reverse=True)

    print("\n=== Average per-position timing (sorted DESC) ===")
    for k, avg in sorted_timings:
        print(f"{k:35s}: {avg*1e6:8.2f} µs")

    if timing_counts["total_static_feature_overhead_s"] > 0:
        avg_overhead = timing_sums["total_static_feature_overhead_s"] / timing_counts["total_static_feature_overhead_s"]
        print("\n=== Controller overhead summary ===")
        print(f"Average static controller overhead per position: {avg_overhead:.8f}s")
        print(f"Average static controller overhead per 1000 positions: {avg_overhead * 1000:.4f}s")


if __name__ == "__main__":
    build_static_dataset()