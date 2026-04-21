from time import perf_counter
from collections import defaultdict
from typing import Dict, Tuple, Optional, List

import pandas as pd
import chess
import chess.engine

# =========================================================
# Config
# =========================================================

STOCKFISH_PATH = r"C:\Users\khoo\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
INPUT_CSV = "dataset_static_tol20.csv"
OUTPUT_CSV = "dataset_probe_tol20.csv"

SMALL_PROBE_NODES = 2_000
LARGE_PROBE_NODES = 5_000

HASH_MB = 128
THREADS = 1

# =========================================================
# Engine helpers
# =========================================================

def open_engine() -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({
        "Threads": THREADS,
        "Hash": HASH_MB,
    })
    return engine


def close_engine(engine: Optional[chess.engine.SimpleEngine]) -> None:
    if engine is not None:
        try:
            engine.quit()
        except Exception:
            pass


# =========================================================
# Score helpers
# =========================================================

def extract_score_info(
    score_obj: Optional[chess.engine.PovScore]
) -> Tuple[Optional[str], Optional[int]]:
    if score_obj is None:
        return None, None

    white_score = score_obj.white()

    if white_score.is_mate():
        mate_val = white_score.mate()
        return "mate", mate_val if mate_val is not None else None

    cp = white_score.score()
    return "cp", cp if cp is not None else None


def score_to_cp(
    score_obj: Optional[chess.engine.PovScore]
) -> Optional[int]:
    """
    Return centipawn score if available; if mate, return None.
    """
    if score_obj is None:
        return None

    white_score = score_obj.white()
    if white_score.is_mate():
        return None

    return white_score.score()


def sign_of_cp(x: Optional[int]) -> int:
    if x is None:
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


# =========================================================
# Probe helpers
# =========================================================

def unpack_probe_infos(infos) -> Tuple[Dict, Dict]:
    """
    Normalize MultiPV output into (pv1, pv2).
    """
    if isinstance(infos, list):
        pv1 = infos[0] if len(infos) >= 1 else {}
        pv2 = infos[1] if len(infos) >= 2 else {}
    else:
        pv1 = infos
        pv2 = {}
    return pv1, pv2


def extract_top_move_uci(pv_info: Dict) -> Optional[str]:
    pv = pv_info.get("pv", [])
    return pv[0].uci() if pv else None


def best_second_gap_cp(pv1: Dict, pv2: Dict) -> int:
    cp1 = score_to_cp(pv1.get("score"))
    cp2 = score_to_cp(pv2.get("score")) if pv2 else None

    if cp1 is not None and cp2 is not None:
        return cp1 - cp2
    return 0


# =========================================================
# Probe extraction
# =========================================================

def extract_probe_features_timed(
    engine: chess.engine.SimpleEngine,
    fen: str,
    small_probe_nodes: int = SMALL_PROBE_NODES,
    large_probe_nodes: int = LARGE_PROBE_NODES
) -> Tuple[Dict, Dict[str, float]]:
    """
    Uses the SAME engine for 2k -> 5k probing.
    This means:
    - the 5k probe naturally follows the earlier 2k search
    - TT warming is preserved inside the staged probe process
    """
    timings: Dict[str, float] = {}
    feats: Dict = {}

    # Board creation
    t0 = perf_counter()
    board = chess.Board(fen)
    timings["probe_board_construction_s"] = perf_counter() - t0

    # -------------------------
    # Small probe (2k)
    # -------------------------
    t0 = perf_counter()
    infos_small = engine.analyse(
        board,
        chess.engine.Limit(nodes=small_probe_nodes),
        multipv=2
    )
    timings["probe_small_engine_analyse_s"] = perf_counter() - t0

    # -------------------------
    # large probe (5k) on same engine
    # -------------------------
    t0 = perf_counter()
    infos_large = engine.analyse(
        board,
        chess.engine.Limit(nodes=large_probe_nodes),
        multipv=2
    )
    timings["probe_large_engine_analyse_s"] = perf_counter() - t0

    # Unpack both
    t0 = perf_counter()
    small_pv1, small_pv2 = unpack_probe_infos(infos_small)
    large_pv1, large_pv2 = unpack_probe_infos(infos_large)
    timings["probe_result_unpack_s"] = perf_counter() - t0

    # =====================================================
    # Probed features from large probe (5k)
    # =====================================================
    t0 = perf_counter()

    large_score_type, large_score_val = extract_score_info(large_pv1.get("score"))

    feats["probe_score_cp"] = large_score_val if large_score_type == "cp" else None
    feats["probe_abs_score_cp"] = abs(large_score_val) if large_score_type == "cp" and large_score_val is not None else None
    feats["probe_is_mate"] = int(large_score_type == "mate")
    feats["probe_depth"] = large_pv1.get("depth")
    feats["probe_seldepth"] = large_pv1.get("seldepth")
    feats["probe_best_second_gap"] = best_second_gap_cp(large_pv1, large_pv2)

    timings["probe_primary_features_s"] = perf_counter() - t0

    # =====================================================
    # 2-search Features
    # =====================================================
    t0 = perf_counter()

    small_cp = score_to_cp(small_pv1.get("score"))
    large_cp = score_to_cp(large_pv1.get("score"))

    small_gap = best_second_gap_cp(small_pv1, small_pv2)
    large_gap = feats["probe_best_second_gap"]

    small_top_move = extract_top_move_uci(small_pv1)
    large_top_move = extract_top_move_uci(large_pv1)

    feats["probe_score_delta_small"] = abs(large_cp - small_cp) if small_cp is not None and large_cp is not None else 0
    feats["probe_gap_delta_small"] = abs(large_gap - small_gap)
    feats["probe_sign_flip"] = int(sign_of_cp(small_cp) != sign_of_cp(large_cp))
    feats["probe_top_move_changed"] = int(
        small_top_move is not None and large_top_move is not None and small_top_move != large_top_move
    )

    timings["probe_stability_features_s"] = perf_counter() - t0

    timings["total_probe_feature_overhead_s"] = sum(timings.values())

    return feats, timings


# =========================================================
# Main dataset build
# =========================================================

def build_probe_dataset(
    input_csv: str = INPUT_CSV,
    output_csv: str = OUTPUT_CSV
) -> None:
    df = pd.read_csv(input_csv)

    if "fen" not in df.columns:
        raise ValueError("Input CSV must contain a 'fen' column.")

    rows: List[Dict] = []
    timing_sums = defaultdict(float)
    timing_counts = defaultdict(int)

    engine = None
    start_all = perf_counter()

    try:
        engine = open_engine()

        total = len(df)
        for i, row in enumerate(df.itertuples(index=False), start=1):
            fen = row.fen
            row_dict = row._asdict()

            try:
                feats, timings = extract_probe_features_timed(
                    engine,
                    fen,
                    small_probe_nodes=SMALL_PROBE_NODES,
                    large_probe_nodes=LARGE_PROBE_NODES
                )

                out = dict(row_dict)   # preserve static features + label_bucket + tau_cp
                out.update(feats)
                out.update(timings)
                rows.append(out)

                for k, v in timings.items():
                    timing_sums[k] += v
                    timing_counts[k] += 1

            except Exception as e:
                out = dict(row_dict)
                out["probe_error"] = str(e)
                rows.append(out)

            if i % 1000 == 0:
                print(f"Processed {i}/{total}")

    finally:
        close_engine(engine)

    total_elapsed = perf_counter() - start_all

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)

    print(f"\nSaved: {output_csv}")
    print(f"Rows: {len(out_df)}")
    print(f"Wall-clock total build time: {total_elapsed:.4f}s")

    avg_timings = {
        k: timing_sums[k] / max(timing_counts[k], 1)
        for k in timing_sums
    }

    sorted_timings = sorted(avg_timings.items(), key=lambda x: x[1], reverse=True)

    print("\n=== Average per-position probe timing (sorted DESC) ===")
    for k, avg in sorted_timings:
        print(f"{k:35s}: {avg:.8f}s ({avg * 1e6:9.2f} µs)")

    if "total_probe_feature_overhead_s" in avg_timings:
        avg_overhead = avg_timings["total_probe_feature_overhead_s"]
        print("\n=== Probe controller overhead summary ===")
        print(f"Average probe overhead per position: {avg_overhead:.8f}s")
        print(f"Average probe overhead per 1000 positions: {avg_overhead * 1000:.4f}s")


if __name__ == "__main__":
    build_probe_dataset()