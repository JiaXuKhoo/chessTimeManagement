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

PROBE_NODES = 5000
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

def extract_score_info(score_obj: Optional[chess.engine.PovScore]) -> Tuple[Optional[str], Optional[int]]:
    if score_obj is None:
        return None, None

    white_score = score_obj.white()

    if white_score.is_mate():
        mate_val = white_score.mate()
        return "mate", mate_val if mate_val is not None else None

    cp = white_score.score()
    return "cp", cp if cp is not None else None


def score_to_cp_for_gap(score_obj: Optional[chess.engine.PovScore]) -> Optional[int]:
    """
    Only use centipawn values for best-second gap.
    If mate, return None.
    """
    if score_obj is None:
        return None

    white_score = score_obj.white()
    if white_score.is_mate():
        return None

    return white_score.score()


# =========================================================
# Probe extraction
# =========================================================

def extract_probe_features_timed(
    engine: chess.engine.SimpleEngine,
    fen: str,
    probe_nodes: int = PROBE_NODES
) -> Tuple[Dict, Dict[str, float]]:
    timings: Dict[str, float] = {}
    feats: Dict = {}

    # Board creation
    t0 = perf_counter()
    board = chess.Board(fen)
    timings["probe_board_construction_s"] = perf_counter() - t0

    # Probe analyse
    t0 = perf_counter()
    infos = engine.analyse(
        board,
        chess.engine.Limit(nodes=probe_nodes),
        multipv=2
    )
    timings["probe_engine_analyse_s"] = perf_counter() - t0

    # Normalize result shape
    t0 = perf_counter()
    if isinstance(infos, list):
        pv1 = infos[0] if len(infos) >= 1 else {}
        pv2 = infos[1] if len(infos) >= 2 else {}
    else:
        pv1 = infos
        pv2 = {}
    timings["probe_result_unpack_s"] = perf_counter() - t0

    # Score features
    t0 = perf_counter()
    score_type, score_val = extract_score_info(pv1.get("score"))
    feats["probe_score_type"] = score_type
    feats["probe_score_cp"] = score_val if score_type == "cp" else None
    feats["probe_abs_score_cp"] = abs(score_val) if score_type == "cp" and score_val is not None else None
    feats["probe_is_mate"] = int(score_type == "mate")
    feats["probe_mate_value"] = score_val if score_type == "mate" else None
    timings["probe_score_features_s"] = perf_counter() - t0

    # Depth features
    t0 = perf_counter()
    feats["probe_depth"] = pv1.get("depth")
    feats["probe_seldepth"] = pv1.get("seldepth")
    timings["probe_depth_features_s"] = perf_counter() - t0

    # Gap feature from MultiPV=2
    t0 = perf_counter()
    cp1 = score_to_cp_for_gap(pv1.get("score"))
    cp2 = score_to_cp_for_gap(pv2.get("score")) if pv2 else None

    if cp1 is not None and cp2 is not None:
        feats["probe_best_second_gap"] = cp1 - cp2
    else:
        feats["probe_best_second_gap"] = 0
    timings["probe_best_second_gap_s"] = perf_counter() - t0

    # Optional bookkeeping
    t0 = perf_counter()
    feats["probe_nodes_requested"] = probe_nodes
    feats["probe_nodes_reported"] = pv1.get("nodes")
    timings["probe_bookkeeping_s"] = perf_counter() - t0

    timings["total_probe_feature_overhead_s"] = sum(timings.values())

    return feats, timings


# =========================================================
# Main dataset build
# =========================================================

def build_probe_dataset(input_csv: str = INPUT_CSV, output_csv: str = OUTPUT_CSV) -> None:
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
                feats, timings = extract_probe_features_timed(engine, fen, probe_nodes=PROBE_NODES)

                out = dict(row_dict)   # keep static features + label if already present
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

    # Average timings
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